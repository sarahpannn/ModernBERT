# Copyright 2024 **AUTHORS_TODO**
# License: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a FlexBERT wrapper around a :class:`.ComposerTransformer`."""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor
import torch.nn as nn

# Add src folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from omegaconf import DictConfig, OmegaConf

import bert_layers as bert_layers_module
import src.bert_layers.configuration_bert as configuration_bert_module
import transformers
from transformers.modeling_outputs import MaskedLMOutput, ModelOutput

import peft
from peft import LoraConfig

from composer.metrics.nlp import BinaryF1Score, LanguageCrossEntropy, MaskedAccuracy
from composer.models.huggingface import HuggingFaceModel
from composer.devices import DeviceCPU

from torchmetrics import MeanSquaredError, Metric
from torchmetrics.classification.accuracy import MulticlassAccuracy, MultilabelAccuracy
from torchmetrics.classification.matthews_corrcoef import MatthewsCorrCoef
from torchmetrics.regression.spearman import SpearmanCorrCoef

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = None

all = ["create_flex_bert_mlm", "create_flex_bert_classification"]


# we want the efficent versions to have the same name as the TorchMetrics' name
def rename_class(new_name):
    def class_renamer(cls):
        cls.__name__ = new_name
        return cls

    return class_renamer


@rename_class("LanguageCrossEntropy")
class FALanguageCrossEntropy(LanguageCrossEntropy):
    """Torchmetric that computes cross entropy on language modeling outputs using flash_attn's Cross Entropy.

    Adds metric state variables:
        sum_loss (float): The sum of the per-example loss in the batch.
        total_items (float): The number of batches to average across.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
        ignore_index (int, optional): The class index to ignore. Default: ``-100``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False, ignore_index: int = -100):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        if CrossEntropyLoss is None:
            raise ImportError("flash_attn is not installed. Please install flash_attn to use FALanguageCrossEntropy.")

        self.ignore_index = ignore_index
        self.loss_fn = CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")


@rename_class("LanguageCrossEntropy")
class EfficientCrossEntropy(Metric):
    """Torchmetric that grabs the precomputed ce_loss value from the model outputs"""

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: Tensor) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            loss (~torch.Tensor): A Tensor of loss values to compare against.
        """
        self.sum_loss += loss
        self.total_items += 1

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_items  # type: ignore (third-party)


@rename_class("ZLoss")
class EfficientZLoss(Metric):
    """Torchmetric that grabs the precomputed z_loss value from the model outputs"""

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_items", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: Tensor) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            loss (~torch.Tensor): A Tensor of loss values to compare against.
        """
        self.sum_loss += loss
        self.total_items += 1

    def compute(self) -> Tensor:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        # Return average loss over entire dataset
        return self.sum_loss / self.total_items  # type: ignore (third-party)


class EfficientHuggingFaceModel(HuggingFaceModel):
    def eval_forward(self, batch, outputs: Optional[Any] = None):
        outputs = self.forward(batch) if outputs is None else outputs
        self.labels = batch.pop("labels")
        return outputs

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> Dict:
        if metric.device.type == "cpu":
            self.labels = DeviceCPU().batch_to_device(self.labels)

        if getattr(metric, "needs_batch", False):
            raise ValueError(f"Unsupported metric {metric=}")

        if getattr(outputs, "ce_loss", False) and isinstance(metric, EfficientCrossEntropy):
            metric_result = metric.update(outputs["ce_loss"])
        elif getattr(outputs, "z_loss", False) and isinstance(metric, EfficientZLoss):
            metric_result = metric.update(outputs["z_loss"])
        elif isinstance(metric, EfficientCrossEntropy):
            metric_result = metric.update(outputs["loss"])
        else:
            metric_result = metric.update(outputs["logits"], outputs.get("labels", self.labels))

        if metric_result is not None:
            # Add the metric name once for each datapoint in the batch
            metric_result["metric_name"] = [metric.__class__.__name__ for _ in range(0, batch["input_ids"].shape[0])]
        else:
            metric_result = {}
        return metric_result
    
@dataclass
class MaskedLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    indices: Optional[torch.LongTensor] = None
    cu_seqlens: Optional[torch.LongTensor] = None
    max_seqlen: Optional[int] = None
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None
    labels: Optional[torch.LongTensor] = None


def create_modern_bert_mixed_mlm(
    pretrained_model_name: str = "answerdotai/ModernBert-large",
    checkpoint_dict: dict = None,
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None,
    recompute_metric_loss: Optional[bool] = False,
    disable_train_metrics: Optional[bool] = False,
    use_dora: Optional[bool] = False,
):
    from transformers import AutoModelForMaskedLM
    from collections import OrderedDict

    base_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name,
                                                        config=model_config)
    
    new_state_dict = base_model.state_dict().copy()

    for param in new_state_dict:
        new_weight = torch.zeros_like(new_state_dict[param])
        new_state_dict[param] = new_weight

    for key in checkpoint_dict:
        model = AutoModelForMaskedLM.from_pretrained(checkpoint_dict[key],
                                                     config=model_config)
        state_dict = model.state_dict()
        for param in state_dict:
            new_state_dict[param] += state_dict[param] / len(checkpoint_dict)
    

    base_model.load_state_dict(new_state_dict)

    return base_model


def create_og_bert_mlm(
    pretrained_model_name: str = "bert-base-uncased",
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None,
    recompute_metric_loss: Optional[bool] = False,
    disable_train_metrics: Optional[bool] = False,
    use_dora: Optional[bool] = False,
    r_dim: Optional[int] = 128,
    mixed_mlm: Optional[bool] = False,
    checkpoint_dict: Optional[dict] = None,
):
    model = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_model_name,
                                                                config=model_config)
    
    if mixed_mlm:
        assert False, "MIXED MLM NOT YET SUPPORTED FOR OG BERT"

    if use_dora:
        print(f"ADDING DORA ADAPTORS WITH DIM {r_dim}")
        linear_layers = ["query", "key", "value", "dense", "decoder"]

        dora_config = LoraConfig(use_dora=True, 
                                 target_modules=linear_layers,
                                 r=r_dim,)

        model = peft.get_peft_model(model, dora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    metrics = [MaskedAccuracy(ignore_index=-100)]

    if recompute_metric_loss or model_config["loss_function"] not in ["fa_cross_entropy", 
                                                                      "cross_entropy"]:
        if CrossEntropyLoss is not None:
            metrics = [FALanguageCrossEntropy(ignore_index=-100)] + metrics
        else:
            metrics = [LanguageCrossEntropy(ignore_index=-100)] + metrics
    else:
        metrics = [EfficientCrossEntropy()] + metrics
    if model_config.get("loss_kwargs", {}).get("return_z_loss", False):
        metrics += [EfficientZLoss()]

    eval_metrics = copy.deepcopy(metrics)
    if disable_train_metrics:
        metrics = None


    hf_model = EfficientHuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=metrics,
        eval_metrics=eval_metrics,
        allow_embedding_resizing=True,
    )

    return hf_model


def create_modern_bert_mlm(
    pretrained_model_name: str = "answerdotai/ModernBERT-large",
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None,
    recompute_metric_loss: Optional[bool] = False,
    disable_train_metrics: Optional[bool] = False,
    use_dora: Optional[bool] = False,
    mixed_mlm: Optional[bool] = False,
    checkpoint_dict: Optional[dict] = None,
):

    model = transformers.AutoModelForMaskedLM.from_pretrained(pretrained_checkpoint if pretrained_checkpoint else pretrained_model_name, 
                                                          config=model_config,
                                                          attn_implementation='flash_attention_2')
    
    
    # print(model.model.state_dict())
    
    if mixed_mlm:
        print("MIXING MLMS")
        model = create_modern_bert_mixed_mlm(pretrained_model_name=pretrained_model_name,
                                     checkpoint_dict=checkpoint_dict,
                                     model_config=model_config,
                                     tokenizer_name=tokenizer_name,
                                     gradient_checkpointing=gradient_checkpointing,
                                     pretrained_checkpoint=pretrained_checkpoint,
                                     recompute_metric_loss=recompute_metric_loss,
                                     disable_train_metrics=disable_train_metrics,
                                     use_dora=use_dora)
        

    if use_dora:
        print("ADDING DORA ADAPTORS")
        linear_layers = ["Wqkv", "Wi", "Wo", "dense", "decoder"]

        dora_config = LoraConfig(use_dora=True, 
                                 target_modules=linear_layers,
                                 r=128,)

        model = peft.get_peft_model(model, dora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        label_copy = labels.clone()
        # label_copy[:, 2:] = -100

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self._maybe_set_compile()

        # if self.config._attn_implementation == "flash_attention_2":
        #     if indices is None and cu_seqlens is None and max_seqlen is None:
        #         batch_size, seq_len = input_ids.shape[:2]
        #         if attention_mask is None:
        #             attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.bool)
        #         with torch.no_grad():
        #             input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
        #                 inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
        #             )

        if use_dora:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=return_dict,
            )
            return MaskedLMOutput(
                loss=outputs.loss,
                logits=outputs.logits,
                hidden_states=outputs.hidden_states,
                labels=label_copy,
            )

        # assert if any of them are inf
        assert not torch.isinf(input_ids).any()
        assert not torch.isinf(attention_mask).any()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            label_copy = label_copy.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]
            label_copy = label_copy[mask_tokens]

        logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.head(last_hidden_state))
        )

        # logits = outputs.logits

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size)

        # if self.config._attn_implementation == "flash_attention_2":
        #     with torch.no_grad():
        #         logits = _pad_modernbert_output(inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            labels=label_copy,
        )
    
    model.forward = forward.__get__(model)

    metrics = [MaskedAccuracy(ignore_index=-100)]

    if recompute_metric_loss or model_config["loss_function"] not in ["fa_cross_entropy", 
                                                                      "cross_entropy"]:
        if CrossEntropyLoss is not None:
            metrics = [FALanguageCrossEntropy(ignore_index=-100)] + metrics
        else:
            metrics = [LanguageCrossEntropy(ignore_index=-100)] + metrics
    else:
        metrics = [EfficientCrossEntropy()] + metrics
    if model_config.get("loss_kwargs", {}).get("return_z_loss", False):
        metrics += [EfficientZLoss()]

    eval_metrics = copy.deepcopy(metrics)
    if disable_train_metrics:
        metrics = None


    hf_model = EfficientHuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=metrics,
        eval_metrics=eval_metrics,
        allow_embedding_resizing=True,
    )

    return hf_model
   

def create_flex_bert_mlm(
    pretrained_model_name: str = "bert-base-uncased",
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None,
    recompute_metric_loss: Optional[bool] = False,
    disable_train_metrics: Optional[bool] = False,
):
    """FlexBERT masked language model based on |:hugging_face:| Transformers.

    For more information, see
    `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a FlexBERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided, the state dictionary
            stored at `pretrained_checkpoint` will be loaded into the model
            after initialization. Default: ``None``.
        disable_train_metrics (bool, optional): Only calculate metrics for 
            validation set when True.
            Default: ``False``.

    .. code-block::

        {
        "_name_or_path": "bert-base-uncased",
        "alibi_starting_size": 512,
        "architectures": ["BertForMaskedLM"],
        "attention_probs_dropout_prob": 0.0,
        "classifier_dropout": null,
        "gradient_checkpointing": false,
        "hidden_act": "silu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.16.0",
        "type_vocab_size": 2,
        "use_cache": true,
        "vocab_size": 30522
        }

    To create a FlexBERT model for Masked Language Model pretraining:

     .. testcode::

         from src.mosaic import create_flex_bert_mlm
         model = create_flex_bert_mlm()
    """
    if not model_config:
        model_config = {}

    if not pretrained_model_name:
        pretrained_model_name = "bert-base-uncased"

    if isinstance(model_config, DictConfig):
        model_config = OmegaConf.to_container(model_config, resolve=True)

    config = configuration_bert_module.FlexBertConfig.from_pretrained(pretrained_model_name, **model_config)

    if "prenorm" in config.bert_layer:
        assert config.final_norm, "Final norm must be used with prenorm attention"
    else:
        assert "postnorm" in config.bert_layer, "config.bert_layer str must contain either prenorm or postnorm"
        assert not config.final_norm, "Final norm should not be used with postnorm attention"

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if pretrained_checkpoint is not None:
        model = bert_layers_module.FlexBertForMaskedLM.from_composer(
            pretrained_checkpoint=pretrained_checkpoint, config=config
        )
    else:
        model = bert_layers_module.FlexBertForMaskedLM(config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

    metrics = [MaskedAccuracy(ignore_index=-100)]

    if recompute_metric_loss or model_config["loss_function"] not in ["fa_cross_entropy", 
                                                                      "cross_entropy"]:
        if CrossEntropyLoss is not None:
            metrics = [FALanguageCrossEntropy(ignore_index=-100)] + metrics
        else:
            metrics = [LanguageCrossEntropy(ignore_index=-100)] + metrics
    else:
        metrics = [EfficientCrossEntropy()] + metrics
    if model_config.get("loss_kwargs", {}).get("return_z_loss", False):
        metrics += [EfficientZLoss()]

    eval_metrics = copy.deepcopy(metrics)
    if disable_train_metrics:
        metrics = None

    hf_model = EfficientHuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        # use_logits=True,
        use_logits=False,
        metrics=metrics,
        eval_metrics=eval_metrics,
        allow_embedding_resizing=True,
    )

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)
    hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model


def create_flex_bert_classification(
    num_labels: int,
    pretrained_model_name: str = "bert-base-uncased",
    model_config: Optional[dict] = None,
    tokenizer_name: Optional[str] = None,
    gradient_checkpointing: Optional[bool] = False,
    pretrained_checkpoint: Optional[str] = None,
    custom_eval_metrics: Optional[list] = [],
    multiple_choice: Optional[bool] = False,
    token_classification: Optional[bool] = False,
):
    """FlexBERT classification model based on |:hugging_face:| Transformers.

    For more information, see `Transformers. <https://huggingface.co/transformers/>`_.

    This function creates a FlexBERT, which includes several throughput
    optimizations not available in |:hugging_face:| BERT as well as
    architecture changes based on ALiBi and Gated Linear Units.

    Args:
        num_labels (int): The number of classes in the classification task.
        pretrained_model_name (str): Name of the Hugging Face model to
            instantiate. This will determine the default model configuration.
            Default: ``bert-base-uncased``.
        model_config (dict): A dictionary of user-specified configurations to
            update/add to the default model configuration.
        tokenizer_name (str, optional): Tokenizer name used to preprocess the
            dataset and validate the models inputs.
        gradient_checkpointing (bool, optional): Use gradient checkpointing.
            Default: ``False``.
        pretrained_checkpoint (str, optional): The pretrained checkpoint to
            initialize the model weights. If provided,
            the state dictionary stored at `pretrained_checkpoint` will be
            loaded into the model after initialization. Default: ``None``.
        custom_eval_metrics (list, optional): Classes of custom metrics to
            evaluate the model. Default: ``[]``.
        multiple_choice (bool, optional): Whether the model is used for
            multiple choice tasks. Default: ``False``.

    .. code-block::
        {
            "_name_or_path": "bert-base-uncased",
            "alibi_starting_size": 512,
            "architectures": ["BertForSequenceClassification"],
            "attention_probs_dropout_prob": 0.0,
            "classifier_dropout": null,
            "gradient_checkpointing": false,
            "hidden_act": "silu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1",
                "2": "LABEL_2"
            },
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1,
                "LABEL_2": 2
            },
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.16.0",
            "type_vocab_size": 2,
            "use_cache": true,
            "vocab_size": 30522
        }

    To create a FlexBERT model for classification:

     .. testcode::
        from flex_bert import create_flex_bert_classification
        model = create_flex_bert_classification(num_labels=3) # if the task has three classes.

    Note:
        This function can be used to construct a BERT model for regression by
        setting ``num_labels == 1``. This will have two noteworthy effects.
        First, it will switch the training loss to :class:`~torch.nn.MSELoss`.
        Second, the returned :class:`.ComposerModel`'s train/validation metrics
        will be :class:`~torchmetrics.MeanSquaredError` and
        :class:`~torchmetrics.SpearmanCorrCoef`. For the classifcation case
        (when ``num_labels > 1``), the training loss is
        :class:`~torch.nn.CrossEntropyLoss`, and the train/validation
        metrics are :class:`~torchmetrics.MulticlassAccuracy` and
        :class:`~torchmetrics.MatthewsCorrCoef`, as well as
        :class:`.BinaryF1Score` if ``num_labels == 2``.
    """
    if not model_config:
        model_config = {}

    # By default, turn off attention dropout in FlexBERT
    # Flash Attention 2 supports dropout in the attention module
    # while our previous Triton Flash Attention layer only works with
    # attention_probs_dropout_prob = 0.
    if "attention_probs_dropout_prob" not in model_config:
        model_config["attention_probs_dropout_prob"] = 0.0

    model_config["num_labels"] = num_labels

    if not pretrained_model_name:
        pretrained_model_name = "bert-base-uncased"

    model_cls = bert_layers_module.FlexBertForSequenceClassification

    if multiple_choice:
        model_cls = bert_layers_module.FlexBertForMultipleChoice

    if token_classification:
        model_cls = bert_layers_module.FlexBertForTokenClassification

    if isinstance(model_config, DictConfig):
        model_config = OmegaConf.to_container(model_config, resolve=True)

    # config = configuration_bert_module.FlexBertConfig.from_pretrained(pretrained_model_name, **model_config)

    # Padding for divisibility by 8
    # if config.vocab_size % 8 != 0:
    #     config.vocab_size += 8 - (config.vocab_size % 8)

    # if pretrained_checkpoint is not None:
    #     model = model_cls.from_composer(pretrained_checkpoint=pretrained_checkpoint, config=config)
    # else:
    #     model = model_cls(config)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, 
                                                                            num_labels=num_labels,
                                                                            config=model_config)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()  # type: ignore

    # setup the tokenizer
    if tokenizer_name:
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name)

    if num_labels == 1:
        # Metrics for a regression model
        metrics = [MeanSquaredError(), SpearmanCorrCoef()]
    else:
        # Metrics for a classification model
        metrics = [
            MulticlassAccuracy(num_classes=num_labels, average="micro", ignore_index=-100),
            MatthewsCorrCoef(task="multiclass", num_classes=num_labels, ignore_index=-100),
        ]
        # if num_labels == 2:
        #     metrics.append(BinaryF1Score())

    if model_config.get("problem_type", "") == "multi_label_classification":
        metrics = [
            MultilabelAccuracy(num_labels=num_labels, average="micro"),
        ]

    hf_model = HuggingFaceModel(
        model=model,
        tokenizer=tokenizer,
        use_logits=True,
        metrics=metrics,
        eval_metrics=[
            *metrics,
            *[metric_cls() for metric_cls in custom_eval_metrics],
        ],
        allow_embedding_resizing=True,
    )

    # Padding for divisibility by 8
    # We have to do it again here because wrapping by HuggingFaceModel changes it
    # if config.vocab_size % 8 != 0:
    #     config.vocab_size += 8 - (config.vocab_size % 8)
    # hf_model.model.resize_token_embeddings(config.vocab_size)

    return hf_model



