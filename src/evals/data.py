# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0


"""from https://arxiv.org/pdf/1905.00537
For classification tasks with sentence-pair inputs (BoolQ, CB, RTE, WiC), we concatenate the
sentences with a [SEP] token, feed the fused input to BERT, and use a logistic regression classifier
that sees the representation corresponding to [CLS]. For WiC, we also concatenate the representation
of the marked word. For COPA, MultiRC, and ReCoRD, for each answer choice, we similarly
concatenate the context with that answer choice and feed the resulting sequence into BERT to produce
an answer representation. For COPA, we project these representations into a scalar, and take as the
answer the choice with the highest associated scalar. For MultiRC, because each question can have
more than one correct answer, we feed each answer representation into a logistic regression classifier.
For ReCoRD, we also evaluate the probability of each candidate independent of other candidates,
and take the most likely candidate as the model’s prediction. For WSC, which is a span-based task,
we use a model inspired by Tenney et al. (2019). Given the BERT representation for each word in the
original sentence, we get span representations of the pronoun and noun phrase via a self-attention
span-pooling operator (Lee et al., 2017), before feeding it into a logistic regression classifier.
"""

import logging
import torch

from composer.utils import MissingConditionalImportError, dist

_glue_task_column_names = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

_superglue_task_column_names = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "copa": ("premise", "choice1", "choice2", "question"),
    "multirc": ("paragraph", "question", "answer"),
    # "record": ("question1", "question2"), ['passage', 'query', 'entities', 'entity_spans', 'answers', 'idx'
    "rte": ("premise", "hypothesis"),
    "wic": (
        "sentence1",
        "sentence2",
    ),  #'word','sentence1'  'sentence2',  'start1',  'start2',  'end1',  'end2',
    # "wsc": ("sentence1", "sentence2"), #'text','span1_index',  'span2_index',  'span1_text',  'span2_text',
    # "wsc.fixed": ("sentence1", "sentence2"), #'text','span1_index',  'span2_index',  'span1_text',  'span2_text',
    # "axb": ("sentence1", "sentence2"),
    # "axg": ("premise", "hypothesis"),
}

log = logging.getLogger(__name__)


def create_eval_dataset(
    task: str,
    tokenizer_name: str,
    split: str, # train, validation, test
    dataset_name: str,
    max_seq_length: int = 256,
    max_retries: int = 10,
    num_workers: int = 0,
    dataset_subset: str = None,
    task_column_names: dict = _glue_task_column_names,
    tokenize_fn_factory: callable = None,
):
    try:
        import datasets
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group="nlp", conda_package="transformers"
        ) from e

    if task not in task_column_names:
        raise ValueError(f"task ({task}) must be one of {task_column_names.keys()}")

    if (max_seq_length % 8) != 0:
        log.warning(
            "For performance, a max_seq_length as a multiple of 8 is recommended."
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)  # type: ignore (thirdparty)

    log.info(f"Loading {task.upper()} on rank {dist.get_global_rank()}")
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        dataset_name,
        dataset_subset if dataset_subset is not None else task, # COMMENT OUT IF PERSONAL DATASET DOES NOT HAVE SUBSET
        split=split,
        download_config=download_config,
    )

    log.info(f"Starting tokenization by preprocessing over {num_workers} threads!")
    text_column_names = task_column_names[task]

    # if tokenize_fn_factory is None:
    #     # Calling the BERT tokenizer in this way will insert [SEP] between the
    #     # inputs, e.g. "[CLS] text [SEP] text_pair [SEP]". Without NSP, BERT is
    #     # not exposed to sequences with two [SEP] tokens during pretraining,
    #     # but finetuning on MNLI before finetuning on smaller datasets can help
    #     # the model get used to this.
    #     tokenize_fn_factory = lambda tokenizer, max_seq_length: lambda inp: tokenizer(
    #         text=inp[text_column_names[0]],
    #         text_pair=(
    #             inp[text_column_names[1]] if text_column_names[1] in inp else None
    #         ),
    #         padding="max_length",
    #         max_length=max_seq_length,
    #         truncation=True,
    #     )
    #     # ).update({"label": [inp["label"]] if "label" in inp else None})

    def tokenize_fn_factory(tokenizer, max_seq_length):
        def tokenizer_fn(inp):
            # print("chosen", inp[text_column_names[0]])
            # print("rejected", inp[text_column_names[1]])

            chosen = tokenizer(
                text=inp[text_column_names[0]],
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
            )

            rejected = tokenizer(
                text=inp[text_column_names[1]],
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
            )

            ret_dict = {
            "input_ids": chosen["input_ids"] + rejected["input_ids"],
            "token_type_ids": chosen["token_type_ids"] + rejected["token_type_ids"],
            "attention_mask": chosen["attention_mask"] + rejected["attention_mask"],
            "label": [1] * len(chosen["input_ids"]) + [0] * len(rejected["input_ids"]),
            # "original_dataset": inp["og_dataset"] + inp["og_dataset"],
            }

            # print(chosen.shape)
            # print(rejected.shape)

            # ret_dict = {
            #     "input_ids": torch.cat((chosen["input_ids"], rejected["input_ids"]), dim=0),
            #     "label": [1] * len(chosen) + [0] * len(rejected),
            # }
            
            return ret_dict

        return tokenizer_fn

    columns_to_remove = [i for i in text_column_names if i is not None]
    # columns_to_remove = [i for i in text_column_names if i not in {None, 'label'}]
    # columns_to_remove = ["chosen", "rejected"]

    # if not tokenizer.chat_template:
    #     llama_chat_template = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").chat_template
    #     tokenizer.chat_template = llama_chat_template

    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.map(
        tokenize_fn_factory(tokenizer, max_seq_length),
        batched=True,
        # batched=False,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        remove_columns=columns_to_remove,
        load_from_cache_file=True,
    )
    return dataset


def create_tok_cls_dataset(    task: str,
    tokenizer_name: str,
    split: str, # train, validation, test
    dataset_name: str,
    max_seq_length: int = 256,
    max_retries: int = 10,
    num_workers: int = 0,
    dataset_subset: str = None,
    task_column_names: dict = _glue_task_column_names,
    tokenize_fn_factory: callable = None,
):
    try:
        import datasets
        import transformers
    except ImportError as e:
        raise MissingConditionalImportError(
            extra_deps_group="nlp", conda_package="transformers"
        ) from e

    if task not in task_column_names:
        raise ValueError(f"task ({task}) must be one of {task_column_names.keys()}")

    if (max_seq_length % 8) != 0:
        log.warning(
            "For performance, a max_seq_length as a multiple of 8 is recommended."
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)  # type: ignore (thirdparty)

    log.info(f"Loading {task.upper()} on rank {dist.get_global_rank()}")
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        dataset_name,
        dataset_subset if dataset_subset is not None else task, # COMMENT OUT IF PERSONAL DATASET DOES NOT HAVE SUBSET
        split=split,
        download_config=download_config,
    )

    log.info(f"Starting tokenization by preprocessing over {num_workers} threads!")
    text_column_names = task_column_names[task]

    def tokenize_fn_factory(tokenizer, max_seq_length):
        def tokenizer_fn(inp):
            # print("chosen", inp[text_column_names[0]])
            # print("rejected", inp[text_column_names[1]])

            chosen = tokenizer(
                text=inp["chosen_labeled"],
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
            )

            rejected = tokenizer(
                text=inp["rejected_labeled"],
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
            )

            chosen_labels = [[-100] * len(example) for example in chosen["input_ids"]]
            rejected_labels = [[-100] * len(example) for example in rejected["input_ids"]]

            cls_token = 50281

            for i, example in enumerate(chosen["input_ids"]):
                for j in range(len(example)):
                    if example[j] == cls_token:
                        chosen_labels[i][j] = 1
    
            for i, example in enumerate(rejected["input_ids"]):
                for j in range(len(example)):
                    if example[j] == cls_token:
                        rejected_labels[i][j] = 0

            ret_dict = {
            "input_ids": chosen["input_ids"] + rejected["input_ids"],
            "token_type_ids": chosen["token_type_ids"] + rejected["token_type_ids"],
            "attention_mask": chosen["attention_mask"] + rejected["attention_mask"],
            "label": chosen_labels + rejected_labels,
            # "original_dataset": inp["og_dataset"] + inp["og_dataset"],
            }

            # print(chosen.shape)
            # print(rejected.shape)

            # ret_dict = {
            #     "input_ids": torch.cat((chosen["input_ids"], rejected["input_ids"]), dim=0),
            #     "label": [1] * len(chosen) + [0] * len(rejected),
            # }
            
            return ret_dict

        return tokenizer_fn
    
    columns_to_remove = [i for i in text_column_names if i is not None]
    # columns_to_remove = [i for i in text_column_names if i not in {None, 'label'}]
    # columns_to_remove = ["chosen", "rejected"]

    # if not tokenizer.chat_template:
    #     llama_chat_template = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").chat_template
    #     tokenizer.chat_template = llama_chat_template

    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.map(
        tokenize_fn_factory(tokenizer, max_seq_length),
        batched=True,
        # batched=False,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        remove_columns=columns_to_remove,
        load_from_cache_file=True,
    )
    return dataset


def create_glue_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs, dataset_name="glue", task_column_names=_glue_task_column_names
    )


def create_superglue_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs,
        dataset_name="aps/super_glue",
        task_column_names=_superglue_task_column_names,
    )


def create_swag_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs,
        dataset_name="swag",
        dataset_subset="regular",
        task_column_names={
            "swag": ("sent1", "sent2", "ending0", "ending1", "ending2", "ending3")
        },
    )

def create_eurlex_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs,
        dataset_name="coastalcph/lex_glue",
        dataset_subset="eurlex",
        task_column_names={"coastalcph/lex_glue": ("text",)},
    )

def create_ultrafeedback_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs,
        dataset_name="rbiswasfc/ultrafeedback-binary-classification",
        dataset_subset="",
        task_column_names={"rbiswasfc/ultrafeedback-binary-classification": ("prompt", "response_a", "response_b")},
    )

def create_mlmmlu_dataset(**kwargs):
    dataset_subset = kwargs.pop("dataset_subset")

    if dataset_subset in ['Amateur', 'Semipro']:
        task_column_names= ("question", "options", "answer", "category", "cot_content", "src", "question_id", "llama_pred", "llama_correct")
    elif dataset_subset in ['Reserve', 'Rookie']:
        task_column_names= ("question", "choices", "category", "question_id", "llama_correct", "id_in_subset")
    else:
        raise NotImplementedError
    
    return create_eval_dataset(
            dataset_name="answerdotai/MLMMLU",
            dataset_subset=dataset_subset,
            task_column_names={"answerdotai/MLMMLU": task_column_names},
            **kwargs,
        )

def create_hhrlhf_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs,
        dataset_name="Anthropic/hh-rlhf",
        dataset_subset="",
        task_column_names={"Anthropic/hh-rlhf": ("chosen", "rejected")}
    )

def create_skywork_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs,
        dataset_name="sarahpann/processed_skywork",
        dataset_subset="",
        task_column_names={"sarahpann/processed_skywork": ("chosen", "rejected")}
    )

def create_helpsteer_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs,
        dataset_name="sarahpann/formatted_help_steer_2",
        dataset_subset="",
        task_column_names={"sarahpann/formatted_help_steer_2": ("chosen", "rejected")}
    )

def create_reward_bench_dataset(**kwargs):
    return create_eval_dataset(
        **kwargs,
        dataset_name="sarahpann/reward_bench_processed",
        dataset_subset="",
        task_column_names={"sarahpann/reward_bench_processed": ("chosen", "rejected","og_dataset")}
    )

def create_tok_cls_skywork_dataset(**kwargs):
    return create_tok_cls_dataset(
        **kwargs,
        dataset_name="sarahpann/processed_skywork_labeled",
        dataset_subset="",
        task_column_names={"sarahpann/processed_skywork_labeled": ('chosen', 'rejected', 'chosen_labeled', 'rejected_labeled', 'num_chosen_labels', 'num_rejected_labels')}
    )

def create_tok_cls_reward_bench_dataset(**kwargs):
    return create_tok_cls_dataset(
        **kwargs,
        dataset_name="sarahpann/reward_bench_processed_labeled",
        dataset_subset="",
        task_column_names={"sarahpann/reward_bench_processed_labeled": ('chosen', 'rejected', 'og_dataset', 'chosen_labeled', 'rejected_labeled', 'num_chosen_labels', 'num_rejected_labels')}
    )