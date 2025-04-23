import os 
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

from huggingface_hub import login
# login(token="hf_xxx", add_to_git_credential=True)

from typing import Union, Any, Dict
import argparse
import datasets
from transformers.utils import logging, check_min_version
from transformers.utils.versions import require_version

from retro_reader import RetroReader
from retro_reader.constants import EXAMPLE_FEATURES
import torch

check_min_version("4.13.0.dev0")
require_version("datasets>=1.8.0")

logger = logging.get_logger(__name__)

def schema_integrate(example) -> Union[Dict, Any]:
    title = example["title"]
    question = example["question"]
    context = example["context"]
    guid = example["id"]
    classtype = [""] * len(title)
    dataset_name = source = ["squad_v2"] * len(title)
    answers, is_impossible = [], []
    for answer_examples in example["answers"]:
        if answer_examples["text"]:
            answers.append(answer_examples)
            is_impossible.append(False)
        else:
            answers.append({"text": [""], "answer_start": [-1]})
            is_impossible.append(True)
    return {
        "guid": guid,
        "question": question,
        "context": context,
        "answers": answers,
        "title": title,
        "classtype": classtype,
        "source": source,
        "is_impossible": is_impossible,
        "dataset": dataset_name,
    }

def data_aug_for_multiple_answers(examples) -> Union[Dict, Any]:
    result = {key: [] for key in examples.keys()}

    def update(i, answers=None):
        for key in result.keys():
            if key == "answers" and answers is not None:
                result[key].append(answers)
            else:
                result[key].append(examples[key][i])

    for i, (answers, unanswerable) in enumerate(zip(examples["answers"], examples["is_impossible"])):
        answerable = not unanswerable
        assert len(answers["text"]) == len(answers["answer_start"]) or answers["answer_start"][0] == -1
        if answerable and len(answers["text"]) > 1:
            for n_ans in range(len(answers["text"])):
                ans = {
                    "text": [answers["text"][n_ans]],
                    "answer_start": [answers["answer_start"][n_ans]],
                }
                update(i, ans)
        elif not answerable:
            update(i, {"text": [], "answer_start": []})
        else:
            update(i)

    return result

def main(args):
    print("Loading SQuAD v2.0 dataset ...")
    squad_v2 = datasets.load_dataset("squad_v2")

    if args.debug:
        squad_v2["train"] = squad_v2["train"].select(range(5))
        squad_v2["validation"] = squad_v2["validation"].select(range(5))

    print("Integrating into the schema used in this library ...")
    squad_v2 = squad_v2.map(
        schema_integrate,
        batched=True,
        remove_columns=squad_v2.column_names["train"],
        features=EXAMPLE_FEATURES,
    )
    num_unanswerable_train = sum(squad_v2["train"]["is_impossible"])
    num_unanswerable_valid = sum(squad_v2["validation"]["is_impossible"])
    logger.warning(f"Number of unanswerable sample for SQuAD v2.0 train dataset: {num_unanswerable_train}")
    logger.warning(f"Number of unanswerable sample for SQuAD v2.0 validation dataset: {num_unanswerable_valid}")

    print("Data augmentation for multiple answers ...")
    squad_v2_train = squad_v2["train"].map(
        data_aug_for_multiple_answers,
        batched=True,
        batch_size=args.batch_size,
        num_proc=5,
    )
    squad_v2 = datasets.DatasetDict({
        "train": squad_v2_train,
        "validation": squad_v2["validation"]
    })

    print("Loading Retro Reader ...")
    retro_reader = RetroReader.load(
        train_examples=squad_v2["train"],
        eval_examples=squad_v2["validation"],
        config_file=args.configs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Resume from checkpoint by updating model weights from path manually
    if args.resume_checkpoint:
        retro_reader.model.load_state_dict(torch.load(os.path.join(args.resume_checkpoint, 'pytorch_model.bin')))

    print("Training ...")
    retro_reader.train(module=args.module)
    logger.warning("Train retrospective reader Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", "-c", type=str, default="configs/train_distilbert.yaml", help="config file path")
    parser.add_argument("--batch_size", "-b", type=int, default=1024, help="batch size")
    parser.add_argument("--resume_checkpoint", "-r", type=str, default=None, help="resume checkpoint path")
    parser.add_argument("--module", "-m", type=str, default="all", choices=["all", "sketch", "intensive"], help="module to train")
    parser.add_argument("--debug", "-d", action="store_true", help="debug mode")
    args = parser.parse_args()
    main(args)
