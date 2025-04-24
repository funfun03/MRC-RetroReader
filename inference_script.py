import json
import datasets
from retro_reader import RetroReader

# Load dev set
dataset = datasets.load_dataset("squad_v2", split="validation")

# Load model
reader = RetroReader.load(
    config_file="/kaggle/working/MRC-RetroReader/configs/inference_en_roberta.yaml",
    device="cuda"
)

# Tạo predictions
preds = {}
for example in dataset:
    qid = example["id"]
    context = example["context"]
    question = example["question"]
    outputs = reader(query=question, context=context)
    answer = outputs[0].get("id-01", "")
    preds[qid] = answer if answer else ""

# Lưu predictions
with open("predictions.json", "w") as f:
    json.dump(preds, f)
