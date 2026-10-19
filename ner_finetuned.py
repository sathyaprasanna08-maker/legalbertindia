# ---------------------------------------------------------
# ⚖️  Fine-Tuning InLegalBERT for Legal NER (with Freeze)
# ---------------------------------------------------------

import json
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    EarlyStoppingCallback
)
import numpy as np
from evaluate import load
from sklearn.model_selection import train_test_split
import torch

# ---------------------------------------------------------
# 1️⃣  Load Dataset
# ---------------------------------------------------------
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

dataset_path = "weak_labels.jsonl"
data = load_jsonl(dataset_path)
print(f"✅ Loaded {len(data)} samples from {dataset_path}")

# ---------------------------------------------------------
# 2️⃣  Split into Train / Validation
# ---------------------------------------------------------
train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
ds = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data)
})

# ---------------------------------------------------------
# 3️⃣  Label Mapping
# ---------------------------------------------------------
unique_labels = sorted(list({label for sample in data for label in sample["ner_tags"]}))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"\n🧾 Unique labels: {len(unique_labels)}")
print(unique_labels)

# ---------------------------------------------------------
# 4️⃣  Tokenizer & Model
# ---------------------------------------------------------
MODEL_NAME = "law-ai/InLegalBERT"
OUTPUT_DIR = "./legal_ner_frozen"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# ---------------------------------------------------------
# 🧊 Freeze Encoder Layers
# ---------------------------------------------------------
def freeze_encoder_layers(model, unfreeze_last_n=1):
    """
    Freezes all encoder layers except the last `n`.
    """
    for name, param in model.bert.named_parameters():
        param.requires_grad = False

    if unfreeze_last_n > 0:
        for layer in model.bert.encoder.layer[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"🧩 Unfroze last {unfreeze_last_n} encoder layer(s)")
    else:
        print("❄️ All encoder layers frozen")

freeze_encoder_layers(model, unfreeze_last_n=1)

# ---------------------------------------------------------
# 5️⃣  Tokenization Function
# ---------------------------------------------------------
def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=256
    )
    labels = []
    word_ids = tokenized.word_ids()
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            labels.append(label2id[example["ner_tags"][word_id]])
        else:
            labels.append(label2id[example["ner_tags"][word_id]])
        prev_word_id = word_id
    tokenized["labels"] = labels
    return tokenized

tokenized_ds = ds.map(tokenize_and_align_labels, batched=False)

# ---------------------------------------------------------
# 6️⃣  Training Arguments
# ---------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # 3 is usually enough when layers are frozen
    weight_decay=0.05,   # Slightly higher to regularize
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    fp16=False
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load("seqeval")

# ---------------------------------------------------------
# 7️⃣  Compute Metrics
# ---------------------------------------------------------
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# ---------------------------------------------------------
# 8️⃣  Trainer Setup
# ---------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ---------------------------------------------------------
# 9️⃣  Main Training
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n🚀 Starting Fine-Tuning (encoder frozen)...\n")
    trainer.train()

    # Evaluate and save model
    metrics = trainer.evaluate(tokenized_ds["validation"])
    print("\n✅ VALIDATION METRICS:", metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Saved fine-tuned model → {OUTPUT_DIR}")

    # Quick test
    nlp = pipeline(
        "token-classification",
        model=OUTPUT_DIR,
        tokenizer=OUTPUT_DIR,
        aggregation_strategy="simple"
    )

    test = "This agreement was signed between Mr. Rajesh Sharma and ABC Pvt Ltd on 15 March 2022 in Mumbai."
    print("\n✅ TEST OUTPUT:")
    print(nlp(test))
