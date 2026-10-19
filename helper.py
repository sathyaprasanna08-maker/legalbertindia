# import json
# import random

# ----------------------------------- DOWNSAMPLING O LABEL -------------------------------------------------------
# def limit_o_tokens(input_path, output_path, max_o_tokens=2000, seed=42):
#     random.seed(seed)
#     total_o = 0
#     kept_data = []

#     with open(input_path, "r", encoding="utf-8") as infile:
#         lines = [json.loads(line) for line in infile]

#     for example in lines:
#         tokens, tags = example["tokens"], example["ner_tags"]

#         new_tokens, new_tags = [], []
#         for tok, tag in zip(tokens, tags):
#             if tag == "O":
#                 if total_o < max_o_tokens:
#                     total_o += 1
#                     new_tokens.append(tok)
#                     new_tags.append(tag)
#                 # else skip extra 'O' tokens
#             else:
#                 new_tokens.append(tok)
#                 new_tags.append(tag)

#         if new_tokens:
#             kept_data.append({"tokens": new_tokens, "ner_tags": new_tags})

#     # Shuffle to avoid label order bias
#     random.shuffle(kept_data)

#     # Save reduced dataset
#     with open(output_path, "w", encoding="utf-8") as out:
#         for entry in kept_data:
#             json.dump(entry, out)
#             out.write("\n")

#     print(f"\n✅ Saved filtered dataset to: {output_path}")
#     print(f"📉 Total examples retained: {len(kept_data)}")
#     print(f"⚖️  O tokens capped at: {max_o_tokens}")
# limit_o_tokens(
#     input_path="weak_labels_final.jsonl",
#     output_path="reduced_final.jsonl",
#     max_o_tokens=5000
# )




import json
from collections import Counter

def check_label_distribution_jsonl(path: str):
    """
    Reads a JSONL file and prints the frequency of each NER tag.
    Expects each line to contain {'tokens': [...], 'ner_tags': [...]}.
    """
    label_counts = Counter()
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            if "ner_tags" in obj:
                label_counts.update(obj["ner_tags"])
                total += len(obj["ner_tags"])

    print(f"\n📊 Label distribution in {path}:")
    for label, count in label_counts.most_common():
        pct = (count / total) * 100
        print(f"  {label:15s} → {count:6d} ({pct:5.2f}%)")

    print(f"\nTotal tokens: {total}")
    print(f"Unique labels: {len(label_counts)}")

    return label_counts
check_label_distribution_jsonl("weak_labels.jsonl")
