import pandas as pd
import random
from itertools import product

# 1️⃣ Load your JSONL dataset
df = pd.read_json("risk_training_data.jsonl", lines=True)
print("Before balancing:\n", df['outcome'].value_counts())

# 2️⃣ Simple paraphrase helper (lightweight)
def simple_paraphrase(text):
    templates = [
        "{} under certain conditions.",
        "In some cases, {}",
        "{} depending on agreement terms.",
        "As per company policy, {}",
        "{} — subject to review.",
        "This clause specifies that {}"
    ]
    return random.choice(templates).format(text)

# 3️⃣ Target samples per risk class
TARGET_SIZE = 150
augmented_rows = []

for outcome, group in df.groupby('outcome'):
    count = len(group)
    if count < TARGET_SIZE:
        needed = TARGET_SIZE - count
        for _ in range(needed):
            row = group.sample(1, replace=True).iloc[0]
            new_clause = simple_paraphrase(row['clause'])
            augmented_rows.append({
                "clause": new_clause,
                "outcome": outcome,
                "clause_type": row['clause_type'],
                "jurisdiction": row['jurisdiction']
            })

# 4️⃣ Combine & shuffle
augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter balancing:\n", augmented_df['outcome'].value_counts())

# 5️⃣ Save the new balanced dataset
augmented_df.to_json("risk_training_data_balanced.jsonl", orient="records", lines=True)
print("\n✅ Saved balanced dataset as risk_training_data_balanced.jsonl")
