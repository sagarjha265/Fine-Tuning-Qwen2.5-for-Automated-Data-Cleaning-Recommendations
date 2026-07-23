"""Split data/train.jsonl into train/val (90/10)."""
import json
import random

random.seed(42)

with open("data/train.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(l) for l in f]

random.shuffle(lines)
n_val = max(20, int(len(lines) * 0.1))
val = lines[:n_val]
train = lines[n_val:]

with open("data/train_split.jsonl", "w", encoding="utf-8") as f:
    for ex in train:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open("data/val_split.jsonl", "w", encoding="utf-8") as f:
    for ex in val:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"train: {len(train)}, val: {len(val)}")
