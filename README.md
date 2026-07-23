# 🧹 LLM Cleaning Advisor

Fine-tunes a small open-source instruction-tuned LLM (**Qwen2.5-0.5B-Instruct**) using **LoRA (PEFT)**
to generate **prioritized, actionable data-cleaning recommendations** from a dataset summary.

Give it a profile of your dataset — columns, dtypes, missing %, and observed issues (the kind of
output you'd get from `df.info()` + `df.describe()` plus a quick profiling pass) — and it returns a
ranked list of specific cleaning steps.

---

## 📌 What this project delivers

| Component | Where |
|---|---|
| 260 instruction-response training examples | `data/generate_dataset.py` → `data/train.jsonl` |
| Fine-tuned small open-source LLM using PEFT (LoRA) on Colab | `Finetune.ipynb` |
| Inference script: dataset summary → cleaning recommendations | `inference/infer.py` |

---

## 🧠 Approach & reasoning

**Why this task?**
"Dataset summary → cleaning recommendations" is structured enough to verify at a glance — does the
recommendation actually match the stated issue? — while still requiring real judgment: prioritizing
fixes, not just naming them. It's also directly reusable in real data-engineering workflows.

**Why Qwen2.5-0.5B-Instruct?**
- Open-source, Apache 2.0 licensed
- Already instruction-tuned, so LoRA only needs to *specialize* behavior rather than teach
  instruction-following from scratch
- Small enough (0.5B params) to fine-tune and run inference on a single free-tier Colab GPU (T4)
  using 4-bit quantization — no expensive hardware required

**Why LoRA (PEFT)?**
Instead of updating all 500M+ parameters, LoRA freezes the base model and trains small rank-decomposition
matrices injected into attention + MLP projection layers (`r=16`, `alpha=32`). This means:
- Only a small fraction of parameters are actually trained
- The saved adapter is a few MB, not gigabytes
- Training fits comfortably in a free Colab session

The base model is loaded in **4-bit (NF4, via bitsandbytes)** for memory efficiency, combining
quantization with LoRA (QLoRA-style fine-tuning).

---

## 📁 Repository structure

```
llm-cleaning-advisor/
├── data/
│   ├── generate_dataset.py   # builds the training set programmatically
│   ├── split_dataset.py      # 90/10 train/val split
│   ├── train.jsonl           # full dataset (260 examples)
│   ├── train_split.jsonl     # training split (234 examples)
│   └── val_split.jsonl       # validation split (26 examples)
├── inference/
│   ├── infer.py               # CLI: dataset summary -> cleaning recommendations
│   └── sample_summary.txt     # example input for a quick demo
├── outputs/
│   └── final_adapter/         # trained LoRA adapter
├── Finetune.ipynb             # run this in Google Colab to fine-tune
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

`data/generate_dataset.py` programmatically builds 260 unique instruction/input/output examples by
combining:

- **10 realistic domains**: e-commerce orders, hospital records, HR, school ERP, IoT sensors, loan
  applications, retail inventory, support tickets, marketing leads, taxi trip logs
- **12 data-quality issue types**: missing values, duplicates, outliers, wrong dtypes, inconsistent
  categorical labels, mixed date formats, high cardinality, class imbalance, whitespace/encoding
  noise, impossible values, unit mismatches, zero-variance columns
- **3–6 randomly combined issues per example**, each paired with a specific, derivable recommendation

Generating examples this way — rather than hand-writing a smaller set — guarantees every training
pair is internally consistent: the output is always a *correct* response to its paired input, which
matters more for teaching a reliable mapping than raw example count alone.

**Example record:**
```json
{
  "instruction": "Given the dataset summary below, list prioritized data-cleaning recommendations.",
  "input": "Dataset: Retail store inventory\nRows: 15000, Columns: 9\n...\n- 'unit_cost' has 18% missing values\n- 'last_restock_date' contains multiple date formats...",
  "output": "Recommended cleaning steps:\n1. Handle missing values in 'unit_cost' (18%): ...\n2. Standardize 'last_restock_date' to ISO 8601 ..."
}
```

**To regenerate or extend the dataset:**
```bash
python data/generate_dataset.py   # writes data/train.jsonl
python data/split_dataset.py      # writes train_split.jsonl / val_split.jsonl
```

---

## 🏋️ Fine-tuning (Google Colab)

1. Open **`Finetune.ipynb`** in [Google Colab](https://colab.research.google.com).
2. **Runtime → Change runtime type → T4 GPU**.
3. Run all cells top to bottom. This will:
   - Load `Qwen/Qwen2.5-0.5B-Instruct` in 4-bit (bitsandbytes/NF4)
   - Attach a LoRA adapter (PEFT) over attention + MLP projection layers
   - Fine-tune for 3 epochs with `trl.SFTTrainer` on 234 training examples (26 held out for validation)
   - Save the adapter and zip it for download
   - Run a quick sanity-check generation at the end of the notebook
4. Download the resulting adapter zip and unzip it locally as `outputs/final_adapter/`.

Training completes in well under Colab's free-tier session limits given the dataset size.

---

## 🔍 Inference

```bash
pip install -r requirements.txt

# From a file
python inference/infer.py --summary inference/sample_summary.txt --adapter_dir outputs/final_adapter

# From a string directly
python inference/infer.py --summary_text "Dataset: ... issues: ..." --adapter_dir outputs/final_adapter

# From stdin
cat my_summary.txt | python inference/infer.py --stdin --adapter_dir outputs/final_adapter
```

`infer.py` loads the base model + the fine-tuned LoRA adapter, formats the input using the same chat
template used during training, and prints the generated cleaning recommendations. Pass `--no_4bit`
if running on a CPU-only machine.


**Input** (`inference/sample_summary.txt`):
```
Dataset: Retail store inventory
Rows: 15000, Columns: 9
Observed data quality issues:
- 'unit_cost' has 18% missing values
- 'last_restock_date' contains multiple date formats (DD/MM/YYYY and YYYY-MM-DD mixed)
- 'supplier_name' has inconsistent category labels (ABC Traders / abc traders / ABC TRADERS)
- 4% of rows appear to be exact or near-duplicates (same sku_id)
```

**Output:**
```
Recommended cleaning steps:
1. Handle missing values in 'unit_cost' (18%): use group-wise imputation or a missing-indicator
   flag rather than dropping rows, since 18% loss would bias the dataset.
2. Standardize 'last_restock_date' to ISO 8601 using pd.to_datetime with dayfirst inference per
   source; flag unparsable rows instead of coercing to NaT silently.
3. Normalize categories in 'supplier_name': lowercase/trim whitespace, then map known synonyms to
   a single canonical label using a lookup table.
4. Deduplicate on 'sku_id' plus a timestamp/id tiebreaker; investigate whether duplicates come from
   repeated ingestion/ETL runs before blindly dropping.
```

---

## 📈 Evaluation notes

- Training/validation loss is tracked via `SFTConfig`'s built-in per-epoch evaluation on the 26-example
  validation split.
- Qualitative checks worth running: does the model (a) reference the correct column names from the
  input, (b) keep the same prioritized, numbered format as training data, and (c) avoid hallucinating
  issues not present in the input summary? Testing on hand-written summaries (not from
  `generate_dataset.py`) is the best way to catch overfitting to the synthetic template style.

---

## ⚖️ Design decisions & trade-offs

- **Synthetic-but-structured dataset over manual curation** — guarantees correctness and full
  coverage of issue types; diversity comes from 10 domains × 12 issue types × randomized
  combinations rather than manually writing individual examples.
- **0.5B model over larger 1–3B options** — fits 4-bit fine-tuning comfortably on a free T4, trains in
  minutes rather than hours, while still fully demonstrating the PEFT pipeline end-to-end. Swapping in
  `Qwen2.5-1.5B-Instruct` or `Phi-3-mini` only requires changing `MODEL_NAME` in the notebook.
- **LoRA over full fine-tuning** — standard approach for consumer-GPU constraints; the adapter is a
  few MB versus the full model, and easy to version or swap.

---

## 🚀 Possible extensions

- Swap in a real profiling library (`ydata-profiling`, `pandas.DataFrame.info()`) to generate summaries
  from actual CSVs instead of synthetic descriptions, paired with real-world cleaning logs.
- Add a small held-out evaluation set of *hand-written* summaries (not generator output) to measure
  genuine generalization rather than memorization of the template style.
- Push the trained adapter to the Hugging Face Hub for easier distribution instead of a repo upload.

---

## 🛠️ Tech stack

`transformers` · `peft` · `trl` · `bitsandbytes` · `accelerate` · `datasets` · Google Colab (T4 GPU)
