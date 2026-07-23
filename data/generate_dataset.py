"""
generate_dataset.py

Generates an instruction-tuning dataset for the task:
  "Given a dataset summary (columns, dtypes, missing %, stats, sample issues),
   produce prioritized, actionable data-cleaning recommendations."

Why synthetic-but-structured generation instead of hand-writing 200 examples:
- Guarantees coverage of many real-world data-quality issue combinations
  (missing values, duplicates, outliers, mixed types, encoding issues,
  imbalanced classes, inconsistent categories, date-format chaos, etc.)
- Keeps every example internally consistent (the "answer" is derived
  programmatically from the same issue-list that generated the "summary"),
  which is important for teaching an LLM a reliable input->output mapping.
- Fully reproducible (fixed seed) and easy to extend with new issue types
  or domains later.

Output: data/train.jsonl  (instruction / input / output triples, Alpaca-style)
"""

import json
import random

random.seed(42)

DOMAINS = [
    ("E-commerce orders", ["order_id", "customer_id", "order_date", "product_category",
                            "unit_price", "quantity", "discount_pct", "payment_method",
                            "shipping_city", "delivery_status", "customer_rating"]),
    ("Hospital patient records", ["patient_id", "admission_date", "age", "gender",
                                   "diagnosis_code", "department", "length_of_stay_days",
                                   "billing_amount", "insurance_provider", "discharge_status"]),
    ("Employee HR dataset", ["employee_id", "join_date", "department", "designation",
                              "monthly_salary", "years_experience", "performance_rating",
                              "attrition_flag", "work_location", "manager_id"]),
    ("School ERP student records", ["student_id", "admission_date", "class", "section",
                                     "attendance_pct", "fee_status", "guardian_phone",
                                     "exam_score", "subject", "remarks"]),
    ("IoT sensor readings", ["sensor_id", "timestamp", "temperature_c", "humidity_pct",
                              "battery_voltage", "location", "signal_strength_dbm",
                              "firmware_version", "status_code"]),
    ("Loan applications", ["applicant_id", "application_date", "income_monthly",
                            "credit_score", "loan_amount", "loan_purpose", "employment_type",
                            "existing_liabilities", "approval_status"]),
    ("Retail store inventory", ["sku_id", "warehouse", "category", "stock_qty",
                                 "reorder_level", "unit_cost", "supplier_name",
                                 "last_restock_date", "expiry_date"]),
    ("Customer support tickets", ["ticket_id", "created_at", "channel", "priority",
                                   "category", "resolution_time_hrs", "agent_id",
                                   "csat_score", "status"]),
    ("Marketing campaign leads", ["lead_id", "source", "campaign_name", "landing_page",
                                   "cost_per_click", "conversion_flag", "region",
                                   "device_type", "utm_medium"]),
    ("Taxi / travel trip logs", ["trip_id", "driver_id", "pickup_time", "drop_time",
                                  "distance_km", "fare_amount", "vehicle_type",
                                  "city", "trip_rating"]),
]

ISSUE_LIBRARY = {
    "missing_values": {
        "summary_tpl": "- '{col}' has {pct}% missing values",
        "rec": ("Handle missing values in '{col}' ({pct}%): if under 5%, drop or impute "
                "with median/mode; if higher, use group-wise imputation or a missing-indicator "
                "flag rather than dropping rows, since {pct}% loss would bias the dataset."),
    },
    "duplicates": {
        "summary_tpl": "- {pct}% of rows appear to be exact or near-duplicates (same {col})",
        "rec": ("Deduplicate on '{col}' plus a timestamp/id tiebreaker; investigate whether "
                "duplicates come from repeated ingestion/ETL runs before blindly dropping, "
                "since {pct}% duplication this high often signals a pipeline bug."),
    },
    "outliers": {
        "summary_tpl": "- '{col}' shows extreme outliers (max value {mult}x the 95th percentile)",
        "rec": ("Investigate outliers in '{col}': cap using IQR or winsorization at the 1st/99th "
                "percentile, or isolate them for separate analysis if they represent genuine rare "
                "events rather than data-entry errors (a {mult}x jump over the 95th percentile "
                "needs manual review, not automatic removal)."),
    },
    "mixed_dtype": {
        "summary_tpl": "- '{col}' is stored as object/string but should be numeric (contains values like '{sample}')",
        "rec": ("Coerce '{col}' to numeric with pd.to_numeric(errors='coerce') after stripping "
                "non-numeric characters (e.g. '{sample}'); log rows that fail conversion instead "
                "of silently dropping them."),
    },
    "inconsistent_categorical": {
        "summary_tpl": "- '{col}' has inconsistent category labels (e.g. '{a}', '{b}', '{c}' all mean the same thing)",
        "rec": ("Normalize categories in '{col}': lowercase/trim whitespace, then map known "
                "synonyms ('{a}', '{b}', '{c}') to a single canonical label using a lookup table "
                "rather than fuzzy-matching blindly, to avoid merging genuinely distinct categories."),
    },
    "date_format_mixed": {
        "summary_tpl": "- '{col}' contains multiple date formats (e.g. 'DD/MM/YYYY' and 'YYYY-MM-DD' mixed)",
        "rec": ("Standardize '{col}' to ISO 8601 using pd.to_datetime with dayfirst inference "
                "per source, or split by source system first if the format correlates with an "
                "upstream system, then re-parse; flag unparsable rows instead of coercing to NaT silently."),
    },
    "high_cardinality": {
        "summary_tpl": "- '{col}' has very high cardinality ({n} unique values out of {rows} rows)",
        "rec": ("For '{col}' with {n} unique values, avoid one-hot encoding directly; use target/"
                "frequency encoding, group rare categories (<1% frequency) into an 'Other' bucket, "
                "or hash-encode if it's a high-cardinality ID-like field not meant for grouping."),
    },
    "imbalanced_target": {
        "summary_tpl": "- Target column '{col}' is highly imbalanced ({maj}% vs {minr}%)",
        "rec": ("Address class imbalance in '{col}' ({maj}%/{minr}%) with stratified sampling for "
                "train/test splits, class-weighted loss, or SMOTE-style oversampling on the "
                "training set only (never on validation/test) to avoid leakage."),
    },
    "whitespace_encoding": {
        "summary_tpl": "- '{col}' contains leading/trailing whitespace and inconsistent casing",
        "rec": ("Clean '{col}' with .str.strip() and a consistent case convention (lower or title "
                "case); also check for non-breaking spaces or encoding artifacts (e.g. '\\xa0') "
                "that plain .strip() may miss."),
    },
    "negative_or_impossible": {
        "summary_tpl": "- '{col}' contains impossible values (e.g. negative {col} or values of 0 where not valid)",
        "rec": ("Validate '{col}' against domain constraints (non-negative, realistic range); "
                "treat impossible values as missing rather than clamping to zero, then apply the "
                "same missing-value strategy used elsewhere in the pipeline."),
    },
    "unit_mismatch": {
        "summary_tpl": "- '{col}' appears to mix units (some values ~1-10, others ~1000+, suggesting different units)",
        "rec": ("Check '{col}' for unit inconsistency (e.g. km vs miles, or raw vs thousands); "
                "confirm with the source system and convert to a single unit before any aggregation "
                "or modeling, since mixed units silently corrupt averages and totals."),
    },
    "constant_column": {
        "summary_tpl": "- '{col}' has only 1 unique value across all {rows} rows (zero variance)",
        "rec": ("Drop or flag '{col}' since it has zero variance and adds no predictive signal, "
                "but first confirm it isn't a broken data feed (e.g. a sensor stuck reporting one "
                "value) rather than a genuinely constant field."),
    },
}

QUALITY_META = [
    "Also run a schema check to lock in expected dtypes going forward so these issues don't recur silently.",
    "Consider adding automated data-quality checks (e.g. Great Expectations) to catch these issues at ingestion time.",
    "Document all transformations in a data-cleaning log so the pipeline is reproducible and auditable.",
    "Version the raw and cleaned datasets separately so cleaning decisions can be revisited or rolled back.",
    "After cleaning, re-profile the dataset to confirm no new issues were introduced by the fixes themselves.",
]

INSTRUCTIONS = [
    "Given the dataset summary below, list prioritized data-cleaning recommendations.",
    "Review this dataset profile and suggest the cleaning steps needed before analysis or modeling.",
    "Analyze the dataset summary and provide actionable cleaning recommendations, ordered by priority.",
    "You are a data-cleaning assistant. Based on the summary below, recommend specific fixes.",
    "Inspect the following dataset summary and produce a step-by-step data-cleaning plan.",
]


def build_example(idx: int) -> dict:
    domain_name, columns = random.choice(DOMAINS)
    n_rows = random.choice([1200, 5000, 8600, 15000, 42000, 98000])
    n_cols = len(columns)

    n_issues = random.randint(3, 6)
    chosen_issue_keys = random.sample(list(ISSUE_LIBRARY.keys()), n_issues)

    summary_lines = [
        f"Dataset: {domain_name}",
        f"Rows: {n_rows}, Columns: {n_cols}",
        f"Column names: {', '.join(columns)}",
        "",
        "Observed data quality issues:",
    ]
    rec_lines = []

    for rank, key in enumerate(chosen_issue_keys, start=1):
        issue = ISSUE_LIBRARY[key]
        col = random.choice(columns)
        params = {
            "col": col,
            "pct": random.choice([2, 4, 7, 12, 18, 25, 33]),
            "mult": random.choice([5, 8, 12, 20]),
            "sample": random.choice(["1,200", "$45.00", "N/A", "12 kg", "--"]),
            "a": random.choice(["NY", "New York", "new york"]),
            "b": random.choice(["Delhi", "delhi", "DEL"]),
            "c": random.choice(["USA", "U.S.A.", "United States"]),
            "n": random.choice([850, 2300, 9100]),
            "rows": n_rows,
            "maj": random.choice([88, 92, 95]),
            "minr": None,
        }
        params["minr"] = 100 - params["maj"]
        summary_lines.append(issue["summary_tpl"].format(**params))
        rec_lines.append(f"{rank}. {issue['rec'].format(**params)}")

    if random.random() < 0.6:
        rec_lines.append(f"{len(chosen_issue_keys) + 1}. {random.choice(QUALITY_META)}")

    instruction = random.choice(INSTRUCTIONS)
    input_text = "\n".join(summary_lines)
    output_text = "Recommended cleaning steps:\n" + "\n".join(rec_lines)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def main(n_examples: int = 260, out_path: str = "data/train.jsonl"):
    examples = []
    seen_inputs = set()
    attempts = 0
    while len(examples) < n_examples and attempts < n_examples * 10:
        attempts += 1
        ex = build_example(len(examples))
        if ex["input"] in seen_inputs:
            continue
        seen_inputs.add(ex["input"])
        examples.append(ex)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
