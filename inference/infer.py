"""
infer.py

Loads the fine-tuned LoRA adapter on top of the base model and generates
data-cleaning recommendations for a given dataset summary.

Usage:
    python inference/infer.py --summary path/to/summary.txt
    python inference/infer.py --summary_text "Dataset: ... issues: ..."
    echo "Dataset: ..." | python inference/infer.py --stdin

Requires the adapter produced by notebooks/finetune_colab.ipynb, downloaded
and unzipped locally (default expected path: outputs/final_adapter).
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER_DIR = "outputs/qwen2.5-0.5b-cleaning-lora/final_adapter"  

SYSTEM_PROMPT = (
    "You are a meticulous data-cleaning assistant. Given a dataset summary, "
    "produce prioritized, actionable cleaning recommendations."
)


def load_model(adapter_dir: str, use_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NOTE: intentionally NOT using device_map="auto" here. For a model this small
    # (0.5B params) accelerate's auto-dispatch can decide to offload part of it to
    # disk/CPU even when unnecessary, and PEFT's adapter loading is incompatible
    # with disk-offloaded weights (raises a KeyError on embed_tokens). Loading the
    # whole model onto a single device directly avoids that entirely.
    kwargs = {}
    if use_4bit and device == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}  # pin the whole model to a single GPU, no auto-splitting
    else:
        kwargs["dtype"] = torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    if "device_map" not in kwargs:
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_recommendations(
    model, tokenizer, dataset_summary: str,
    instruction: str = "Given the dataset summary below, list prioritized data-cleaning recommendations.",
    max_new_tokens: int = 350, temperature: float = 0.3,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n\n{dataset_summary}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Generate data-cleaning recommendations from a dataset summary.")
    parser.add_argument("--summary", type=str, help="Path to a text file containing the dataset summary.")
    parser.add_argument("--summary_text", type=str, help="Dataset summary passed directly as a string.")
    parser.add_argument("--stdin", action="store_true", help="Read the dataset summary from stdin.")
    parser.add_argument("--adapter_dir", type=str, default=DEFAULT_ADAPTER_DIR,
                         help="Path to the saved LoRA adapter directory.")
    parser.add_argument("--max_new_tokens", type=int, default=350)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit loading (e.g. CPU-only machines).")
    args = parser.parse_args()

    if args.summary:
        with open(args.summary, "r", encoding="utf-8") as f:
            dataset_summary = f.read()
    elif args.summary_text:
        dataset_summary = args.summary_text
    elif args.stdin:
        dataset_summary = sys.stdin.read()
    else:
        parser.error("Provide one of --summary, --summary_text, or --stdin")

    print("Loading model + adapter from:", args.adapter_dir)
    model, tokenizer = load_model(args.adapter_dir, use_4bit=not args.no_4bit)

    print("\n--- Dataset summary ---")
    print(dataset_summary.strip())

    result = generate_recommendations(
        model, tokenizer, dataset_summary,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
    )

    print("\n--- Recommended cleaning steps ---")
    print(result)


if __name__ == "__main__":
    main()