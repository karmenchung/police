import argparse
import os
import sys
import torch
from pathlib import Path

# Force Hugging Face caches onto F: to avoid filling C:
os.environ["HF_HOME"] = r"F:\python\警察\.hf_cache"
os.environ["HF_HUB_CACHE"] = r"F:\python\警察\.hf_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"F:\python\警察\.hf_cache\transformers"

try:
    import accelerate
except Exception:
    accelerate = None

from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as hf_version


def extract_text_blocks(content: str):
    blocks = content.splitlines()
    current = []
    for line in blocks:
        if line.strip() == "":
            if current:
                yield current
                current = []
            continue
        current.append(line.rstrip("\n"))
    if current:
        yield current


def text_from_block(lines):
    if not lines:
        return ""
    idx = 0
    if lines and lines[0].strip().isdigit():
        idx = 1
    if idx < len(lines) and "-->" in lines[idx]:
        idx += 1
    text_lines = [ln.strip() for ln in lines[idx:] if ln.strip()]
    return " ".join(text_lines)


def run_prompt(prompt, system, tokenizer, model, max_new_tokens):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    input_len = inputs.shape[-1]
    new_tokens = output[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Translate text parts from an SRT file.")
    parser.add_argument("srt_path", nargs="?", default=None, help="Path to .srt file")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name or path")
    parser.add_argument("--system", default="你是一个可靠的翻译助手，我给你的所有英文请翻译成中文，不要添加任何解释。必须翻译。可以添加一些幽默感。", help="System prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("CUDA not available. Install a CUDA-enabled PyTorch and GPU driver.", file=sys.stderr)
        raise SystemExit(2)

    if accelerate is None:
        print(
            "accelerate is required for device_map loading. Install it with:\n"
            "  pip install accelerate",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    except ValueError as exc:
        msg = str(exc)
        if "Tokenizer class" in msg and "does not exist" in msg:
            print(
                "Tokenizer class not found. Please upgrade transformers:\n"
                "  pip install -U transformers\n"
                f"Current transformers version: {hf_version}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        raise

    if args.srt_path:
        path = Path(args.srt_path)
    else:
        # Try to find a single .srt in current directory
        srt_files = list(Path(".").glob("*.srt"))
        if len(srt_files) == 1:
            path = srt_files[0]
        else:
            raise SystemExit("Please provide an .srt path, e.g. python translate.py origin_english.srt")

    content = path.read_text(encoding="utf-8-sig", errors="ignore")

    for block in extract_text_blocks(content):
        text = text_from_block(block)
        if text:
            reply = run_prompt(text, args.system, tokenizer, model, args.max_new_tokens)
            print(text, "|", reply)


if __name__ == "__main__":
    main()
