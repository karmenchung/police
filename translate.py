import argparse
import os
import sys
import torch
from pathlib import Path

try:
    import accelerate
except Exception:
    accelerate = None

from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as hf_version


def read_srt_text(path: Path) -> str:
    data = path.read_bytes()
    if not data:
        return ""
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = None
    if text is None or "\x00" in text:
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    if text is None:
        return data.decode("utf-8", errors="ignore")
    return text


def extract_text_blocks(content: str):
    lines = content.splitlines()
    if any(line.strip() == "" for line in lines):
        current = []
        for line in lines:
            if line.strip() == "":
                if current:
                    yield current
                    current = []
                continue
            current.append(line.rstrip("\n"))
        if current:
            yield current
        return

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit() and i + 1 < len(lines) and "-->" in lines[i + 1]:
            block = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines):
                next_line = lines[i].strip()
                if next_line.isdigit() and i + 1 < len(lines) and "-->" in lines[i + 1]:
                    break
                block.append(lines[i])
                i += 1
            yield block
            continue
        i += 1


def parse_block(lines, fallback_index):
    if not lines:
        return None
    idx = 0
    block_index = None
    if lines[0].strip().isdigit():
        block_index = lines[0].strip()
        idx = 1
    else:
        block_index = str(fallback_index)
    timestamp = None
    if idx < len(lines) and "-->" in lines[idx]:
        timestamp = lines[idx].strip()
        idx += 1
    text_lines = [ln.strip() for ln in lines[idx:] if ln.strip()]
    text = " ".join(text_lines)
    return block_index, timestamp, text


def parse_timestamp_range(timestamp):
    if not timestamp or "-->" not in timestamp:
        return None, None
    parts = [p.strip() for p in timestamp.split("-->")]
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def merge_timestamps(first_ts, last_ts):
    start, _ = parse_timestamp_range(first_ts)
    _, end = parse_timestamp_range(last_ts)
    if start and end:
        return f"{start} --> {end}"
    if first_ts:
        return first_ts
    return last_ts


def run_prompt(prompt, system, tokenizer, model, max_new_tokens):
    print("prompt:",prompt) 
    print("-"*60)
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


def choose_grouping(text_a, text_b, text_c, tokenizer, model):
    selection_system = (
        "You choose the most fluent combined subtitle. Reply with only A, B, or C."
    )
    prompt = (
        "A:\n"
        f"{text_a}\n\n"
        "B:\n"
        f"{text_a} {text_b}\n\n"
        "C:\n"
        f"{text_a} {text_b} {text_c}\n\n"
        "Reply with only A, B, or C."
    )
    result = run_prompt(prompt, selection_system, tokenizer, model, 8)
    result = result.strip().upper()
    if result.startswith("C"):
        return 3
    if result.startswith("B"):
        return 2
    return 1


def main():
    parser = argparse.ArgumentParser(description="Translate text parts from an SRT file.")
    parser.add_argument("srt_path", nargs="?", default=None, help="Path to .srt file")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model name or path")
    parser.add_argument("--system", default="你是一个可靠的翻译助手，我给你的所有英文请翻译成中文，不要添加任何解释。必须翻译。日常口吻。", help="System prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens")
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

    content = read_srt_text(path)
    raw_blocks = list(extract_text_blocks(content))

    parsed_blocks = []
    fallback_index = 1
    for block in raw_blocks:
        parsed = parse_block(block, fallback_index)
        if not parsed:
            continue
        block_index, timestamp, text = parsed
        fallback_index += 1
        if not text:
            continue
        parsed_blocks.append(
            {"index": block_index, "timestamp": timestamp, "text": text}
        )

    output_path = Path("translated.srt")
    out_index = 1
    i = 0
    with output_path.open("w", encoding="utf-8") as out_file:
        while i < len(parsed_blocks):
            a = parsed_blocks[i]
            b = parsed_blocks[i + 1] if i + 1 < len(parsed_blocks) else None
            c = parsed_blocks[i + 2] if i + 2 < len(parsed_blocks) else None

            text_a = a["text"]
            text_b = b["text"] if b else ""
            text_c = c["text"] if c else ""

            if c:
                count = choose_grouping(text_a, text_b, text_c, tokenizer, model)
            elif b:
                count = 2
            else:
                count = 1

            selected = parsed_blocks[i : i + count]
            combined_text = " ".join(item["text"] for item in selected)
            combined_timestamp = merge_timestamps(
                selected[0]["timestamp"], selected[-1]["timestamp"]
            )

            reply = run_prompt(
                combined_text, args.system, tokenizer, model, args.max_new_tokens
            )

            print(out_index)
            if combined_timestamp:
                print(combined_timestamp)
            print(reply)
            print("")

            out_file.write(str(out_index) + "\n")
            if combined_timestamp:
                out_file.write(combined_timestamp + "\n")
            out_file.write(reply + "\n\n")

            out_index += 1
            i += count


if __name__ == "__main__":
    main()
