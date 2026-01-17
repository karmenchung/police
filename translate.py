import argparse
from pathlib import Path


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


def main():
    parser = argparse.ArgumentParser(description="Print text parts from an SRT file.")
    parser.add_argument("srt_path", nargs="?", default=None, help="Path to .srt file")
    args = parser.parse_args()

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
            print(text)


if __name__ == "__main__":
    main()
