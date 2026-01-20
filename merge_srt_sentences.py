import re
import sys
from pathlib import Path

SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*$")
ELLIPSIS_RE = re.compile(r"\.\.\.[\"')\]]*$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_time(ts: str) -> int:
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def format_time(ms: int) -> str:
    total_seconds, millis = divmod(ms, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def is_sentence_end(text: str) -> bool:
    t = text.strip()
    return bool(SENTENCE_END_RE.search(t) or ELLIPSIS_RE.search(t))


def split_sentences(text: str):
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text.strip())]
    return [p for p in parts if p]


def split_block(start: int, end: int, text: str):
    parts = split_sentences(text)
    if len(parts) <= 1:
        return [(start, end, text)]

    total_len = sum(len(p) for p in parts)
    duration = max(1, end - start)
    out = []
    cur = start

    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            part_end = end
        else:
            part_dur = max(1, int(round(duration * len(part) / total_len)))
            remaining = len(parts) - i - 1
            latest_end = end - remaining  # keep 1ms per remaining part
            part_end = min(latest_end, cur + part_dur)
            if part_end <= cur:
                part_end = min(latest_end, cur + 1)
        out.append((cur, part_end, part))
        cur = part_end

    return out


def parse_srt(path: Path):
    raw = path.read_text(encoding="utf-8-sig")
    parts = re.split(r"\r?\n\r?\n+", raw.strip())
    blocks = []
    for part in parts:
        lines = [ln.rstrip() for ln in part.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        if re.match(r"^\d+$", lines[0]):
            time_line = lines[1]
            text_lines = lines[2:]
        else:
            time_line = lines[0]
            text_lines = lines[1:]
        m = re.match(r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)", time_line)
        if not m:
            continue
        start = parse_time(m.group(1))
        end = parse_time(m.group(2))
        text = " ".join(ln.strip() for ln in text_lines)
        text = re.sub(r"\s+", " ", text).strip()
        blocks.append((start, end, text))
    return blocks


def merge_blocks(blocks, max_gap_ms=900):
    merged = []
    cur_start = None
    cur_end = None
    cur_text = []

    for start, end, text in blocks:
        if not text:
            continue
        if cur_start is None:
            cur_start, cur_end, cur_text = start, end, [text]
            continue

        gap = start - cur_end
        prev_text = " ".join(cur_text).strip()
        if is_sentence_end(prev_text) or gap >= max_gap_ms:
            merged.extend(split_block(cur_start, cur_end, prev_text))
            cur_start, cur_end, cur_text = start, end, [text]
        else:
            cur_text.append(text)
            cur_end = end

    if cur_start is not None:
        merged.extend(split_block(cur_start, cur_end, " ".join(cur_text).strip()))

    return merged


def write_srt(path: Path, blocks):
    lines = []
    for i, (start, end, text) in enumerate(blocks, 1):
        lines.append(str(i))
        lines.append(f"{format_time(start)} --> {format_time(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv):
    if len(argv) > 1:
        inputs = [Path(p) for p in argv[1:]]
    else:
        inputs = [p for p in Path.cwd().glob("*.srt") if not p.name.endswith(".sentences.srt")]

    if not inputs:
        print("No input .srt files found.")
        return 1

    for src in inputs:
        blocks = parse_srt(src)
        merged = merge_blocks(blocks)
        dst = src.with_suffix(".sentences.srt")
        write_srt(dst, merged)
        print(f"{src.name} -> {dst.name} ({len(blocks)} blocks -> {len(merged)} sentences)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
