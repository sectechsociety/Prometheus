import json
import sys

REQUIRED_FIELDS = [
    "input_prompt",
    "enhanced_prompt",
    "source",
    "chunk_id",
    "created_at",
    "target_model",
]


def validate(path: str):
    total = 0
    bad = 0
    stats: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"Line {total}: JSON parse error: {e}")
                    bad += 1
                    continue
                missing = [f for f in REQUIRED_FIELDS if f not in obj]
                if missing:
                    print(f"Line {total}: missing fields: {missing}")
                    bad += 1
                    continue
                # simple length checks
                if not obj.get("input_prompt") or len(obj.get("input_prompt", "")) < 10:
                    print(f"Line {total}: input_prompt too short")
                    bad += 1
                    continue
                tm = obj.get("target_model")
                stats[tm] = stats.get(tm, 0) + 1
    except FileNotFoundError:
        print("File not found:", path)
        return 2

    print(f"Total lines: {total}, valid: {total - bad}, invalid: {bad}")
    print("Counts per target_model:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_jsonl.py path/to/file.jsonl")
        sys.exit(2)
    sys.exit(validate(sys.argv[1]))
