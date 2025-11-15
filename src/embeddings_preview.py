import sys
import json
from pathlib import Path

FIELDS = [
    ("important", "title"),
    ("important", "category"),
    ("important", "sub_category"),
    ("important", "tags"),             # list -> space-joined
    ("important", "targets", "os"),    # list -> space-joined
    ("important", "targets", "system"),# list -> space-joined
    ("important", "risk"),
]

def get_in(d: dict, path: tuple):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur

def to_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        # join all items with spaces; convert non-strings safely
        return " ".join(str(x) for x in value if x is not None)
    return str(value)

def build_embedding_text(record: dict) -> str:
    parts = []
    for path in FIELDS:
        v = get_in(record, path)
        parts.append(to_text(v))
    # Filter empty chunks and join with a single space
    return " ".join(p for p in parts if p).strip()

def preview(jsonl_path: Path, max_examples: int = 2):
    total = 0
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Skipping invalid JSON line: {e}")
                continue
            total += 1
            if len(examples) < max_examples:
                examples.append(build_embedding_text(obj))

    print(f"Records: {total}")
    for i, ex in enumerate(examples, 1):
        print(f"Example embedding text #{i}:")
        print(ex)
        print("-" * 40)

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.embeddings_preview <path-to-jsonl>")
        sys.exit(1)
    path = Path(sys.argv[1]).resolve()
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)
    preview(path)

if __name__ == "__main__":
    main()
