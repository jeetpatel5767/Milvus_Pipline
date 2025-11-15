# src/normalize_and_move.py
import sys, os, json, re, shutil
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INBOX = ROOT / "data" / "inbox"
STAGING = ROOT / "data" / "processed" / "staging"
ERRORS = ROOT / "data" / "processed" / "errors"
DATASETS = ROOT / "data" / "datasets"
LOGS = ROOT / "data" / "logs"

def log_line(msg: str):
    LOGS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    log_file = LOGS / f"{ts.date()}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{ts.strftime('%H:%M:%S')}] {msg}\n")

def to_slug(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "record"

def normalize_to_jsonl(src_path: Path) -> tuple[Path, int]:
    STAGING.mkdir(parents=True, exist_ok=True)
    out_path = STAGING / (src_path.stem + ".jsonl")
    count = 0

    if src_path.suffix.lower() == ".jsonl":
        with open(src_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError("Line is not a JSON object")
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    raise ValueError(f"Invalid JSONL line: {e}")
    else:
        with open(src_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                raise ValueError(f"Invalid JSON: {e}")

        with open(out_path, "w", encoding="utf-8") as fout:
            if isinstance(data, dict):
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                count = 1
            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError("Array contains non-object item")
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
            else:
                raise ValueError("Top-level JSON must be object or array")

    return out_path, count

def extract_keys(first_record: dict):
    try:
        imp = first_record["important"]
        category = imp["category"]
        sub_category = imp["sub_category"]
        title = imp.get("title")
    except Exception:
        raise ValueError("Missing important.category or important.sub_category")
    return category, sub_category, title

def first_record_from_jsonl(jsonl_path: Path) -> dict:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        if not line:
            raise ValueError("Empty JSONL file")
        return json.loads(line)

def next_version_path(folder: Path, slug: str) -> Path:
    existing = list(folder.glob(f"{slug}_v*.jsonl"))
    if not existing:
        return folder / f"{slug}_v1.jsonl"
    versions = []
    for p in existing:
        m = re.search(r"_v(\d+)\.jsonl$", p.name)
        if m:
            versions.append(int(m.group(1)))
    v = (max(versions) + 1) if versions else 1
    return folder / f"{slug}_v{v}.jsonl"

def append_manifest(folder: Path, filename: str, records: int):
    manifest = folder / "MANIFEST.jsonl"
    entry = {
        "file": filename,
        "records": records,
        "moved_at": datetime.now(UTC).isoformat()
    }
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m src.normalize_and_move <path-to-json-or-jsonl>")
        sys.exit(1)

    src = Path(sys.argv[1]).resolve()
    try:
        jsonl_path, count = normalize_to_jsonl(src)
        first = first_record_from_jsonl(jsonl_path)
        category, sub_category, title = extract_keys(first)

        dest_folder = DATASETS / category / sub_category
        dest_folder.mkdir(parents=True, exist_ok=True)

        base_slug = to_slug(title) if title else to_slug(f"{category}_{sub_category}")
        dest_path = next_version_path(dest_folder, base_slug)

        shutil.move(str(jsonl_path), str(dest_path))
        append_manifest(dest_folder, dest_path.name, count)

        log_line(f"Moved inbox/{src.name} → datasets/{category}/{sub_category}/{dest_path.name} (records={count})")
        print(f"OK: {dest_path}")
    except Exception as e:
        ERRORS.mkdir(parents=True, exist_ok=True)
        failed = ERRORS / (src.name + ".failed")
        try:
            shutil.move(str(src), str(failed))
        except Exception:
            pass
        log_line(f"ERROR processing {src.name}: {e}")
        print(f"ERROR: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
