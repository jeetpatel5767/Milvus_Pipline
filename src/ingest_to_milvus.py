# src/ingest_to_milvus.py
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, UTC
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

def load_config():
    import yaml
    here = Path(__file__).resolve()
    repo_root = here.parent.parent  # src/ -> repo root
    cfg_path = repo_root / "data" / "config" / "embedding.yaml"
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        sys.exit(2)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def build_embedding_text(record):
    FIELDS = [
        ("important","title"),
        ("important","category"),
        ("important","sub_category"),
        ("important","tags"),
        ("important","targets","os"),
        ("important","targets","system"),
        ("important","risk"),
    ]
    def get_in(d, path):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur
    def to_text(v):
        if v is None:
            return ""
        if isinstance(v, list):
            return " ".join(str(x) for x in v if x is not None)
        return str(v)
    return " ".join(to_text(get_in(record, p)) for p in FIELDS).strip()

def extract_scalar_fields(record):
    important = record.get("important", {})
    targets = important.get("targets", {})
    def list_to_str(v):
        if isinstance(v, list):
            return " ".join(str(x) for x in v if x is not None)
        return str(v) if v is not None else ""
    return {
        "id": record.get("id", ""),
        "title": important.get("title", ""),
        "category": important.get("category", ""),
        "sub_category": important.get("sub_category", ""),
        "risk": important.get("risk", ""),
        "tags": list_to_str(important.get("tags", [])),
        "os": list_to_str(targets.get("os", [])),
        "system": list_to_str(targets.get("system", [])),
    }

def ensure_collection(config):
    c = config.get("collection", {})
    name = c.get("name", "attacks_v2")
    dim  = c.get("dim", 384)

    if utility.has_collection(name):
        print(f"Collection '{name}' exists.")
        coll = Collection(name)
        # If 'raw' JSON field is missing, advise to drop & recreate
        field_names = {f.name for f in coll.schema.fields}
        if "raw" not in field_names:
            print(f"Warning: collection '{name}' lacks JSON field 'raw'; drop the collection to recreate with full JSON storage.")
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="sub_category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="risk", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="os", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="system", dtype=DataType.VARCHAR, max_length=500),
            # New: visible JSON field that will show in Attu and store the entire source record
            FieldSchema(name="raw", dtype=DataType.JSON),
        ]
        # Dynamic field can stay enabled, but 'raw' guarantees visibility in UI
        schema = CollectionSchema(fields, description="Attack patterns with embeddings and full JSON", enable_dynamic_field=True)
        coll = Collection(name=name, schema=schema)
        print(f"Collection '{name}' created.")

    # Ensure vector index exists (safe to call even if it already exists)
    idx_type = c.get("index_type", "HNSW")
    metric   = c.get("metric_type", "COSINE")
    params   = c.get("index_params", {"M": 16, "efConstruction": 200})
    try:
        coll.create_index(
            field_name="embedding",
            index_params={"index_type": idx_type, "metric_type": metric, "params": params},
        )
    except Exception:
        pass
    return coll

def write_checkpoint_atomic(checkpoints_root: Path, stem: str, payload: dict):
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    done_path = checkpoints_root / f"{stem}.done"
    tmp_path = checkpoints_root / f"{stem}.done.tmp"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(done_path)
    return done_path

def stream_records(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="19530")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path).resolve()
    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}")
        sys.exit(2)

    config = load_config()
    connections.connect("default", host=args.host, port=args.port)
    collection = ensure_collection(config)

    checkpoints_root = Path(config.get("paths", {}).get("checkpoints_root", "./data/checkpoints")).resolve()
    stem = jsonl_path.stem
    lock_path = checkpoints_root / f"{stem}.lock"
    done_path = checkpoints_root / f"{stem}.done"

    if done_path.exists():
        print(f"Already ingested (found): {done_path}")
        return

    if lock_path.exists():
        print(f"Another ingest is in progress for {stem}, skipping.")
        return
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("lock", encoding="utf-8")

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        batch_size = int(config.get("batch_size", 512))

        texts, records = [], []
        total_count = 0

        def flush_batch():
            nonlocal texts, records
            if not texts:
                return 0
            embeddings = model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            rows = []
            for r, e in zip(records, embeddings):
                s = extract_scalar_fields(r)
                row = {
                    "id": s["id"],
                    "embedding": e.tolist(),
                    "title": s["title"],
                    "category": s["category"],
                    "sub_category": s["sub_category"],
                    "risk": s["risk"],
                    "tags": s["tags"],
                    "os": s["os"],
                    "system": s["system"],
                    # New: store full source JSON so Attu shows everything
                    "raw": r,
                }
                rows.append(row)
            collection.insert(rows)
            print(f"Inserted batch of {len(rows)} records")
            texts.clear()
            records.clear()
            return len(rows)

        for rec in stream_records(jsonl_path):
            total_count += 1
            texts.append(build_embedding_text(rec))
            records.append(rec)
            if len(texts) >= batch_size:
                flush_batch()

        flush_batch()

        payload = {
            "source_file": str(jsonl_path),
            "collection_name": collection.name,
            "records_ingested": total_count,
            "model_name": config.get("model_name", "unknown"),
            "vector_dim": config.get("collection", {}).get("dim", 384),
            "ingested_at": datetime.now(UTC).isoformat(),
        }
        written = write_checkpoint_atomic(checkpoints_root, stem, payload)
        print(f"Checkpoint written: {written}")
        print(f"Total records ingested: {total_count}")
        print("Ingestion completed successfully!")
    finally:
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    main()
