import sys
import json
import argparse
from pathlib import Path

def load_config():
    # Robust YAML loader with sane defaults
    try:
        import yaml
    except ImportError:
        print("Missing dependency: pyyaml. Install with: pip install pyyaml")
        sys.exit(2)

    cfg_path = Path(__file__).resolve().parents[1] / "data" / "config" / "embedding.yaml"
    model_name, batch_size = "all-MiniLM-L6-v2", 512
    if not cfg_path.exists():
        return model_name, batch_size

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if isinstance(data, dict):
        if "model_name" in data:
            model_name = str(data["model_name"])
        if "batch_size" in data:
            try:
                batch_size = int(data["batch_size"])
            except Exception:
                pass
    return model_name, batch_size

FIELDS = [
    ("important", "title"),
    ("important", "category"),
    ("important", "sub_category"),
    ("important", "tags"),
    ("important", "targets", "os"),
    ("important", "targets", "system"),
    ("important", "risk"),
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

def build_embedding_text(rec):
    parts = []
    for p in FIELDS:
        parts.append(to_text(get_in(rec, p)))
    return " ".join(x for x in parts if x).strip()

def stream_records(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                yield obj
            except Exception:
                continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--out", default="", help="Optional debug JSONL output")
    args = parser.parse_args()

    jsonl = Path(args.jsonl_path).resolve()
    if not jsonl.exists():
        print(f"File not found: {jsonl}")
        sys.exit(2)

    model_name, batch_size = load_config()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Missing dependency: sentence-transformers. Install with: pip install sentence-transformers")
        sys.exit(2)

    model = SentenceTransformer(model_name)

    texts, metas = [], []
    total = 0
    out_f = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.out, "w", encoding="utf-8")

    def flush_batch():
        nonlocal texts, metas
        if not texts:
            return None
        vecs = model.encode(
            texts,
            batch_size=min(batch_size, len(texts)),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vecs

    first_vec = None

    for rec in stream_records(jsonl):
        total += 1
        emb_text = build_embedding_text(rec)
        texts.append(emb_text)
        metas.append(
            {
                "id": rec.get("id"),
                "title": get_in(rec, ("important", "title")),
                "len": len(emb_text),
            }
        )
        if len(texts) >= batch_size:
            vecs = flush_batch()
            if first_vec is None and vecs is not None and len(vecs) > 0:
                # store as a list to simplify preview logic
                first_vec = vecs[0].tolist() if hasattr(vecs, "tolist") else list(vecs)
            if out_f:
                for m, v in zip(metas, vecs):
                    v_head = v[:8].tolist() if hasattr(v, "tolist") else list(v[:8])
                    out_f.write(
                        json.dumps(
                            {
                                "id": m["id"],
                                "title": m["title"],
                                "text_len": m["len"],
                                "embedding_head": [float(x) for x in v_head],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            texts, metas = [], []

    # last batch
    if texts:
        vecs = flush_batch()
        if first_vec is None and vecs is not None and len(vecs) > 0:
            first_vec = vecs[0].tolist() if hasattr(vecs, "tolist") else list(vecs)
        if out_f and vecs is not None:
            for m, v in zip(metas, vecs):
                v_head = v[:8].tolist() if hasattr(v, "tolist") else list(v[:8])
                out_f.write(
                    json.dumps(
                        {
                            "id": m["id"],
                            "title": m["title"],
                            "text_len": m["len"],
                            "embedding_head": [float(x) for x in v_head],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if out_f:
        out_f.close()

    # Preview
    dim = len(first_vec) if first_vec is not None else 0
    head = []
    if first_vec:
        seq = first_vec[:5] if isinstance(first_vec, (list, tuple)) else [first_vec]
        head = [round(float(x), 6) for x in seq]

    print(f"Records: {total}")
    print(f"Vector dim: {dim}")
    if head:
        print(f"First vector (first 5 vals): {head}")

if __name__ == "__main__":
    main()
