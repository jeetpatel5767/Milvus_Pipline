# src/watcher.py
import time, subprocess, sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

INBOX = Path("data/inbox").resolve()
DATASETS = Path("data/datasets").resolve()
CHECKPOINTS = Path("data/checkpoints").resolve()
HOST = "localhost"
PORT = "19530"

IGNORES = (".tmp", ".part", ".crdownload", "~")
EXCLUDE_NAMES = {"MANIFEST.jsonl"}  # do not ingest audit manifests

def is_ready(p: Path, wait_ms=500):
    """Wait until file size is stable."""
    if not p.exists():
        return False
    s1 = p.stat().st_size
    time.sleep(wait_ms / 1000)
    return p.exists() and p.stat().st_size == s1

def ingest_pending():
    """Find any datasets/*.jsonl without a .done checkpoint and ingest them."""
    for jsonl in DATASETS.rglob("*.jsonl"):
        if jsonl.name in EXCLUDE_NAMES:
            continue
        done = CHECKPOINTS / f"{jsonl.stem}.done"
        lock = CHECKPOINTS / f"{jsonl.stem}.lock"
        # Skip if already ingested or currently being ingested
        if done.exists() or lock.exists():
            continue
        print(f"[watcher] Ingesting: {jsonl}")
        subprocess.run(
            [sys.executable, "-m", "src.ingest_to_milvus", str(jsonl), "--host", HOST, "--port", PORT],
            check=True,
        )

def normalize_one(src: Path):
    print(f"[watcher] Normalizing: {src}")
    subprocess.run([sys.executable, "-m", "src.normalize_and_move", str(src)], check=True)

class InboxHandler(FileSystemEventHandler):
    def __init__(self, debounce_ms=1200):
        super().__init__()
        self.debounce_ms = debounce_ms
        self.last_ts = {}  # path -> last processed time (ms)

    def on_created(self, event):
        self._maybe_process(event)
    def on_moved(self, event):
        self._maybe_process(event)
    def on_modified(self, event):
        # some editors write, then rename; debounce by checking readiness
        self._maybe_process(event)

    def _maybe_process(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() not in {".json", ".jsonl"}:
            return
        if any(str(p).endswith(s) for s in IGNORES):
            return
        if not is_ready(p):
            return

        # Debounce rapid duplicate events for the same path
        now_ms = int(time.time() * 1000)
        key = str(p.resolve())
        last = self.last_ts.get(key, 0)
        if now_ms - last < self.debounce_ms:
            return
        self.last_ts[key] = now_ms

        try:
            normalize_one(p)
            ingest_pending()
        except subprocess.CalledProcessError as e:
            print(f"[watcher] Error: {e}", file=sys.stderr)

def main():
    INBOX.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)

    obs = Observer()
    handler = InboxHandler()
    obs.schedule(handler, str(INBOX), recursive=True)
    obs.start()
    print(f"[watcher] Watching: {INBOX}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()

if __name__ == "__main__":
    main()
