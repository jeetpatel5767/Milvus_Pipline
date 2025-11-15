Step-by-step behavior

Detect new file event

Trigger when a .json or .jsonl appears in ./data/inbox or is modified.

Validate and normalize

If .json:

Parse:

If it’s a single object → wrap as a single record

If it’s an array → multiple records

Output JSONL: one record per line

If .jsonl:

Validate each line is valid JSON object

Copy as-is to staging JSONL

Extract categorization info

Read the first record from the normalized data.

Extract:

category = record.important.category

sub_category = record.important.sub_category

title = record.important.title (for slug)

Auto-create destination folders

Ensure:

./data/datasets/<category>/<sub_category>/

If they don’t exist, create them.

Compute destination filename

slug = lowercase(title), spaces/specials → underscores

If no title, fallback to lowercase(category_subcategory)

Version:

If no file with _v1.jsonl exists in that folder → use _v1.jsonl

Else increment: _v2.jsonl, _v3.jsonl, etc.

Example:

./data/datasets/SQL Injection/Authentication Bypass/sql_injection_login_bypass_v1.jsonl

Move normalized file

Move from staging to the destination path.

Log the move with source → destination.

Write a small manifest (optional but helpful)

In the same destination folder, create/append a MANIFEST.jsonl with an entry:

{ "file": "<name>.jsonl", "records": <count>, "moved_at": "<timestamp>" }

Error handling

If parsing fails, write an error log in ./data/logs and move the original file to ./data/processed/errors/ with a .failed suffix.

Do not crash; keep watching.

Do NOT ingest yet

In this phase, we stop after organizing the file. In the next phase, we’ll wire ingestion (embeddings + Milvus) to run automatically after the move.