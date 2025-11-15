Embedding text rule:

For each record, build one string by concatenating these fields in order:

important.title

important.category

important.sub_category

important.tags (joined by spaces)

important.targets.os (joined by spaces)

important.targets.system (joined by spaces)

important.risk

Do not include variants, usage, or system fields in the embedding text.

Example using your sample:
"SQL injection login bypass SQL Injection Authentication Bypass database login auth sql Any Windows Linux Web App MySQL MSSQL medium"


Scalar mapping (what we store alongside vectors):

id → id

important.title → title

important.category → category

important.sub_category → sub_category

important.tags → tags

important.targets.os → os

important.targets.system → system

important.risk → risk

Reason: these fields are enough for filters (e.g., category=XSS, os=Windows) and for showing basic info in results. The full JSONL on disk remains the source of truth.