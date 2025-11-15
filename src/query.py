from pymilvus import connections, Collection

connections.connect(alias="default", host="localhost", port="19530")
c = Collection("attacks_v2")
c.load()
rows = c.query(expr="*", limit=5, output_fields=["id", "title", "$meta"])
for r in rows:
    print(r)
