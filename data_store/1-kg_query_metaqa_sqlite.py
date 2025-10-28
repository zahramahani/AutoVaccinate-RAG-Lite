# build_kg.py
import sqlite3
import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace

CSV_TRIPLES = "./data/Meta_qa/Meta_qa_kb.csv"
TTL_OUT = "./kg_small.ttl"
SQLITE_DB = "./kg_small.db"

EX = Namespace("http://example.org/")

# load CSV
df = pd.read_csv(CSV_TRIPLES, keep_default_na=False).fillna("")

# build RDF graph
g = Graph()
for _, row in df.iterrows():
    subj_qid = row.get("subject_qid") or row["subject_label"]
    obj_qid  = row.get("object_qid") or row["object_label"]
    pred = row["predicate"].replace(" ", "_")
    s = URIRef(EX + str(subj_qid))
    p = URIRef(EX + pred)
    o = Literal(row["object_label"]) if row.get("object_qid", "") == "" else URIRef(EX + str(obj_qid))
    g.add((s, p, o))

# persist TTL
g.serialize(TTL_OUT, format="turtle")
print("Saved RDF TTL:", TTL_OUT)

# build sqlite triple store (subject, predicate, object, subj_label, obj_label)
con = sqlite3.connect(SQLITE_DB)
cur = con.cursor()
cur.execute("DROP TABLE IF EXISTS triples;")
cur.execute("""
CREATE TABLE IF NOT EXISTS triples (
    subject TEXT,
    predicate TEXT,
    object TEXT,
    subj_label TEXT,
    obj_label TEXT
);
""")
cur.execute("CREATE INDEX IF NOT EXISTS idx_subj ON triples(subject);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_pred ON triples(predicate);")
cur.execute("CREATE INDEX IF NOT EXISTS idx_obj ON triples(object);")

rows = []
for _, row in df.iterrows():
    rows.append((
        str(row.get("subject_qid") or row["subject_label"]),
        row["predicate"],
        str(row.get("object_qid") or row["object_label"]),
        row["subject_label"],
        row["object_label"]
    ))
cur.executemany("INSERT INTO triples VALUES (?,?,?,?,?)", rows)
con.commit()
con.close()
print("Saved sqlite DB:", SQLITE_DB)