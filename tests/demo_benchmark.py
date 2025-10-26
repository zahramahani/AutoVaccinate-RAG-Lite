# demo_benchmark.py
import time, sqlite3, os, json
from build_lmdb_from_sqlite import build_label_and_qid_lmdb, build_sp_lmdb, LMDB_LABEL, LMDB_QID, LMDB_SP
from linker import link_entity
from detector import sp_has_object

# Create small demo SQLite (metaqa-like)
DEMO_DB = "demo_small.db"
if os.path.exists(DEMO_DB):
    os.remove(DEMO_DB)
con = sqlite3.connect(DEMO_DB)
cur = con.cursor()
cur.execute("CREATE TABLE triples (subject TEXT, predicate TEXT, object TEXT, subj_label TEXT, obj_label TEXT)")
rows = [
    ("Q1", "born_in", "Q2", "Alan Turing", "London"),
    ("Q3", "played_for", "Q4", "Messi", "PSG"),
    ("Q1", "died_in", "Q5", "Alan Turing", "Princeton"),
]
cur.executemany("INSERT INTO triples VALUES (?,?,?,?,?)", rows)
con.commit()
con.close()

# Build LMDBs for demo (use metaqa path for both functions)
build_label_and_qid_lmdb("", DEMO_DB)
build_sp_lmdb(DEMO_DB)

# Benchmark label lookup
start = time.time()
res = link_entity("Alan Turing")
t = time.time() - start
print("link_entity('Alan Turing') ->", res, "time:", t)

# Benchmark SP membership
start = time.time()
print("sp_has_object Q1 born_in Q2:", sp_has_object("Q1", "born_in", "Q2"))
print("time:", time.time()-start)
