# build_lmdb_from_sqlite.py
import sqlite3
import lmdb
from collections import defaultdict
from pathlib import Path
import os
from tqdm import tqdm
import zlib, json
from itertools import islice

# Config: paths (adjust as needed)
BASE = os.path.dirname(os.path.abspath(__file__))
WIKIDATA_DB = os.path.join(BASE,"data_store", "kg_wikidata5m.db")   # has triples + aliases
META_QA_DB   = os.path.join(BASE, "data_store","kg_small.db")       # triples with subj_label/obj_label
OUT_DIR = os.path.join(BASE, "data_store","lmdb")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

LMDB_LABEL = os.path.join(OUT_DIR, "lmdb_label_index")
LMDB_QID = os.path.join(OUT_DIR, "lmdb_qid_labels")
LMDB_FIRST = os.path.join(OUT_DIR, "lmdb_first_letter")
LMDB_SP = os.path.join(OUT_DIR, "lmdb_sp")
LMDB_QID_INT = os.path.join(OUT_DIR, "lmdb_qid_int")    # qid->int mapping
LMDB_INT_QID = os.path.join(OUT_DIR, "lmdb_int_qid")    # int->qid mapping

# tune map sizes based on available disk (bytes)
MAP_SIZE = 40 * 1024**3  # 40GB - adjust to your disk size

def pack(obj):
    return zlib.compress(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

def unpack(b):
    return json.loads(zlib.decompress(b).decode("utf-8"))

def normalize_label(s: str) -> str:
    import unicodedata, re
    if not s:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --------------- Build label_index & qid->labels ---------------
# Strategy:
# - For Wikidata: aliases table provides many labels per qid
# - For MetaQA: subj_label / obj_label provide labels
# We'll stream aliases and triples and write to LMDB.

def build_label_and_qid_lmdb(wikidb_path, metaqa_path):
    # Open LMDB envs
    env_label = lmdb.open(LMDB_LABEL, map_size=MAP_SIZE)
    env_qid = lmdb.open(LMDB_QID, map_size=MAP_SIZE)
    env_first = lmdb.open(LMDB_FIRST, map_size=MAP_SIZE)
    # optional qid->int mapping (build if you will use roaring bitmaps)
    # env_qid_int = lmdb.open(LMDB_QID_INT, map_size=1*(1024**3))
    # env_int_qid = lmdb.open(LMDB_INT_QID, map_size=1*(1024**3))

    # temporary in-memory accumulators for labels -> qids and qid -> aliases
    label_to_qids = defaultdict(list)
    qid_to_labels = defaultdict(list)
    first_letter_buckets = defaultdict(list)
    next_int = 1

    # Process Wikidata aliases first (if exists)
    if os.path.exists(wikidb_path):
        con = sqlite3.connect(wikidb_path)
        cur = con.cursor()
        print("Reading aliases from Wikidata DB...")
        cur.execute("SELECT id, alias FROM aliases")
        for qid, alias in tqdm(cur.fetchall(), desc="aliases"):
            if not alias:
                continue
            qid = str(qid)
            norm = normalize_label(alias)
            label_to_qids[norm].append(qid)
            qid_to_labels[qid].append(alias)
            if norm:
                first_letter_buckets[norm[0]].append(norm)
        con.close()

    # Process MetaQA labels (subj_label / obj_label)
    if os.path.exists(metaqa_path):
        con = sqlite3.connect(metaqa_path)
        cur = con.cursor()
        print("Reading MetaQA triple labels...")
        cur.execute("SELECT subject, subj_label, object, obj_label FROM triples")
        for subj, subj_lbl, obj, obj_lbl in tqdm(cur.fetchall(), desc="metaqa_triples"):
            for qid, lbl in ((subj, subj_lbl), (obj, obj_lbl)):
                if not lbl or not qid:
                    continue
                qid = str(qid)
                norm = normalize_label(lbl)
                label_to_qids[norm].append(qid)
                qid_to_labels[qid].append(lbl)
                if norm:
                    first_letter_buckets[norm[0]].append(norm)
        con.close()

    # Deduplicate and write to LMDB in batches
    
    def pack_ids(qids):
        return zlib.compress(json.dumps(qids, ensure_ascii=False).encode("utf-8"))

    def unpack_ids(data):
        return json.loads(zlib.decompress(data).decode("utf-8"))

    def chunked(iterable, n):
        it = iter(iterable)
        while chunk := list(islice(it, n)):
            yield chunk

    print("Writing label->qids and qid->labels to LMDB...")

    with env_label.begin(write=True) as txn_label, \
        env_qid.begin(write=True) as txn_qid, \
        env_first.begin(write=True) as txn_first:

        # label -> qids
        for lbl, qids in tqdm(label_to_qids.items(), desc="labels"):
            uniq = sorted(set(qids))
            blob = pack_ids(uniq)
            if len(blob) > 3900:  # avoid MDB_BAD_VALSIZE
                for i, chunk in enumerate(chunked(uniq, 500)):
                    key = f"{lbl}#{i}".encode("utf-8")
                    txn_label.put(key, pack_ids(chunk))
            else:
                txn_label.put(lbl.encode("utf-8"), blob)

        # qid -> labels
        for qid, labels in tqdm(qid_to_labels.items(), desc="qids"):
            uniq = sorted(set(labels))
            txn_qid.put(qid.encode("utf-8"), pack(uniq))

        # first-letter buckets
        for ch, labels in tqdm(first_letter_buckets.items(), desc="first_letters"):
            txn_first.put(ch.encode("utf-8"), pack(sorted(set(labels))))

    env_label.sync(); env_qid.sync(); env_first.sync()
    env_label.close(); env_qid.close(); env_first.close()

    print("Label LMDBs built.")

# --------------- Build SP store ---------------
# We'll build SP keys: "SP:{subject}|{predicate}" -> compressed list of object qids
# For large-degree nodes, consider updating to roaring bitmaps (requires int mapping).

def build_sp_lmdb(db_path, batch_size=2_000_000):
    """
    Build LMDB for SP lookups using first alias of subject, predicate, and object IDs.
    Keys: "subject_qid|normalized_pred_alias" -> list of object qids
    """
    env_sp = lmdb.open(LMDB_SP, map_size=MAP_SIZE)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Load all predicate aliases into memory for fast lookup
    pred_alias_map = {}
    try:
        cur.execute("SELECT id, alias FROM aliases WHERE id LIKE 'P%'")
        for pid, alias in cur.fetchall():
            if pid not in pred_alias_map:
                pred_alias_map[pid] = []
            pred_alias_map[pid].append(alias)
    except sqlite3.OperationalError:
        print("⚠️ No property aliases found, using raw predicate IDs")

    cur.execute("SELECT subject, predicate, object FROM triples")
    sp_map = defaultdict(list)
    count = 0
    pbar = tqdm(desc="SP rows")

    for subj, pred, obj in cur:
        # Normalize predicate using first alias if exists
        if pred in pred_alias_map and pred_alias_map[pred]:
            pred_text = normalize_label(pred_alias_map[pred][0])
        else:
            pred_text = normalize_label(pred)

        key = f"{subj}|{pred_text}"
        sp_map[key].append(obj)
        count += 1

        if count % batch_size == 0:
            # write batch to LMDB
            with env_sp.begin(write=True) as txn:
                for k, objs in sp_map.items():
                    txn.put(k.encode("utf-8"), pack(sorted(set(objs))))
            sp_map.clear()
            pbar.update(batch_size)

    # write remaining
    with env_sp.begin(write=True) as txn:
        for k, objs in sp_map.items():
            txn.put(k.encode("utf-8"), pack(sorted(set(objs))))
    pbar.update(count % batch_size)

    env_sp.sync()
    env_sp.close()
    con.close()
    print("SP LMDB built with predicate aliases.")


# --------------- main ---------------
if __name__ == "__main__":
    build_label_and_qid_lmdb(WIKIDATA_DB, META_QA_DB)
    # build SP store from both DBs: prefer to merge both triples into one source (or run both)
    if os.path.exists(WIKIDATA_DB):
        build_sp_lmdb(WIKIDATA_DB)
    if os.path.exists(META_QA_DB):
        # append metaqa triples to the same SP store (open and write)
        # for brevity you can call build_sp_lmdb on metaqa too, but that will overwrite; better to stream both into same function.
        pass

    print("All LMDB stores created.")
