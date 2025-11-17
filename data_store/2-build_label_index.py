# build_label_index.py
import sqlite3
import json
import unicodedata
import re
from collections import defaultdict
from pathlib import Path
import os
from tqdm import tqdm  # progress bar

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WIKIDATA_DB = os.path.join(BASE_DIR, "kg_wikidata5m.db")  # Wikidata5M
META_QA_DB = os.path.join(BASE_DIR, "kg_small.db")        # Meta QA
LABEL_INDEX_OUT = os.path.join(BASE_DIR, "label_index.json")
QID_TO_LABELS_OUT = os.path.join(BASE_DIR, "qid_to_labels.json")
  # optional JSON labels

# --------------------
def normalize_label(s: str) -> str:
    """Normalize a label: remove diacritics, lowercase, remove punctuation, collapse spaces."""
    if not s:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

Path(LABEL_INDEX_OUT).parent.mkdir(parents=True, exist_ok=True)

label_to_ids = defaultdict(set)  # normalized_label -> set(QIDs)
id_to_labels = defaultdict(set)  # QID -> set(labels/aliases)

# --------------------
def process_wikidata(db_path):
    """Process Wikidata DB with triples + aliases."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Load all aliases once
    aliases_dict = defaultdict(list)
    cur.execute("SELECT id, alias FROM aliases")
    for qid, alias in cur.fetchall():
        aliases_dict[str(qid)].append(alias)

    # Process all QIDs in triples
    cur.execute("SELECT DISTINCT subject FROM triples UNION SELECT DISTINCT object FROM triples")
    qids = [row[0] for row in cur.fetchall()]
    print(f"Processing {len(qids)} QIDs from Wikidata DB...")
    for qid in tqdm(qids, desc="Wikidata QIDs"):
        labels = aliases_dict.get(str(qid), [])
        for lbl in labels:
            if not lbl:
                continue
            norm = normalize_label(lbl)
            label_to_ids[norm].add(qid)
            id_to_labels[str(qid)].add(lbl)

    con.close()

# --------------------
def process_metaqa(db_path):
    """Process MetaQA DB (triples with subj_label / obj_label, no aliases table)."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("SELECT subject, subj_label, object, obj_label FROM triples")
    rows = cur.fetchall()
    print(f"Processing {len(rows)} triples from Meta QA DB...")
    for subj, subj_lbl, obj, obj_lbl in tqdm(rows, desc="Meta QA triples"):
        for qid, lbl in [(subj, subj_lbl), (obj, obj_lbl)]:
            if not lbl or not qid:
                continue
            qid = str(qid)
            norm = normalize_label(lbl)
            label_to_ids[norm].add(qid)
            id_to_labels[qid].add(lbl)

    con.close()

# --------------------
# Process both DBs
if Path(WIKIDATA_DB).exists():
    process_wikidata(WIKIDATA_DB)
if Path(META_QA_DB).exists():
    process_metaqa(META_QA_DB)

# --- Convert sets to sorted lists ---
label_index = {k: sorted(list(v)) for k, v in label_to_ids.items()}
qid_labels = {k: sorted(list(v)) for k, v in id_to_labels.items()}

# --- Save JSON ---
with open(LABEL_INDEX_OUT, "w", encoding="utf-8") as f:
    json.dump(label_index, f, ensure_ascii=False, indent=2)

with open(QID_TO_LABELS_OUT, "w", encoding="utf-8") as f:
    json.dump(qid_labels, f, ensure_ascii=False, indent=2)

print(f"✅ Label index saved: {LABEL_INDEX_OUT} ({len(label_index)} normalized labels)")
print(f"✅ QID→labels saved: {QID_TO_LABELS_OUT} ({len(qid_labels)} QIDs)")

