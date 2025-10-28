# # build_label_index.py
# import sqlite3
# import json
# import unicodedata
# import re
# from collections import defaultdict
# from pathlib import Path

# SQLITE_DB = "./kg_small.db"
# LABEL_INDEX_OUT = "./label_index.json"
# QID_TO_LABELS_OUT = "./qid_to_labels.json"
# MANUAL_ALIASES = "./manual_aliases.json"

# def normalize_label(s: str) -> str:
#     """Normalize a label for matching: unicode-normalize, remove diacritics, lowercase,
#        remove punctuation (keep word chars & spaces), collapse whitespace."""
#     if not s:
#         return ""
#     s = str(s).strip()
#     s = unicodedata.normalize("NFKD", s)
#     s = "".join(ch for ch in s if not unicodedata.combining(ch))
#     s = s.lower()
#     s = re.sub(r"[^\w\s]", " ", s)      # punctuation -> space
#     s = re.sub(r"\s+", " ", s).strip()  # collapse spaces
#     return s

# Path(LABEL_INDEX_OUT).parent.mkdir(parents=True, exist_ok=True)

# con = sqlite3.connect(SQLITE_DB)
# cur = con.cursor()

# # SELECT subj_label and subject (subject is canonical id stored in DB)
# cur.execute("SELECT subj_label, subject FROM triples UNION SELECT obj_label, object FROM triples")
# rows = cur.fetchall()

# label_to_ids = defaultdict(set)   # normalized_label -> set(subject_value)
# id_to_labels = defaultdict(set)   # subject_value -> set(original_labels)

# for raw_label, subject_value in rows:
#     if not raw_label:
#         continue
#     norm = normalize_label(raw_label)
#     subj_val = subject_value if subject_value is not None else ""
#     subj_val = str(subj_val)
#     label_to_ids[norm].add(subj_val)
#     id_to_labels[subj_val].add(raw_label)

# # convert sets to lists for JSON serialization (sorted for determinism)
# label_index = {k: sorted(list(v)) for k, v in label_to_ids.items()}
# qid_labels = {k: sorted(list(v)) for k, v in id_to_labels.items()}

# with open(LABEL_INDEX_OUT, "w", encoding="utf-8") as f:
#     json.dump(label_index, f, ensure_ascii=False, indent=2)

# with open(QID_TO_LABELS_OUT, "w", encoding="utf-8") as f:
#     json.dump(qid_labels, f, ensure_ascii=False, indent=2)

# # create a manual alias file if not exists (user can edit to add domain aliases)
# if not Path(MANUAL_ALIASES).exists():
#     # example: map common short forms to canonical labels (normalized)
#     example = {
#         # format: "normalized_alias": ["normalized_canonical_label1", "normalized_canonical_label2"]
#         # add your own entries, e.g. "49ers": ["san francisco 49ers"]
#         "49ers": ["san francisco 49ers"],
#         "sf": ["san francisco"]
#     }
#     with open(MANUAL_ALIASES, "w", encoding="utf-8") as f:
#         json.dump(example, f, ensure_ascii=False, indent=2)

# print(f"Saved label index: {LABEL_INDEX_OUT} ({len(label_index)} normalized labels)")
# print(f"Saved id->labels: {QID_TO_LABELS_OUT} ({len(qid_labels)} ids)")
# print(f"Manual aliases file (editable): {MANUAL_ALIASES}")
# con.close()
# build_label_index.py
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

