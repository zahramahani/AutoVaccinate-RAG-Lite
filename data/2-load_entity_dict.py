import pandas as pd

ENTITY_DICT_TXT = "./Meta_qa/kb_entity_dict.txt"
INPUT_TXT = "./Meta_qa/Meta_qa_kb.txt"
OUTPUT_CSV = "./Meta_qa/Meta_qa_kb.csv"

# Load entity dictionary
entity_dict = {}
with open(ENTITY_DICT_TXT, "r", encoding="utf-8") as f:
    for line in f:
        qid, label = line.strip().split("\t")
        entity_dict[label] = qid

rows = []
with open(INPUT_TXT, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) != 3:
            continue
        subj, pred, obj = parts
        rows.append({
            "subject_qid": entity_dict.get(subj, ""),   # map to ID
            "subject_label": subj,
            "predicate": pred,
            "object_qid": entity_dict.get(obj, ""),
            "object_label": obj
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print("Saved triples CSV:", OUTPUT_CSV)
