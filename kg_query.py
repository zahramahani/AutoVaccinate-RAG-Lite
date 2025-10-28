#kg_query.py
from lmdb_connector import lmdb_qid_get, get_triples_for_subjects
import logging

def get_kg_triples(qid_label_pairs, limit_per_q=10):
    """
    LMDB-based KG retrieval:
    - qids: list of subject QIDs
    - limit_per_q: max triples per subject
    Returns: list of (subject_aliases, predicate_text, object_aliases)
    """
    triples_context = []
    qids = [qid for qid, _ in qid_label_pairs]

    raw_triples = get_triples_for_subjects(qids, limit_per_q)
    if not raw_triples:
        logging.debug("⚠️ No triples retrieved from LMDB.")
        return []

    for triple in raw_triples:
        try:
            s_data = lmdb_qid_get(triple[0])
            o_data = lmdb_qid_get(triple[2])

            if not s_data or not o_data:
                logging.debug(f"⚠️ Missing LMDB entity for triple: {triple}")
                continue  # skip incomplete triples

            s = s_data[0]
            p = triple[1]
            o = o_data[0]
            triples_context.append((s, p, o))

        except Exception as e:
            logging.warning(f"❌ Error processing triple {triple}: {e}")
            continue

    return triples_context


def facts_to_text(triples):
    """
    Convert triples to readable text.
    """
    if not triples:
        return []

    return [f"{s} -- {p} --> {o}" for s, p, o in triples]


if __name__ == "__main__":
    test_qids = [("Q42", "sometext"), ("Q1", "some text")]
    triples = get_kg_triples(test_qids, limit_per_q=5)
    for line in facts_to_text(triples):
        print(line)
