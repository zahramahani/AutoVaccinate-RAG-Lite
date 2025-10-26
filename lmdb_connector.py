#lmdb_connector.py
import re
import lmdb, zlib, json
from functools import lru_cache
import logging
try:
    from pyroaring import BitMap
    ROARING = False # impelement bit map later
except Exception:
    ROARING = False



logger = logging.getLogger(__name__)
BASE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))

LMDB_LABEL = lmdb.open(f"{BASE}/data_store/lmdb/lmdb_label_index", readonly=True, lock=False,map_size=10 * 1024 * 1024 * 1024 )
LMDB_QID = lmdb.open(f"{BASE}/data_store/lmdb/lmdb_qid_labels", readonly=True, lock=False,map_size=10 * 1024 * 1024 * 1024 )
LMDB_FIRST = lmdb.open(f"{BASE}/data_store/lmdb/lmdb_first_letter", readonly=True, lock=False,map_size=10 * 1024 * 1024 * 1024 )
LMDB_SP = lmdb.open(f"{BASE}/data_store/lmdb/lmdb_sp", readonly=True, lock=False, map_size=10 * 1024 * 1024 * 1024)
if ROARING:
    LMDB_QID_INT = lmdb.open(f"{BASE}/data_store/lmdb/lmdb_qid_int", readonly=True, lock=False, map_size=10 * 1024 * 1024 * 1024)
    LMDB_INT_QID = lmdb.open(f"{BASE}/data_store/lmdb/lmdb_int_qid", readonly=True, lock=False, map_size=10 * 1024 * 1024 * 1024)

# -------------------- Helpers --------------------
def normalize_pred(pred: str) -> str:
    """Normalize predicate text for KG lookup."""
    return re.sub(r"[\s\-]+", "_", pred.strip().lower())

# --------------------- utilities ---------------------
def unpack(data):
    return json.loads(zlib.decompress(data).decode("utf-8"))

@lru_cache(maxsize=200_000)
def lmdb_label_get(norm_label):
    """Return all QIDs for a normalized label, handling chunked keys."""
    # --- Validate input ---
    if not norm_label or not isinstance(norm_label, str):
        logger.warning(f"[lmdb_label_get] Invalid label: {norm_label!r}")
        return []

    norm_label = norm_label.strip()
    if not norm_label:
        logger.warning("[lmdb_label_get] Empty label after stripping.")
        return []

    qids = []
    try:
        with LMDB_LABEL.begin() as txn:
            i = 0
            while True:
                key = f"{norm_label}#{i}".encode("utf-8") if i > 0 else norm_label.encode("utf-8")
                if not key:
                    break  # LMDB disallows empty keys
                v = txn.get(key)
                if not v:
                    break
                qids.extend(unpack(v))
                i += 1
    except lmdb.BadValsizeError as e:
        logger.error(f"[lmdb_label_get] Bad value size for label {norm_label!r}: {e}")
        return []
    except Exception as e:
        logger.exception(f"[lmdb_label_get] Unexpected error for label {norm_label!r}: {e}")
        return []

    return qids

@lru_cache(maxsize=200_000)
def lmdb_qid_get(qid):
    with LMDB_QID.begin() as txn:
        v = txn.get(qid.encode("utf-8"))
        return unpack(v) if v else []

@lru_cache(maxsize=10_000)
def lmdb_first_bucket(ch):
    with LMDB_LABEL.begin() as txn:
        v = txn.get(ch.encode("utf-8"))
        return unpack(v) if v else []

# If you used QID->int mapping and roaring bitmaps:
@lru_cache(maxsize=1_000_00)
def qid_to_int(qid):
    with LMDB_QID_INT.begin() as txn:
        v = txn.get(qid.encode("utf-8"))
        return int(v.decode("utf-8")) if v else None
    
@lru_cache(maxsize=200_000)
def get_all_aliases(qid):
    """
    Return all aliases for a QID from LMDB.
    """
    with LMDB_QID.begin() as txn:
        v = txn.get(qid.encode("utf-8"))
        # print("all aliases are",unpack(v))
        return unpack(v) if v else []

def get_qids_for_label(label: str):
    """
    Look up normalized label in LMDB and return list of QIDs (or property IDs).
    """
    label_norm = label.lower().strip()
    with LMDB_LABEL.begin() as txn:
        data = txn.get(label_norm.encode("utf-8"))
        if data:
            return unpack(data)
    return []

def get_triples_for_subjects(qids, limit_per_q=10):
    """
    Efficiently retrieve triples for multiple QIDs using a single LMDB cursor scan.
    Each key in LMDB_SP is expected to be 'QID|predicate' or just 'QID'.
    """
    triples = []
    with LMDB_SP.begin() as txn:
        cursor = txn.cursor()
        for qid in qids:
            prefix = f"{qid}|".encode("utf-8")
            # Efficiently position cursor at prefix start
            if not cursor.set_range(prefix):
                continue
            while True:
                key, value = cursor.item()
                if not key.startswith(prefix):
                    break
                try:
                    key_decoded = key.decode("utf-8")
                    parts = key_decoded.split("|", 1)
                    pred = parts[1] if len(parts) == 2 else "related_to"
                    objs = unpack(value)
                    for obj in objs[:limit_per_q]:
                        triples.append((qid, pred, obj))
                except Exception:
                    pass
                if not cursor.next():
                    break
    return triples

def sp_has_object(subject, predicate, object_qid):
    # fast path: if roaring bitmaps exist
    if ROARING:
        key = f"{subject}|{predicate}".encode("utf-8")
        with LMDB_SP.begin() as txn:
            v = txn.get(key)
            if not v:
                return False
            bm = BitMap.deserialize(v)
            oi = qid_to_int(object_qid)
            if oi is None:
                return False
            return oi in bm
    # fallback: list membership (still fast if lists are small)
    objs = sp_get_objects_as_list(subject, predicate)
    return object_qid in set(objs)

@lru_cache(maxsize=100_000)
def sp_get_objects_as_list(subject, predicate):
    key = f"{subject}|{predicate}".encode("utf-8")
    with LMDB_SP.begin() as txn:
        v = txn.get(key)
        if not v: 
            return []
        # if roaring stored: try to detect
        if ROARING:
            try:
                # detect by attempt to deserialize
                bm = BitMap.deserialize(v)
                # convert to ints list (but better to test membership directly)
                return bm.to_list()
            except Exception:
                return unpack(v)
        else:
            return unpack(v)

@lru_cache(maxsize=10_000)
def text_to_pred_qids(pred_text: str):
    """
    Given predicate text (like 'born in'), return all matching predicate QIDs (like ['P17']).
    """
    pred_norm = normalize_pred(pred_text)
    qids = get_qids_for_label(pred_norm)
    return qids or [pred_text]  # fallback to raw text if not found
