# linker.py
import re, unicodedata
from rapidfuzz import process, fuzz
import spacy
import logging
from lmdb_connector import lmdb_label_get,lmdb_qid_get,lmdb_first_bucket
nlp = spacy.load("en_core_web_sm")

import re
import unicodedata

def normalize_label(s: str) -> str:
    """
    Normalize a label for LMDB lookups:
    - Removes accents, punctuation, and extra spaces
    - Converts to lowercase
    - Ensures safe ASCII/UTF-8 key form
    - Returns '' for invalid or non-informative inputs
    """
    if not s or not isinstance(s, str):
        return ""

    # Strip and normalize Unicode
    s = s.strip()
    if not s:
        return ""

    try:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    except Exception:
        # In case unicodedata fails on weird surrogate characters
        s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # Lowercase, remove punctuation except underscores and alphanumerics
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)  # replace non-word chars with space
    s = re.sub(r"\s+", " ", s).strip()

    # Avoid dangerous keys (LMDB disallows zero-length keys)
    if not s or len(s) > 2000:  # also avoid huge normalization results
        return ""

    # Remove leading/trailing underscores and numerics (optional)
    s = s.strip("_ ")

    return s

logger = logging.getLogger(__name__)
from functools import lru_cache

def link_entity(label, topk=3, fuzz_threshold=80):
    """Link a text label to candidate QIDs using LMDB lookup and fuzzy match fallback."""
    if not label or not isinstance(label, str):
        logger.debug(f"[link_entity] Skipping invalid label: {label!r}")
        return []

    label_n = normalize_label(label)
    if not label_n:
        logger.debug(f"[link_entity] Normalized label is empty for {label!r}")
        return []

    # --- Exact lookup ---
    try:
        qids = lmdb_label_get(label_n)
    except Exception as e:
        logger.error(f"[link_entity] Error in exact lookup for '{label_n}': {e}")
        qids = []

    if qids:
        results = []
        for qid in qids[:topk]:
            qid_label = lmdb_qid_get(qid)
            results.append((qid, qid_label[0] if qid_label else qid))
        return results

    # --- Fuzzy matching fallback ---
    first = label_n[0] if label_n else ''
    try:
        candidates = lmdb_first_bucket(first)
    except Exception as e:
        logger.error(f"[link_entity] Failed to fetch bucket for '{first}': {e}")
        candidates = []

    if not candidates:
        logger.debug(f"[link_entity] No candidates found for bucket '{first}'")
        return []

    results = process.extract(label_n, candidates, scorer=fuzz.WRatio, limit=20)
    collected = []
    seen = set()

    for cand_label, score, _ in results:
        if score < fuzz_threshold:
            continue

        try:
            qids_c = lmdb_label_get(cand_label)
        except Exception as e:
            logger.warning(f"[link_entity] Error fetching qids for '{cand_label}': {e}")
            continue

        for q in qids_c:
            if q not in seen:
                seen.add(q)
                collected.append(q)

        if len(collected) >= topk:
            break

    out = []
    for q in collected[:topk]:
        q_label = lmdb_qid_get(q)
        out.append((q, q_label[0] if q_label else q))

    return out

@lru_cache(maxsize=50000)
def link_entity_cached(label, topk=3, fuzz_threshold=80):
    return link_entity(label, topk, fuzz_threshold)

def extract_entities(text: str):
    doc = nlp(text)
    ents = [ent.text for ent in doc.ents]
    return list(dict.fromkeys(ents))
