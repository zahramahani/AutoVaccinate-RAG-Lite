# detector.py

import re
import json
import spacy
import unicodedata
import logging
from typing import List, Tuple, Optional
from transformers import pipeline
from difflib import SequenceMatcher
from linker import link_entity
from lmdb_connector import (
    sp_has_object,
    get_all_aliases,
    normalize_pred,
    text_to_pred_qids
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INFO] %(message)s")

# -----------------------------
# MODELS
# -----------------------------
nlp = spacy.load("en_core_web_sm")
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")


# -----------------------------
# BASIC UTILITIES
# -----------------------------
def normalize_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    return " ".join(t.split())


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _strip_code_fences(text: str) -> str:
    """
    Remove leading/trailing ``` or ```json fences but keep internal JSON intact.
    """
    if not text:
        return ""
    s = text.strip()
    # Leading fence: ``` or ```json etc.
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    # Trailing fence
    if s.endswith("```"):
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

# -----------------------------
# 1) Triple Parser for KB facts (supports ANY input)
# -----------------------------
def fact_to_triple(fact: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse explicit KG-style facts of the form:
        Subject -- predicate --> Object
    that can appear anywhere in the string.
    """
    if not fact:
        return None
    fact = fact.strip()
    # Use search, not match, so we handle "... Scott Derrickson -- country ... "
    m = re.search(r"(.+?)\s*--\s*(.+?)\s*-->\s*(.+)", fact)
    if not m:
        return None

    s, p, o = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    o = o.strip().rstrip(".;")
    return s, p, o

# -----------------------------
# 2) Triple Extraction from Natural Language
# -----------------------------
def extract_entities(sent: str) -> List[str]:
    doc = nlp(sent)
    return list({ent.text for ent in doc.ents})


def extract_predicate(sent: str) -> Optional[str]:
    doc = nlp(sent)
    # ROOT verb preferred
    for t in doc:
        if t.dep_ == "ROOT" and t.pos_ == "VERB":
            return t.lemma_

    # fallback
    verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"]
    return verbs[0] if verbs else None


def extract_triples(sent: str) -> List[Tuple[str, str, str]]:
    """
    High-precision SVO extractor for a single sentence.

    Strategy:
      - find ROOT verb
      - take nsubj/nsubjpass as subject
      - take dobj/attr as object
      - subject must contain a PROPN or named entity
      - filter obvious meta-nouns as subject/object ("facts", "evidence", etc.)
    """
    sent = sent.strip()
    if not sent:
        return []
    doc = nlp(sent)
    triples: List[Tuple[str, str, str]] = []

    META_NOUNS = {
        "facts", "evidence", "information", "question",
        "answer", "explanation", "genres", "series", "narration", "position"
    }

    for token in doc:
        if token.dep_ != "ROOT" or token.pos_ != "VERB":
            continue

        # candidate subjects & objects
        subj_tokens = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
        obj_tokens = [w for w in token.rights if w.dep_ in ("dobj", "attr")]

        if not subj_tokens or not obj_tokens:
            continue

        subj_text = " ".join(w.text for w in subj_tokens).strip()
        obj_text = " ".join(w.text for w in obj_tokens).strip()
        if not subj_text or not obj_text:
            continue

        if subj_text.lower() in META_NOUNS or obj_text.lower() in META_NOUNS:
            continue

        # require subject to contain a PROPN or named entity
        subj_doc = nlp(subj_text)
        has_propn = any(t.pos_ == "PROPN" for t in subj_doc)
        has_ent = any(e.label_ in ("PERSON", "ORG", "GPE", "LOC", "WORK_OF_ART", "FAC")
                      for e in subj_doc.ents)
        if not (has_propn or has_ent):
            continue

        triples.append((subj_text, token.lemma_, obj_text))

    return triples


# -----------------------------
# 3) Assertion Extractor (model answer or retrieved docs)
# -----------------------------
def extract_candidate_assertions(text: str) -> List[Tuple[str, str, str]]:
    """
    For arbitrary text (e.g., retrieved docs, non-JSON answers):
      1. extract explicit 'A -- P --> B' patterns anywhere
      2. if none found, run SVO extraction sentence-by-sentence
    """
    assertions: List[Tuple[str, str, str]] = []

    if not text:
        return assertions

    # 1) explicit KG-style triples
    triple_pattern = re.compile(r"(.+?)\s*--\s*(.+?)\s*-->\s*(.+)")
    for m in triple_pattern.finditer(text):
        s = m.group(1).strip()
        p = m.group(2).strip()
        o = m.group(3).strip().rstrip(".; ")
        o = o.strip("()").strip()
        assertions.append((s, p, o))

    # 2) fallback SVO triples if no explicit triples
    if not assertions:
        for sent in re.split(r"[.?!]", text):
            sent = sent.strip()
            if not sent:
                continue
            for tri in extract_triples(sent):
                assertions.append(tri)

    return assertions
# -----------------------------
# 6b) Parse model answer JSON facts
# -----------------------------
def normalize_model_fact_to_triple(fact: str) -> Optional[Tuple[str, str, str]]:
    """
    General patterns for natural-language facts.

    Supported:
      - "X's P is Y"          -> (X, P, Y)
      - "X is the/an R of Y"  -> (X, "R of", Y)
        (e.g. "X is a citizen of Y" -> (X, "citizen of", Y))
      - "X is Y"              -> (X, "is", Y) as a fallback
    """
    if not fact:
        return None
    f = fact.strip()

    # 1) "X's P is Y"
    #    e.g. "Scott Derrickson's country of citizenship is (US)"
    m = re.match(r"(.+?)'s\s+(.+?)\s+is\s+(.+)", f, flags=re.IGNORECASE)
    if m:
        subj = m.group(1).strip()
        pred = m.group(2).strip()
        obj = m.group(3).strip().strip(".;")
        obj = obj.strip("()").strip()
        return subj, pred, obj

    # 2) "X is (a/the) R of Y" (generic relational pattern)
    #    e.g. "Paris is the capital of France"
    #         -> (Paris, "capital of", France)
    m = re.match(r"(.+?)\s+is\s+(an?|the)?\s*(.+?)\s+of\s+(.+)", f, flags=re.IGNORECASE)
    if m:
        subj = m.group(1).strip()
        role = m.group(3).strip()
        obj = m.group(4).strip().strip(".;")
        obj = obj.strip("()").strip()
        pred = f"{role} of"
        return subj, pred, obj

    # 3) "X is Y" generic copula
    #    e.g. "Ed Wood is a film."
    m = re.match(r"(.+?)\s+is\s+(.+)", f, flags=re.IGNORECASE)
    if m:
        subj = m.group(1).strip()
        obj = m.group(2).strip().strip(".;")
        obj = obj.strip("()").strip()
        return subj, "is", obj

    return None

def extract_assertions_from_model_answer(raw_answer: str) -> List[Tuple[str, str, str]]:
    """
    Extract (s,p,o) assertions from a model answer.

    Works generically across many formats:
      - JSON with 'facts' / 'extracted_facts' / 'evidence_facts'
      - code-fenced JSON (```json ... ```)
      - plain text with or without 'A -- P --> B' patterns

    Algorithm:
      1) Strip code fences.
      2) Isolate JSON object between first '{' and last '}' (if present).
      3) If JSON parse OK:
          - collect fact strings from 'facts', 'extracted_facts', 'evidence_facts'
          - for each fact string:
              a) try 'A -- P --> B'
              b) try pattern 'X's P is Y', 'X is R of Y', 'X is Y'
              c) else, SVO triples from that sentence only
      4) If no JSON or no triples:
          - fallback: 'A -- P --> B' anywhere in raw answer
          - if still nothing: SVO triples sentence-by-sentence on raw answer
    """
    if not raw_answer:
        return []

    triples: List[Tuple[str, str, str]] = []

    # ---------- 1) strip code fences ----------
    txt = _strip_code_fences(raw_answer)

    # ---------- 2) try to isolate JSON ----------
    start = txt.find("{")
    end = txt.rfind("}")
    json_str = None
    if start != -1 and end != -1 and end > start:
        json_str = txt[start:end + 1]

    fact_strings: List[str] = []

    # ---------- 3) JSON parse & fact extraction ----------
    if json_str is not None:
        try:
            data = json.loads(json_str)
        except Exception:
            data = None
    else:
        data = None

    if isinstance(data, dict):
        # collect lists of strings from typical fact keys
        for key in ("facts", "extracted_facts", "evidence_facts"):
            val = data.get(key)
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        fact_strings.append(item.strip())

    # process fact strings if we found any
    for f in fact_strings:
        if not f:
            continue

        # a) KG-style explicit triple
        t = fact_to_triple(f)
        if t:
            triples.append(t)
            continue

        # b) NL patterns
        t = normalize_model_fact_to_triple(f)
        if t:
            triples.append(t)
            continue

        # c) fallback SVO for this single fact sentence
        for s, p, o in extract_triples(f):
            triples.append((s, p, o))

    if triples:
        return triples

    # ---------- 4) Fallbacks when JSON absent / useless ----------

    # 4a) explicit 'A -- P --> B' anywhere in raw answer
    fallback_triples: List[Tuple[str, str, str]] = []
    triple_pattern = re.compile(r"(.+?)\s*--\s*(.+?)\s*-->\s*(.+)")
    for m in triple_pattern.finditer(raw_answer):
        s = m.group(1).strip()
        p = m.group(2).strip()
        o = m.group(3).strip().rstrip(".; ")
        o = o.strip("()").strip()
        fallback_triples.append((s, p, o))

    if fallback_triples:
        return fallback_triples

    # 4b) last resort: SVO over sentences in raw answer
    for sent in re.split(r"[.?!]", raw_answer):
        sent = sent.strip()
        if not sent:
            continue
        for s, p, o in extract_triples(sent):
            fallback_triples.append((s, p, o))

    return fallback_triples
# -----------------------------
# 4) KG Checking
# -----------------------------

def link_to_ids(name: str, topk: int = 5) -> List[str]:
    """
    Wrapper around link_entity that returns a list of candidate ids/strings.
    Accepts whatever link_entity returns:
      - If it returns a list of (qid, score) -> returns [qid, ...]
      - If it returns a single id/string -> returns [id]
      - If None -> returns []
    """
    if not name:
        return []
    try:
        res = link_entity(name, topk=topk)
    except TypeError:
        # maybe link_entity doesn't accept topk
        res = link_entity(name)

    if not res:
        return []

    # If it's already a list of tuples like [(qid, score), ...]
    if isinstance(res, list):
        out = []
        for item in res:
            if item is None:
                continue
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                out.append(item[0])
            else:
                out.append(item)
        return [str(x) for x in out if x is not None]
    # single string/id
    return [str(res)]

def _safe_get_all_aliases(key: str) -> List[str]:
    """Return list of aliases (possibly empty) and ensure strings."""
    try:
        aliases = get_all_aliases(key) or []
    except Exception:
        aliases = []
    # make sure canonical key is also included as alias
    out = []
    for a in aliases:
        if a:
            out.append(str(a))
    if key and key not in out:
        out.append(str(key))
    # normalize whitespace
    return list({normalize_text(x) for x in out})

def kg_check_assertion(s: str, p: str, o: str,
                       topk: int = 5) -> str:
    """
    KG check using QIDs + predicate QIDs:

      - link subject and object strings -> entity QIDs via LMDB_LABEL/index
      - map predicate text -> predicate QIDs via text_to_pred_qids
      - query LMDB_SP with (subject_qid, predicate_qid) and see if object_qid is there

    Returns:
      - KG_CONSISTENT: at least one (s_qid, p_qid, o_qid) exists
      - KG_UNKNOWN: no such triple found (we do NOT treat this as contradiction)
    """
    s_norm = normalize_text(s)
    o_norm = normalize_text(o)

    # 1) Subject / object QIDs
    s_qids = link_to_ids(s_norm, topk=topk)
    o_qids = link_to_ids(o_norm, topk=topk)

    if not s_qids or not o_qids:
        return "KG_UNKNOWN"

    # 2) Predicate QIDs
    # Skip too-generic predicates
    if normalize_pred(p) == "is":
        return "KG_UNKNOWN"

    p_qids = text_to_pred_qids(p)  # e.g. "country of citizenship" -> ["P27", ...]

    # If any real predicate QID exists, prefer those over the fallback raw text
    has_real_pred = any(pid.upper().startswith("P") for pid in p_qids)


    for s_qid in s_qids:
        for o_qid in o_qids:
            for p_qid in p_qids:
                if has_real_pred and not p_qid.upper().startswith("P"):
                    # Ignore non-QID fallback if we have real predicate IDs
                    continue

                try:
                    if sp_has_object(s_qid, p_qid, o_qid):
                        return "KG_CONSISTENT"
                except Exception:
                    # If something is off with one combination, just skip it
                    continue

    return "KG_UNKNOWN"

# -----------------------------
# 5) NLI Check
# -----------------------------
def nli_entailment(retrieved_texts: List[str], claim: str, max_docs: int = 5) -> str:
    """
    Robust NLI:
      - Calls transformers pipeline with {"text": premise, "text_pair": hypothesis}
      - Iterates over retrieved_texts (up to max_docs)
      - If any doc is ENTAILMENT -> ENTAILS
      - If any doc is CONTRADICTION -> CONTRADICTS
      - Else -> UNKNOWN

    Handles exceptions and normalizes label names.
    """
    if not retrieved_texts:
        return "UNKNOWN"

    claim_h = normalize_text(claim)
    found_entail = False
    found_contradict = False

    for doc in retrieved_texts[:max_docs]:
        prem = normalize_text(doc)
        if not prem:
            continue

        # call pipeline with correct input shape
        try:
            out = nli_model({"text": prem, "text_pair": claim_h})
        except TypeError:
            # some older transformers versions accept positional args; support both
            try:
                out = nli_model(prem, claim_h)
            except Exception as e:
                logging.warning(f"NLI call failed for a pair; skipping. Err: {e}")
                continue
        except Exception as e:
            logging.warning(f"NLI pipeline exception, skipping doc. Err: {e}")
            continue

        # pipeline may return list of dicts or a dict
        if isinstance(out, list) and len(out) > 0:
            label_raw = out[0].get("label", "")
        elif isinstance(out, dict):
            label_raw = out.get("label", "")
        else:
            label_raw = ""

        label = str(label_raw).upper().strip()

        # Normalize common label variants
        if label in ("ENTAILMENT", "ENTAILS", "LABEL_0"):  # LABEL_0 might be model-specific
            found_entail = True
        elif label in ("CONTRADICTION", "CONTRADICTS", "LABEL_1"):
            found_contradict = True
        elif label in ("NEUTRAL", "LABEL_2"):
            # neutral -> no-op
            pass
        else:
            # sometimes labels are like "entails" / "contradiction"
            if "ENTAIL" in label:
                found_entail = True
            if "CONTRADICT" in label:
                found_contradict = True

        logging.debug(f"NLI doc: {prem} | claim: {claim_h} -> label: {label}")

        # If we found entailment we can short-circuit
        if found_entail:
            return "ENTAILS"

    if found_contradict:
        return "CONTRADICTS"
    if found_entail:
        return "ENTAILS"
    return "UNKNOWN"
# -----------------------------
# 6) Clean model answer
# -----------------------------
def preprocess_model_answer(ans: str) -> str:
    """
    Light, non-destructive cleaning for logging / display.
    Does NOT touch '--' or '-->' so we don't break KG-style facts.
    """
    if not ans:
        return ""
    ans = _strip_code_fences(ans)
    # Normalize unicode & whitespace, but keep punctuation/hyphens
    ans = unicodedata.normalize("NFKC", ans)
    ans = re.sub(r"\s+", " ", ans).strip()
    return ans


# -----------------------------
# 7) Final Failure Detection
# -----------------------------
def detect_failures(claim: str,
                    retrieved_texts: List[str],
                    model_answer: str) -> dict:
    """
    FULL PIPELINE:
        NLI check
        Try assertions from model answer
        Fallback: assertions from retrieved docs
        KG check
        Label: OK / NLI_FAILURE / KG_MISMATCH / BOTH_FAIL / NO_EVIDENCE
    """

    # --------------------------------------
    # 1) NLI
    # --------------------------------------
    nli_label = nli_entailment(retrieved_texts, claim)


    logging.info(f"[DEBUG] Model answer: {model_answer}")
    cleaned_answer = preprocess_model_answer(model_answer)
    logging.info(f"[DEBUG] Cleaned answer: {cleaned_answer}")

    # --------------------------------------
    # 2) Assertions from model answer
    # --------------------------------------
    assertions = extract_assertions_from_model_answer(model_answer)
    print("[DEBUG] assertions are", assertions)
    # --------------------------------------
    # 3) Fallback to retrieved docs (ONLY explicit triples)
    # --------------------------------------
    if not assertions and retrieved_texts:
        logging.info("[DEBUG] No model assertions, using retrieved docs fallback...")
        combined = " ".join(retrieved_texts[:3])
        
        # Extract explicit triples
        assertions = []
        triple_pattern = re.compile(r"(.+?)\s*--\s*(.+?)\s*-->\s*(.+)")
        for match in triple_pattern.finditer(combined):
            s, p, o = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            o = o.rstrip('.; ')
            assertions.append((s, p, o))

        # NLP-based triples fallback
        nlp_triples = extract_triples(combined)
        for t in nlp_triples:
            if t not in assertions:
                assertions.append(t)
        
        logging.info(f"[DEBUG] Extracted {len(assertions)} fallback assertions (explicit + NLP triples)")

    if not assertions:
        for doc in retrieved_texts:
            if cleaned_answer.lower() in doc.lower():
                logging.info("[DEBUG] Fallback textual evidence found")
                kg_agg = "KG_CONSISTENT_FUZZY"
                break
    print("[DEBUG]  retrival assertions are", assertions)
    # --------------------------------------
    # 4) KG checks
    # --------------------------------------
    kg_results = []
    kg_details = []

    for (s, p, o) in assertions:
        result = kg_check_assertion(s, p, o)
        kg_results.append(result)
        kg_details.append({
            "subject": s,
            "predicate": p,
            "object": o,
            "result": result
        })
    
    # --------------------------------------
    # 5) Aggregate KG results
    # --------------------------------------
    if not kg_results:
        kg_agg = "KG_UNKNOWN"
    elif all(r == "KG_CONSISTENT" for r in kg_results):
        kg_agg = "KG_CONSISTENT"
    elif any(r == "KG_INCONSISTENT" for r in kg_results):
        kg_agg = "KG_INCONSISTENT"
    else:
        kg_agg = "KG_UNKNOWN"

    # --------------------------------------
    # 6) Failure Label
    # --------------------------------------
    if kg_agg == "KG_CONSISTENT":
        if nli_label == "CONTRADICTS":
            failure = "KG_MATCH"   # KG says true, docs say no
        else:
            failure = "OK"            # docs don’t support but don’t contradict
    elif kg_agg == "KG_INCONSISTENT":
        if nli_label == "ENTAILS":
            failure = "NLI_ONLY"
        else:
            failure = "BOTH_FAIL"
    else:
        # kg_agg == KG_UNKNOWN
        if nli_label == "ENTAILS":
            failure = "NLI_ONLY"
        else:
            failure = "NO_EVIDENCE"
    

    logging.info(f"[DEBUG] Final assertions ({len(assertions)}): {assertions}")
    logging.info(f"[DEBUG] KG aggregate after fallback: {kg_agg}")
    logging.info(f"[DEBUG] Failure label: {failure}")
    return {
        "nli": nli_label,
        "kg_results": kg_results,
        "kg_details": kg_details,
        "kg_aggregate": kg_agg,
        "failure_label": failure,
        "num_assertions": len(assertions),
        "assertions_source": "model" if cleaned_answer else "retrieved_docs"
    }
