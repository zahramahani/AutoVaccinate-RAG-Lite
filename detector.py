
# detector.py
import re
import spacy
from transformers import pipeline
from linker import link_entity
from typing import List, Tuple
from difflib import SequenceMatcher
from lmdb_connector import sp_has_object, get_all_aliases,normalize_pred

# -------------------- SpaCy & NLI --------------------

_nlp = None
_nli = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def get_nli():
    global _nli
    if _nli is None:
        _nli = pipeline("text-classification", model="roberta-large-mnli",
                        device=0 if __import__("torch").cuda.is_available() else -1)
    return _nli

# -------------------- KG Consistency --------------------

def fuzzy_match(a: str, b: str, threshold=0.8) -> bool:
    """Return True if strings a and b are similar enough."""
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def kg_check_assertion(subject_text: str, predicate_text: str, object_text: str, topk=3, fuzzy_threshold=0.8) -> str:
    """
    Check KG consistency using LMDB backend, with fuzzy matching.
    Returns: "KG_CONSISTENT", "KG_INCONSISTENT", or "KG_UNKNOWN".
    """
    subj_candidates = link_entity(subject_text, topk=topk)
    obj_candidates = link_entity(object_text, topk=topk)
    pred_norm = normalize_pred(predicate_text)

    if not subj_candidates or not obj_candidates:
        return "KG_UNKNOWN"

    for s_qid, _ in subj_candidates:
        for o_qid, _ in obj_candidates:
            # Exact LMDB check
            if sp_has_object(s_qid, pred_norm, o_qid):
                return "KG_CONSISTENT"

            # Check using aliases (exact)
            s_aliases = get_all_aliases(s_qid)
            o_aliases = get_all_aliases(o_qid)
            for sa in s_aliases:
                for oa in o_aliases:
                    if sp_has_object(sa, pred_norm, oa):
                        return "KG_CONSISTENT"

            # Fuzzy match check
            for sa in s_aliases:
                for oa in o_aliases:
                    if fuzzy_match(subject_text, sa, threshold=fuzzy_threshold) and \
                       fuzzy_match(object_text, oa, threshold=fuzzy_threshold):
                        # Optional: also fuzzy-match predicate_text against known predicate aliases
                        pred_aliases = get_all_aliases(pred_norm)
                        for pa in pred_aliases:
                            if fuzzy_match(predicate_text, pa, threshold=fuzzy_threshold):
                                return "KG_CONSISTENT"

    return "KG_INCONSISTENT"


# -------------------- Assertion Extraction --------------------
def extract_candidate_assertions(text: str) -> List[Tuple[str, str, str]]:
    """
    Extract SVO triples using spaCy; fallback heuristics if necessary.
    """
    triples = []
    nlp = get_nlp()
    doc = nlp(text)

    # spaCy SVO extraction
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ("nsubj", "nsubjpass"):
                subject = token.text
                verb = token.head.text
                obj = None
                for child in token.head.children:
                    if child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                        obj = child.text
                        triples.append((subject.lower(), verb.lower(), obj.lower()))

    # fallback 1: use consecutive entities and verb in-between
    if not triples:
        ents = [ent.text for ent in doc.ents]
        for i in range(len(ents)-1):
            subj, obj = ents[i], ents[i+1]
            between = text[text.find(subj)+len(subj):text.find(obj)]
            pred = ""
            for token in nlp(between):
                if token.pos_ == "VERB":
                    pred = token.lemma_
                    break
            triples.append((subj, pred or "related_to", obj))

    # fallback 2: regex heuristic
    if not triples:
        ents = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        verbs = re.findall(r"\b(?:is|was|are|were|directed|played|born|starred|acts|act|has|had)\b", text.lower())
        if len(ents) >= 2 and verbs:
            triples.append((ents[0].lower(), verbs[0].lower(), ents[-1].lower()))

    return triples

# -------------------- NLI Entailment --------------------
def nli_entailment(premises: List[str], hypothesis: str) -> str:
    """
    Checks entailment using NLI model.
    """
    nli = get_nli()
    premise = " ".join(premises)
    result = nli(f"{premise}\n\nHypothesis: {hypothesis}", truncation=True)[0]
    label = result["label"].upper()
    if label.startswith("ENTAIL"):
        return "ENTAILS"
    elif label.startswith("CONTRADICT"):
        return "CONTRADICTS"
    else:
        return "NOT_ENOUGH_INFO"

# -------------------- Combined Detector --------------------
def detect_failures(claim: str, retrieved_texts: List[str], model_answer: str) -> dict:
    """
    Runs NLI + KG consistency checks.
    Returns full failure report.
    """
    nli_label = nli_entailment(retrieved_texts, claim)
    assertions = extract_candidate_assertions(model_answer)
    kg_results = [kg_check_assertion(s, p, o) for s, p, o in assertions]

    # Aggregate KG results
    if not kg_results:
        kg_agg = "KG_UNKNOWN"
    elif all(r == "KG_CONSISTENT" for r in kg_results):
        kg_agg = "KG_CONSISTENT"
    elif any(r == "KG_INCONSISTENT" for r in kg_results):
        kg_agg = "KG_INCONSISTENT"
    else:
        kg_agg = "KG_UNKNOWN"

    # Final failure label
    if nli_label == "ENTAILS" and kg_agg == "KG_CONSISTENT":
        failure = "OK"
    elif nli_label != "ENTAILS" and kg_agg == "KG_CONSISTENT":
        failure = "NLI_FAILURE"
    elif nli_label == "ENTAILS" and kg_agg == "KG_INCONSISTENT":
        failure = "KG_MISMATCH"
    elif nli_label != "ENTAILS" and kg_agg == "KG_INCONSISTENT":
        failure = "BOTH_FAIL"
    else:
        failure = "NO_EVIDENCE"

    return {
        "nli": nli_label,
        "kg_results": kg_results,
        "kg_aggregate": kg_agg,
        "failure_label": failure
    }
