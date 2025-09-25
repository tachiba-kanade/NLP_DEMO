# from __future__ import annotations
# import spacy
# import re
# from datetime import datetime
# from functools import lru_cache
# from typing import Dict, Optional, Tuple, List


# import dateparser
# from dateparser.search import search_dates

# StructuredNote = Dict[str, object]

# # Expanded action keywords
# _ACTION_PATTERN = re.compile(
#     r"\b(meet|call|schedule|remind|lunch|brunch|appointment|sync|pickup|game|visit|check|groceries|message)\b",
#     re.I,
# )

# _DEFAULT_SPACY_MODEL = "en_core_web_sm"
# _DEFAULT_HF_MODEL_ALIAS = "bert-base-ner"
# _FLAIR_PREFIX = "flair/"

# AVAILABLE_HF_MODELS = {
#     "bert-base-ner": "dslim/bert-base-NER",
#     "roberta-large-ner": "Jean-Baptiste/roberta-large-ner-english",
#     "flair": "flair/ner-english-large",
# }


# class SpaCyModelError(RuntimeError):
#     pass


# class HuggingFaceModelError(RuntimeError):
#     pass


# #newly added spacy model
# def _load_spacy_model(model_name: str):
#     """Load a spaCy model safely with error handling."""
#     try:
#         return spacy.load(model_name)
#     except OSError as exc:
#         raise SpaCyModelError(
#             f"SpaCy model '{model_name}' is not installed. "
#             f"Run: python -m spacy download {model_name}"
#         ) from exc
    


# def extract_structured_notes(note: str, model_name: str = _DEFAULT_SPACY_MODEL) -> StructuredNote:
#     """Extract structured data from a free-form note using spaCy."""
#     nlp = _load_spacy_model(model_name)
#     doc = nlp(note)

#     structured = _empty_structure()

#     # Sentence-level loop makes spaCy more reliable
#     for sent in doc.sents:
#         for ent in sent.ents:
#             if ent.label_ in {"TIME", "DATE"}:
#                 _apply_parsed_datetime(structured, dateparser.parse(ent.text))
#             else:
#                 structured["entities"].append({ent.label_: ent.text})

#         action = _find_action(sent.text)
#         if action and not structured["action"]:
#             structured["action"] = action

#     _fill_temporal_from_search(note, structured)
#     return structured


# def extract_structured_notes_hf(
#     note: str, model_name: Optional[str] = None
# ) -> Tuple[str, StructuredNote]:
#     """Extract structured data using Hugging Face Transformers or Flair models."""
#     resolved_model = _resolve_hf_model_name(model_name)
#     structured = _empty_structure()

#     # Chunk text into ~250 token pieces (NER models handle this better)
#     chunks = _chunk_text(note, 250)

#     for chunk in chunks:
#         if resolved_model.startswith(_FLAIR_PREFIX):
#             _populate_from_flair(chunk, resolved_model, structured)
#         else:
#             _populate_from_transformers(chunk, resolved_model, structured)

#         action = _find_action(chunk)
#         if action and not structured["action"]:
#             structured["action"] = action

#     _fill_temporal_from_search(note, structured)
#     return resolved_model, structured


# def _empty_structure() -> StructuredNote:
#     return {"time": None, "date": None, "action": None, "entities": []}


# def _find_action(note: str) -> Optional[str]:
#     match = _ACTION_PATTERN.search(note)
#     return match.group(0) if match else None


# def _apply_parsed_datetime(structured: StructuredNote, parsed: Optional[datetime]) -> None:
#     if not parsed:
#         return

#     if not structured["date"]:
#         structured["date"] = parsed.strftime("%Y-%m-%d")

#     if parsed.time() != datetime.min.time() and not structured["time"]:
#         structured["time"] = parsed.strftime("%H:%M")


# def _fill_temporal_from_search(note: str, structured: StructuredNote) -> None:
#     if structured["date"] and structured["time"]:
#         return

#     results = search_dates(note, settings={"RETURN_AS_TIMEZONE_AWARE": False})
#     if not results:
#         return

#     for _, parsed in results:
#         _apply_parsed_datetime(structured, parsed)
#         if structured["date"] and structured["time"]:
#             break


# def _chunk_text(text: str, max_words: int) -> List[str]:
#     """Split text into smaller pieces for better NER accuracy."""
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), max_words):
#         chunks.append(" ".join(words[i : i + max_words]))
#     return chunks


# # --- existing spaCy / HF / Flair loaders and helpers remain unchanged ---

# def format_structured_note(structured: dict) -> str:
#     """Convert a structured note dict into a readable string summary."""
#     parts = []
#     if structured.get("action"):
#         parts.append(f"Action: {structured['action']}")
#     if structured.get("date"):
#         parts.append(f"Date: {structured['date']}")
#     if structured.get("time"):
#         parts.append(f"Time: {structured['time']}")

#     if structured.get("entities"):
#         entities_str = ", ".join(
#             [f"{list(ent.keys())[0]}: {list(ent.values())[0]}" for ent in structured["entities"]]
#         )
#         parts.append(f"Entities: {entities_str}")

#     return " | ".join(parts) if parts else "No structured data found"







# extractor.py
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import spacy
import dateparser
from dateparser.search import search_dates

StructuredNote = Dict[str, object]

# Action keywords
_ACTION_PATTERN = re.compile(
    r"\b(meet|call|schedule|remind|lunch|brunch|appointment|sync|pickup|game|visit|check|groceries|message|book)\b",
    re.I,
)

_DEFAULT_SPACY_MODEL = "en_core_web_sm"
_FLAIR_PREFIX = "flair/"

AVAILABLE_HF_MODELS = {
    "bert-base-ner": "dslim/bert-base-NER",
    "roberta-large-ner": "Jean-Baptiste/roberta-large-ner-english",
    "flair": "flair/ner-english-large",
}


# --- Exceptions ---
class SpaCyModelError(RuntimeError):
    pass

class HuggingFaceModelError(RuntimeError):
    pass


# --- SpaCy Extraction ---
def _load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise SpaCyModelError(
            f"SpaCy model '{model_name}' not installed. Run: python -m spacy download {model_name}"
        ) from exc


def extract_structured_notes(note: str, model_name: str = _DEFAULT_SPACY_MODEL) -> StructuredNote:
    nlp = _load_spacy_model(model_name)
    doc = nlp(note)
    structured = _empty_structure()

    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ in {"TIME", "DATE"}:
                _apply_parsed_datetime(structured, dateparser.parse(ent.text))
            else:
                structured["entities"].append({ent.label_: ent.text})

        action = _find_action(sent.text)
        if action and not structured["action"]:
            structured["action"] = action

    _fill_temporal_from_search(note, structured)
    return structured


# --- Hugging Face / Flair Extraction ---
def extract_structured_notes_hf(note: str, model_name: Optional[str] = None) -> Tuple[str, StructuredNote]:
    resolved_model = _resolve_hf_model_name(model_name)
    structured = _empty_structure()

    chunks = _chunk_text(note, 250)

    for chunk in chunks:
        if resolved_model.startswith(_FLAIR_PREFIX):
            _populate_from_flair(chunk, resolved_model, structured)
        else:
            _populate_from_transformers(chunk, resolved_model, structured)

        action = _find_action(chunk)
        if action and not structured["action"]:
            structured["action"] = action

    _fill_temporal_from_search(note, structured)
    return resolved_model, structured


# --- Helpers ---
def _resolve_hf_model_name(model_name: Optional[str]) -> str:
    if not model_name:
        return "bert-base-ner"
    if model_name not in AVAILABLE_HF_MODELS:
        raise HuggingFaceModelError(f"Model '{model_name}' not supported")
    return model_name


def _populate_from_transformers(chunk, model_name, structured):
    # Stub for testing: adds dummy entity
    structured["entities"].append({"TRANSFORMERS_ENTITY": "example"})


def _populate_from_flair(chunk, model_name, structured):
    # Stub for testing: adds dummy entity
    structured["entities"].append({"FLAIR_ENTITY": "example"})


def _empty_structure() -> StructuredNote:
    return {"time": None, "date": None, "action": None, "entities": []}


def _find_action(note: str) -> Optional[str]:
    match = _ACTION_PATTERN.search(note)
    return match.group(0) if match else None


def _apply_parsed_datetime(structured: StructuredNote, parsed: Optional[datetime]) -> None:
    if not parsed:
        return
    if not structured["date"]:
        structured["date"] = parsed.strftime("%Y-%m-%d")
    if parsed.time() != datetime.min.time() and not structured["time"]:
        structured["time"] = parsed.strftime("%H:%M")


def _fill_temporal_from_search(note: str, structured: StructuredNote) -> None:
    if structured["date"] and structured["time"]:
        return

    results = search_dates(note, settings={"RETURN_AS_TIMEZONE_AWARE": False})
    if not results:
        return

    for _, parsed in results:
        _apply_parsed_datetime(structured, parsed)
        if structured["date"] and structured["time"]:
            break


def _chunk_text(text: str, max_words: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def format_structured_note(structured: StructuredNote) -> str:
    """Convert a structured note dict into a human-readable string."""
    parts = []
    if structured.get("action"):
        parts.append(f"Action: {structured['action']}")
    if structured.get("date"):
        parts.append(f"Date: {structured['date']}")
    if structured.get("time"):
        parts.append(f"Time: {structured['time']}")

    if structured.get("entities"):
        entities_str = ", ".join(
            [f"{list(ent.keys())[0]}: {list(ent.values())[0]}" for ent in structured["entities"]]
        )
        parts.append(f"Entities: {entities_str}")

    return " | ".join(parts) if parts else "No structured data found"