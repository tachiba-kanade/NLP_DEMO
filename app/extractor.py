from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Dict, Optional, Tuple

import dateparser
from dateparser.search import search_dates


StructuredNote = Dict[str, object]

_ACTION_PATTERN = re.compile(r"\b(meet|call|schedule|remind)\b", re.I)
_DEFAULT_SPACY_MODEL = "en_core_web_sm"
_DEFAULT_HF_MODEL_ALIAS = "bert-base-ner"
_FLAIR_PREFIX = "flair/"

AVAILABLE_HF_MODELS = {
    "bert-base-ner": "dslim/bert-base-NER",
    "roberta-large-ner": "Jean-Baptiste/roberta-large-ner-english",
    "flair": "flair/ner-english-large",
}


class SpaCyModelError(RuntimeError):
    """Raised when spaCy or its language model are unavailable."""


class HuggingFaceModelError(RuntimeError):
    """Raised when Hugging Face or Flair models cannot be loaded."""


def extract_structured_notes(note: str, model_name: str = _DEFAULT_SPACY_MODEL) -> StructuredNote:
    """Extract structured data from a free-form note using spaCy."""
    nlp = _load_spacy_model(model_name)
    doc = nlp(note)

    structured = _empty_structure()

    for ent in doc.ents:
        if ent.label_ in {"TIME", "DATE"}:
            _apply_parsed_datetime(structured, dateparser.parse(ent.text))
        else:
            structured["entities"].append({ent.label_: ent.text})

    structured["action"] = _find_action(note)
    _fill_temporal_from_search(note, structured)

    return structured


def extract_structured_notes_hf(
    note: str, model_name: Optional[str] = None
) -> Tuple[str, StructuredNote]:
    """Extract structured data using Hugging Face Transformers or Flair models."""
    resolved_model = _resolve_hf_model_name(model_name)
    structured = _empty_structure()

    if resolved_model.startswith(_FLAIR_PREFIX):
        _populate_from_flair(note, resolved_model, structured)
    else:
        _populate_from_transformers(note, resolved_model, structured)

    structured["action"] = _find_action(note)
    _fill_temporal_from_search(note, structured)

    return resolved_model, structured


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


@lru_cache(maxsize=2)
def _load_spacy_model(model_name: str):
    try:
        import spacy
    except ModuleNotFoundError as exc:
        raise SpaCyModelError(
            "spaCy is not installed. Run `pip install -r requirements.txt` to install dependencies."
        ) from exc

    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise SpaCyModelError(
            f"spaCy model `{model_name}` is not available. Download it with `python -m spacy download {model_name}`."
        ) from exc


@lru_cache(maxsize=4)
def _load_hf_pipeline(model_name: str):
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ModuleNotFoundError as exc:
        raise HuggingFaceModelError(
            "transformers is not installed. Install it with `pip install transformers torch`."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    except OSError as exc:
        raise HuggingFaceModelError(
            f"Unable to load Hugging Face model `{model_name}`. Ensure the identifier is correct and the model is available."
        ) from exc

    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")


@lru_cache(maxsize=2)
def _load_flair_model(model_name: str):
    try:
        from flair.models import SequenceTagger
    except ModuleNotFoundError as exc:
        raise HuggingFaceModelError(
            "flair is not installed. Install it with `pip install flair` to use Flair models."
        ) from exc

    try:
        return SequenceTagger.load(model_name)
    except (OSError, ValueError) as exc:
        raise HuggingFaceModelError(
            f"Unable to load Flair model `{model_name}`. Verify the identifier and that the model is downloaded."
        ) from exc


def _populate_from_transformers(note: str, model_name: str, structured: StructuredNote) -> None:
    ner_pipeline = _load_hf_pipeline(model_name)
    predictions = ner_pipeline(note)

    for entity in predictions:
        label = entity.get("entity_group") or entity.get("entity") or "ENTITY"
        text = entity.get("word") or entity.get("entity") or ""
        normalized = _normalize_entity_text(text)
        if not normalized:
            continue

        structured["entities"].append({label: normalized})

        if label.upper() in {"TIME", "DATE"}:
            _apply_parsed_datetime(structured, dateparser.parse(normalized))


def _populate_from_flair(note: str, model_name: str, structured: StructuredNote) -> None:
    tagger = _load_flair_model(model_name)

    try:
        from flair.data import Sentence
    except ModuleNotFoundError as exc:
        raise HuggingFaceModelError(
            "flair is partially installed. Install full flair support with `pip install flair`."
        ) from exc

    sentence = Sentence(note)
    tagger.predict(sentence)

    for span in sentence.get_spans("ner"):
        label = span.get_label("ner").value
        structured["entities"].append({label: span.text})

        if label.upper() in {"TIME", "DATE"}:
            _apply_parsed_datetime(structured, dateparser.parse(span.text))


def _normalize_entity_text(text: str) -> str:
    return text.replace("##", "").strip()


def _resolve_hf_model_name(candidate: Optional[str]) -> str:
    if not candidate:
        candidate = _DEFAULT_HF_MODEL_ALIAS

    key = candidate.lower()
    return AVAILABLE_HF_MODELS.get(key, candidate)


def format_structured_note(structured: StructuredNote) -> str:
    """Build a human-readable summary of the structured note."""
    date = structured.get("date")
    time = structured.get("time")
    action = structured.get("action")
    entities = structured.get("entities") or []

    lines = ["Structured Summary:"]

    schedule_text = None
    if date and time:
        schedule_text = f"{date} at {time}"
    elif date:
        schedule_text = date
    elif time:
        schedule_text = time

    if schedule_text:
        lines.append(f"- Schedule: {schedule_text}")

    if action:
        lines.append(f"- Action: {action}")

    entity_lines = []
    for entity in entities:
        for label, value in entity.items():
            entity_lines.append(f"  - {label}: {value}")

    if entity_lines:
        lines.append("- Entities:")
        lines.extend(entity_lines)

    if len(lines) == 1:
        return "Structured Summary: No structured information detected."

    return "\n".join(lines)
