from typing import Optional
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.extractor import (
    AVAILABLE_HF_MODELS,
    HuggingFaceModelError,
    SpaCyModelError,
    format_structured_note,
    extract_structured_notes,
    extract_structured_notes_hf,
)

app = FastAPI(title="Structured Notes Extractor")

class NoteInput(BaseModel):
    text: str


class HFNoteInput(BaseModel):
    text: str
    model: Optional[str] = None


@app.post("/extract")
def extract_notes(note: NoteInput):
    try:
        result = extract_structured_notes(note.text)
    except SpaCyModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "input": note.text,
        "structured": result,
        "structured_text": format_structured_note(result),
    }


@app.post("/extract/hf")
def extract_notes_hf(note: HFNoteInput):
    try:
        model_used, result = extract_structured_notes_hf(note.text, note.model)
    except HuggingFaceModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "input": note.text,
        "model": model_used,
        "aliases": AVAILABLE_HF_MODELS,
        "structured": result,
        "structured_text": format_structured_note(result),

    }
StructuredNote = Dict[str, object] 

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