from typing import Optional

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
