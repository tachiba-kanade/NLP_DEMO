# Structured Notes Extractor

REST API for turning messy notes into structured data. The service now supports
two extraction engines:

- spaCy (`POST /extract`) for lightweight parsing.
- Hugging Face / Flair models (`POST /extract/hf`) for deeper named-entity extraction.

## Setup

```bash
python3 -m venv env
source ./env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> The first Hugging Face or Flair request downloads model weights the first
> time you call it. Make sure your environment can reach the Hugging Face Hub
> (or pre-download models in advance).

## Run the API

```bash
uvicorn app.main:app --reload
```

Docs: http://127.0.0.1:8000/docs

## API Reference

### `POST /extract`

- Body: `{"text": "Hey, let's meet tomorrow at 6:30 for dinner"}`
- Engine: spaCy (`en_core_web_sm`)
- Returns the structured dictionary plus a `structured_text` field with a
  human-readable summary.

### `POST /extract/hf`

Payload schema:

```json
{
  "text": "Daily standup moved to 9am tomorrow in Berlin.",
  "model": "roberta-large-ner"
}
```

- `model` (optional) accepts either a short alias or a full Hugging Face ID.
- Response includes the resolved model, detected entities, temporal hints, and a
  human-readable `structured_text` summary.
- Built-in aliases (see response `aliases` field):
  - `bert-base-ner` → `dslim/bert-base-NER`
  - `roberta-large-ner` → `Jean-Baptiste/roberta-large-ner-english`
  - `flair` → `flair/ner-english-large`

> Flair models require the `flair` package (already listed in requirements) and
> will download weights the first time they are used.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'spacy'`** – Activate the virtual
  environment and reinstall dependencies: `pip install -r requirements.txt`.
- **`OSError: [E050] Can't find model 'en_core_web_sm'`** – Download the spaCy
  English model with `python -m spacy download en_core_web_sm`.
- **Model download failures** – Ensure your environment has network access or
  pre-cache the weights on the machine before running the API.
