# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import json

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var.")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

class ParsedSet(BaseModel):
    date: Optional[str]                  # YYYY-MM-DD (LLM extracts)
    exercise_name: str                   # Standardized name WITH equipment (e.g., DB Bench Press)
    weight: float                        # numeric
    reps_unassisted: int
    reps_assisted: int
    reps_total: int
    tempo_notes: Optional[str] = ""
    injury_flag: bool = False
    injury_notes: Optional[str] = ""
    equipment: Optional[str] = ""        # DB / BB / BW, etc.
    source_line: Optional[str] = ""      # The raw line the LLM parsed
    notes: Optional[str] = ""            # Any misc leftover notes

class ParseRequest(BaseModel):
    text: str

class ParseResponse(BaseModel):
    sets: List[ParsedSet]


# ------------------------------------------------------------
# LLM PROMPT
# ------------------------------------------------------------

PROMPT = """
You are a strict workout-log parser. Your job:

INPUT:
A raw block of text containing informal workout notes, often messy.
Each "line" usually includes:
- exercise name (with typos)
- weight
- reps (may include assisted)
- date sometimes
- tempo notes (e.g., 3-1-1)
- injury notes (like "hurt back", "twinge", "ankle pain")
- DB means dumbbells (weight is PER HAND unless single arm mentioned)
- BB means barbell

OUTPUT:
You MUST return a JSON OBJECT with a single key "sets" whose value is an array
of objects, each object describing ONE set, using this schema:

{
  "sets": [
    {
      "date": "YYYY-MM-DD or empty if not present",
      "exercise_name": "Standardized name including equipment; e.g., 'DB Bench Press'",
      "weight": number,
      "reps_unassisted": number,
      "reps_assisted": number,
      "reps_total": number,
      "tempo_notes": "freeform string",
      "injury_flag": boolean,
      "injury_notes": "string if injury_flag true",
      "equipment": "DB or BB or BW or Machine, etc.",
      "source_line": "raw input line",
      "notes": "anything leftover"
    }
  ]
}

RULES:
- Standardize exercise names; fix typos.
- If DB: weight applies PER HAND unless "single arm" or similar.
- Assisted reps appear in parentheses: e.g. "8(2)" means 8 total, 2 assisted, 6 unassisted.
- If no assisted reps, reps_assisted = 0.
- reps_total = reps_unassisted + reps_assisted.
- Identify injury indicators: hurt, pain, tweak, strain, back issue, etc.
  If any present, set injury_flag=true and describe it in injury_notes.
- Tempo like "3-1-1" goes to tempo_notes.
- If no explicit date, leave date empty.
- equipment: DB, BB, BW, Machine, Cable, etc.

Your output MUST be valid JSON with NO commentary, NO trailing text.
Only return the JSON object with the "sets" array.
"""


# ------------------------------------------------------------
# /parse ENDPOINT
# ------------------------------------------------------------

@app.post("/parse", response_model=ParseResponse)
def parse_text(req: ParseRequest):
    """Parse a chunk of workout text into normalized sets."""
    text = req.text.strip()
    if not text:
        return ParseResponse(sets=[])

    # Call OpenAI
    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # small, cheap, good enough for parsing
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format={"type": "json_object"},  # enforce strict JSON
    )

    raw = completion.choices[0].message.content or ""

    # Try to parse JSON strictly, with code-fence stripping fallback
    parsed_sets: List[dict] = []
    try:
        obj = json.loads(raw)
    except Exception:
        # strip code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            try:
                cleaned = cleaned.split("```", 2)[1]
            except Exception:
                pass
        try:
            obj = json.loads(cleaned)
        except Exception:
            obj = {"sets": []}

    if isinstance(obj, dict) and "sets" in obj:
        candidate = obj.get("sets") or []
    else:
        # allow bare array fallback
        candidate = obj if isinstance(obj, list) else []

    # Validate into Pydantic models
    out: List[ParsedSet] = []
    for item in candidate:
        try:
            out.append(ParsedSet(**item))
        except:
            # skip malformed entries but do not error the whole batch
            continue

    return ParseResponse(sets=out)


# ------------------------------------------------------------
# ROOT CHECK
# ------------------------------------------------------------

@app.get("/")
def root():
    return {"ok": True, "message": "Workout parser backend running"}
