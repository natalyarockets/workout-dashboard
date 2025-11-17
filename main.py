from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from models import ParseRequest, ParseResponse

# OpenAI client (OPENAI_API_KEY required in Railway environment)
client = OpenAI()

app = FastAPI()

# Allow Apps Script fetch (can lock down later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/parse", response_model=ParseResponse)
def parse(payload: ParseRequest):
    text = payload.text

    # --- Minimal LLM call just to check wiring ---
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a simple system doing a connectivity check."},
                {"role": "user", "content": f"Received {len(text)} characters from the document."}
            ]
        )
        llm_output = resp.choices[0].message.content
    except Exception as e:
        llm_output = f"LLM error: {str(e)}"

    # --- Dummy rows for now ---
    rows = [
        {"dummy": "row 1"},
        {"dummy": "row 2"}
    ]

    return ParseResponse(
        success=True,
        message=f"Backend OK. LLM said: {llm_output}",
        rows=rows
    )
