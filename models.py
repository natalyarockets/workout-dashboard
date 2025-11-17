from pydantic import BaseModel
from typing import List

class ParseRequest(BaseModel):
    text: str

class DummyRow(BaseModel):
    dummy: str

class ParseResponse(BaseModel):
    success: bool
    message: str
    rows: List[DummyRow]
