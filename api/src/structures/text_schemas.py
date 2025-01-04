from pydantic import BaseModel

class PhonemeRequest(BaseModel):
    text: str
    language: str = "a"  # Default to American English

class PhonemeResponse(BaseModel):
    phonemes: str
    tokens: list[int]
