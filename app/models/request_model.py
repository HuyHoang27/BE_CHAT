from pydantic import BaseModel

class TextRequest(BaseModel):
    input_text: str
    id: int
