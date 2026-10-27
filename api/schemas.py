from pydantic import BaseModel 

#esto es lo que la api recibe del bot 
class MessageInput(BaseModel):
    message: str

#esto es lo que la api le responde al bot 
class PredictionOutput(BaseModel):
    label: str
    confidence: float
    probabilities: dict
