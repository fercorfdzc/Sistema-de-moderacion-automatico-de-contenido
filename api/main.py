from fastapi import FastAPI
from schemas import MessageInput, PredictionOutput
from model_loader import load_model, predict

#crea la api
app= FastAPI(title="MODERADOR DE CHAT IA", 
    description="API clasifica mensajes de chat como clean, toxicos o de hate", 
    version="1.0.0")

#carga el modelo una vez cuando arranca la api
@app.on_event("startup")
def startup_event():
    load_model()

#endpoint para verificar que al api funciona
@app.get("/")
def root():
    return {"message": "API del moderador de chat IA funcionando"}

#recibe un mensaje y devuelve la clasificacion, el bot llama este endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict_message(data:MessageInput):
    result=predict(data.message)
    return result
















