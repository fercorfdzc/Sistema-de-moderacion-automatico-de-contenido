import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import os 

class BERTModerador(nn.Module):
    def __init__(self, n_classes):
        super(BERTModerador, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.linear(x)
        return logits


model = None
tokenizer=None
device =None

id2label={
    0:"clean",
    1:"hate"
}

"FUNCION PARA CARGAR EL MODELO , se ejecuta solo una vez cuando arranca la api"
def load_model():
    global model, tokenizer, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer=BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = BERTModerador(n_classes=2)
    model_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "moderador_bert.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only =True))
    model=model.to(device)
    model.eval()
    print("Modelo cargado")

"FUNCION PARA CLASIFICAR UN MENSAJE, se ejecuta cada que lelga un mensaje del bot "
def predict(text: str)-> dict:
    encoding =tokenizer(
        text,
        max_length=150,
        truncation=True,
        add_special_tokens=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids= encoding["input_ids"].to(device)
    attention_mask= encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs=model(input_ids, attention_mask)
        probs=torch.softmax(outputs, dim=1).squeeze()
        predicted_class= torch.argmax(probs).item()
    #construye el resultado en formato diccionario
    result={
        "label" : id2label[predicted_class],
        "confidence":round(probs[predicted_class].item(),4),
        "probabilities":{
            "clean":round(probs[0].item(),4),
            "toxic":0.0,
            "hate":round(probs[1].item(),4)
        }
    }
    return result
