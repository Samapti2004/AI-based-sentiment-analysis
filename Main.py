from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from contextlib import asynccontextmanager
from keras.models import load_model
from keras.utils import pad_sequences
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import pickle
import re



model_path = "Artifacts/BiGRU_Modle.keras"

tokenizer_path = "Artifacts/tokenizer.pkl"

max_sequence_length = 50

emotion_labels = ['sadness','joy','love','anger','fear','surprise']

def preprocess_text(text : str) -> str:
    text = text.lower()
    text = re.sub(r"'","",text)
    text = re.sub(r"[^a-z0-9\s]"," ",text)
    text = re.sub(r"\s+"," ",text).strip()
    return text

#Text Input
class textInput(BaseModel):
    text:str = Field(
        ...,
        min_length= 1,
        max_length= 2000,
        description= "The sentence to be analyzed",
        json_schema_extra= {"Example":"I feel so happy and excited"}
    )

#Output response
class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence : float
    all_probalities: dict[str,float]


#Health check
class HealthResponse(BaseModel):
    status : str
    model_loaded: bool


#Model loading and Lifespan management
#Asychronous codding

dl_model = {}

@asynccontextmanager
async def lifespan(app : FastAPI):
    print('Loading the model and tokenizer...')
    dl_model["BiGRU"] = load_model(model_path)                  #BiGRU Model
    with open(tokenizer_path,'rb') as file:
        dl_model["Tokenizer"] = pickle.load(file)               #Tokenzier model
    print('MOdel are loaded successfully')

    yield

    dl_model.clear()


app = FastAPI(
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_credentials= True,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)

app.mount('/static',StaticFiles(directory ="static"), name="static")

#api Endpoints
@app.get('/', include_in_schema= False)
def serve_ui():
    return FileResponse('static/index.html')

@app.get('/health', response_model = HealthResponse)
def health_check():
    return HealthResponse(status="Server is running", model_loaded=bool(dl_model))

@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(text_input : textInput):
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer is None:
        return HTTPException(status_code=503, detail="Model is not loaded. Please try again later...")

    cleanned_text = preprocess_text(text_input.text)
    tokenized_text = tokenizer.texts_to_sequences([cleanned_text])
    padded_text = pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating="post"
    )

    probabilities = BiGRU_model.predict(padded_text)[0]
    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {
        label: float(prob) for prob, label in zip(probabilities, emotion_labels)
    }

    return PredictionResponse(
        text= text_input.text,
        predicted_emotion= emotion_labels[top_emotion_index],
        confidence= probabilities[top_emotion_index],
        all_probalities= all_probabilities
    )