from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle
import re




'''
note - pydantic is use for validate the input and use basemodel for it check input is text or not
'''
# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/DL_GRU.keras"
TOKENIZER_PATH = "model/tokenzer.pkl"

MAX_SEQUENCE_LENGTH = 50

# ============================================================
# EMOTION LABELS
# IMPORTANT:
# The order must exactly match the order used during training.
# ============================================================

EMOTION_NAMES = [
    "sadness",
    "joy",
    "love",
    "fear",
    "surprise",
    "anger",
]

EMOTION_EMOJIS = {
    "sadness": "😔",
    "joy": "😊",
    "love": "💕",
    "fear": "😥",
    "surprise": "😎",
    "anger": "😡",
}


# ============================================================
# MODEL STORAGE
# ============================================================

dl_model = {
    "BIGRU": None,
    "tokenizer": None,
}


'''
preprocess the text
clean raw text so it matches the format used while training.
1. convert the text to lowercase
2. remove apostrophes (e.g can't - cant)
3. remove special character and punctuation
4. remove extra spaces


'''

# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text: str) -> str:
    """
    Clean the input text in the same way as during training.

    Steps:
    1. Convert to lowercase
    2. Remove special characters
    3. Remove extra spaces
    """

    text = text.lower()

    # Keep only letters, numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text



'''
Request and response schemas
1. text input
2. prediction response
3. health response

'''


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to recognize",
        json_schema_extra={
            "example": "I feel so happy and excited"
        },
    )

    

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probability: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

    
 
 

# ============================================================
# LIFESPAN - LOAD MODEL AND TOKENIZER
# ============================================================
# ============================================================
# LIFESPAN - LOAD MODEL WHEN SERVER STARTS
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Loading Emotion Detection Model...")
    print("=" * 60)

    try:

        # Load trained BiGRU model
        dl_model["BIGRU"] = load_model(MODEL_PATH)

        print("✓ BiGRU model loaded successfully")

        # Load tokenizer
        with open(TOKENIZER_PATH, "rb") as file:
            dl_model["tokenizer"] = pickle.load(file)

        print("✓ Tokenizer loaded successfully")

        print("=" * 60)
        print("Model and tokenizer are ready!")
        print("=" * 60)

    except Exception as e:

        print("=" * 60)
        print("ERROR WHILE LOADING MODEL")
        print("=" * 60)
        print(str(e))

        # Keep server running so /health can report the problem
        dl_model["BIGRU"] = None
        dl_model["tokenizer"] = None

    yield

    # Cleanup
    dl_model["BIGRU"] = None
    dl_model["tokenizer"] = None

    print("Model resources released.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Emotion Detection API",
    description="Deep Learning based emotion detection using BiGRU",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    # For development and deployment with same server
    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# STATIC FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", include_in_schema=False)
async def server_ui():

    return FileResponse(
        "static/index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse
)
async def health_check():

    model_loaded = (
        dl_model.get("BIGRU") is not None
        and dl_model.get("tokenizer") is not None
    )

    return HealthResponse(
        status="Server is Running",
        model_loaded=model_loaded,
    )


# ============================================================
# EMOTION PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict_emotion(
    text_input: TextInput
):

    # --------------------------------------------------------
    # Get model and tokenizer
    # --------------------------------------------------------

    bigru_model = dl_model.get("BIGRU")
    tokenizer_model = dl_model.get("tokenizer")

    if bigru_model is None or tokenizer_model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check the model files and server logs.",
        )

    # --------------------------------------------------------
    # Clean input
    # --------------------------------------------------------

    clean_text = preprocess_text(
        text_input.text
    )

    if not clean_text:

        raise HTTPException(
            status_code=400,
            detail="Please enter meaningful text.",
        )

    # --------------------------------------------------------
    # Convert text to sequence
    # --------------------------------------------------------

    # IMPORTANT:
    # text_to_sequences expects a LIST of texts.
    tokenized_text = tokenizer_model.texts_to_sequences(
        [clean_text]
    )

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    probability = bigru_model.predict(
        padded_sequence,
        verbose=0,
    )[0]

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if len(probability) != len(EMOTION_NAMES):

        raise HTTPException(
            status_code=500,
            detail=(
                f"Model returned {len(probability)} classes, "
                f"but {len(EMOTION_NAMES)} emotion labels are configured."
            ),
        )

    # --------------------------------------------------------
    # Find top emotion
    # --------------------------------------------------------

    top_emotion_index = int(
        np.argmax(probability)
    )

    top_emotion = EMOTION_NAMES[
        top_emotion_index
    ]

    # --------------------------------------------------------
    # All probabilities
    # --------------------------------------------------------

    all_probability = {
        emotion: float(prob)
        for emotion, prob in zip(
            EMOTION_NAMES,
            probability
        )
    }

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return PredictionResponse(

        text=text_input.text,

        predicted_emotion=top_emotion,

        confidence=float(
            probability[top_emotion_index]
        ),

        all_probability=all_probability,
    )