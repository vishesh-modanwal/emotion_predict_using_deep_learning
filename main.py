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

EMOTION_LABELS = [
    "sadness",
    "joy",
    "love",
    "fear",
    "surprise",
    "anger"
]

EMOTION_EMOJIS = {
    "sadness": "😔",
    "joy": "😊",
    "love": "💕",
    "fear": "😥",
    "surprise": "😎",
    "anger": "😡"
}

'''
preprocess the text
clean raw text so it matches the format used while training.
1. convert the text to lowercase
2. remove apostrophes (e.g can't - cant)
3. remove special character and punctuation
4. remove extra spaces


'''

def preprocess_text(text: str) -> str:
    """
    Clean raw text so it matches the format used during training.

    Steps:
    1. Convert text to lowercase
    2. Remove apostrophes
    3. Remove special characters and punctuation
    4. Remove extra spaces
    """

    # Convert to lowercase
    text = text.lower()

    # Remove apostrophes
    # Example:
    # can't -> cant
    # don't -> dont
    text = re.sub(r"'", "", text)

    # Remove special characters and punctuation
    # Keep only letters, numbers and whitespace
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


class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to recognize",
        json_schema_extra={
            "example": "I feel so happy and excited"
        }
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
# MODEL STORAGE
# ============================================================

dl_model = {}

# ============================================================
# LIFESPAN - LOAD MODEL AND TOKENIZER
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):

    print("=" * 60)
    print("Loading model and tokenizer...")
    print("=" * 60)

    try:

        # Load GRU model
        dl_model["BIGRU"] = load_model(MODEL_PATH)

        print("✓ GRU model loaded successfully")

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

        dl_model.clear()

    yield

    # Cleanup when application shuts down
    dl_model.clear()

    print("Model resources cleared.")
    
    
    
    
'''
Mount the static files to the fastapi app
enable cors (cross origin resource sharing ) to allow request from different origins.
'''




# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Emotion Detection API",
    description="Emotion detection using a trained GRU deep learning model",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)

# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


"""
API Endpoint    
1. server ui at homepage ('/')
2. health check endpoint ('/health')
3. predict emotion endpoint ('/predict')
"""

# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", include_in_schema=False)
def server_ui():

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
def health_check():

    model_loaded = (
        dl_model.get("BIGRU") is not None
        and dl_model.get("tokenizer") is not None
    )

    return HealthResponse(
        status="Server is Running",
        model_loaded=model_loaded
    )


# ============================================================
# PREDICT EMOTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_emotion(text_input: TextInput):

    # Get model and tokenizer
    bigru_model = dl_model.get("BIGRU")
    tokenizer_model = dl_model.get("tokenizer")

    # Check model availability
    if bigru_model is None or tokenizer_model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model or tokenizer is not loaded. "
                "Please check the model loader."
            )
        )

    # --------------------------------------------------------
    # Preprocess input text
    # --------------------------------------------------------

    clean_text = preprocess_text(
        text_input.text
    )

    # Check whether text is empty after preprocessing
    if not clean_text:

        raise HTTPException(
            status_code=400,
            detail=(
                "The provided text contains no valid "
                "characters after preprocessing."
            )
        )

    # --------------------------------------------------------
    # Convert text into numerical sequence
    # --------------------------------------------------------

    tokenized_text = tokenizer_model.texts_to_sequences(
        [clean_text]
    )

    # --------------------------------------------------------
    # Pad sequence
    # --------------------------------------------------------

    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = bigru_model.predict(
        padded_sequence,
        verbose=0
    )

    # Get first prediction
    probability = prediction[0]

    # --------------------------------------------------------
    # Validate model output
    # --------------------------------------------------------

    if len(probability) != len(EMOTION_LABELS):

        raise HTTPException(
            status_code=500,
            detail=(
                f"Model returned {len(probability)} classes, "
                f"but {len(EMOTION_LABELS)} emotion labels "
                f"are configured."
            )
        )

    # --------------------------------------------------------
    # Find highest probability emotion
    # --------------------------------------------------------

    top_index = int(
        np.argmax(probability)
    )

    top_label = EMOTION_LABELS[top_index]

    confidence = float(
        probability[top_index]
    )

    # --------------------------------------------------------
    # Create probability dictionary
    # --------------------------------------------------------

    all_probability = {
        label: float(prob)
        for label, prob in zip(
            EMOTION_LABELS,
            probability
        )
    }

    # --------------------------------------------------------
    # Add emoji
    # --------------------------------------------------------

    predicted_emotion = (
        f"{top_label} "
        f"{EMOTION_EMOJIS.get(top_label, '')}"
    )

    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=predicted_emotion,
        confidence=confidence,
        all_probability=all_probability
    )