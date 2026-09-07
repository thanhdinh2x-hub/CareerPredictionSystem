from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# Tạo FastAPI server
app = FastAPI()


# ==========================================
# LOAD MODEL
# ==========================================

# Lấy thư mục hiện tại của app.py
ML_SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đi lên thư mục CareerPredictionSystem
PROJECT_DIR = os.path.dirname(ML_SERVICE_DIR)

# Đường dẫn tới model
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "career_level_model.pkl"
)


# Load model một lần khi server khởi động
model = joblib.load(MODEL_PATH)


print("Model loaded successfully")


# ==========================================
# INPUT DATA
# ==========================================

class JobInput(BaseModel):

    title: str
    location: str
    description: str
    function: str
    industry: str


# ==========================================
# TEST SERVER
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Career prediction API is running"
    }


# ==========================================
# PREDICTION API
# ==========================================

@app.post("/predict")
def predict(data: JobInput):

    # Tạo DataFrame giống dữ liệu lúc train
    input_data = pd.DataFrame(
        [{
            "title": data.title,
            "location": data.location,
            "description": data.description,
            "function": data.function,
            "industry": data.industry
        }]
    )

    # Dự đoán
    prediction = model.predict(input_data)

    return {
        "career_level": prediction[0]
    }