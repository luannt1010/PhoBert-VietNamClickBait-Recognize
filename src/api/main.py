import pandas as pd
from pydantic import BaseModel
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File
from clickbait_detector.inference import ClickBaitPredictor

predictor = ClickBaitPredictor(config_dir=r".\configs\phobert-base-v2",
                                   weight_path=r".\artifacts\models\last.pth",
                                   max_len=256, threshold=0.5)

class TitleInput(BaseModel):
    title: str

class UrlInput(BaseModel):
    url: str


class Output(BaseModel):
    sentence: str
    label: str
    score: float

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Clickbait Detection API"}

@app.post("/predict/title", response_model=Output)
def predict_title(request: TitleInput):
    if not request.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    try:
        title = request.title
        predictions = predictor.predict_one_title(title)
        return predictions
    except Exception:
        raise HTTPException(status_code=500, detail="Predict Failed")

@app.post("/predict/url", response_model=Output)
def predict_url(request: UrlInput):
    if not request.url.strip():
        raise HTTPException( status_code=400, detail="URL cannot be empty")
    try:
        url = request.url
        predictions = predictor.predict_url(url)
        if predictions is None:
            raise HTTPException(status_code=400, detail="Cannot extract title from URL")
        return predictions
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Predict Failed")

@app.post("/predict/file", response_model=list[Output])
async def predict_file(file: UploadFile=File(...)):
    contents = await file.read()
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(contents))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        if "title" not in df.columns:
            raise HTTPException(status_code=400, detail="File must contain a 'title' column")
        predictions = predictor.predict_file(df)
        return predictions
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Predict Failed")
