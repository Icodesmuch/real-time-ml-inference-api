from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.models.inference import load_model, predict as model_predict


app = FastAPI()


class PredictionRequest(BaseModel):
    user_id: str
    context: dict


class PredictionResponse(BaseModel):
    user_id: str
    prediction: float
    model_version: str


@app.on_event("startup")
def startup():
    """Optionally load model at startup (e.g. after downloading from S3)."""
    try:
        load_model()
    except FileNotFoundError:
        pass  # Model will be loaded on first /predict or after S3 download


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        prediction, model_version = model_predict(request.context)
        return PredictionResponse(
            user_id=request.user_id,
            prediction=prediction,
            model_version=model_version,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))