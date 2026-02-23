from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class PredictionRequest(BaseModel):
    user_id: str
    context: dict


class PredictionResponse(BaseModel):
    user_id: str
    prediction: float
    model_version: str

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    return PredictionResponse(
        user_id=request.user_id,
        prediction=0.5,
        model_version="1.0"
    )