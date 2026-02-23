# Real-Time ML Inference API

A simple FastAPI-based service for real-time predictions using a scikit-learn model, plus a separate training script to generate and update the model.

## Project Structure

- `app/`
  - `__init__.py`
  - `main.py` – FastAPI application and API endpoints
- `training/`
  - `__init__.py`
  - `train_model.py` – training script that generates a churn model and saves it as a `.pkl` file
- `models/`
  - `model_v1.pkl` – trained model artifact (created by the training script; not checked into git)
- `requirements.txt` – Python dependencies
- `README.md` – project documentation

## Prerequisites

- Python 3.10+ (recommended)
- Git (optional but recommended)
- `pip` for installing dependencies

## Setup

1. **Clone the repository (if you haven’t already)**

   ```bash
   git clone git@github.com:Icodesmuch/real-time-ml-inference-api.git
   cd "Real-Time ML Inference API"
   ```

2. **Create and activate a virtual environment (recommended)**

   On Windows (PowerShell):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   On macOS / Linux:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

The training script generates a synthetic churn dataset, trains a `RandomForestClassifier`, evaluates it, and saves the trained model.

From the project root:

```bash
python -m training.train_model
```

This will:

- Generate synthetic churn data using `sklearn.datasets.make_classification`
- Train a Random Forest model
- Print accuracy and feature importances
- Save the trained model to `models/model_v1.pkl`

If the `models/` directory doesn’t exist, create it before running the script:

```bash
mkdir models
```

On Windows PowerShell you can also use:

```powershell
New-Item -ItemType Directory models
```

## Running the API

The FastAPI app lives in `app/main.py` and exposes basic endpoints.

From the project root (with your virtual environment activated):

```bash
uvicorn app.main:app --reload
```

By default, Uvicorn will run on `http://127.0.0.1:8000`.

### Available Endpoints

- `GET /health` – simple health check returning `{"status": "ok"}`.
- `POST /predict` – returns a dummy prediction for now (placeholder for real model inference).

### Interactive Docs

FastAPI automatically provides interactive documentation:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Example Prediction Request

Once the server is running, you can send a request (e.g. using `curl` or a tool like Postman).

Example `curl` request on Windows (PowerShell `^` for line continuation):

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\": \"user_123\", \"context\": {\"account_age_months\": 12}}"
```

Example JSON request body:

```json
{
  "user_id": "user_123",
  "context": {
    "account_age_months": 12,
    "monthly_charges": 50.0,
    "total_charges": 600.0,
    "support_calls": 2,
    "contract_length": 12,
    "payment_method_score": 0.8,
    "usage_frequency": 0.6,
    "feature_8": 0.1
  }
}
```

At the moment, the `/predict` endpoint returns a fixed prediction (e.g. `0.5`) and a model version string. You can later update it to load `models/model_v1.pkl` and run real inference.

## Development Workflow

A simple recommended workflow:

1. **Update / retrain the model**
   - Edit `training/train_model.py` as needed.
   - Run `python -m training.train_model` to generate a new `model_vX.pkl`.
2. **Wire up or update inference in the API**
   - Add/load the model in `app/main.py` (or a separate module, e.g. `app/models/inference.py`).
   - Expose / update the `/predict` endpoint.
3. **Run and test locally**
   - Start the API: `uvicorn app.main:app --reload`.
   - Hit `/health` and `/predict` from your browser, `curl`, or an API client.
4. **Commit changes**
   - Use `git status` to review changes.
   - Run `git add app training README.md`.
   - Commit with a clear message, e.g. `git commit -m "docs: update README and project structure"`.

## Notes

- It’s recommended to **ignore model artifacts** (`models/*.pkl`) in `.gitignore` and store them in a model registry or artifact store for real projects.
- This project is intentionally minimal and suitable as a learning scaffold for:
  - Training a basic ML model with scikit-learn
  - Exposing predictions via a FastAPI service
  - Practicing basic project structuring and git workflow
