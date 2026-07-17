# Adult Income Prediction API

An end-to-end machine learning project that predicts whether a person's annual income is:

* `<=50K`
* `>50K`

The project includes data exploration, preprocessing, model comparison, XGBoost tuning, FastAPI serving, automated tests, Docker, GitHub Actions CI, and cloud deployment.

## Live API

Base URL:

```text
<YOUR_RENDER_URL>
```

Interactive documentation:

```text
<YOUR_RENDER_URL>/docs
```

## Model Performance

The best model was XGBoost.

| Metric              | Result |
| ------------------- | -----: |
| Accuracy            |  87.2% |
| `>50K` Precision    |   0.78 |
| `>50K` Recall       |   0.65 |
| `>50K` F1-score     |   0.71 |
| Cross-validation F1 |  0.715 |

## Models Compared

* Logistic Regression
* K-Nearest Neighbors
* Random Forest
* Gradient Boosting
* XGBoost

XGBoost produced the best balance between precision and recall for the minority `>50K` class.

## Preprocessing

The saved pipeline includes:

* Median imputation for numeric features
* Standard scaling
* Constant-value imputation for missing categorical features
* One-hot encoding
* Handling of unseen categories
* Tuned XGBoost classifier

## API Endpoints

### Health Check

```http
GET /health
```

### Model Information

```http
GET /model-info
```

### Prediction

```http
POST /predict
```

Example request:

```json
{
  "age": 40,
  "workclass": "Private",
  "fnlwgt": 200000,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Married-civ-spouse",
  "occupation": "Exec-managerial",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 0,
  "capital_loss": 0,
  "hours_per_week": 45,
  "native_country": "United-States"
}
```

Example response:

```json
{
  "prediction": 1,
  "label": ">50K",
  "probabilities": {
    "less_equal_50k": 0.2186,
    "greater_50k": 0.7814
  }
}
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Run Tests

```bash
python -m pytest -v
```

The tests cover:

* Health endpoint
* Model-information endpoint
* Valid predictions
* Invalid numeric values
* Missing required fields
* Unseen categorical values
* Prediction probability structure

## Docker

Build the image:

```bash
docker build -t adult-income-api:v1 .
```

Run the container:

```bash
docker run --name adult-income-api -p 8000:8000 adult-income-api:v1
```

Using Docker Compose:

```bash
docker compose up --build
```

## Continuous Integration

GitHub Actions automatically:

1. Installs project dependencies
2. Runs all API tests
3. Builds the Docker image when tests pass

## Project Structure

```text
adult_income/
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── model_service.py
├── models/
│   └── adult_income_v1.pkl
├── training/
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── README.md
```

## Responsible Use

This project is educational. The dataset contains sensitive demographic features such as sex and race. The model should not be used for real hiring, lending, insurance, or other high-impact decisions without extensive fairness analysis, legal review, and human oversight.
