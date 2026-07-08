# 3-5 Minute Video Script

Hello, my name is Raja. This video demonstrates my Machine Learning Operations Assignment 01 project: an end-to-end heart disease risk prediction system using the official UCI Heart Disease dataset.

The objective is not only to train a model, but to build a production-style MLOps workflow. The solution covers data acquisition, EDA, preprocessing, model training, experiment tracking, API development, Docker containerization, CI/CD, deployment, and monitoring.

First, the dataset is downloaded from the official UCI Machine Learning Repository using the reproducible data acquisition script. The original target variable contains multiple disease classes, and I convert it into a binary classification problem: zero means no heart disease, and one means heart disease risk is present.

For EDA, I review missing values, data types, class balance, feature distributions, and correlations. The purpose of EDA is to understand the patient health variables before modeling and to identify any preprocessing needs such as missing value imputation and scaling.

For model development, I use a scikit-learn pipeline with a ColumnTransformer. Numerical variables are imputed and scaled, while categorical variables are imputed and one-hot encoded. I compare Logistic Regression and Random Forest using cross-validation and hyperparameter tuning. Because this is a healthcare-style risk screening problem, I prioritize recall and ROC-AUC rather than accuracy alone.

MLflow is used for experiment tracking. Each model run logs parameters, hyperparameters, cross-validation score, test metrics, confusion matrix, ROC curve, classification report, and model artifact. This makes the experiment reproducible and auditable.

The final model is packaged as a reusable joblib artifact. The FastAPI service exposes `/`, `/health`, `/predict`, and `/metrics`. The `/predict` endpoint accepts patient health data in JSON format and returns the predicted class, risk label, confidence score, model name, and model version.

Next, I containerize the service with Docker. The Dockerfile installs dependencies, runs data acquisition, trains the model, and starts the FastAPI API with Uvicorn. This ensures the application can run consistently across environments.

For CI/CD, I use GitHub Actions. The workflow checks out the repository, sets up Python, installs dependencies, runs linting, executes unit tests, validates training, and performs a Docker build check. The pipeline fails if tests or linting fail.

For deployment, the recommended public option is Render or Railway because both are simple, low-cost, and suitable for academic evidence. I also provide Kubernetes manifests for Minikube or Docker Desktop Kubernetes as a local reproducible fallback.

Finally, monitoring is implemented using structured API logs and Prometheus metrics. The `/metrics` endpoint exposes request count, prediction count, error count, and latency metrics. These metrics can be visualized in Grafana.

In conclusion, this project demonstrates a complete MLOps lifecycle from official data acquisition to monitored deployment. It is reproducible, testable, containerized, and ready for academic demonstration.
