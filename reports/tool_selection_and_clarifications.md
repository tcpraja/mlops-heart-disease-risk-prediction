# Tool Selection and Clarifications - Raw UCI MLOps Project

The project uses Python, pandas, scikit-learn, MLflow, FastAPI, pytest, Docker, GitHub Actions, Kubernetes/Minikube, logging, Prometheus, Git, and GitHub.

Raw-only clarification: the processed UCI files are not used for model training. The acquisition script extracts the selected 14 variables directly from the four original raw 76-attribute source files: `cleveland.data`, `hungarian.data`, `switzerland.data`, and `long-beach-va.data`.
