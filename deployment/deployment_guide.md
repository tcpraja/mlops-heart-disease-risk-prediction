# Deployment Guide

## Recommended path for academic demonstration

Use Docker + FastAPI + GitHub + GitHub Actions + Render. This is the simplest route to produce public API evidence while keeping cost and complexity low.

## Render

1. Push repository to GitHub.
2. Create Render Web Service.
3. Select Docker environment.
4. Confirm `/health` returns `status: ok`.
5. Open `/docs` and execute `/predict`.
6. Save screenshots to `screenshots/deployment/` and `screenshots/api_testing/`.

## Railway

```bash
railway login
railway init
railway up
```

Verify:

```bash
curl https://<railway-url>/health
```

## Local Kubernetes with Minikube

```bash
minikube start
docker build -t heart-disease-mlops:latest .
minikube image load heart-disease-mlops:latest
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl get pods
kubectl port-forward service/heart-disease-service 8000:80
curl http://127.0.0.1:8000/health
```

## Screenshot evidence

- Docker build success
- Container running
- `/health` response
- `/predict` response
- GitHub Actions successful pipeline
- Render/Railway public endpoint
- Kubernetes pod and service status
