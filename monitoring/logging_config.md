# Logging and Monitoring Configuration

The FastAPI application logs every request through middleware.

Logged fields:

- HTTP method
- endpoint path
- status code
- request latency
- prediction output
- probability of heart disease
- errors and stack traces

Prometheus metrics are exposed at:

```text
/metrics
```

Key metrics:

- `api_requests_total`
- `model_predictions_total`
- `api_errors_total`
- `api_request_latency_seconds`

Local Prometheus test:

```bash
docker run --rm -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

Open `http://127.0.0.1:9090` and query `api_requests_total`.
