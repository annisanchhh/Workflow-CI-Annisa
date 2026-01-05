from prometheus_client import start_http_server, Counter, Histogram
import time

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests"
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency"
)

ERROR_COUNT = Counter(
    "inference_errors_total",
    "Total inference errors"
)

if __name__ == "__main__":
    start_http_server(8000)
    while True:
        REQUEST_COUNT.inc()
        with INFERENCE_LATENCY.time():
            time.sleep(0.2)
        time.sleep(5)
