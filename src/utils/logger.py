import logging
import time
from prometheus_client import Counter, Histogram

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(module)s", "message": "%(message)s"}'
)
logger = logging.getLogger("nexus_search")

# --- Prometheus Metrics Definitions ---
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests processed",
    ["method", "endpoint", "status"]
)

QUERY_LATENCY = Histogram(
    "query_latency_seconds",
    "Latency of search and chat RAG operations in seconds",
    ["operation"]
)

CACHE_ACCESS = Counter(
    "cache_access_total",
    "Total cache checks partitioned by hit or miss outcome",
    ["outcome"]
)

DOCUMENTS_INGESTED = Counter(
    "documents_ingested_total",
    "Total count of documents successfully ingested and indexed"
)
