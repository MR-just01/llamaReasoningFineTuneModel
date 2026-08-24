import time
import statistics
import requests


# ---------------------------------------------------------
# API CONFIGURATION
# ---------------------------------------------------------

API_URL = "http://127.0.0.1:8000/generate"

# Number of requests to send.
NUM_REQUESTS = 5

# Maximum time to wait for one API request.
REQUEST_TIMEOUT = 120


# ---------------------------------------------------------
# TEST REQUEST
# ---------------------------------------------------------

PAYLOAD = {
    "instruction": "Solve the following reasoning problem.",
    "input": """
A farmer has 17 sheep. All but 9 die. How many sheep are left?

A) 8
B) 9
C) 17
D) 26
""",
    "max_new_tokens": 256,
}


# ---------------------------------------------------------
# SEND BENCHMARK REQUESTS
# ---------------------------------------------------------

latencies = []
successful_requests = 0
failed_requests = 0

print("=" * 60)
print("API BENCHMARK")
print("=" * 60)

print(f"Endpoint: {API_URL}")
print(f"Requests: {NUM_REQUESTS}")
print()


for i in range(NUM_REQUESTS):

    print(f"Running request {i + 1}/{NUM_REQUESTS}...")

    start = time.perf_counter()

    try:

        response = requests.post(
            API_URL,
            json=PAYLOAD,
            timeout=REQUEST_TIMEOUT,
        )

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        if response.status_code == 200:

            successful_requests += 1
            latencies.append(latency_ms)

            print(
                f"Latency: {latency_ms:.2f} ms"
            )

        else:

            failed_requests += 1

            print(
                f"Request failed: HTTP {response.status_code}"
            )

    except requests.RequestException as exc:

        failed_requests += 1

        print(
            f"Request failed: {exc}"
        )


# ---------------------------------------------------------
# CALCULATE METRICS
# ---------------------------------------------------------

if latencies:

    latencies_sorted = sorted(latencies)

    mean_latency = statistics.mean(latencies)

    p50 = statistics.median(latencies)

    # For only 5 requests, these percentile values
    # are approximate. They become more meaningful
    # with larger benchmark sizes.
    p95 = statistics.quantiles(
        latencies,
        n=20,
        method="inclusive",
    )[18]

    p99 = statistics.quantiles(
        latencies,
        n=100,
        method="inclusive",
    )[98]

    min_latency = min(latencies)
    max_latency = max(latencies)

    print()
    print("=" * 60)
    print("API BENCHMARK RESULTS")
    print("=" * 60)

    print(f"Requests:          {NUM_REQUESTS}")
    print(f"Successful:        {successful_requests}")
    print(f"Failed:            {failed_requests}")

    print(f"Mean latency:      {mean_latency:.2f} ms")
    print(f"P50 latency:       {p50:.2f} ms")
    print(f"P95 latency:       {p95:.2f} ms")
    print(f"P99 latency:       {p99:.2f} ms")
    print(f"Min latency:       {min_latency:.2f} ms")
    print(f"Max latency:       {max_latency:.2f} ms")

    print("=" * 60)

else:

    print()
    print("No successful requests were recorded.")