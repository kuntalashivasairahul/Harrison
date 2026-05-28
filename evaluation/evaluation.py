import json
import requests
import time

API_URL = "http://127.0.0.1:8000/ask"

with open("evaluation/test_queries.json") as f:
    queries = json.load(f)

results = []

for q in queries:
    print(f"Running query: {q['query']}")

    start = time.time()

    response = requests.post(API_URL, json=q)
    answer = response.json()

    latency = time.time() - start

    result = {
        "query": q["query"],
        "latency_seconds": round(latency, 3),
        "answer": answer["answer"]
    }

    results.append(result)

    print("Latency:", latency)
    print("Answer preview:", answer["answer"][:120])
    print("--------------------------------------------------")

with open("evaluation/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete. Results saved to evaluation/results.json")