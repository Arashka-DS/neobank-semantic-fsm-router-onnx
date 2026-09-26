import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def run_benchmark():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
    session = ort.InferenceSession("models/semantic_router_int8.onnx", providers=['CPUExecutionProvider'])
    
    text = "دویست هزار تومان به رضا کارت به کارت کن"
    inputs = tokenizer(text, max_length=32, padding="max_length", return_tensors="np")
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

    # Warmup
    for _ in range(50):
        _ = session.run(None, ort_inputs)

    latencies = []
    iterations = 500
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, ort_inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"--- ONNX INT8 Benchmark Results ({iterations} runs) ---")
    print(f"Median (P50) Latency: {p50:.2f} ms")
    print(f"95th Percentile (P95): {p95:.2f} ms")
    print(f"99th Percentile (P99): {p99:.2f} ms")

if __name__ == "__main__":
    run_benchmark()
