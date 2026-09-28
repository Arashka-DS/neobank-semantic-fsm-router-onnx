import time
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_PATH = "models/router_int8.onnx"
TOKENIZER_DIR = "models/tokenizer"

def run_benchmark(n_iterations=500):
    print(f"Running latency benchmark over {n_iterations} samples...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(MODEL_PATH, sess_opts, providers=['CPUExecutionProvider'])
    
    sample_text = "پونصد هزار تومن به علی کارت به کارت کن"
    encoded = tokenizer(sample_text, return_tensors="np", truncation=True, max_length=32)
    inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64)
    }
    
    # Warmup
    for _ in range(25):
        sess.run(None, inputs)
        
    latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        latencies.append((time.perf_counter() - t0) * 1000)
        
    latencies = np.array(latencies)
    print("\n--- ONNX INT8 CPU BENCHMARK RESULTS ---")
    print(f"P50 Latency : {np.percentile(latencies, 50):.2f} ms")
    print(f"P95 Latency : {np.percentile(latencies, 95):.2f} ms")
    print(f"P99 Latency : {np.percentile(latencies, 99):.2f} ms")
    print(f"Mean Latency: {np.mean(latencies):.2f} ms")
    print(f"Throughput  : {1000 / np.mean(latencies):.1f} queries/sec/core")

if __name__ == "__main__":
    run_benchmark()
