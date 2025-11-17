import json

def compute_stats(jsonl_path):
    latencies = []
    memories = []

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print("Skipping malformed line:", line[:80], "...")
                continue

            if "total_latency" in data:
                latencies.append(data["total_latency"])
            if "memory_gb" in data:
                memories.append(data["memory_gb"])

    mean_latency = sum(latencies) / len(latencies) if latencies else 0
    mean_memory = sum(memories) / len(memories) if memories else 0

    print(f"Mean Latency: {mean_latency:.4f} seconds")
    print(f"Mean RAM Usage: {mean_memory:.4f} GB")

    return mean_latency, mean_memory


# Example usage:
compute_stats("./cost_breakdown.jsonl")