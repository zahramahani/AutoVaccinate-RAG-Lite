"""
cost_model.py
Enhanced cost calculation with real-time profiling and RAGAS integration.
"""

import time
import psutil
import logging
import math
from typing import Dict, Optional
from collections import defaultdict
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("CostModel")

# --- SYNTHETIC COST MODEL (fallback) ---
COST_MODEL = {
    "dense": {"latency": 0.8, "vram": 2.5, "api_calls": 0},
    "bm25": {"latency": 0.3, "vram": 0.5, "api_calls": 0},
    "rerank_on": {"latency": 0.6, "vram": 1.0, "api_calls": 0},
    "prompt_verbose": {"latency": 0.1, "vram": 0.1, "api_calls": 0.2},
    "prompt_verifier": {"latency": 0.1, "vram": 0.1, "api_calls": 0.3},
    "prompt_default": {"latency": 0.05, "vram": 0.05, "api_calls": 0.1},
    "prompt_concise": {"latency": 0.05, "vram": 0.05, "api_calls": 0.05},
}

# Normalization constants
MAX_LATENCY = 2.0
MAX_VRAM = 4.0
MAX_API_CALLS = 0.5


class CostProfiler:
    """
    Real-time cost profiling with support for:
    - Latency measurement
    - VRAM (GPU) or RAM (CPU) tracking
    - API token counting (Mistral)
    """
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_vram = None
        self.api_tokens = 0
        self.measurements = defaultdict(list)
        
    def start(self):
        """Start profiling session."""
        self.start_time = time.time()
        
        # Track memory
        if TORCH_AVAILABLE:
            torch.cuda.reset_peak_memory_stats()
            self.start_vram = torch.cuda.memory_allocated() / (1024**3)  # GB
        else:
            self.start_memory = psutil.virtual_memory().used / (1024**3)  # GB
        
        logger.debug("⏱️ Cost profiling started")
    
    def add_api_tokens(self, token_count: int):
        """Add API token usage (e.g., from Mistral response)."""
        self.api_tokens += token_count
    
    def stop(self) -> Dict[str, float]:
        """
        Stop profiling and return measured metrics.
        Returns: {"latency": float, "memory_gb": float, "api_tokens": int}
        """
        if self.start_time is None:
            logger.warning("⚠️ Profiler.stop() called without start()")
            return {"latency": 0.0, "memory_gb": 0.0, "api_tokens": 0}
        
        # Measure latency
        latency = time.time() - self.start_time
        
        # Measure memory
        if TORCH_AVAILABLE:
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            memory_used = max(peak_vram - (self.start_vram or 0.0), 0.0)
        else:
            current_memory = psutil.virtual_memory().used / (1024**3)
            memory_used = max(current_memory - (self.start_memory or 0.0), 0.0)
        
        metrics = {
            "latency": round(latency, 4),
            "memory_gb": round(memory_used, 4),
            "api_tokens": self.api_tokens
        }
        
        logger.debug(
            f"⏹️ Profiling stopped: latency={metrics['latency']:.3f}s, "
            f"memory={metrics['memory_gb']:.3f}GB, "
            f"tokens={metrics['api_tokens']}"
        )
        
        return metrics
    
    def log_measurement(self, component: str):
        """Log a measurement checkpoint (for component-level profiling)."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.measurements[component].append(elapsed)
    
    def get_component_breakdown(self) -> Dict[str, float]:
        """Get per-component timing breakdown."""
        return {k: sum(v) for k, v in self.measurements.items()}


def calculate_patch_cost_synthetic(patch_config: Dict) -> float:
    """
    Calculate normalized cost using synthetic model (fallback).
    Returns float in [0, 1].
    """
    cost = {"latency": 0.0, "vram": 0.0, "api_calls": 0.0}
    
    # Retriever cost
    rtype = patch_config["retriever_type"]
    cost["latency"] += COST_MODEL[rtype]["latency"]
    cost["vram"] += COST_MODEL[rtype]["vram"]
    
    # Reranker cost
    if patch_config.get("rerank_on", False):
        cost["latency"] += COST_MODEL["rerank_on"]["latency"]
        cost["vram"] += COST_MODEL["rerank_on"]["vram"]
    
    # Prompt cost
    pid = patch_config.get("prompt_id", "default")
    prompt_key = f"prompt_{pid}"
    if prompt_key in COST_MODEL:
        cost["latency"] += COST_MODEL[prompt_key]["latency"]
        cost["vram"] += COST_MODEL[prompt_key]["vram"]
        cost["api_calls"] += COST_MODEL[prompt_key]["api_calls"]
    
    # Normalize
    norm_latency = min(cost["latency"] / MAX_LATENCY, 1.0)
    norm_vram = min(cost["vram"] / MAX_VRAM, 1.0)
    norm_api = min(cost["api_calls"] / MAX_API_CALLS, 1.0)
    
    return 0.4 * norm_latency + 0.3 * norm_vram + 0.3 * norm_api


def calculate_patch_cost_real(real_metrics: Dict) -> float:
    """
    Calculate normalized cost from real profiling measurements.
    
    :param real_metrics: {"latency": float, "memory_gb": float, "api_tokens": int}
    :return: Normalized cost in [0, 1]
    """
    # Normalize measured values
    norm_latency = min(real_metrics.get("latency", 0.0) / MAX_LATENCY, 1.0)
    norm_memory = min(real_metrics.get("memory_gb", 0.0) / MAX_VRAM, 1.0)
    
    # Normalize API tokens (assume ~1000 tokens = 0.5 cost units)
    api_tokens = real_metrics.get("api_tokens", 0)
    norm_api = min(api_tokens / 1000 * 0.5, 1.0)
    
    total_cost = 0.4 * norm_latency + 0.3 * norm_memory + 0.3 * norm_api
    
    logger.debug(
        f"💰 Real cost: latency={norm_latency:.3f}, "
        f"memory={norm_memory:.3f}, api={norm_api:.3f}, "
        f"total={total_cost:.3f}"
    )
    
    return total_cost


def calculate_patch_cost(patch_config: Dict, real_metrics: Optional[Dict] = None) -> float:
    """
    Unified cost calculation:
    - Uses real metrics if available
    - Falls back to synthetic model
    
    :param patch_config: Patch configuration dict
    :param real_metrics: Optional real profiling data
    :return: Normalized cost [0, 1]
    """
    if real_metrics:
        return calculate_patch_cost_real(real_metrics)
    else:
        return calculate_patch_cost_synthetic(patch_config)


def calculate_ragas_reward(
    ragas_scores: Dict, 
    weights: Optional[Dict] = None
) -> float:
    """
    Calculate reward from RAGAS scores with configurable weights.
    
    :param ragas_scores: Dict with keys like "faithfulness", "answer_relevancy", etc.
    :param weights: Optional weights for each metric (default: equal weighting)
    :return: Weighted reward in [0, 1]
    """
    if weights is None:
        # Default: equal weight to faithfulness, answer_relevancy, context_precision
        weights = {
            "faithfulness": 0.4,
            "nv_accuracy": 0.3,
            "nv_context_relevance": 0.2,
            # "context_recall": 0.1
        }
    
    reward = 0.0
    total_weight = 0.0
    
    for metric, weight in weights.items():
        if metric in ragas_scores:
            score = ragas_scores[metric]
            # Handle NaN
            if isinstance(score, float) and not math.isnan(score):
                reward += weight * score
                total_weight += weight
    
    # Normalize by actual total weight (in case some metrics are missing)
    if total_weight > 0:
        reward /= total_weight
    
    logger.debug(f"🎯 RAGAS reward: {reward:.4f} (from {ragas_scores})")
    return reward


def calculate_combined_reward(
    failure_report: Dict,
    ragas_scores: Optional[Dict] = None,
    use_ragas: bool = True,
    ragas_weight: float = 0.7
) -> float:
    """
    Calculate combined reward from failure detection + RAGAS.
    
    :param failure_report: Output from detect_failures()
    :param ragas_scores: Optional RAGAS scores dict
    :param use_ragas: Whether to use RAGAS in reward calculation
    :param ragas_weight: Weight for RAGAS vs binary (default: 0.7 RAGAS, 0.3 binary)
    :return: Combined reward [0, 1]
    """
    # Binary reward from failure detection
    binary_reward = 1.0 if (
        failure_report.get("nli") == "ENTAILS" and
        failure_report.get("kg_aggregate") == "KG_CONSISTENT"
    ) else 0.0
    
    # If not using RAGAS or no scores available, return binary
    if not use_ragas or not ragas_scores:
        return binary_reward
    
    # Calculate RAGAS reward
    ragas_reward = calculate_ragas_reward(ragas_scores)
    
    # Weighted combination
    combined = ragas_weight * ragas_reward + (1 - ragas_weight) * binary_reward
    
    logger.debug(
        f"🎯 Combined reward: {combined:.4f} "
        f"(RAGAS: {ragas_reward:.4f}, Binary: {binary_reward:.4f})"
    )
    
    return combined
