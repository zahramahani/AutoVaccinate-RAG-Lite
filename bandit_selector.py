"""
bandit_selector.py
LinUCB contextual bandit for adaptive patch selection.
NOW SUPPORTS: RAGAS scores in context encoding for better learning.
"""

import numpy as np
import logging
import math
from typing import List, Dict, Optional

class BanditPatchSelector:
    """
    Contextual bandit using LinUCB algorithm.
    Selects patches based on failure context (NLI + KG + RAGAS scores).
    """

    def __init__(self, patch_space: List[Dict], alpha=0.5, dim=8, use_ragas=True):
        """
        :param patch_space: List of patch configurations
        :param alpha: Exploration coefficient (UCB tuning)
        :param dim: Context vector dimensionality (8D with RAGAS, 5D without)
        :param use_ragas: Whether to include RAGAS scores in context
        """
        self.patch_space = patch_space
        self.alpha = alpha
        self.use_ragas = use_ragas
        self.dim = dim if use_ragas else 5  # 8D with RAGAS, 5D without
        self.n_arms = len(patch_space)
        
        # LinUCB parameters: A_a and b_a for each arm
        self.A = [np.eye(self.dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.dim) for _ in range(self.n_arms)]
        
        self.rewards = []  # History: [(context, arm, reward, cost)]
        self.logger = logging.getLogger("BanditSelector")
        
        self.logger.info(
            f"🤖 Initialized LinUCB bandit: {self.n_arms} arms, "
            f"dim={self.dim}, alpha={alpha}, use_ragas={use_ragas}"
        )

    def encode_context(
        self, 
        failure_report: Dict, 
        ragas_scores: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Encode failure report + optional RAGAS scores into context vector.
        
        Without RAGAS (5D):
        [nli_entails, nli_contradicts, kg_consistent, kg_inconsistent, nli_unknown]
        
        With RAGAS (8D):
        [nli_entails, nli_contradicts, kg_consistent, kg_inconsistent, nli_unknown,
         faithfulness, answer_relevancy, context_precision]
        
        :param failure_report: Output from detect_failures()
        :param ragas_scores: Optional dict with keys like "faithfulness", "answer_relevancy"
        :return: Context vector (numpy array)
        """
        nli = failure_report.get("nli", "NOT_ENOUGH_INFO")
        kg_agg = failure_report.get("kg_aggregate", "KG_UNKNOWN")

        vec = np.zeros(self.dim)
        
        # --- NLI encoding (dimensions 0-1, 4) ---
        if nli == "ENTAILS":
            vec[0] = 1.0
        elif nli == "CONTRADICTS":
            vec[1] = 1.0
        else:
            vec[4] = 1.0  # unknown/not enough info
        
        # --- KG encoding (dimensions 2-3) ---
        if kg_agg == "KG_CONSISTENT":
            vec[2] = 1.0
        elif kg_agg == "KG_INCONSISTENT":
            vec[3] = 1.0
        
        # --- RAGAS encoding (dimensions 5-7) ---
        if self.use_ragas and ragas_scores:
            # Normalize RAGAS scores to [0, 1] (they're already in that range)
            faithfulness = ragas_scores.get("faithfulness", 0.0)
            nv_accuracy = ragas_scores.get("nv_accuracy", 0.0)
            nv_context_relevance = ragas_scores.get("nv_context_relevance", 0.0)
            
            # Handle NaN values
            vec[5] = 0.0 if (isinstance(faithfulness, float) and math.isnan(faithfulness)) else faithfulness
            vec[6] = 0.0 if (isinstance(nv_accuracy, float) and math.isnan(nv_accuracy)) else nv_accuracy
            vec[7] = 0.0 if (isinstance(nv_context_relevance, float) and math.isnan(nv_context_relevance)) else nv_context_relevance
        
        return vec

    def select_arm(self, context: np.ndarray) -> int:
        """
        LinUCB: Select arm with highest upper confidence bound.
        """
        scores = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv @ self.b[a]
            
            # Mean reward estimate
            mu_a = theta_a.T @ context
            
            # Upper confidence bound
            ucb = mu_a + self.alpha * np.sqrt(context.T @ A_inv @ context)
            scores.append(ucb)
        
        chosen_arm = int(np.argmax(scores))
        self.logger.info(
            f"📈 LinUCB selected arm {chosen_arm} "
            f"(patch: {self.patch_space[chosen_arm]}) "
            f"with UCB score {scores[chosen_arm]:.4f}"
        )
        return chosen_arm

    def update(
        self, 
        arm: int, 
        context: np.ndarray, 
        reward: float, 
        cost: float,
        ragas_scores: Optional[Dict] = None
    ):
        """
        Update LinUCB model with observed reward.
        
        :param arm: Selected arm index
        :param context: Context vector used
        :param reward: Observed reward (0–1 scale, can be RAGAS-based)
        :param cost: Observed cost (0–1 scale, for logging)
        :param ragas_scores: Optional RAGAS scores for logging
        """
        # LinUCB update
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Log decision
        log_entry = {
            "arm": arm,
            "context": context.tolist(),
            "reward": reward,
            "cost": cost,
            "patch": self.patch_space[arm]
        }
        
        if ragas_scores:
            log_entry["ragas_scores"] = ragas_scores
        
        self.rewards.append(log_entry)
        
        self.logger.debug(
            f"✅ Updated arm {arm}: reward={reward:.3f}, cost={cost:.3f}"
        )

    def get_best_patch(
        self, 
        failure_report: Dict, 
        ragas_scores: Optional[Dict] = None
    ) -> tuple:
        """
        High-level interface: given failure report + RAGAS, return best patch + context.
        
        Returns: (patch_config, context_vector)
        """
        context = self.encode_context(failure_report, ragas_scores)
        arm = self.select_arm(context)
        return self.patch_space[arm], context

    def get_history(self) -> List[Dict]:
        """Return full decision history."""
        return self.rewards.copy()

    def reset(self):
        """Reset bandit state (for cross-validation)."""
        self.A = [np.eye(self.dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.dim) for _ in range(self.n_arms)]
        self.rewards.clear()
        self.logger.info("🔄 Bandit state reset")