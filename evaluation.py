import numpy as np
import pandas as pd
from typing import Dict, List


class Evaluator:
    """
    Offline evaluation module for the Gym Recommendation System.
    Computes ranking metrics:
      - Precision@K
      - Recall@K
      - F1@K
      - MAP@K
      - nDCG@K
      - HitRate@K
    """

    # -----------------------------------------------------------
    # Precision@K
    # -----------------------------------------------------------
    def precision_at_k(self, recommended: List[str], relevant: List[str], k: int):
        rec_k = recommended[:k]
        hits = len(set(rec_k) & set(relevant))
        return hits / k

    # -----------------------------------------------------------
    # Recall@K
    # -----------------------------------------------------------
    def recall_at_k(self, recommended: List[str], relevant: List[str], k: int):
        if len(relevant) == 0:
            return 0
        rec_k = recommended[:k]
        hits = len(set(rec_k) & set(relevant))
        return hits / len(relevant)

    # -----------------------------------------------------------
    # F1@K
    # -----------------------------------------------------------
    def f1_at_k(self, precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    # -----------------------------------------------------------
    # Hit Rate @K
    # -----------------------------------------------------------
    def hit_rate_at_k(self, recommended: List[str], relevant: List[str], k: int):
        rec_k = recommended[:k]
        return 1 if len(set(rec_k) & set(relevant)) > 0 else 0

    # -----------------------------------------------------------
    # MAP@K
    # -----------------------------------------------------------
    def average_precision(self, recommended, relevant, k):
        score = 0.0
        hits = 0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                hits += 1
                score += hits / (i + 1)
        return score / max(1, len(relevant))

    # -----------------------------------------------------------
    # nDCG@K
    # -----------------------------------------------------------
    def ndcg_at_k(self, recommended, relevant, k):
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                dcg += 1 / np.log2(i + 2)

        ideal_hits = min(len(relevant), k)
        idcg = sum(1 / np.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg > 0 else 0

    # -----------------------------------------------------------
    # Evaluate system across ALL users
    # -----------------------------------------------------------
    def evaluate(self, recommendations: Dict[str, List[str]], ground_truth: Dict[str, List[str]], k: int):
        rows = []

        for user_id, recs in recommendations.items():
            relevant = ground_truth.get(user_id, [])

            p = self.precision_at_k(recs, relevant, k)
            r = self.recall_at_k(recs, relevant, k)
            f1 = self.f1_at_k(p, r)
            map_k = self.average_precision(recs, relevant, k)
            ndcg = self.ndcg_at_k(recs, relevant, k)
            hit = self.hit_rate_at_k(recs, relevant, k)

            rows.append({
                "user_id": user_id,
                "precision@k": p,
                "recall@k": r,
                "f1@k": f1,
                "map@k": map_k,
                "ndcg@k": ndcg,
                "hit_rate@k": hit
            })

        return pd.DataFrame(rows)
