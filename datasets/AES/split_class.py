"""
Utility functions for mapping raw AES scores to model-friendly class labels.

The dataset has different score ranges per prompt/domain. We map the overall
score into 4 classes [0,1,2,3] with simple binning rules consistent with
openprompt expectations in this repo.
"""
from typing import Union


def _bins_for_domain(domain: int):
    """Return (min_score, max_score, num_bins) for overall score per domain.

    These are inferred from openprompt.trainer.get_min_max_scores().
    For domains with larger ranges (e.g., 7,8) we still bin into 4 classes.
    """
    ranges = {
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 30),
        8: (0, 60),
    }
    return ranges.get(int(domain), (0, 3))


def get_model_friendly_scores(score: Union[int, float], domain: int) -> int:
    """Map raw score to an integer class in [0,3].

    We linearly split the valid range into 4 equal-width bins and clamp.
    """
    smin, smax = _bins_for_domain(domain)
    # Avoid division by zero
    span = max(1, smax - smin)
    norm = (float(score) - smin) / span
    # 4 classes: 0,1,2,3
    idx = int(norm * 4)
    if idx < 0:
        idx = 0
    if idx > 3:
        idx = 3
    return idx
