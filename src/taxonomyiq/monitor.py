"""Population Stability Index: monitoring proxy, not a performance estimate."""
import math

def population_stability_index(reference: dict[str, float], observed: dict[str, int]) -> dict:
    total = sum(observed.values())
    if not total:
        return {"psi": None, "n": 0, "warning": "no_predictions_recorded"}
    psi = 0.0
    for key in set(reference) | set(observed):
        p = max(reference.get(key, 0), 1e-6)
        q = max(observed.get(key, 0) / total, 1e-6)
        psi += (q-p)*math.log(q/p)
    return {"psi": round(psi, 5), "n": total,
            "warning": "distribution_shift_review" if psi >= .2 else None}
