"""Metrics helpers placeholder."""
def compute_basic_stats(values):
    if not values:
        return {}
    return {"min": min(values), "max": max(values), "mean": sum(values)/len(values)}

__all__ = ["compute_basic_stats"]
