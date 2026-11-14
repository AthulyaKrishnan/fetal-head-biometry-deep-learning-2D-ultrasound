"""Statistics used in the report: bootstrap confidence intervals, paired tests, multiplicity."""
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(x, stat=np.mean, n=5000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    vals = stat(x[idx], axis=1)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def paired_diff_ci(a, b, stat=np.mean, n=5000, seed=0):
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), stat, n, seed)


def wilcoxon_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.all(d == 0):
        return 1.0
    return float(wilcoxon(a, b).pvalue)


def holm(pvalues):
    """Holm step-down adjustment; returns adjusted p-values in the original order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
