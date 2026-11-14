"""Patient-grouped, stratified split that keeps every sample.

  * No leakage: every image of one patient goes to the same split (checked by the caller too).
  * No loss: the split only ASSIGNS patients. Every patient, and therefore every image, is placed.
  * Balance: each split's image share matches its target fraction as closely as patient sizes
    allow, and each stratum (for example image size x OFD-length group) is shared between the
    splits in the same proportions.

Method: randomised greedy assignment of patients (largest first) with many random restarts, then
single-patient move passes that accept any move lowering the total cost. The cost is in images:
  size deviation     = sum over splits of |images in split - target|
  stratum deviation  = sum over splits and strata of |count - ideal share of that stratum|
Every split must be non-empty, which is enforced with a large penalty.
"""
import numpy as np


def size_strata(values, n_bins=3):
    """Quantile bins of a per-patient number (for example median OFD length).
    values: dict key -> float. Returns dict key -> bin label string '0'..'n_bins-1'."""
    keys = list(values)
    v = np.asarray([values[k] for k in keys], dtype=float)
    if n_bins <= 1 or len(v) == 0:
        return {k: "0" for k in keys}
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(v, edges)
    return {k: str(int(b)) for k, b in zip(keys, bins)}


def _cost(C, targets, fracs, stratum_totals, counts):
    size_dev = np.abs(C.sum(axis=1) - targets).sum()
    stratum_dev = np.abs(C - np.outer(fracs, stratum_totals)).sum()
    empty_penalty = 1e9 * int(np.sum(counts == 0))
    return size_dev + stratum_dev + empty_penalty


def stratified_group_split(n_img, strata, fracs, seed=42, n_trials=2000):
    """n_img: dict patient -> number of images. strata: dict patient -> stratum label.
    fracs: dict split name -> target fraction (e.g. {'train': .8, 'val': .1, 'test': .1}).
    Returns dict patient -> split name."""
    names = list(fracs)
    fr = np.array([fracs[s] for s in names], dtype=float)
    fr = fr / fr.sum()
    pids = sorted(n_img)
    labels = sorted({strata[p] for p in pids})
    pi = {p: i for i, p in enumerate(pids)}
    gi = {g: j for j, g in enumerate(labels)}
    M = np.zeros((len(pids), len(labels)))
    for p in pids:
        M[pi[p], gi[strata[p]]] = n_img[p]
    n_per_patient = M.sum(axis=1)
    stratum_totals = M.sum(axis=0)
    N = M.sum()
    targets = fr * N
    rng = np.random.default_rng(seed)
    K = len(names)

    def full_cost(assign):
        C = np.zeros((K, len(labels)))
        counts = np.zeros(K, int)
        np.add.at(C, assign, M)
        np.add.at(counts, assign, 1)
        return _cost(C, targets, fr, stratum_totals, counts)

    best_cost, best_assign = np.inf, None
    for _ in range(n_trials):                                     # randomised greedy restarts
        perm = rng.permutation(len(pids))
        order = perm[np.argsort(-n_per_patient[perm], kind="stable")]
        C, counts = np.zeros((K, len(labels))), np.zeros(K, int)
        assign = np.zeros(len(pids), int)
        for p in order:
            best_s, best_c = 0, np.inf
            for s in range(K):
                C[s] += M[p]; counts[s] += 1
                c = _cost(C, targets, fr, stratum_totals, counts)
                C[s] -= M[p]; counts[s] -= 1
                if c < best_c:
                    best_s, best_c = s, c
            C[best_s] += M[p]; counts[best_s] += 1
            assign[p] = best_s
        cost = full_cost(assign)
        if cost < best_cost:
            best_cost, best_assign = cost, assign.copy()

    assign, cur = best_assign, full_cost(best_assign)             # single-patient move passes
    for _ in range(20):
        improved = False
        for p in range(len(pids)):
            home = assign[p]
            for s in range(K):
                if s == home:
                    continue
                assign[p] = s
                c = full_cost(assign)
                if c < cur - 1e-9:
                    cur, home, improved = c, s, True
                else:
                    assign[p] = home
            assign[p] = home
        if not improved:
            break
    return {p: names[assign[pi[p]]] for p in pids}


def split_quality(assign, n_img, strata, fracs):
    """Per-split sizes and balance of an assignment. 'max_fraction_deviation' is the largest gap
    between a split's share of the images and its target share."""
    names = list(fracs)
    total_fr = sum(fracs.values())
    N = sum(n_img.values())
    out = {}
    for s in names:
        ps = [p for p in n_img if assign[p] == s]
        imgs = sum(n_img[p] for p in ps)
        share_strata = {}
        for p in ps:
            share_strata[strata[p]] = share_strata.get(strata[p], 0) + n_img[p]
        out[s] = dict(images=int(imgs), patients=len(ps), share=imgs / N,
                      target_share=fracs[s] / total_fr, strata={k: int(v) for k, v in sorted(share_strata.items())})
    out["max_fraction_deviation"] = float(max(abs(out[s]["share"] - out[s]["target_share"]) for s in names))
    return out
