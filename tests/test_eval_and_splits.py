import numpy as np
import pytest

from qkhosts.active.loop import labels_to_recall, run_al
from qkhosts.data.splits import (assert_disjoint_groups, chemsys, cluster_out, group_kfold, leave_element_out,
                                 subsample, tail_split)
from qkhosts.eval.metrics import enrichment_factor, mae, r2, topk_precision
from qkhosts.eval.stats import bootstrap_ci, corrected_resampled_ttest, h1_decision, holm
from qkhosts.kernels.classical import rbf
from qkhosts.models.kernel_models import GP, KRR, select_and_fit
from qkhosts.data.splits import random_kfold

rng = np.random.default_rng(0)


def test_group_kfold_no_leakage_and_seeded():
    groups = [chemsys(s) for s in (["Si", "C"], ["C", "Si"], ["O", "Si"], ["Zn", "O"], ["Ca", "O"], ["Mg", "O"])] * 3
    f1, f2 = group_kfold(groups, 3, 0), group_kfold(groups, 3, 1)
    assert_disjoint_groups(f1, groups)
    assert any(not np.array_equal(a[1], b[1]) for a, b in zip(f1, f2))


def test_other_splits():
    sets = [{"Si", "C"}, {"Si", "O"}, {"Zn", "O"}, {"Ca", "S"}]
    folds = leave_element_out(sets, ["O", "Si"])
    assert set(folds[0][1]) == {1, 2}
    X = rng.normal(size=(30, 2))
    assert sum(len(te) for _, te in cluster_out(X, 3)) == 30
    tr, te = tail_split(np.arange(100.0), 0.9)[0]
    assert len(te) == 10 and te.min() == 90
    assert len(subsample(np.arange(50), 10, 0)) == 10


def test_metrics():
    y = np.arange(100.0)
    assert mae(y, y) == 0 and r2(y, y) == 1
    assert topk_precision(y, y, 0.1) == 1.0
    assert topk_precision(y, -y, 0.1) == 0.0
    assert enrichment_factor(y, y, 0.1, 90) == pytest.approx(10.0)


def test_stats():
    d = rng.normal(-1, 0.1, 25)
    t, p = corrected_resampled_ttest(d, 400, 100)
    assert t < 0 and p < 1e-3
    lo, hi = bootstrap_ci(d, n_boot=2000)
    assert lo < d.mean() < hi
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_h1_decision_paths():
    J = 25
    c = rng.normal(1.0, 0.02, J)
    adv = {N: {"mae_q": c * 0.7, "mae_c": c, "n_train": N, "n_test": 50} for N in (50, 100, 200, 400)}
    assert h1_decision(adv)["verdict"] == "advantage"
    par = {N: {"mae_q": c * (1 + rng.normal(0, 0.003, J)), "mae_c": c, "n_train": N, "n_test": 50} for N in (50, 100, 200, 400)}
    assert h1_decision(par)["verdict"] == "parity"


def test_models_and_selection():
    X = rng.uniform(-1, 1, (60, 2))
    y = np.sin(3 * X[:, 0])
    K = rbf(X, lengthscale=0.5)
    assert mae(y, KRR(1e-4).fit(K, y).predict(K)) < 0.05
    gp = GP(noise=1e-4).fit(K, y)
    m, s = gp.predict(K, return_std=True)
    assert np.all(s > 0) and np.isfinite(gp.log_marginal_likelihood())
    kfn = lambda kp, A, B: rbf(A, A if B is None else B, kp["ls"])
    model, kp, ep, table = select_and_fit(kfn, [{"ls": l} for l in (0.1, 0.5, 2)], "krr",
                                          [{"alpha": a} for a in (1e-4, 1e-1)], X, y, random_kfold(60, 3), budget=4)
    assert len(table) == 4


def test_active_learning_beats_random_on_smooth_target():
    X = rng.uniform(-1, 1, (150, 2))
    y = -((X[:, 0] - 0.5) ** 2 + (X[:, 1] + 0.3) ** 2)
    K = rbf(X, lengthscale=0.5)
    ucb = [labels_to_recall(run_al(K, y, 5, 5, 20, "ucb", seed=s, gp_kwargs={"noise": 1e-3})["recall"], 5, 5) or 999 for s in range(3)]
    rnd = [labels_to_recall(run_al(K, y, 5, 5, 20, "random", seed=s)["recall"], 5, 5) or 999 for s in range(3)]
    assert np.median(ucb) < np.median(rnd)
