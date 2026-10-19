import numpy as np
from scipy import stats


def difference(control, treatment, alpha=0.05):
    """Welch difference in means, independent randomisation units."""
    a, b = np.asarray(control, float), np.asarray(treatment, float)
    if len(a) < 2 or len(b) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Need at least two finite observations in each arm")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    effect = float(b.mean() - a.mean())
    se = float(np.sqrt(va + vb))
    if se == 0:
        return dict(
            effect=effect,
            se=0.0,
            p=1.0 if effect == 0 else 0.0,
            low=effect,
            high=effect,
        )
    df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
    width = stats.t.ppf(1 - alpha / 2, df) * se
    return dict(
        effect=effect,
        se=se,
        p=float(2 * stats.t.sf(abs(effect / se), df)),
        low=float(effect - width),
        high=float(effect + width),
    )


def sample_size(sd, mde, power=0.8, alpha=0.05):
    if sd <= 0 or mde <= 0 or not 0 < power < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid planning parameters")
    return int(
        np.ceil(
            2
            * (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) ** 2
            * sd**2
            / mde**2
        )
    )


def srm(assignment, expected_treatment=0.5):
    z = np.asarray(assignment)
    if not len(z) or not np.isin(z, [0, 1]).all() or not 0 < expected_treatment < 1:
        raise ValueError("Invalid assignment or expected allocation")
    counts = np.bincount(z.astype(int), minlength=2)
    return float(
        stats.chisquare(
            counts,
            f_exp=len(z) * np.array([1 - expected_treatment, expected_treatment]),
        ).pvalue
    )


def cuped(y, pre, theta):
    y, pre = np.asarray(y, float), np.asarray(pre, float)
    if y.shape != pre.shape or not np.isfinite(theta):
        raise ValueError("Invalid CUPED inputs")
    return y - theta * (pre - pre.mean())


def bh_adjust(pvalues):
    p = np.asarray(pvalues, float)
    if not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p values")
    order = np.argsort(p)
    corrected = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    output = np.empty(len(p))
    output[order] = np.minimum(corrected, 1)
    return output.tolist()


def analyse(pre, outcome, assignment, guardrail, theta, planned_per_arm=2000, mde=0.15):
    z = np.asarray(assignment)
    if not np.isin(z, [0, 1]).all():
        raise ValueError("Assignment must be binary")
    if len({len(pre), len(outcome), len(z), len(guardrail)}) != 1:
        raise ValueError("Length mismatch")
    control = z == 0
    treatment = z == 1
    raw = difference(np.asarray(outcome)[control], np.asarray(outcome)[treatment])
    adjusted = cuped(outcome, pre, theta)
    effect = difference(adjusted[control], adjusted[treatment])
    # One-sided upper confidence bound: a nonsignificant harm test is NOT evidence of safety.
    guard = difference(
        np.asarray(guardrail)[control], np.asarray(guardrail)[treatment], alpha=0.10
    )
    allocation_p = srm(z)
    complete = min(control.sum(), treatment.sum()) >= planned_per_arm
    balanced = allocation_p >= 0.001
    safe = guard["high"] < 0.05
    if not balanced:
        decision = "INVALID_RANDOMISATION"
    elif not complete:
        decision = "CONTINUE_TO_FIXED_HORIZON"
    elif effect["low"] > mde and safe:
        decision = "SHIP"
    elif effect["high"] < 0 or guard["low"] > 0.05:
        decision = "REJECT"
    else:
        decision = "INCONCLUSIVE"
    pooled = np.sqrt(
        (
            np.var(np.asarray(pre)[control], ddof=1)
            + np.var(np.asarray(pre)[treatment], ddof=1)
        )
        / 2
    )
    return dict(
        raw=raw,
        cuped=effect,
        srm_p=allocation_p,
        decision=decision,
        guardrail=guard,
        guardrail_margin=0.05,
        guardrail_safe=bool(safe),
        fixed_horizon_complete=bool(complete),
        baseline_standardised_difference=float(
            (np.mean(np.asarray(pre)[treatment]) - np.mean(np.asarray(pre)[control]))
            / pooled
        ),
        variance_reduction=float(1 - effect["se"] ** 2 / raw["se"] ** 2),
        counts=[int(control.sum()), int(treatment.sum())],
    )


def simulate(seed=17, n=6000, effect=0.4, broken=False):
    rng = np.random.default_rng(seed)
    pre = rng.normal(size=n)
    z = rng.binomial(1, 0.5, n)
    outcome = 2 * pre + effect * z + rng.normal(size=n)
    guard = rng.normal(0, 0.3, n)
    # Post-assignment logging loss removes low-outcome treatment users.
    keep = ~((z == 1) & (pre < 0)) if broken else np.ones(n, bool)
    return pre[keep], outcome[keep], z[keep], guard[keep]


def peeking_simulation(seed=44, repetitions=300):
    rng = np.random.default_rng(seed)
    any_naive = any_corrected = final = 0
    for _ in range(repetitions):
        a = rng.normal(size=1000)
        b = rng.normal(size=1000)
        p = np.array([difference(a[:n], b[:n])["p"] for n in range(100, 1001, 100)])
        any_naive += bool((p < 0.05).any())
        any_corrected += bool((p < 0.05 / 10).any())
        final += bool(p[-1] < 0.05)
    return dict(
        repetitions=repetitions,
        looks=10,
        naive_any_significant=any_naive / repetitions,
        bonferroni_any_significant=any_corrected / repetitions,
        fixed_horizon_significant=final / repetitions,
    )


def subgroup_analysis(pre, outcome, assignment, guardrail, theta):
    # Prespecified pre-treatment activity strata; exploratory multiplicity correction.
    adjusted = cuped(outcome, pre, theta)
    z = np.asarray(assignment)
    groups = np.asarray(pre) >= 0
    rows = []
    for value, name in [(False, "low_pre_activity"), (True, "high_pre_activity")]:
        mask = groups == value
        rows.append(
            dict(
                group=name,
                n=int(mask.sum()),
                **difference(adjusted[mask & (z == 0)], adjusted[mask & (z == 1)]),
            )
        )
    for row, q in zip(rows, bh_adjust([r["p"] for r in rows])):
        row["bh_q"] = q
    return rows
