"""Calibration-only linear controllers over validation-frozen score banks."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from bandha_external_criteria_io import baseline, criterion, plug_in_score


def plugin_threshold(data):
    score = plug_in_score(data)
    order = np.argsort(-score, kind="stable")
    value = np.cumsum(data[order, 2])
    target = 0.1 * value[-1]
    index = min(int(np.searchsorted(value, target, side="left")), len(order) - 1)
    return float(score[order[index]])


def bank(data, mask, baseline_threshold, limit=4096):
    scores = np.unique(data[mask, 1])
    if len(scores) > limit:
        scores = scores[np.unique(np.linspace(0, len(scores) - 1, limit).astype(int))]
    return [None, "full", baseline_threshold] + scores.tolist()


def cut_moments(data, mask, multiplier, thresholds):
    order = np.argsort(data[mask, 1], kind="stable")
    scores = data[mask, 1][order]
    amounts = data[mask, 2:4] * multiplier[mask, None]
    sums = np.vstack([np.zeros(2), np.cumsum(amounts[order], axis=0)])
    locations = [
        (
            len(scores)
            if t is None
            else 0 if t == "full" else np.searchsorted(scores, t, side="left")
        )
        for t in thresholds
    ]
    return sums[-1] - sums[locations]


def upper_hull(moments, thresholds, scales):
    best = {}
    for index, (cost, capture) in enumerate(moments):
        previous = best.get(float(cost))
        if previous is None or capture > moments[previous, 1]:
            best[float(cost)] = index
    hull = []
    for index in sorted(best.values(), key=lambda i: moments[i, 0]):
        while len(hull) >= 2:
            a, b, c = moments[[hull[-2], hull[-1], index]] / scales
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cross < 0:
                break
            hull.pop()
        hull.append(index)
    return moments[hull], [thresholds[i] for i in hull]


def solve_problem(cost, capture, anchor, qmass, scales, floor, objective, cap=0.1):
    cost_row = np.r_[cost, qmass[0]] / scales[0]
    capture_row = np.r_[capture, qmass[1]] / scales[1]
    rhs = [cap - anchor[0] / scales[0], (anchor[1] - floor) / scales[1]]
    result = linprog(
        objective,
        A_ub=np.array([cost_row, -capture_row]),
        b_ub=rhs,
        A_eq=[np.r_[np.ones(len(cost)), 0]],
        b_eq=[1],
        bounds=[(0, None)] * len(cost) + [(0, 1)],
        method="highs-ipm",
        options={
            "presolve": False,
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
        },
    )
    if result.status not in (0, 2):
        raise RuntimeError(f"LP failed: {result.message}")
    return result


def calibrated_weights(
    moments, anchor, qmass, scales, baseline_capture, cap=0.1, rho=0.95
):
    n = len(moments)
    objective = np.r_[np.zeros(n), 1]
    fitted = solve_problem(
        *moments.T, anchor, qmass, scales, rho * baseline_capture, objective, cap
    )
    status = "feasible"
    if fitted.status == 2:
        status = "capture_infeasible"
        objective = -np.r_[moments[:, 1], qmass[1]] / scales[1]
        fitted = solve_problem(*moments.T, anchor, qmass, scales, 0, objective, cap)
    if fitted.status == 2:
        return np.r_[1.0, np.zeros(n)], "resource_infeasible"
    weights = np.maximum(fitted.x[:-1], 0)
    weights /= np.sum(weights)
    return np.r_[weights, np.clip(fitted.x[-1], 0, 1)], status


def fit_common(v, c, case, name, flexible, plugin, cap=0.1, rho=0.95):
    qv, qc = criterion(v, name, plugin), criterion(c, name, plugin)
    thresholds = bank(v, ~qv, case["baseline_threshold"])
    p0 = baseline(c, case)
    multiplier = np.ones(len(c)) if flexible else 1 - p0
    moments = cut_moments(c, ~qc, multiplier, thresholds)
    scales = np.maximum(np.sum(c[:, 2:4], axis=0), 1e-30)
    moments, thresholds = upper_hull(moments, thresholds, scales)
    base_moments = p0[:, None] * c[:, 2:4]
    anchor = np.zeros(2) if flexible else np.sum(base_moments[~qc], axis=0)
    qmass = np.sum(base_moments[qc], axis=0)
    weights, status = calibrated_weights(
        moments, anchor, qmass, scales, np.sum(base_moments[:, 1]), cap, rho
    )
    selected = np.flatnonzero(weights[:-1] > 1e-12)
    return {
        "mode": "flexible" if flexible else "common",
        "criterion": name,
        "plugin_threshold": plugin,
        "s": float(weights[-1]),
        "fit_status": status,
        "thresholds": [thresholds[i] for i in selected],
        "weights": weights[selected].tolist(),
        "V_bank_size": len(bank(v, ~qv, case["baseline_threshold"])),
        "C_hull_size": len(moments),
        "fitted_value_cap": cap,
        "fitted_capture_floor": rho,
    }


def fit_no_addition(c, case, cap=0.1, rho=0.95):
    q, p0 = criterion(c, "native", 0), baseline(c, case)
    totals = np.sum(p0[:, None] * c[:, 2:4], axis=0)
    qmass = np.sum((p0 * q)[:, None] * c[:, 2:4], axis=0)
    nonq = totals - qmass
    lower = max(0.0, (rho * totals[1] - nonq[1]) / qmass[1]) if qmass[1] > 0 else 0.0
    value_limit = cap * np.sum(c[:, 2])
    upper = min(1.0, (value_limit - nonq[0]) / qmass[0]) if qmass[0] > 0 else 1.0
    feasible = nonq[0] <= value_limit + 1e-9 and lower <= upper + 1e-9
    return {
        "mode": "no_addition",
        "criterion": "native",
        "plugin_threshold": 0.0,
        "s": min(lower, max(0.0, upper)),
        "fit_status": "feasible" if feasible else "infeasible",
        "fitted_value_cap": cap,
        "fitted_capture_floor": rho,
    }


def fit_case(v, c, case):
    plugin = plugin_threshold(v)
    result = {
        "published": {
            "mode": "published",
            "fit_status": "original",
            "s": case["original_common"]["contraction_s"],
        },
        "thinning": {"mode": "uniform", "fit_status": "fixed", "s": 0.95},
        "no_addition": fit_no_addition(c, case),
    }
    for name in ("native", "native_wbce", "native_plugin", "native_joint"):
        for flexible in (False, True):
            key = ("flexible_" if flexible else "common_") + name
            result[key] = fit_common(v, c, case, name, flexible, plugin)
    for name in ("native", "native_joint"):
        result["reserved_" + name] = fit_common(
            v, c, case, name, False, plugin, 0.095, 0.98
        )
    result["no_addition_reserved"] = fit_no_addition(c, case, 0.095, 0.98)
    return {"plugin_threshold": plugin, "policies": result}
