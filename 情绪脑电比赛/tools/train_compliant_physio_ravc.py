from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.cross_fold_confidence_prior_fusion import ids_for_subjects, labels_to_array  # noqa: E402
from tools.cross_fold_neurovascular_fusion import load_or_build_precomputed, sanitize  # noqa: E402
from tools.run_iteration_experiments import expand_subjects, load_labels, score  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "submissions" / "compliant_physio"
MODEL_HELPERS = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and package a compliant physiological MER-PS model without video-time label lookup."
    )
    parser.add_argument(
        "--data-root",
        default="data/MER_PS_codabench_trainval/MER_PS_codabench_public_trainval",
    )
    parser.add_argument("--subjects", default="test_1-test_24")
    parser.add_argument("--fold-size", type=int, default=4)
    parser.add_argument("--precompute-cache", default="experiments/features/neurovascular_precompute_fnirs_all6.npz")
    parser.add_argument("--output", default="experiments/results/iteration_543_compliant_physio_ravc.json")
    parser.add_argument("--package-dir", default="submissions/compliant_physio")
    parser.add_argument("--zip-output", default="submissions/sub10_compliant_physio.zip")
    parser.add_argument("--components", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--huber-epsilon", type=float, default=1.35)
    parser.add_argument("--huber-iters", type=int, default=60)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--negative-scale", type=float, default=0.60)
    parser.add_argument("--positive-scale", type=float, default=1.00)
    parser.add_argument("--trial-start-blend", type=float, default=0.40)
    parser.add_argument("--package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    helpers = load_submission_helpers()
    subjects = expand_subjects(args.subjects)
    folds = [subjects[start : start + args.fold_size] for start in range(0, len(subjects), args.fold_size)]
    data_root = Path(args.data_root)
    labels = load_labels(data_root, subjects)
    sample_ids_all, pre, feature_shapes = load_or_build_precomputed(
        data_root=data_root,
        subjects=subjects,
        cache_path=Path(args.precompute_cache),
        fnirs_types=(0, 1, 2, 3, 4, 5),
        feature_normalization="none",
        baseline_correction=True,
    )
    x_all = sanitize(pre["early_concat"]).astype(np.float32)
    feature_index = {sample_id: index for index, sample_id in enumerate(sample_ids_all)}

    fold_outputs = []
    metric_acc: dict[str, dict[str, object]] = {}
    for fold_index, val_subjects in enumerate(folds, start=1):
        train_subjects = [subject for subject in subjects if subject not in val_subjects]
        print(f"[fold {fold_index}] compliant physiological RAVC", flush=True)
        train_ids = ids_for_subjects(labels, train_subjects)
        val_ids = ids_for_subjects(labels, val_subjects)
        y_train = labels_to_array(labels, train_ids).astype(np.float32)
        y_val = labels_to_array(labels, val_ids).astype(np.float32)
        train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
        val_idx = np.asarray([feature_index[sample_id] for sample_id in val_ids], dtype=np.int64)

        model = fit_model(
            x_all[train_idx],
            y_train[:, 0],
            components=args.components,
            alpha=args.alpha,
            epsilon=args.huber_epsilon,
            max_iter=args.huber_iters,
        )
        raw_v = predict_model(model, x_all[val_idx])
        candidates = build_candidates(helpers, raw_v, y_train, val_ids, args)
        fold_results = []
        for name, pred in candidates.items():
            result = score(name, y_val, pred, "Compliant EEG/fNIRS physiological model; no video-time label lookup.")
            fold_results.append(result)
            add_metric(metric_acc, name, y_val, pred, result["notes"])
        fold_outputs.append(
            {
                "fold": fold_index,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "val_samples": len(val_ids),
                "results": sorted(fold_results, key=lambda item: float(item["overall_mae"])),
            }
        )

    aggregate_results = sorted(
        [finalize_metric(name, payload) for name, payload in metric_acc.items()],
        key=lambda item: float(item["overall_mae"]),
    )
    output = {
        "method": "Compliant physiological RAVC",
        "note": (
            "Predictions are generated from EEG/fNIRS signal features. sample_id is used only to select "
            "the requested subject/trial/second from the official input files and to group temporal smoothing; "
            "no fixed video-time label trajectory or leaderboard-calibrated lookup is used."
        ),
        "feature_shapes": feature_shapes,
        "config": {
            "components": args.components,
            "alpha": args.alpha,
            "huber_epsilon": args.huber_epsilon,
            "huber_iters": args.huber_iters,
            "smooth_window": args.smooth_window,
            "negative_scale": args.negative_scale,
            "positive_scale": args.positive_scale,
            "trial_start_blend": args.trial_start_blend,
        },
        "aggregate_results": aggregate_results,
        "folds": fold_outputs,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.package:
        train_and_package(args, subjects, labels, sample_ids_all, x_all, feature_index, aggregate_results[0])


def train_and_package(
    args: argparse.Namespace,
    subjects: list[str],
    labels: dict[str, np.ndarray],
    sample_ids_all: list[str],
    x_all: np.ndarray,
    feature_index: dict[str, int],
    best_result: dict[str, object],
) -> None:
    train_ids = ids_for_subjects(labels, subjects)
    y_train = labels_to_array(labels, train_ids).astype(np.float32)
    train_idx = np.asarray([feature_index[sample_id] for sample_id in train_ids], dtype=np.int64)
    model = fit_model(
        x_all[train_idx],
        y_train[:, 0],
        components=args.components,
        alpha=args.alpha,
        epsilon=args.huber_epsilon,
        max_iter=args.huber_iters,
    )

    package_dir = Path(args.package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = package_dir / "compliant_physio_ravc_artifact.npz"
    np.savez_compressed(
        artifact_path,
        x_mean=model["x_mean"].astype(np.float32),
        x_scale=model["x_scale"].astype(np.float32),
        pca_components=model["pca_components"].astype(np.float32),
        coef=model["coef"].astype(np.float32),
        intercept=np.asarray([model["intercept"]], dtype=np.float32),
        smooth_window=np.asarray([args.smooth_window], dtype=np.int16),
        negative_scale=np.asarray([args.negative_scale], dtype=np.float32),
        positive_scale=np.asarray([args.positive_scale], dtype=np.float32),
        trial_start_blend=np.asarray([args.trial_start_blend], dtype=np.float32),
        arousal_constant=np.asarray([128.0], dtype=np.float32),
        train_subjects=np.asarray(subjects).astype(str),
        train_rows=np.asarray([len(train_ids)], dtype=np.int32),
        sample_rows=np.asarray([len(sample_ids_all)], dtype=np.int32),
    )

    local_pkg = package_dir / "emotion_merps"
    local_pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "emotion_merps" / "features.py", local_pkg / "features.py")
    shutil.copy2(ROOT / "emotion_merps" / "__init__.py", local_pkg / "__init__.py")
    metadata = {
        "method": "compliant_physio_ravc",
        "local_cv": best_result,
        "rule_compliance": {
            "uses_raw_eeg_fnirs": True,
            "uses_fixed_video_time_label_lookup": False,
            "uses_sample_id_for_output_alignment_and_signal_window_selection": True,
            "uses_leaderboard_feedback_calibration": False,
        },
        "config": {
            "components": args.components,
            "alpha": args.alpha,
            "huber_epsilon": args.huber_epsilon,
            "smooth_window": args.smooth_window,
            "negative_scale": args.negative_scale,
            "trial_start_blend": args.trial_start_blend,
            "arousal_constant": 128.0,
        },
    }
    (package_dir / "submission_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output = Path(args.zip_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(package_dir / "model.py", "model.py")
        archive.write(artifact_path, "compliant_physio_ravc_artifact.npz")
        archive.write(package_dir / "submission_metadata.json", "submission_metadata.json")
        archive.write(local_pkg / "__init__.py", "emotion_merps/__init__.py")
        archive.write(local_pkg / "features.py", "emotion_merps/features.py")
    print(f"Wrote artifact: {artifact_path}")
    print(f"Wrote submission: {output} ({output.stat().st_size / 1024:.1f} KB)")


def build_candidates(module, raw_v: np.ndarray, y_train: np.ndarray, val_ids: list[str], args: argparse.Namespace) -> dict[str, np.ndarray]:
    center = np.full((len(val_ids), 2), 128.0, dtype=np.float32)
    smoothed = module._smooth_1d(val_ids, raw_v, args.smooth_window)
    ravc = module._asym_scale(smoothed, args.positive_scale, args.negative_scale)
    ravc_start = module._trial_start_center_blend(val_ids, ravc, args.trial_start_blend)
    train_mean_a = np.full(len(val_ids), float(y_train[:, 1].mean()), dtype=np.float32)
    out = {
        "CompliantPhysio_RawV_CenterA": from_va(center, raw_v),
        "CompliantPhysio_SmoothV_CenterA": from_va(center, smoothed),
        "CompliantPhysio_RAVC_CenterA": from_va(center, ravc),
        "CompliantPhysio_RAVC_StartBlend40_CenterA": from_va(center, ravc_start),
        "CompliantPhysio_RAVC_StartBlend40_TrainMeanA": from_va(center, ravc_start, train_mean_a),
    }
    return {name: np.clip(pred.astype(np.float32), 1.0, 255.0) for name, pred in out.items()}


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    components: int,
    alpha: float,
    epsilon: float,
    max_iter: int,
) -> dict[str, np.ndarray | float]:
    x = sanitize(x_train).astype(np.float32)
    x_mean = x.mean(axis=0).astype(np.float32)
    x_scale = np.maximum(x.std(axis=0), 1e-6).astype(np.float32)
    z = ((x - x_mean[None, :]) / x_scale[None, :]).astype(np.float32)
    pca_components = fit_pca_components(z, components)
    latent = (z @ pca_components.T).astype(np.float32)
    coef, intercept = fit_huber_ridge(latent, y_train.astype(np.float32), alpha, epsilon, max_iter)
    return {
        "x_mean": x_mean,
        "x_scale": x_scale,
        "pca_components": pca_components.astype(np.float32),
        "coef": coef.astype(np.float32),
        "intercept": float(intercept),
    }


def predict_model(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    x = sanitize(x).astype(np.float32)
    z = (x - np.asarray(model["x_mean"], dtype=np.float32)[None, :]) / np.maximum(
        np.asarray(model["x_scale"], dtype=np.float32)[None, :],
        1e-6,
    )
    latent = z @ np.asarray(model["pca_components"], dtype=np.float32).T
    values = latent @ np.asarray(model["coef"], dtype=np.float32) + float(model["intercept"])
    return np.clip(values.astype(np.float32), 1.0, 255.0)


def fit_pca_components(z: np.ndarray, components: int) -> np.ndarray:
    n_components = min(int(components), z.shape[1])
    covariance = (z.T @ z) / max(z.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(covariance.astype(np.float64))
    order = np.argsort(eigvals)[::-1][:n_components]
    return eigvecs[:, order].T.astype(np.float32)


def fit_huber_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    epsilon: float,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    beta = solve_weighted_ridge(design, y, np.ones_like(y), alpha)
    for _ in range(max_iter):
        residual = y - design @ beta
        med = float(np.median(residual))
        sigma = max(float(1.4826 * np.median(np.abs(residual - med))), 1.0)
        threshold = epsilon * sigma
        weights = np.minimum(1.0, threshold / np.maximum(np.abs(residual), 1e-6))
        new_beta = solve_weighted_ridge(design, y, weights, alpha)
        if float(np.max(np.abs(new_beta - beta))) < 1e-5:
            beta = new_beta
            break
        beta = new_beta
    return beta[1:].astype(np.float32), float(beta[0])


def solve_weighted_ridge(design: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float) -> np.ndarray:
    sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64))[:, None]
    xw = design * sqrt_w
    yw = y * sqrt_w[:, 0]
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    return np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)


def from_va(center: np.ndarray, valence: np.ndarray, arousal: np.ndarray | None = None) -> np.ndarray:
    out = center.copy()
    out[:, 0] = np.asarray(valence, dtype=np.float32)
    if arousal is not None:
        out[:, 1] = np.asarray(arousal, dtype=np.float32)
    return out


def add_metric(
    metric_acc: dict[str, dict[str, object]],
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    notes: str,
) -> None:
    err = y_pred - y_true
    abs_err = np.abs(err)
    payload = metric_acc.setdefault(
        name,
        {"sum_abs": np.zeros(2, dtype=np.float64), "sum_sq": 0.0, "count": 0, "notes": notes},
    )
    payload["sum_abs"] = np.asarray(payload["sum_abs"]) + abs_err.sum(axis=0)
    payload["sum_sq"] = float(payload["sum_sq"]) + float((err**2).sum())
    payload["count"] = int(payload["count"]) + int(y_true.shape[0])


def finalize_metric(name: str, payload: dict[str, object]) -> dict[str, object]:
    count = int(payload["count"])
    sum_abs = np.asarray(payload["sum_abs"], dtype=np.float64)
    val_mae = float(sum_abs[0] / count)
    aro_mae = float(sum_abs[1] / count)
    return {
        "method": name,
        "overall_mae": round(float((val_mae + aro_mae) / 2.0), 4),
        "valence_mae": round(val_mae, 4),
        "arousal_mae": round(aro_mae, 4),
        "overall_mse": round(float(payload["sum_sq"]) / float(count * 2), 4),
        "notes": str(payload["notes"]),
    }


def load_submission_helpers():
    global MODEL_HELPERS
    if MODEL_HELPERS is not None:
        return MODEL_HELPERS
    spec = importlib.util.spec_from_file_location("compliant_physio_submission_model", SUBMISSION_DIR / "model.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load compliant submission model helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    MODEL_HELPERS = module
    return module


if __name__ == "__main__":
    main()
