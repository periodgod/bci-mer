from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from emotion_merps.features import build_prediction_features, read_sample_rows


ARTIFACT_NAME = "compliant_physio_ravc_artifact.npz"
SAMPLE_RE = re.compile(r"^(?P<subject>.+)_V(?P<video>\d+)_T(?P<timestamp>\d+)$")


def predict(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    rows = read_sample_rows(input_dir)
    sample_ids, eeg, fnirs = build_prediction_features(
        input_dir,
        rows,
        fnirs_types=(0, 1, 2, 3, 4, 5),
        baseline_correction=True,
    )
    artifact = _PhysioRAVCArtifact(Path(__file__).resolve().parent / ARTIFACT_NAME)
    x = _build_signal_features(eeg, fnirs, sample_ids)
    valence = artifact.predict_valence(x, sample_ids)
    arousal = np.full_like(valence, artifact.arousal_constant, dtype=np.float32)
    prediction = np.stack([valence, arousal], axis=1)
    _write_predictions(sample_ids, prediction, output_dir)


class _PhysioRAVCArtifact:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing physiological model artifact: {path}")
        data = np.load(path, allow_pickle=False)
        self.x_mean = data["x_mean"].astype(np.float32)
        self.x_scale = data["x_scale"].astype(np.float32)
        self.pca_components = data["pca_components"].astype(np.float32)
        self.coef = data["coef"].astype(np.float32)
        self.intercept = float(data["intercept"][0])
        self.smooth_window = int(data["smooth_window"][0])
        self.negative_scale = float(data["negative_scale"][0])
        self.positive_scale = float(data["positive_scale"][0])
        self.trial_start_blend = float(data["trial_start_blend"][0])
        self.arousal_constant = float(data["arousal_constant"][0])

    def predict_valence(self, x: np.ndarray, sample_ids: list[str]) -> np.ndarray:
        x = _sanitize(x)
        z = (x - self.x_mean[None, :]) / np.maximum(self.x_scale[None, :], 1e-6)
        latent = z @ self.pca_components.T
        values = latent @ self.coef + self.intercept
        values = np.clip(values.astype(np.float32), 1.0, 255.0)
        if self.smooth_window > 1:
            values = _smooth_1d(sample_ids, values, self.smooth_window)
        values = _asym_scale(values, self.positive_scale, self.negative_scale)
        if self.trial_start_blend > 0.0:
            values = _trial_start_center_blend(sample_ids, values, self.trial_start_blend)
        return np.clip(values.astype(np.float32), 1.0, 255.0)


def _build_signal_features(eeg: np.ndarray, fnirs: np.ndarray, sample_ids: list[str]) -> np.ndarray:
    eeg_core = _eeg_core_features(eeg)
    fnirs_core = _fnirs_core_features(fnirs)
    eeg_lag = _lagged_features(sample_ids, eeg_core, lags=(0, 1, 2, 4), include_delta=True)
    fnirs_slow = np.concatenate(
        [
            fnirs_core,
            _rolling_past_features(sample_ids, fnirs_core, windows=(3, 5, 9)),
            _lagged_features(sample_ids, fnirs_core, lags=(1, 3, 5), include_delta=True),
        ],
        axis=1,
    )
    return _sanitize(np.concatenate([eeg_lag, fnirs_slow], axis=1))


def _eeg_core_features(eeg: np.ndarray) -> np.ndarray:
    mean_band = eeg.mean(axis=1)
    std_band = eeg.std(axis=1)
    half = eeg.shape[1] // 2
    spatial_diff = eeg[:, :half].mean(axis=1) - eeg[:, half:].mean(axis=1)
    beta_alpha = mean_band[:, 3:4] - mean_band[:, 2:3]
    gamma_alpha = mean_band[:, 4:5] - mean_band[:, 2:3]
    return _sanitize(np.concatenate([mean_band, std_band, spatial_diff, beta_alpha, gamma_alpha], axis=1))


def _fnirs_core_features(fnirs: np.ndarray) -> np.ndarray:
    mean_feat = fnirs.mean(axis=1)
    std_feat = fnirs.std(axis=1)
    half = fnirs.shape[1] // 2
    spatial_diff = fnirs[:, :half].mean(axis=1) - fnirs[:, half:].mean(axis=1)
    hbo_hbr = mean_feat[:, 0:1] - mean_feat[:, 1:2]
    hbt_slope = mean_feat[:, 8:9] if mean_feat.shape[1] > 8 else np.zeros_like(hbo_hbr)
    return _sanitize(np.concatenate([mean_feat, std_feat, spatial_diff, hbo_hbr, hbt_slope], axis=1))


def _lagged_features(sample_ids: list[str], values: np.ndarray, lags: tuple[int, ...], include_delta: bool) -> np.ndarray:
    groups = _group_indices(sample_ids)
    lagged = np.zeros((len(sample_ids), values.shape[1] * len(lags)), dtype=np.float32)
    delta_lags = [lag for lag in lags if lag > 0]
    delta_out = np.zeros((len(sample_ids), values.shape[1] * len(delta_lags)), dtype=np.float32)
    for items in groups.values():
        items = sorted(items)
        indices = [index for _, index in items]
        index_array = np.asarray(indices, dtype=np.int64)
        seq = values[indices]
        for lag_pos, lag in enumerate(lags):
            shifted = _shift_past(seq, lag)
            lagged[index_array, lag_pos * values.shape[1] : (lag_pos + 1) * values.shape[1]] = shifted
            if include_delta and lag > 0:
                delta_pos = delta_lags.index(lag)
                start = delta_pos * values.shape[1]
                stop = start + values.shape[1]
                delta_out[index_array, start:stop] = seq - shifted
    if include_delta and delta_lags:
        return _sanitize(np.concatenate([lagged, delta_out], axis=1))
    return _sanitize(lagged)


def _rolling_past_features(sample_ids: list[str], values: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    out = np.zeros((len(sample_ids), values.shape[1] * len(windows)), dtype=np.float32)
    for items in _group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        seq = values[indices]
        cumsum = np.concatenate([np.zeros((1, values.shape[1]), dtype=np.float32), np.cumsum(seq, axis=0)], axis=0)
        for win_pos, window in enumerate(windows):
            rolled = np.zeros_like(seq)
            for local_index in range(seq.shape[0]):
                start = max(0, local_index - window + 1)
                stop = local_index + 1
                rolled[local_index] = (cumsum[stop] - cumsum[start]) / float(stop - start)
            out[np.asarray(indices), win_pos * values.shape[1] : (win_pos + 1) * values.shape[1]] = rolled
    return _sanitize(out)


def _smooth_1d(sample_ids: list[str], values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    radius = window // 2
    for items in _group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        seq = values[indices]
        smoothed = np.zeros_like(seq)
        for local_index in range(len(seq)):
            start = max(0, local_index - radius)
            stop = min(len(seq), local_index + radius + 1)
            smoothed[local_index] = float(np.mean(seq[start:stop]))
        out[np.asarray(indices, dtype=np.int64)] = smoothed
    return out


def _trial_start_center_blend(sample_ids: list[str], values: np.ndarray, amount: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = values.copy()
    for items in _group_indices(sample_ids).values():
        items = sorted(items)
        indices = [index for _, index in items]
        for local_index, global_index in enumerate(indices[:10]):
            weight = amount * (1.0 - local_index / 10.0)
            out[global_index] = (1.0 - weight) * values[global_index] + weight * 128.0
    return out


def _asym_scale(values: np.ndarray, pos: float, neg: float) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float32) - 128.0
    scale = np.where(residual >= 0.0, pos, neg)
    return 128.0 + scale * residual


def _group_indices(sample_ids: list[str]) -> dict[tuple[str, int], list[tuple[int, int]]]:
    groups: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        parsed = _parse_sample_id(sample_id)
        if parsed is None:
            groups[(sample_id, 0)].append((0, index))
            continue
        subject, video, timestamp = parsed
        groups[(subject, video)].append((timestamp, index))
    return groups


def _shift_past(seq: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        return seq.copy()
    shifted = np.empty_like(seq)
    shifted[:lag] = seq[:1]
    shifted[lag:] = seq[:-lag]
    return shifted


def _parse_sample_id(sample_id: str) -> tuple[str, int, int] | None:
    match = SAMPLE_RE.match(sample_id)
    if match is None:
        return None
    return match.group("subject"), int(match.group("video")), int(match.group("timestamp"))


def _sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _write_predictions(sample_ids: list[str], prediction: np.ndarray, output_dir: str | Path) -> None:
    output_path = Path(output_dir) / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction = np.rint(np.clip(prediction, 1.0, 255.0)).astype(np.int16)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "valence", "arousal"])
        writer.writeheader()
        for sample_id, values in zip(sample_ids, prediction):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "valence": int(values[0]),
                    "arousal": int(values[1]),
                }
            )
