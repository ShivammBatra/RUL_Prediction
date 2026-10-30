"""Inference helpers for the interactive, saved-model demonstration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "Setting_1", "Setting_2",
    "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_6", "Sensor_7", "Sensor_8", "Sensor_9",
    "Sensor_11", "Sensor_12", "Sensor_13", "Sensor_14", "Sensor_15", "Sensor_17",
    "Sensor_20", "Sensor_21",
]
WINDOW = 30
MC_PASSES = 100


def load_runtime_artifacts(model_path: str | Path, artifact_path: str | Path):
    """Load the saved BiLSTM and reliability-gate artifact on demand."""
    import joblib
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    artifact = joblib.load(artifact_path)

    if tuple(model.input_shape[1:]) != (WINDOW, len(FEATURES)):
        raise ValueError(
            f"Saved model expects {model.input_shape[1:]}, but the live pipeline supplies "
            f"({WINDOW}, {len(FEATURES)})."
        )
    if not np.isfinite(float(artifact.get("mc_threshold", np.nan))):
        raise ValueError("The reliability artifact has no valid MC-Dropout threshold.")
    if list(artifact.get("features", [])) != FEATURES:
        raise ValueError("The saved consistency model feature order does not match the BiLSTM input order.")
    return model, artifact


def extract_window(engine_data: pd.DataFrame, end_cycle: int) -> pd.DataFrame:
    """Return exactly the same rolling window used by the reliability notebook."""
    window = engine_data[
        (engine_data["Cycle"] > end_cycle - WINDOW)
        & (engine_data["Cycle"] <= end_cycle)
    ].sort_values("Cycle").copy()
    if len(window) != WINDOW:
        raise ValueError(f"Cycle {end_cycle} has {len(window)} observations; exactly {WINDOW} are required.")
    return window.reset_index(drop=True)


def inject_window_fault(
    full_engine: pd.DataFrame,
    window: pd.DataFrame,
    fault_type: str,
    sensor: str,
    severity_per_cycle: float = 0.01,
) -> pd.DataFrame:
    """Apply the reliability notebook's fault definitions inside the selected window."""
    changed = window.copy()
    if fault_type == "none":
        return changed
    if sensor not in changed.columns:
        raise ValueError(f"{sensor} is not present in the selected sensor history.")

    onset_cycle = int(changed["Cycle"].min())
    if fault_type in {"drift_up", "drift_down"}:
        direction = 1.0 if fault_type == "drift_up" else -1.0
        steps = changed["Cycle"].to_numpy(dtype=float) - onset_cycle + 1
        changed[sensor] = changed[sensor].to_numpy(dtype=float) + direction * severity_per_cycle * steps
    elif fault_type == "stuck_at":
        prior = full_engine.loc[full_engine["Cycle"] == onset_cycle - 1, sensor]
        frozen_value = float(prior.iloc[-1]) if not prior.empty else float(changed.iloc[0][sensor])
        changed[sensor] = frozen_value
    else:
        raise ValueError(f"Unsupported fault scenario: {fault_type}")
    return changed


def _consistency_check(window: pd.DataFrame, artifact: dict) -> tuple[bool, str | None, list[str]]:
    recent = window.tail(int(artifact["persistence_window"]))
    persistence_window = int(artifact["persistence_window"])
    persistence_min_count = int(artifact["persistence_min_count"])
    trips: list[tuple[str, float]] = []

    for sensor in artifact["monitored_sensors"]:
        predictors = [feature for feature in artifact["features"] if feature != sensor]
        predicted_sensor = artifact["models"][sensor].predict(recent[predictors])
        residual = np.abs(recent[sensor].to_numpy() - predicted_sensor) / artifact["sigmas"][sensor]
        flags = (residual > artifact["thresholds"][sensor]).astype(int)
        persistent = pd.Series(flags).rolling(persistence_window, min_periods=1).sum().to_numpy() >= persistence_min_count
        if persistent.any():
            trips.append((sensor, float(np.max(residual))))

    trips.sort(key=lambda item: item[1], reverse=True)
    return not trips, trips[0][0] if trips else None, [sensor for sensor, _ in trips]


def run_prediction(window: pd.DataFrame, model, artifact: dict) -> dict:
    """Run 100 stochastic forward passes and the saved reliability gate."""
    missing = [feature for feature in FEATURES if feature not in window.columns]
    if missing:
        raise ValueError(f"Input is missing model features: {', '.join(missing)}")
    values = window[FEATURES].to_numpy(dtype=np.float32)
    if values.shape != (WINDOW, len(FEATURES)):
        raise ValueError(f"Expected input shape ({WINDOW}, {len(FEATURES)}); received {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("The selected input window contains missing or non-finite feature values.")

    # Each batch row is a stochastic dropout pass, keeping the saved notebook's
    # 100-pass uncertainty estimate while avoiding 100 separate model calls.
    batch = np.repeat(values[np.newaxis, :, :], MC_PASSES, axis=0)
    predictions = np.asarray(model(batch, training=True)).reshape(-1)
    predicted_rul = max(0.0, float(np.mean(predictions)))
    uncertainty = float(np.std(predictions))
    consistent, worst_sensor, tripped_sensors = _consistency_check(window, artifact)
    uncertainty_ok = uncertainty <= float(artifact["mc_threshold"])
    decision = "ACCEPT" if uncertainty_ok and consistent else "ABSTAIN"

    reasons = []
    if not uncertainty_ok:
        reasons.append("MC-Dropout uncertainty exceeds the validation-calibrated threshold")
    if not consistent:
        reasons.append("persistent sensor-consistency violation" + (f" ({worst_sensor})" if worst_sensor else ""))

    return {
        "predicted_rul": predicted_rul,
        "uncertainty": uncertainty,
        "mc_threshold": float(artifact["mc_threshold"]),
        "decision": decision,
        "uncertainty_ok": uncertainty_ok,
        "consistent": consistent,
        "worst_sensor": worst_sensor,
        "tripped_sensors": tripped_sensors,
        "reasons": reasons,
    }
