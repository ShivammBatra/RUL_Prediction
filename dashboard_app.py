from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from live_inference import FEATURES as LIVE_FEATURES
from live_inference import WINDOW as LIVE_WINDOW
from live_inference import inject_window_fault, load_runtime_artifacts, run_prediction

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATASET = ROOT / "Dataset"
MISSING_TEST = DATASET / "Missing_Test"

st.set_page_config(
    page_title="Turbofan RUL | Capstone Explorer",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "navy": "#10243A",
    "blue": "#3678A8",
    "teal": "#19A895",
    "amber": "#F2B84B",
    "red": "#D95D54",
    "muted": "#64748B",
}


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_prediction_data() -> pd.DataFrame:
    frames = []
    for path in sorted(RESULTS.glob("*__*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if {"Engine_ID", "Actual_RUL", "Predicted_RUL", "Uncertainty"}.issubset(df.columns):
            method, scenario = path.stem.split("__", 1)
            df["Method"] = method
            df["Scenario"] = scenario
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def available_file(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        st.warning(f"Could not find {label}: `{path.relative_to(ROOT)}`")
        return None
    return read_csv(str(path))


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def format_metric(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def show_table(frame: pd.DataFrame, max_rows: int = 100) -> None:
    """Render a small HTML table without Streamlit's Arrow/PyArrow path."""
    if frame.empty:
        st.info("No rows to display.")
        return
    visible = frame.head(max_rows)
    st.markdown(
        '<div class="table-wrap">' + visible.to_html(index=False, escape=True, border=0) + '</div>',
        unsafe_allow_html=True,
    )
    if len(frame) > max_rows:
        st.caption(f"Showing the first {max_rows:,} of {len(frame):,} rows. Download the CSV for the full result.")


def section_intro(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def apply_chart_style(fig, height: int = 420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color=COLORS["navy"]),
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E8EEF3", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E8EEF3", zeroline=False)
    return fig


def load_baseline_metrics() -> pd.DataFrame:
    path = RESULTS / "baseline_model_metrics.csv"
    baseline = available_file(path, "baseline model metrics")
    return baseline if baseline is not None else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_live_runtime():
    """Load the large saved inference artifacts once for this app process."""
    return load_runtime_artifacts(
        ROOT / "bilstm_reliability_gate.keras",
        ROOT / "consistency_models.pkl",
    )


@st.cache_data(show_spinner=False)
def load_official_test_rul() -> list[float]:
    labels = pd.read_csv(ROOT / "RUL_FD001.txt", sep=r"\s+", header=None).iloc[:, 0]
    return labels.astype(float).tolist()


def overview_page():
    st.title("Remaining Useful Life under Incomplete Sensor Data")
    st.markdown(
        "An interactive guide to the capstone experiments on **turbofan RUL prediction, "
        "uncertainty, and sensor reliability**. Use the sections in the sidebar to explore "
        "the evidence behind the project."
    )
    st.divider()

    train_path = DATASET / "processed_train.csv"
    test_path = DATASET / "processed_test.csv"
    train = available_file(train_path, "processed training data")
    test = available_file(test_path, "processed test data")
    baseline = load_baseline_metrics()
    preds = load_prediction_data()

    a, b, c, d = st.columns(4)
    a.metric("Training records", f"{len(train):,}" if train is not None else "—")
    b.metric("Test records", f"{len(test):,}" if test is not None else "—")
    c.metric("Baseline models", str(baseline["Model"].nunique()) if not baseline.empty else "6")
    d.metric("UQ prediction sets", str(len(preds[["Method", "Scenario"]].drop_duplicates())) if not preds.empty else "12")

    st.subheader("How to read this project")
    steps = st.columns(4)
    cards = [
        ("01 · Data", "NASA C-MAPSS FD001 turbofan runs are represented as cycle-by-cycle sensor measurements."),
        ("02 · Missingness", "Sensor values are synthetically masked at 5%, 10%, 20%, and 30% to study robustness."),
        ("03 · Prediction", "Six model families estimate remaining useful life from the sensor history."),
        ("04 · Reliability", "Uncertainty methods and consistency checks help show when an estimate may need caution."),
    ]
    for col, (heading, body) in zip(steps, cards):
        with col:
            st.markdown(f"#### {heading}")
            st.write(body)

    st.subheader("A few headline findings")
    findings = st.columns(3)
    findings[0].info("**Best capped UQ RMSE:** Deep Ensembles, 13.26 on clean data and 14.26 at 30% missingness.")
    findings[1].info("**Baseline winner changes with missingness:** different architectures lead at different corruption levels.")
    findings[2].info("**Fault experiment:** 89.4% of injected faults were detected; detected faults were correctly localized 95.6% of the time.")

    st.caption(
        "These are results from controlled experiments on FD001 and synthetic corruption. "
        "They describe this evaluation setup, not a deployed aircraft monitoring system."
    )


def dataset_page():
    section_intro("Dataset & preprocessing", "Explore an engine's sensor history and compare the available missing-data variants.")
    data = available_file(DATASET / "processed_test.csv", "processed test data")
    if data is None:
        return
    engine_ids = sorted(data["Engine_ID"].unique().tolist())
    engine_id = st.selectbox("Test engine", engine_ids, index=0)
    engine = data[data["Engine_ID"] == engine_id].sort_values("Cycle")
    sensor_cols = [c for c in engine.columns if c.startswith("Sensor_")]
    useful_sensors = [c for c in sensor_cols if engine[c].nunique(dropna=False) > 1]
    selected = st.multiselect("Sensors to display", sensor_cols, default=useful_sensors[:3])

    left, right = st.columns([1.2, 1])
    with left:
        if selected:
            long = engine.melt(id_vars="Cycle", value_vars=selected, var_name="Sensor", value_name="Normalized value")
            fig = px.line(long, x="Cycle", y="Normalized value", color="Sensor", title=f"Sensor history · Engine {engine_id}")
            st.plotly_chart(apply_chart_style(fig, 430), use_container_width=True)
        else:
            st.info("Choose at least one sensor to display its history.")
    with right:
        st.markdown("#### Dataset at a glance")
        c1, c2 = st.columns(2)
        c1.metric("Cycles for engine", f"{len(engine):,}")
        c2.metric("Sensor channels", str(len(sensor_cols)))
        st.write("Each row is one engine cycle. Sensor values in the processed files are normalized.")
        st.write("The prediction feature set uses 15 retained sensors and 2 operating settings; missing-data neural models also receive observation masks.")
        st.write("The test set contains sensor history; the official RUL labels are stored separately in `RUL_FD001.txt`.")
        with st.expander("Show sample records"):
            show_table(engine[["Engine_ID", "Cycle", *selected]].head(15) if selected else engine.head(15), max_rows=15)

    st.divider()
    st.subheader("Missing-data variants")
    st.write("The project includes preprocessed test data with MCAR missingness at four rates.")
    missingness = st.select_slider("Missingness level", options=[5, 10, 20, 30], value=10, format_func=lambda x: f"{x}%")
    missing_path = MISSING_TEST / f"processed_test_missing_{missingness}.csv"
    missing = available_file(missing_path, f"{missingness}% missing test data")
    if missing is not None:
        missing_engine = missing[missing["Engine_ID"] == engine_id].sort_values("Cycle")
        show_sensors = selected or sensor_cols[:3]
        if not missing_engine.empty:
            sensor = st.selectbox("Sensor to compare", show_sensors, key="missing_sensor")
            clean_series = engine[["Cycle", sensor]].copy()
            missing_series = missing_engine[["Cycle", sensor]].copy()
            missing_values = missing_series[sensor].isna() | missing_series[sensor].eq(0)
            masked_cycles = clean_series.loc[(clean_series[sensor].abs() > 1e-12) & missing_values.to_numpy(), "Cycle"]
            compare = pd.concat(
                [clean_series.assign(Condition="Clean"), missing_series.assign(Condition=f"{missingness}% missing")],
                ignore_index=True,
            )
            fig = px.line(compare, x="Cycle", y=sensor, color="Condition", title=f"{sensor} · clean versus {missingness}% missing")
            if not masked_cycles.empty:
                masked = missing_series[missing_series["Cycle"].isin(masked_cycles)]
                fig.add_trace(go.Scatter(
                    x=masked["Cycle"], y=masked[sensor], mode="markers", name="Likely masked readings",
                    marker=dict(color=COLORS["red"], symbol="x", size=10),
                ))
            st.plotly_chart(apply_chart_style(fig, 350), use_container_width=True)
            st.caption("Red x markers identify unavailable readings (stored as NaN or zero) where the corresponding clean value is nonzero. This distinguishes injected masks from naturally zero-valued normalized readings.")


def baseline_page():
    section_intro("Baseline model comparison", "Compare six point-prediction models across clean data and four missingness levels.")
    baseline = load_baseline_metrics()
    if baseline.empty:
        st.warning("The baseline comparison file is missing. See `results/baseline_model_metrics.csv`.")
        return
    levels = sorted(baseline["Missingness_pct"].unique().tolist())
    level = st.select_slider("Sensor missingness", options=levels, value=0, format_func=lambda x: "Clean" if x == 0 else f"{x}%")
    metric = st.selectbox("Metric", ["RMSE", "MAE", "R2", "NASA_PHM08_score"], format_func=lambda x: "R²" if x == "R2" else ("NASA/PHM08 score" if x == "NASA_PHM08_score" else x))
    subset = baseline[baseline["Missingness_pct"] == level].sort_values(metric, ascending=(metric != "R2"))
    winner = subset.iloc[0]
    if metric == "R2":
        winner = subset.loc[subset[metric].idxmax()]
    a, b, c = st.columns(3)
    a.metric("Best model at this setting", winner["Model"])
    b.metric("Best " + ("R²" if metric == "R2" else metric), format_metric(float(winner[metric]), 3))
    c.metric("Models compared", str(subset["Model"].nunique()))

    fig = px.bar(
        subset.sort_values(metric, ascending=(metric != "R2")),
        x=metric, y="Model", color="Model", orientation="h", text_auto=".2f",
        title=f"{metric.replace('_', ' ')} · {'clean' if level == 0 else f'{level}% missing'}",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(apply_chart_style(fig), use_container_width=True)

    st.subheader("How performance changes as data goes missing")
    trend = baseline[baseline["Model"].isin(st.multiselect("Models to track", baseline["Model"].unique(), default=baseline["Model"].unique().tolist()))]
    trend_fig = px.line(trend, x="Missingness_pct", y=metric, color="Model", markers=True, title=f"{metric.replace('_', ' ')} across missingness levels")
    st.plotly_chart(apply_chart_style(trend_fig, 390), use_container_width=True)
    with st.expander("Full baseline results table"):
        show_table(baseline.sort_values(["Missingness_pct", metric]), max_rows=30)


def uncertainty_page():
    section_intro("Uncertainty & RUL results", "Explore saved test predictions, uncertainty estimates, and selective risk as uncertain cases are withheld.")
    preds = load_prediction_data()
    if preds.empty:
        st.warning("No UQ prediction CSVs were found in the `results` folder.")
        return
    methods = sorted(preds["Method"].unique())
    method_labels = {"mc_dropout": "MC Dropout", "deep_ensembles": "Deep Ensembles", "conformal": "Conformal Prediction"}
    scenarios = sorted(preds["Scenario"].unique())
    a, b = st.columns(2)
    method = a.selectbox("Uncertainty method", methods, format_func=lambda x: method_labels.get(x, x.replace("_", " ").title()))
    scenario = b.selectbox("Evaluation condition", scenarios, format_func=lambda x: x.replace("_", " + ").title())
    df = preds[(preds["Method"] == method) & (preds["Scenario"] == scenario)].copy()
    if df.empty:
        st.info("No records match this selection.")
        return
    error = df["Predicted_RUL"] - df["Actual_RUL"]
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    interval = (df["Actual_RUL"] >= df["Predicted_RUL"] - df["Uncertainty"]) & (df["Actual_RUL"] <= df["Predicted_RUL"] + df["Uncertainty"])
    coverage = float(interval.mean())
    summary = available_file(RESULTS / "summary_table.csv", "uncertainty summary metrics")
    score_value = None
    if summary is not None:
        score_row = summary[(summary["Method"] == method) & (summary["Scenario"] == scenario)]
        if not score_row.empty:
            score_value = float(score_row.iloc[0]["Score"])
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("RMSE", format_metric(rmse))
    k2.metric("MAE", format_metric(mae))
    k3.metric("Empirical band coverage", f"{coverage:.0%}", help="Share of engines whose actual RUL falls within Predicted_RUL ± saved Uncertainty. For MC Dropout and Deep Ensembles this is descriptive, not a calibrated coverage guarantee.")
    k4.metric("NASA/PHM08 score", format_metric(score_value) if score_value is not None else "—", help="Saved score at 100% coverage; lower is better. It is summed over all 100 test engines.")
    k5.metric("Test engines", str(df["Engine_ID"].nunique()))

    left, right = st.columns(2)
    with left:
        scatter = px.scatter(df, x="Actual_RUL", y="Predicted_RUL", color="Uncertainty", hover_data=["Engine_ID"], color_continuous_scale="Viridis", title="Predicted versus actual RUL")
        upper = max(float(df["Actual_RUL"].max()), float(df["Predicted_RUL"].max()))
        scatter.add_trace(go.Scatter(x=[0, upper], y=[0, upper], mode="lines", name="Ideal prediction", line=dict(color=COLORS["red"], dash="dash")))
        st.plotly_chart(apply_chart_style(scatter), use_container_width=True)
    with right:
        by_engine = df.sort_values("Actual_RUL").reset_index(drop=True)
        x = np.arange(1, len(by_engine) + 1)
        lower_band = (by_engine["Predicted_RUL"] - by_engine["Uncertainty"]).clip(lower=0)
        upper_band = by_engine["Predicted_RUL"] + by_engine["Uncertainty"]
        band = go.Figure()
        band.add_trace(go.Scatter(x=x, y=upper_band, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        band.add_trace(go.Scatter(x=x, y=lower_band, mode="lines", fill="tonexty", fillcolor="rgba(54,120,168,0.18)", name="Predicted ± uncertainty", hoverinfo="skip"))
        band.add_trace(go.Scatter(x=x, y=by_engine["Predicted_RUL"], mode="lines+markers", name="Predicted RUL", line=dict(color=COLORS["blue"]), marker=dict(size=5)))
        band.add_trace(go.Scatter(x=x, y=by_engine["Actual_RUL"], mode="markers", name="Actual RUL", marker=dict(color=COLORS["navy"], size=6)))
        band.update_layout(title="Prediction and uncertainty band · engines sorted by actual RUL", xaxis_title="Test engine rank", yaxis_title="RUL (cycles)")
        st.plotly_chart(apply_chart_style(band), use_container_width=True)

    st.subheader("Risk–coverage view")
    st.write("Engines are ranked from lower to higher uncertainty. The curve shows RMSE among the retained fraction.")
    ordered = df.sort_values("Uncertainty").reset_index(drop=True)
    coverages = np.linspace(1.0, 0.5, 26)
    risk = []
    for cov in coverages:
        count = max(2, int(np.ceil(len(ordered) * cov)))
        accepted = ordered.iloc[:count]
        risk.append(float(np.sqrt(np.mean((accepted["Actual_RUL"] - accepted["Predicted_RUL"]) ** 2))))
    risk_fig = px.line(x=coverages * 100, y=risk, markers=True, labels={"x": "Retained predictions (%)", "y": "RMSE on retained predictions"}, title="Selective prediction: error versus retained coverage")
    st.plotly_chart(apply_chart_style(risk_fig, 350), use_container_width=True)
    st.caption("This curve ranks the test predictions by their saved uncertainty to visualize the relationship between uncertainty and error. It is an exploratory view of the held-out results.")

    with st.expander("Inspect or download the selected engine predictions"):
        show_table(df.sort_values("Engine_ID"), max_rows=100)
        st.download_button("Download selected predictions as CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{method}__{scenario}.csv", mime="text/csv")


def live_prediction_page():
    section_intro(
        "Live RUL prediction demo",
        "Run the saved BiLSTM on a selected 30-cycle test window, then apply the trained uncertainty and sensor-consistency gate.",
    )
    st.info(
        "This page performs fresh inference when you click **Run prediction**. "
        "The model is loaded from the saved project artifact; it is not retrained. "
        "Faults are synthetic demonstrations applied to the selected window."
    )

    test = available_file(DATASET / "processed_test.csv", "processed FD001 test data")
    model_path = ROOT / "bilstm_reliability_gate.keras"
    artifact_path = ROOT / "consistency_models.pkl"
    label_path = ROOT / "RUL_FD001.txt"
    missing_paths = [path.name for path in [model_path, artifact_path, label_path] if not path.exists()]
    if test is None:
        return
    if missing_paths:
        st.error("Live inference files are missing: " + ", ".join(missing_paths))
        return

    absent_features = [feature for feature in LIVE_FEATURES if feature not in test.columns]
    if absent_features:
        st.error("Processed test data is missing model features: " + ", ".join(absent_features))
        return

    engine_ids = sorted(pd.to_numeric(test["Engine_ID"], errors="coerce").dropna().astype(int).unique().tolist())
    if not engine_ids:
        st.error("No valid engine IDs were found in processed_test.csv.")
        return

    left, right = st.columns(2)
    with left:
        demo_engine = 20 if 20 in engine_ids else engine_ids[0]
        engine_id = st.selectbox("Test engine", engine_ids, index=engine_ids.index(demo_engine), key="live_engine")
    engine = test[pd.to_numeric(test["Engine_ID"], errors="coerce") == engine_id].sort_values("Cycle").copy()
    max_cycle = int(engine["Cycle"].max())
    if max_cycle < LIVE_WINDOW:
        st.error(f"Engine {engine_id} has fewer than {LIVE_WINDOW} cycles, so it cannot form a model input window.")
        return
    with right:
        end_cycle = st.slider(
            "Prediction cycle",
            min_value=LIVE_WINDOW,
            max_value=max_cycle,
            value=max_cycle,
            help="The model receives the 30 cycles ending at this cycle.",
            key=f"live_cycle_{engine_id}",
        )

    scenario_col, sensor_col = st.columns(2)
    with scenario_col:
        fault_type = st.selectbox(
            "Demonstration condition",
            ["none", "drift_up", "drift_down", "stuck_at"],
            format_func=lambda value: {
                "none": "Clean sensor history",
                "drift_up": "Inject upward drift",
                "drift_down": "Inject downward drift",
                "stuck_at": "Inject stuck-at reading",
            }[value],
            key="live_fault_type",
        )
    monitored_sensors = [
        "Sensor_9", "Sensor_14", "Sensor_11", "Sensor_12", "Sensor_8", "Sensor_13",
        "Sensor_4", "Sensor_7", "Sensor_15", "Sensor_20", "Sensor_21", "Sensor_17",
        "Sensor_2", "Sensor_3",
    ]
    with sensor_col:
        sensor = st.selectbox(
            "Sensor to use",
            monitored_sensors,
            index=monitored_sensors.index("Sensor_2"),
            disabled=fault_type == "none",
            key="live_fault_sensor",
        )

    st.caption(
        f"Input: cycles {max(1, end_cycle - LIVE_WINDOW + 1)}–{end_cycle} · {LIVE_WINDOW} rows · "
        f"{len(LIVE_FEATURES)} normalized model features. Drift severity is ±0.01 normalized units per cycle, "
        "matching the severe-fault setting in the reliability-gate experiment."
    )

    ready = st.session_state.get("live_runtime_ready", False)
    if st.button("Prepare model for the demo", disabled=ready, key="prepare_live_model"):
        try:
            with st.spinner("Loading TensorFlow, the BiLSTM, and the sensor-consistency models…"):
                load_live_runtime()
            st.session_state.live_runtime_ready = True
            st.success("Model and reliability artifacts are loaded and ready for live predictions.")
            st.rerun()
        except Exception as exc:
            st.error(f"The saved inference artifacts could not be loaded: {exc}")
            return
    if ready:
        st.caption("Live model is loaded and ready.")

    if "live_runs" not in st.session_state:
        st.session_state.live_runs = []
    run_clicked = st.button("Run prediction", type="primary", key="run_live_prediction")
    if run_clicked:
        try:
            input_window = engine[(engine["Cycle"] > end_cycle - LIVE_WINDOW) & (engine["Cycle"] <= end_cycle)].copy()
            if len(input_window) != LIVE_WINDOW:
                st.error(f"Could not form a complete {LIVE_WINDOW}-cycle input window.")
                return
            input_window = inject_window_fault(engine, input_window, fault_type, sensor)
            with st.spinner("Loading the saved model and running 100 MC-Dropout passes…"):
                model, artifact = load_live_runtime()
                st.session_state.live_runtime_ready = True
                output = run_prediction(input_window, model, artifact)
            actual_labels = load_official_test_rul()
            if int(engine_id) > len(actual_labels):
                raise ValueError(f"No official RUL label is available for test engine {engine_id}.")
            final_observed_cycle = int(engine["Cycle"].max())
            actual_rul = float(actual_labels[int(engine_id) - 1] + final_observed_cycle - int(end_cycle))
            condition_label = {
                "none": "Clean",
                "drift_up": f"Drift up · {sensor}",
                "drift_down": f"Drift down · {sensor}",
                "stuck_at": f"Stuck-at · {sensor}",
            }[fault_type]
            run = {
                "engine_id": int(engine_id),
                "end_cycle": int(end_cycle),
                "fault_type": fault_type,
                "sensor": sensor,
                "condition": condition_label,
                "actual_rul": actual_rul,
                "window": input_window,
                "clean_window": engine[(engine["Cycle"] > end_cycle - LIVE_WINDOW) & (engine["Cycle"] <= end_cycle)].copy(),
                **output,
            }
            st.session_state.live_last_run = run
            history = [item for item in st.session_state.live_runs if not (
                item["engine_id"] == run["engine_id"]
                and item["end_cycle"] == run["end_cycle"]
                and item["condition"] == run["condition"]
            )]
            st.session_state.live_runs = (history + [run])[-8:]
        except Exception as exc:
            st.error(f"Live prediction could not be completed: {exc}")
            st.caption("Check the dashboard setup instructions and confirm the saved model/artifact files are present.")

    last_run = st.session_state.get("live_last_run")
    if last_run:
        st.divider()
        st.subheader(f"Fresh prediction · Engine {last_run['engine_id']} at cycle {last_run['end_cycle']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted RUL", f"{last_run['predicted_rul']:.1f} cycles")
        m2.metric("MC-Dropout uncertainty", f"±{last_run['uncertainty']:.2f} cycles")
        m3.metric("Uncertainty threshold", f"{last_run['mc_threshold']:.2f} cycles")
        m4.metric("Reliability decision", last_run["decision"])
        if last_run["decision"] == "ACCEPT":
            st.success("The estimate passes both checks: uncertainty is below its validation threshold and no persistent monitored-sensor inconsistency was detected.")
        else:
            st.warning("The gate recommends abstaining from this estimate: " + "; ".join(last_run["reasons"]) + ".")

        with st.expander("Reveal official RUL and compare", expanded=False):
            absolute_error = abs(last_run["predicted_rul"] - last_run["actual_rul"])
            a, b, c = st.columns(3)
            a.metric("Official RUL at selected cycle", f"{last_run['actual_rul']:.1f} cycles")
            b.metric("Absolute prediction error", f"{absolute_error:.1f} cycles")
            c.metric("Gate outcome", last_run["decision"])
            st.caption("The official label is revealed only after inference. It was not included in the model input.")

        plot_sensor = last_run["sensor"] if last_run["fault_type"] != "none" else "Sensor_2"
        chart_data = pd.concat([
            last_run["clean_window"][["Cycle", plot_sensor]].assign(Condition="Original test data"),
            last_run["window"][["Cycle", plot_sensor]].assign(Condition=last_run["condition"]),
        ], ignore_index=True)
        fig = px.line(chart_data, x="Cycle", y=plot_sensor, color="Condition", markers=True,
                      title=f"Selected 30-cycle input · {plot_sensor}")
        st.plotly_chart(apply_chart_style(fig, 330), use_container_width=True)

        if st.session_state.live_runs:
            st.subheader("Runs in this demo session")
            comparison = pd.DataFrame([{
                "Engine": run["engine_id"],
                "Cycle": run["end_cycle"],
                "Condition": run["condition"],
                "Predicted RUL": round(run["predicted_rul"], 2),
                "Uncertainty": round(run["uncertainty"], 2),
                "Gate": run["decision"],
            } for run in st.session_state.live_runs])
            show_table(comparison, max_rows=8)
            if st.button("Clear demo runs", key="clear_live_runs"):
                st.session_state.live_runs = []
                st.session_state.pop("live_last_run", None)
                st.rerun()

    with st.expander("How this live prediction works"):
        st.markdown(
            "1. The app takes the selected engine's normalized sensor history and extracts the last 30 cycles ending at the chosen cycle.\n"
            "2. If selected, it injects one synthetic fault into that window using the reliability experiment's fault definition.\n"
            "3. The saved BiLSTM runs 100 stochastic dropout passes; their mean is the RUL estimate and their standard deviation is the uncertainty.\n"
            "4. The saved validation threshold and sensor-consistency models produce the ACCEPT/ABSTAIN result.\n\n"
            "The model and gate run on this laptop. This is an educational FD001 demonstration, not a deployed aircraft-monitoring system."
        )


def faults_page():
    section_intro("Sensor fault detection", "Review the controlled fault-injection experiment and the reliability gate's accept/abstain decisions.")
    fault_path = ROOT / "fault_detection_results.csv"
    faults = available_file(fault_path, "fault detection evaluation")
    gate_path = ROOT / "reliability_gate_results.csv"
    gate = available_file(gate_path, "reliability gate evaluation")
    if faults is None:
        return
    clean = faults[faults["Fault_type"] == "none"].copy()
    injected = faults[faults["Fault_type"] != "none"].copy()
    detected = injected[injected["Detected"].astype(str).str.lower() == "true"]
    det_rate = float(injected["Detected"].astype(str).str.lower().eq("true").mean()) if not injected.empty else 0
    false_rate = float(clean["False_alarm"].astype(str).str.lower().eq("true").mean()) if not clean.empty else 0
    delay = pd.to_numeric(detected["Detection_delay"], errors="coerce").mean()
    sensor_acc = float(detected["Correct_sensor"].astype(str).str.lower().eq("true").mean()) if not detected.empty else 0

    a, b, c, d = st.columns(4)
    a.metric("Injected-fault detection", f"{det_rate:.1%}")
    b.metric("Clean-run false alarms", f"{false_rate:.1%}")
    c.metric("Mean delay when detected", f"{delay:.1f} cycles" if pd.notna(delay) else "—")
    d.metric("Sensor ID accuracy", f"{sensor_acc:.1%}")
    st.caption(f"Evaluation includes {len(clean):,} clean runs and {len(injected):,} injected-fault runs. Sensor identification accuracy is calculated among detected faults.")

    fault_types = sorted(injected["Fault_type"].dropna().unique().tolist())
    selected_types = st.multiselect("Fault types", fault_types, default=fault_types)
    detail = injected[injected["Fault_type"].isin(selected_types)].copy()
    detail["Detected_bool"] = detail["Detected"].astype(str).str.lower().eq("true")
    detail["Severity_label"] = detail["Severity"].fillna("not applicable")
    grouped = detail.groupby(["Fault_type", "Severity_label"], as_index=False).agg(
        Runs=("Detected_bool", "size"),
        Detection_rate=("Detected_bool", "mean"),
        Mean_delay=("Detection_delay", "mean"),
    )
    st.caption("Severity changes the gradual-drift magnitude. The stuck-at fault is the same injected condition under both severity labels because severity does not apply to a fixed reading.")
    left, right = st.columns(2)
    with left:
        fig = px.bar(grouped, x="Fault_type", y="Detection_rate", color="Severity_label", barmode="group", text=grouped["Detection_rate"].map(lambda x: f"{x:.0%}"), title="Detection rate by fault type and severity", labels={"Detection_rate": "Detection rate", "Fault_type": "Fault type", "Severity_label": "Severity"}, range_y=[0, 1.08])
        st.plotly_chart(apply_chart_style(fig, 380), use_container_width=True)
    with right:
        fig = px.bar(grouped, x="Fault_type", y="Mean_delay", color="Severity_label", barmode="group", title="Mean detection delay among detected runs", labels={"Mean_delay": "Cycles after onset", "Fault_type": "Fault type", "Severity_label": "Severity"})
        st.plotly_chart(apply_chart_style(fig, 380), use_container_width=True)

    st.subheader("Reliability gate")
    st.write("The gate combines prediction uncertainty and sensor-consistency checks to accept or abstain on an RUL estimate.")
    if gate is not None:
        gate_summary = gate.groupby(["Scenario", "Decision"], as_index=False).size()
        totals = gate.groupby("Scenario").size().rename("Total")
        gate_summary = gate_summary.join(totals, on="Scenario")
        gate_summary["Decision_rate"] = gate_summary["size"] / gate_summary["Total"]
        chart = px.bar(gate_summary, x="Scenario", y="Decision_rate", color="Decision", barmode="stack", text=gate_summary.apply(lambda r: f"{int(r['size'])} · {r['Decision_rate']:.0%}", axis=1), title="Gate decisions by fault timing scenario", labels={"Decision_rate": "Share of decisions", "Scenario": "Fault timing"}, color_discrete_map={"ACCEPT": COLORS["teal"], "ABSTAIN": COLORS["amber"]}, range_y=[0, 1.12])
        st.plotly_chart(apply_chart_style(chart, 350), use_container_width=True)
        with st.expander("Show fault-detection records"):
            show_table(detail.drop(columns="Detected_bool", errors="ignore"), max_rows=200)
            st.download_button("Download filtered fault runs", detail.to_csv(index=False).encode("utf-8"), file_name="fault_detection_filtered.csv", mime="text/csv")

    with st.expander("Definitions and interpretation"):
        st.markdown(
            "- **Drift:** a sensor reading gradually moves away from its expected value.\n"
            "- **Stuck-at:** a sensor reading becomes fixed at a value.\n"
            "- **Detection delay:** cycles between injected fault onset and first persistent alarm.\n"
            "- **ACCEPT / ABSTAIN:** whether the reliability gate allows the RUL prediction to be used.\n\n"
            "Faults in this experiment are synthetic. Results depend on the chosen sensors, severities, onset cycles, and persistence rule."
        )


def about_page():
    section_intro("Methods & limitations", "A concise reference for the examiner while exploring the results.")
    with st.expander("Dataset and task", expanded=True):
        st.write("The project uses NASA C-MAPSS FD001 turbofan run-to-failure data. It is a single operating-condition subset with a simulated degradation trajectory. The target is remaining useful life in cycles.")
    with st.expander("Missing data"):
        st.write("Missingness is generated synthetically using missing completely at random (MCAR) masks at 5%, 10%, 20%, and 30%. This does not represent every real sensor-failure pattern.")
    with st.expander("Prediction models"):
        st.write("Baseline comparison: GRU, LSTM, BiLSTM, TCN, Random Forest, and XGBoost. The uncertainty comparison evaluates MC Dropout, Deep Ensembles, and Conformal Prediction using BiLSTM-based predictions.")
    with st.expander("Metrics"):
        st.write("RMSE and MAE summarize error in RUL cycles. R² describes variance explained. NASA/PHM08 score applies asymmetric penalties to early versus late predictions; scores are summed over the evaluated engines in the saved comparisons.")
    with st.expander("Scope and limitations"):
        st.write("The analysis pages present saved experiment results. The Live RUL prediction demo runs fresh inference with the saved BiLSTM and reliability-gate artifacts; it does not retrain models. Results use one C-MAPSS subset, synthetic missingness, and specific injected fault scenarios; they should not be interpreted as field validation or operational certification.")


st.markdown(
    """
    <style>
    .stApp {background: #0E1117; color: #E6EDF3;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px;}
    [data-testid="stSidebar"] {background: #121A24;}
    [data-testid="stSidebar"] * {color: #E6EDF3;}
    [data-testid="stMetric"] {background: #192635; border: 1px solid #2A3B4D; padding: 14px 16px; border-radius: 12px;}
    [data-testid="stMetric"] label, [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {color: #F3F7FA !important;}
    h1, h2, h3, h4 {color: #F3F7FA;}
    .table-wrap {overflow-x: auto; max-height: 460px; overflow-y: auto; border: 1px solid #344252; border-radius: 8px;}
    .table-wrap table {width: 100%; border-collapse: collapse; color: #E6EDF3; font-size: 0.88rem;}
    .table-wrap th {position: sticky; top: 0; background: #223245; color: #FFFFFF; text-align: left; padding: 0.55rem;}
    .table-wrap td {padding: 0.45rem 0.55rem; border-top: 1px solid #2A3948;}
    .table-wrap tr:nth-child(even) {background: #151E28;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ✈️ Capstone Explorer")
    st.caption("RUL prediction · uncertainty · sensor reliability")
    page = st.radio(
        "Navigate",
        ["Overview", "Live prediction demo", "Dataset & preprocessing", "Baseline comparison", "Uncertainty & RUL", "Sensor fault detection", "Methods & limitations"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Uses saved project data, experiment results, and model artifacts.")

if page == "Overview":
    overview_page()
elif page == "Live prediction demo":
    live_prediction_page()
elif page == "Dataset & preprocessing":
    dataset_page()
elif page == "Baseline comparison":
    baseline_page()
elif page == "Uncertainty & RUL":
    uncertainty_page()
elif page == "Sensor fault detection":
    faults_page()
else:
    about_page()
