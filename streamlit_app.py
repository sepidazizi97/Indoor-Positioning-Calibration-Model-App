import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False


# ---------------------------------------------------
# Page setup
# ---------------------------------------------------

st.set_page_config(
    page_title="Indoor Positioning Calibration App",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #1f3b57;
    font-size: 2.4rem;
    font-weight: 700;
}

h2, h3 {
    color: #24496b;
}

.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    border: none;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #155d8b;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

[data-testid="stTabs"] {
    background-color: white;
    border-radius: 14px;
    padding: 12px;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #1f3b57, #3c7fa6);
            padding: 30px;
            border-radius: 18px;
            margin-bottom: 25px;
            color: white;">
    <h1 style="color:white; margin-bottom: 5px;">Indoor Positioning Calibration App</h1>
    <p style="font-size:18px; margin-bottom:0;">
        A regression-based spatial calibration tool for improving indoor path tracking accuracy
        using SVM, Random Forest, XGBoost, and Gaussian Process Regression.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# Functions
# ---------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def reshape_calibration_excel(df):
    long_rows = []

    for _, row in df.iterrows():
        for i in range(1, 11):
            lat_col = f"Lat_test_{i}"
            lon_col = f"Lon_test_{i}"

            if lat_col in df.columns and lon_col in df.columns:
                if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                    long_rows.append({
                        "Path_Points": row.get("Path-Points"),
                        "Point_ID": row.get("Point-ID"),
                        "Trial": i,
                        "Lat_true": row["Lat_true"],
                        "Lon_true": row["Lon_true"],
                        "Lat_test": row[lat_col],
                        "Lon_test": row[lon_col],
                    })

    long_df = pd.DataFrame(long_rows)

    long_df["Delta_Lat"] = long_df["Lat_true"] - long_df["Lat_test"]
    long_df["Delta_Lon"] = long_df["Lon_true"] - long_df["Lon_test"]

    return long_df


def get_models():
    return {
        "Random Forest": (
            RandomForestRegressor(n_estimators=400, random_state=42),
            RandomForestRegressor(n_estimators=400, random_state=42)
        ),

        "GPR": (
            make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(
                    kernel=C(1.0) * RBF(length_scale=1.0),
                    normalize_y=True,
                    random_state=42
                )
            ),
            make_pipeline(
                StandardScaler(),
                GaussianProcessRegressor(
                    kernel=C(1.0) * RBF(length_scale=1.0),
                    normalize_y=True,
                    random_state=42
                )
            )
        ),

        "XGBoost": (
            XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ) if XGBOOST_AVAILABLE else GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ) if XGBOOST_AVAILABLE else GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        ),

        "SVM": (
            make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=10, gamma=0.5, epsilon=1e-5)
            ),
            make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=10, gamma=0.5, epsilon=1e-5)
            )
        )
    }


def run_model(model_name, lat_model, lon_model, X_train, X_test, y_lat_train, y_lon_train, test_data):
    lat_model.fit(X_train, y_lat_train)
    lon_model.fit(X_train, y_lon_train)

    pred_delta_lat = lat_model.predict(X_test)
    pred_delta_lon = lon_model.predict(X_test)

    calibrated = test_data.copy()

    calibrated["Pred_Delta_Lat"] = pred_delta_lat
    calibrated["Pred_Delta_Lon"] = pred_delta_lon

    calibrated["Lat_calibrated"] = calibrated["Lat_test"] + calibrated["Pred_Delta_Lat"]
    calibrated["Lon_calibrated"] = calibrated["Lon_test"] + calibrated["Pred_Delta_Lon"]

    calibrated["Error_Before_m"] = haversine(
        calibrated["Lat_test"],
        calibrated["Lon_test"],
        calibrated["Lat_true"],
        calibrated["Lon_true"]
    )

    calibrated["Error_After_m"] = haversine(
        calibrated["Lat_calibrated"],
        calibrated["Lon_calibrated"],
        calibrated["Lat_true"],
        calibrated["Lon_true"]
    )

    calibrated["Improvement_m"] = calibrated["Error_Before_m"] - calibrated["Error_After_m"]

    summary = {
        "Model": model_name,
        "Mean Error": round(calibrated["Error_After_m"].mean(), 2),
        "Median Error": round(calibrated["Error_After_m"].median(), 2),
        "P90 Reliability": round(np.percentile(calibrated["Error_After_m"], 90), 2),
        "Std Dev": round(calibrated["Error_After_m"].std(), 2),
        "Mean Error Before": round(calibrated["Error_Before_m"].mean(), 2),
        "Improvement (%)": round(
            (
                calibrated["Error_Before_m"].mean()
                - calibrated["Error_After_m"].mean()
            )
            / calibrated["Error_Before_m"].mean()
            * 100,
            2
        )
    }

    return calibrated, summary


# ---------------------------------------------------
# Upload and run section
# ---------------------------------------------------

with st.container():
    st.markdown("### Upload Calibration File")
    st.write(
        "Upload your Excel file containing ground-truth points and repeated observed/test points."
    )

    uploaded_file = st.file_uploader(
        "Upload Excel or CSV file",
        type=["xlsx", "csv"]
    )

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file, sheet_name="CM")

    data = reshape_calibration_excel(raw_df)

    if len(data) == 0:
        st.error("No valid calibration points were found. Please check your column names.")
        st.stop()

    st.info(f"{len(data)} valid calibration observations were detected.")

    run_button = st.button("Run Calibration Models")

    if run_button:

        X = data[["Lat_test", "Lon_test"]]
        y_lat = data["Delta_Lat"]
        y_lon = data["Delta_Lon"]

        (
            X_train,
            X_test,
            y_lat_train,
            y_lat_test,
            y_lon_train,
            y_lon_test,
            idx_train,
            idx_test
        ) = train_test_split(
            X,
            y_lat,
            y_lon,
            data.index,
            test_size=0.2,
            random_state=42
        )

        test_data = data.loc[idx_test].copy()

        models = get_models()

        calibrated_outputs = {}
        summaries = []

        with st.spinner("Running calibration models..."):
            for model_name, (lat_model, lon_model) in models.items():
                calibrated, summary = run_model(
                    model_name,
                    lat_model,
                    lon_model,
                    X_train,
                    X_test,
                    y_lat_train,
                    y_lon_train,
                    test_data
                )

                calibrated_outputs[model_name] = calibrated
                summaries.append(summary)

        results_df = pd.DataFrame(summaries)

        st.session_state["calibrated_outputs"] = calibrated_outputs
        st.session_state["results_df"] = results_df

        st.success("Calibration completed successfully.")


# ---------------------------------------------------
# Results section
# ---------------------------------------------------

if "calibrated_outputs" in st.session_state:

    calibrated_outputs = st.session_state["calibrated_outputs"]
    results_df = st.session_state["results_df"]

    st.markdown("---")
    st.header("Results")

    st.write(
        "This section presents the comparative performance of four calibration models "
        "in reducing positional error for indoor path tracking. Results are reported "
        "using multiple accuracy and reliability metrics to evaluate average performance "
        "and robustness to spatial outliers."
    )

    best_model = results_df.sort_values("Mean Error").iloc[0]["Model"]

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Best Model", best_model)

    with metric_col2:
        best_mean = results_df.sort_values("Mean Error").iloc[0]["Mean Error"]
        st.metric("Lowest Mean Error", f"{best_mean} m")

    with metric_col3:
        best_p90 = results_df.sort_values("P90 Reliability").iloc[0]["P90 Reliability"]
        st.metric("Best P90 Reliability", f"{best_p90} m")

    st.subheader("Comparison of Positional Error Metrics")

    display_results = results_df.copy()
    display_results["Mean Error"] = display_results["Mean Error"].astype(str) + " m"
    display_results["Median Error"] = display_results["Median Error"].astype(str) + " m"
    display_results["P90 Reliability"] = display_results["P90 Reliability"].astype(str) + " m"
    display_results["Std Dev"] = display_results["Std Dev"].astype(str) + " m"

    st.dataframe(
        display_results[
            ["Model", "Mean Error", "Median Error", "P90 Reliability", "Std Dev"]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Cumulative Distribution Function of Positioning Error")

    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, df_model in calibrated_outputs.items():
        errors = np.sort(df_model["Error_After_m"])
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, label=model_name)

    ax.axhline(y=0.9, linestyle="--", label="P90 Reliability")
    ax.set_xlabel("Positioning Error After Calibration (meters)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Positioning Error by Calibration Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.subheader("Box-and-Whisker Plot of Residual Positional Errors")

    boxplot_data = [
        calibrated_outputs[model]["Error_After_m"]
        for model in calibrated_outputs.keys()
    ]

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.boxplot(boxplot_data, labels=list(calibrated_outputs.keys()))
    ax2.set_ylabel("Positioning Error After Calibration (meters)")
    ax2.set_title("Distribution of Residual Positional Errors")
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)

    st.markdown("---")
    st.header("Calibration Results by Method")

    tab_rf, tab_gpr, tab_xgb, tab_svm = st.tabs(
        ["Random Forest", "GPR", "XGBoost", "SVM"]
    )

    model_tabs = {
        "Random Forest": tab_rf,
        "GPR": tab_gpr,
        "XGBoost": tab_xgb,
        "SVM": tab_svm
    }

    for model_name, tab in model_tabs.items():

        with tab:
            model_df = calibrated_outputs[model_name]
            model_summary = results_df[results_df["Model"] == model_name].iloc[0]

            st.subheader(f"{model_name} Calibration Results")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Mean Error", f"{model_summary['Mean Error']} m")
            c2.metric("Median Error", f"{model_summary['Median Error']} m")
            c3.metric("P90 Reliability", f"{model_summary['P90 Reliability']} m")
            c4.metric("Std Dev", f"{model_summary['Std Dev']} m")

            st.write("### Spatial Distribution of Calibrated Points")

            fig3, ax3 = plt.subplots(figsize=(8, 6))

            ax3.scatter(
                model_df["Lon_true"],
                model_df["Lat_true"],
                label="Ground-Truth Points",
                marker="x",
                s=70
            )

            ax3.scatter(
                model_df["Lon_calibrated"],
                model_df["Lat_calibrated"],
                label=f"{model_name} Calibrated Points",
                alpha=0.7,
                s=45
            )

            ax3.set_xlabel("Longitude")
            ax3.set_ylabel("Latitude")
            ax3.set_title(f"Spatial Distribution of {model_name} Calibrated Points")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            st.pyplot(fig3)

            st.write("### Calibrated Points Table")

            st.dataframe(
                model_df[
                    [
                        "Path_Points",
                        "Point_ID",
                        "Trial",
                        "Lat_true",
                        "Lon_true",
                        "Lat_test",
                        "Lon_test",
                        "Lat_calibrated",
                        "Lon_calibrated",
                        "Error_Before_m",
                        "Error_After_m",
                        "Improvement_m"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            csv = model_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label=f"Download {model_name} Calibrated Points",
                data=csv,
                file_name=f"{model_name.replace(' ', '_')}_calibrated_points.csv",
                mime="text/csv"
            )

    comparison_csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Model Comparison Table",
        data=comparison_csv,
        file_name="model_comparison_table.csv",
        mime="text/csv"
    )
