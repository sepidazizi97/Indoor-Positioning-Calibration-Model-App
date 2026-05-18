import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

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


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Indoor Positioning Calibration App",
    layout="wide"
)


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f4f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero-box {
    background: linear-gradient(135deg, #1f3b57, #3c7fa6);
    padding: 34px;
    border-radius: 20px;
    margin-bottom: 28px;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.12);
}

.hero-title {
    color: white !important;
    font-size: 44px;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero-subtitle {
    color: white !important;
    font-size: 18px;
    line-height: 1.5;
    margin-bottom: 0px;
}

.info-card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 18px;
}

h2, h3 {
    color: #24496b;
}

.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.3rem;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #155d8b;
    color: white;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.08);
}

[data-testid="stTabs"] {
    background-color: white;
    border-radius: 14px;
    padding: 10px;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero-box">
    <div class="hero-title">Indoor Positioning Calibration App</div>
    <div class="hero-subtitle">
        Regression-based spatial calibration tool for improving indoor positioning accuracy
        using Random Forest, GPR, XGBoost, and SVM.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# APP DESCRIPTION
# =========================================================

st.markdown("""
<div class="info-card">
    <h3>About This App</h3>
    <p>
    This app calibrates indoor positioning points by comparing observed/test locations
    with known ground-truth control points. It applies four regression-based calibration
    methods: Random Forest, Gaussian Process Regression, XGBoost, and Support Vector Machine.
    </p>
    <p>
    The app calculates residual coordinate offsets, applies model-based corrections,
    and compares model performance using mean error, median error, P90 reliability,
    and standard deviation.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# EXCEL STRUCTURE DESCRIPTION
# =========================================================

with st.expander("Required Excel File Structure"):

    st.write(
        "The calibration file should be in wide format. Each row represents one "
        "ground-control point. The true coordinates are stored once, and repeated "
        "observed/test coordinates are stored across multiple test columns."
    )

    example_structure = pd.DataFrame({
        "Path-Points": ["Path 1"],
        "Point-ID": [1],
        "Lat_true": [42.407123],
        "Lon_true": [-71.119456],
        "Lat_test_1": [42.407100],
        "Lon_test_1": [-71.119430],
        "Lat_test_2": [42.407090],
        "Lon_test_2": [-71.119420],
        "...": ["..."],
        "Lat_test_10": [42.407080],
        "Lon_test_10": [-71.119410],
    })

    st.dataframe(
        example_structure,
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Required columns: Lat_true, Lon_true, Lat_test_1, Lon_test_1 through "
        "Lat_test_10, Lon_test_10. Optional columns include Path-Points and Point-ID."
    )


# =========================================================
# MODE SELECTION
# =========================================================

mode = st.radio(
    "Choose Calibration Mode",
    [
        "Train and Evaluate Calibration Models",
        "Apply Existing Calibration Model to New Location"
    ]
)


if mode == "Apply Existing Calibration Model to New Location":
    st.info(
        "This mode applies a previously trained calibration model to new-location points "
        "without requiring ground-truth points."
    )

    st.warning(
        "Results may contain positional inaccuracies because different buildings may have "
        "different layouts, wall materials, signal conditions, and environmental interference. "
        "This should be interpreted as estimated calibration, not validated calibration."
    )


# =========================================================
# FUNCTIONS
# =========================================================

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

    if len(long_df) > 0:
        long_df["Delta_Lat"] = long_df["Lat_true"] - long_df["Lat_test"]
        long_df["Delta_Lon"] = long_df["Lon_true"] - long_df["Lon_test"]

    return long_df


def get_models():

    return {

        "Random Forest": (
            RandomForestRegressor(
                n_estimators=400,
                random_state=42
            ),
            RandomForestRegressor(
                n_estimators=400,
                random_state=42
            )
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
                SVR(
                    kernel="rbf",
                    C=10,
                    gamma=0.5,
                    epsilon=1e-5
                )
            ),
            make_pipeline(
                StandardScaler(),
                SVR(
                    kernel="rbf",
                    C=10,
                    gamma=0.5,
                    epsilon=1e-5
                )
            )
        )
    }


def save_model(model_name, lat_model, lon_model):
    safe_name = model_name.replace(" ", "_")

    with open(f"{safe_name}_lat_model.pkl", "wb") as f:
        pickle.dump(lat_model, f)

    with open(f"{safe_name}_lon_model.pkl", "wb") as f:
        pickle.dump(lon_model, f)


def load_model(model_name):
    safe_name = model_name.replace(" ", "_")

    lat_path = f"{safe_name}_lat_model.pkl"
    lon_path = f"{safe_name}_lon_model.pkl"

    if not os.path.exists(lat_path) or not os.path.exists(lon_path):
        return None, None

    with open(lat_path, "rb") as f:
        lat_model = pickle.load(f)

    with open(lon_path, "rb") as f:
        lon_model = pickle.load(f)

    return lat_model, lon_model


# =========================================================
# TRAIN AND EVALUATE MODE
# =========================================================

if mode == "Train and Evaluate Calibration Models":

    st.subheader("Upload or Use Sample Calibration Points")

    use_example = st.checkbox("Use sample calibration points from GitHub")

    GITHUB_SAMPLE_URL = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPOSITORY/main/Callibration_Model.xlsx"

    raw_df = None

    if use_example:
        try:
            raw_df = pd.read_excel(GITHUB_SAMPLE_URL, sheet_name="CM")
            st.success("Sample calibration file loaded from GitHub.")
        except Exception as e:
            st.error(
                "Could not load the sample file from GitHub. Please check that your GitHub raw file URL is correct."
            )
            st.stop()

    else:
        uploaded_file = st.file_uploader(
            "Upload Calibration Excel or CSV File",
            type=["xlsx", "csv"]
        )

        if uploaded_file is not None:

            if uploaded_file.name.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file, sheet_name="CM")

    if raw_df is not None:

        data = reshape_calibration_excel(raw_df)

        if len(data) == 0:
            st.error(
                "No valid calibration observations were detected. Please check that your file includes "
                "Lat_true, Lon_true, Lat_test_1, Lon_test_1 through Lat_test_10 and Lon_test_10."
            )
            st.stop()

        st.success(f"{len(data)} valid calibration observations detected.")

        if st.button("Run Calibration Models"):

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
            summaries = {}

            with st.spinner("Running calibration models..."):

                for model_name, (lat_model, lon_model) in models.items():

                    lat_model.fit(X_train, y_lat_train)
                    lon_model.fit(X_train, y_lon_train)

                    pred_lat = lat_model.predict(X_test)
                    pred_lon = lon_model.predict(X_test)

                    calibrated = test_data.copy()

                    calibrated["Pred_Delta_Lat"] = pred_lat
                    calibrated["Pred_Delta_Lon"] = pred_lon

                    calibrated["Lat_calibrated"] = (
                        calibrated["Lat_test"] + pred_lat
                    )

                    calibrated["Lon_calibrated"] = (
                        calibrated["Lon_test"] + pred_lon
                    )

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

                    calibrated["Improvement_m"] = (
                        calibrated["Error_Before_m"]
                        - calibrated["Error_After_m"]
                    )

                    calibrated_outputs[model_name] = calibrated

                    summaries[model_name] = {
                        "Mean Error": round(calibrated["Error_After_m"].mean(), 2),
                        "Median Error": round(calibrated["Error_After_m"].median(), 2),
                        "P90 Reliability": round(
                            np.percentile(calibrated["Error_After_m"], 90),
                            2
                        ),
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

                    save_model(model_name, lat_model, lon_model)

            st.success("Calibration completed successfully.")

            comparison_df = pd.DataFrame(summaries).T.reset_index()
            comparison_df.columns = [
                "Model",
                "Mean Error",
                "Median Error",
                "P90 Reliability",
                "Std Dev",
                "Mean Error Before",
                "Improvement (%)"
            ]

            st.session_state["comparison_df"] = comparison_df
            st.session_state["calibrated_outputs"] = calibrated_outputs


# =========================================================
# DISPLAY RESULTS
# =========================================================

if "comparison_df" in st.session_state and mode == "Train and Evaluate Calibration Models":

    comparison_df = st.session_state["comparison_df"]
    calibrated_outputs = st.session_state["calibrated_outputs"]

    st.markdown("---")
    st.header("Results")

    st.write(
        "This section presents the comparative performance of four calibration models "
        "in reducing positional error for indoor path tracking. Results are reported "
        "using multiple accuracy and reliability metrics to evaluate both average "
        "performance and robustness to spatial outliers."
    )

    best_model = comparison_df.sort_values("Mean Error").iloc[0]["Model"]
    best_error = comparison_df.sort_values("Mean Error").iloc[0]["Mean Error"]
    best_p90 = comparison_df.sort_values("P90 Reliability").iloc[0]["P90 Reliability"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Best Model", best_model)
    col2.metric("Lowest Mean Error", f"{best_error} m")
    col3.metric("Best P90 Reliability", f"{best_p90} m")

    st.subheader("Comparison of Positional Error Metrics")

    display_df = comparison_df.copy()
    for col in ["Mean Error", "Median Error", "P90 Reliability", "Std Dev", "Mean Error Before"]:
        display_df[col] = display_df[col].astype(str) + " m"

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Cumulative Distribution Function of Positioning Error")

    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, df_model in calibrated_outputs.items():
        errors = np.sort(df_model["Error_After_m"])
        cdf = np.arange(1, len(errors) + 1) / len(errors)

        ax.plot(errors, cdf, label=model_name)

    ax.axhline(
        y=0.9,
        linestyle="--",
        label="P90 Reliability"
    )

    ax.set_xlabel("Positioning Error After Calibration (meters)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Positioning Error by Calibration Model")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    st.subheader("Box-and-Whisker Plot of Residual Positional Errors")

    fig2, ax2 = plt.subplots(figsize=(9, 6))

    ax2.boxplot(
        [
            calibrated_outputs[m]["Error_After_m"]
            for m in calibrated_outputs.keys()
        ],
        labels=list(calibrated_outputs.keys())
    )

    ax2.set_ylabel("Positioning Error After Calibration (meters)")
    ax2.set_title("Distribution of Residual Positional Errors")
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)

    st.markdown("---")
    st.header("Calibration Results by Method")

    tab_rf, tab_gpr, tab_xgb, tab_svm = st.tabs([
        "Random Forest",
        "GPR",
        "XGBoost",
        "SVM"
    ])

    tabs = {
        "Random Forest": tab_rf,
        "GPR": tab_gpr,
        "XGBoost": tab_xgb,
        "SVM": tab_svm
    }

    for model_name, tab in tabs.items():

        with tab:

            df_model = calibrated_outputs[model_name]
            model_summary = comparison_df[
                comparison_df["Model"] == model_name
            ].iloc[0]

            st.subheader(f"{model_name} Calibration Results")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Mean Error", f"{model_summary['Mean Error']} m")
            c2.metric("Median Error", f"{model_summary['Median Error']} m")
            c3.metric("P90 Reliability", f"{model_summary['P90 Reliability']} m")
            c4.metric("Std Dev", f"{model_summary['Std Dev']} m")

            st.write("### Spatial Distribution of Calibrated Points")

            fig3, ax3 = plt.subplots(figsize=(8, 6))

            ax3.scatter(
                df_model["Lon_true"],
                df_model["Lat_true"],
                label="Ground-Truth Points",
                marker="x",
                s=75
            )

            ax3.scatter(
                df_model["Lon_test"],
                df_model["Lat_test"],
                label="Original Test Points",
                alpha=0.35,
                s=40
            )

            ax3.scatter(
                df_model["Lon_calibrated"],
                df_model["Lat_calibrated"],
                label=f"{model_name} Calibrated Points",
                alpha=0.8,
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
                df_model[
                    [
                        "Path_Points",
                        "Point_ID",
                        "Trial",
                        "Lat_true",
                        "Lon_true",
                        "Lat_test",
                        "Lon_test",
                        "Pred_Delta_Lat",
                        "Pred_Delta_Lon",
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

            csv = df_model.to_csv(index=False).encode("utf-8")

            st.download_button(
                label=f"Download {model_name} Results",
                data=csv,
                file_name=f"{model_name.replace(' ', '_')}_calibrated_results.csv",
                mime="text/csv"
            )

    comparison_csv = comparison_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Model Comparison Table",
        data=comparison_csv,
        file_name="model_comparison_table.csv",
        mime="text/csv"
    )


# =========================================================
# APPLY EXISTING MODEL MODE
# =========================================================

if mode == "Apply Existing Calibration Model to New Location":

    st.subheader("Upload New Location Points")

    st.write(
        "This file should include observed coordinates only. Required columns: "
        "`Lat_test` and `Lon_test`."
    )

    new_example = pd.DataFrame({
        "Point_ID": [1, 2, 3],
        "Lat_test": [42.407100, 42.407090, 42.407080],
        "Lon_test": [-71.119430, -71.119420, -71.119410]
    })

    with st.expander("Example Structure for New-Location Points"):
        st.dataframe(
            new_example,
            hide_index=True,
            use_container_width=True
        )

    new_file = st.file_uploader(
        "Upload CSV or Excel file containing new observed points",
        type=["csv", "xlsx"]
    )

    if new_file is not None:

        if new_file.name.endswith(".csv"):
            new_df = pd.read_csv(new_file)
        else:
            new_df = pd.read_excel(new_file)

        required_cols = ["Lat_test", "Lon_test"]

        missing_cols = [
            col for col in required_cols
            if col not in new_df.columns
        ]

        if missing_cols:
            st.error(
                f"Missing required columns: {missing_cols}. "
                "Please make sure your file includes Lat_test and Lon_test."
            )
            st.stop()

        model_choice = st.selectbox(
            "Select Calibration Model",
            [
                "Random Forest",
                "GPR",
                "XGBoost",
                "SVM"
            ]
        )

        if st.button("Apply Estimated Calibration"):

            lat_model, lon_model = load_model(model_choice)

            if lat_model is None or lon_model is None:
                st.error(
                    "No saved model was found for this method. Please first run "
                    "the Train and Evaluate mode to create saved calibration models."
                )
                st.stop()

            X_new = new_df[["Lat_test", "Lon_test"]]

            pred_lat = lat_model.predict(X_new)
            pred_lon = lon_model.predict(X_new)

            new_df["Pred_Delta_Lat"] = pred_lat
            new_df["Pred_Delta_Lon"] = pred_lon

            new_df["Lat_calibrated"] = (
                new_df["Lat_test"] + pred_lat
            )

            new_df["Lon_calibrated"] = (
                new_df["Lon_test"] + pred_lon
            )

            st.success("Estimated calibration completed.")

            st.warning(
                "Because no ground-truth points were provided for this new location, "
                "the app cannot calculate true positional error. These results are estimated."
            )

            c1, c2 = st.columns(2)

            with c1:
                st.metric("Number of Points", len(new_df))

            with c2:
                st.metric("Calibration Type", "Estimated Cross-Location")

            st.subheader("Estimated Calibrated Points")

            st.dataframe(
                new_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Original vs Estimated Calibrated Points")

            fig4, ax4 = plt.subplots(figsize=(8, 6))

            ax4.scatter(
                new_df["Lon_test"],
                new_df["Lat_test"],
                label="Original Points",
                alpha=0.5,
                s=45
            )

            ax4.scatter(
                new_df["Lon_calibrated"],
                new_df["Lat_calibrated"],
                label="Estimated Calibrated Points",
                alpha=0.8,
                s=45
            )

            ax4.set_xlabel("Longitude")
            ax4.set_ylabel("Latitude")
            ax4.set_title("Estimated Cross-Location Calibration")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            st.pyplot(fig4)

            csv = new_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Estimated Calibrated Points",
                data=csv,
                file_name="estimated_cross_location_calibrated_points.csv",
                mime="text/csv"
            )
