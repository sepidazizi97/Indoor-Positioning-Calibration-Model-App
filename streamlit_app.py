import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

h1 {
    color: white;
    font-weight: 700;
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

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div style="
background: linear-gradient(135deg,#1f3b57,#3c7fa6);
padding: 30px;
border-radius: 18px;
margin-bottom: 25px;
">
<h1>Indoor Positioning Calibration App</h1>

<p style="font-size:18px; color:white;">
Regression-based spatial calibration tool for improving indoor positioning accuracy
using Random Forest, GPR, XGBoost, and SVM.
</p>
</div>
""", unsafe_allow_html=True)

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

# =========================================================
# WARNING FOR CROSS LOCATION
# =========================================================

if mode == "Apply Existing Calibration Model to New Location":

    st.info(
        "This mode applies a model trained on a previous location "
        "to a new location without ground-truth points."
    )

    st.warning(
        "Results may contain positional inaccuracies because different buildings "
        "can have different signal conditions, wall materials, layouts, and "
        "environmental interference. This should be interpreted as estimated calibration."
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


# =========================================================
# TRAIN MODE
# =========================================================

if mode == "Train and Evaluate Calibration Models":

    uploaded_file = st.file_uploader(
        "Upload Calibration Excel File",
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file, sheet_name="CM")

        data = reshape_calibration_excel(raw_df)

        st.success(f"{len(data)} calibration observations detected.")

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

                    calibrated["Lat_calibrated"] = (
                        calibrated["Lat_test"] + pred_lat
                    )

                    calibrated["Lon_calibrated"] = (
                        calibrated["Lon_test"] + pred_lon
                    )

                    calibrated["Error_Before"] = haversine(
                        calibrated["Lat_test"],
                        calibrated["Lon_test"],
                        calibrated["Lat_true"],
                        calibrated["Lon_true"]
                    )

                    calibrated["Error_After"] = haversine(
                        calibrated["Lat_calibrated"],
                        calibrated["Lon_calibrated"],
                        calibrated["Lat_true"],
                        calibrated["Lon_true"]
                    )

                    calibrated_outputs[model_name] = calibrated

                    summaries[model_name] = {
                        "Mean Error": round(calibrated["Error_After"].mean(), 2),
                        "Median Error": round(calibrated["Error_After"].median(), 2),
                        "P90 Reliability": round(
                            np.percentile(calibrated["Error_After"], 90),
                            2
                        ),
                        "Std Dev": round(
                            calibrated["Error_After"].std(),
                            2
                        )
                    }

                    # SAVE MODEL
                    with open(f"{model_name}_lat_model.pkl", "wb") as f:
                        pickle.dump(lat_model, f)

                    with open(f"{model_name}_lon_model.pkl", "wb") as f:
                        pickle.dump(lon_model, f)

            st.success("Calibration completed successfully.")

            # =========================================================
            # COMPARISON TABLE
            # =========================================================

            st.header("Results")

            comparison_df = pd.DataFrame(summaries).T.reset_index()
            comparison_df.columns = [
                "Model",
                "Mean Error",
                "Median Error",
                "P90 Reliability",
                "Std Dev"
            ]

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            # =========================================================
            # CDF PLOT
            # =========================================================

            st.subheader("CDF of Positioning Error")

            fig, ax = plt.subplots(figsize=(9, 6))

            for model_name, df_model in calibrated_outputs.items():

                errors = np.sort(df_model["Error_After"])
                cdf = np.arange(1, len(errors) + 1) / len(errors)

                ax.plot(errors, cdf, label=model_name)

            ax.axhline(
                y=0.9,
                linestyle="--",
                label="P90 Reliability"
            )

            ax.set_xlabel("Error (meters)")
            ax.set_ylabel("Cumulative Probability")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)

            # =========================================================
            # BOXPLOT
            # =========================================================

            st.subheader("Residual Error Distribution")

            fig2, ax2 = plt.subplots(figsize=(9, 6))

            ax2.boxplot(
                [
                    calibrated_outputs[m]["Error_After"]
                    for m in calibrated_outputs.keys()
                ],
                labels=list(calibrated_outputs.keys())
            )

            ax2.set_ylabel("Error (meters)")
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig2)

            # =========================================================
            # TABS
            # =========================================================

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

                    st.subheader(f"{model_name} Results")

                    # METRICS

                    c1, c2, c3, c4 = st.columns(4)

                    c1.metric(
                        "Mean Error",
                        f"{summaries[model_name]['Mean Error']} m"
                    )

                    c2.metric(
                        "Median Error",
                        f"{summaries[model_name]['Median Error']} m"
                    )

                    c3.metric(
                        "P90 Reliability",
                        f"{summaries[model_name]['P90 Reliability']} m"
                    )

                    c4.metric(
                        "Std Dev",
                        f"{summaries[model_name]['Std Dev']} m"
                    )

                    # MAP

                    fig3, ax3 = plt.subplots(figsize=(8, 6))

                    ax3.scatter(
                        df_model["Lon_true"],
                        df_model["Lat_true"],
                        label="Ground Truth",
                        marker="x",
                        s=70
                    )

                    ax3.scatter(
                        df_model["Lon_calibrated"],
                        df_model["Lat_calibrated"],
                        label=f"{model_name} Calibrated",
                        alpha=0.7
                    )

                    ax3.legend()
                    ax3.grid(True, alpha=0.3)

                    st.pyplot(fig3)

                    # DOWNLOAD

                    csv = df_model.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label=f"Download {model_name} Results",
                        data=csv,
                        file_name=f"{model_name}_results.csv",
                        mime="text/csv"
                    )

# =========================================================
# APPLY EXISTING MODEL MODE
# =========================================================

if mode == "Apply Existing Calibration Model to New Location":

    st.subheader("Upload New Location Points")

    new_file = st.file_uploader(
        "Upload CSV or Excel file containing only observed points",
        type=["csv", "xlsx"]
    )

    if new_file is not None:

        if new_file.name.endswith(".csv"):
            new_df = pd.read_csv(new_file)
        else:
            new_df = pd.read_excel(new_file)

        st.write(
            "Required columns: Lat_test and Lon_test"
        )

        model_choice = st.selectbox(
            "Select Calibration Model",
            [
                "Random Forest",
                "GPR",
                "XGBoost",
                "SVM"
            ]
        )

        if st.button("Apply Calibration"):

            with open(f"{model_choice}_lat_model.pkl", "rb") as f:
                lat_model = pickle.load(f)

            with open(f"{model_choice}_lon_model.pkl", "rb") as f:
                lon_model = pickle.load(f)

            X_new = new_df[["Lat_test", "Lon_test"]]

            pred_lat = lat_model.predict(X_new)
            pred_lon = lon_model.predict(X_new)

            new_df["Lat_calibrated"] = (
                new_df["Lat_test"] + pred_lat
            )

            new_df["Lon_calibrated"] = (
                new_df["Lon_test"] + pred_lon
            )

            st.success("Estimated calibration completed.")

            st.dataframe(
                new_df,
                use_container_width=True,
                hide_index=True
            )

            fig4, ax4 = plt.subplots(figsize=(8, 6))

            ax4.scatter(
                new_df["Lon_test"],
                new_df["Lat_test"],
                label="Original Points",
                alpha=0.5
            )

            ax4.scatter(
                new_df["Lon_calibrated"],
                new_df["Lat_calibrated"],
                label="Calibrated Points",
                alpha=0.7
            )

            ax4.legend()
            ax4.grid(True, alpha=0.3)

            st.pyplot(fig4)

            csv = new_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Calibrated Points",
                data=csv,
                file_name="estimated_calibrated_points.csv",
                mime="text/csv"
            )
