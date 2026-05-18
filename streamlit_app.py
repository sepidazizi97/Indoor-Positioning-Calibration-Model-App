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


st.set_page_config(
    page_title="Indoor Positioning Calibration App",
    layout="wide"
)

GITHUB_CALIBRATION_FILE = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPOSITORY/main/Callibration_Model.xlsx"


st.markdown("""
<style>
.hero-box {
    background: linear-gradient(135deg, #1f3b57, #3c7fa6);
    padding: 34px;
    border-radius: 20px;
    margin-bottom: 28px;
}
.hero-title {
    color: white !important;
    font-size: 42px;
    font-weight: 800;
}
.hero-subtitle {
    color: white !important;
    font-size: 18px;
}
.info-card {
    background-color: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}
.warning-card {
    background-color: #fff8e6;
    padding: 18px;
    border-left: 5px solid #f0a500;
    border-radius: 14px;
    margin-bottom: 18px;
}
.method-card {
    background-color: #f8fbff;
    padding: 18px;
    border-left: 5px solid #3c7fa6;
    border-radius: 14px;
    margin-bottom: 18px;
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
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero-box">
    <div class="hero-title">Indoor Positioning Calibration App</div>
    <div class="hero-subtitle">
        Upload indoor positioning data, apply regression-based calibration methods,
        and compare corrected locations using clear spatial accuracy metrics.
    </div>
</div>
""", unsafe_allow_html=True)


# ======================================================
# FUNCTIONS
# ======================================================

def fix_swapped_coordinates(df, lat_col="Lat_test", lon_col="Lon_test"):
    """
    Detects rows where latitude/longitude appear reversed.
    This is useful when latitude appears around -70 and longitude appears around 40.
    """

    df = df.copy()

    swapped_mask = (
        (df[lat_col].abs() > 60) &
        (df[lon_col].abs() < 60)
    )

    if swapped_mask.sum() > 0:
        temp_lat = df.loc[swapped_mask, lat_col].copy()
        df.loc[swapped_mask, lat_col] = df.loc[swapped_mask, lon_col]
        df.loc[swapped_mask, lon_col] = temp_lat

    return df, swapped_mask.sum()


def remove_extreme_outliers(df):
    """
    Removes impossible latitude/longitude values after the swap check.
    """

    df = df.copy()

    valid_mask = (
        df["Lat_test"].between(-90, 90) &
        df["Lon_test"].between(-180, 180) &
        df["Lat_true"].between(-90, 90) &
        df["Lon_true"].between(-180, 180)
    )

    removed_count = len(df) - valid_mask.sum()

    return df[valid_mask].copy(), removed_count


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


def reshape_training_excel(df):
    long_rows = []

    for _, row in df.iterrows():
        for i in range(1, 11):
            lat_col = f"Lat_test_{i}"
            lon_col = f"Lon_test_{i}"

            if lat_col in df.columns and lon_col in df.columns:
                if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                    long_rows.append({
                        "Path_Points": row.get("Path-Points", None),
                        "Point_ID": row.get("Point-ID", None),
                        "Trial": i,
                        "Lat_true": row["Lat_true"],
                        "Lon_true": row["Lon_true"],
                        "Lat_test": row[lat_col],
                        "Lon_test": row[lon_col],
                    })

    long_df = pd.DataFrame(long_rows)

    if len(long_df) > 0:
        long_df, swapped_count = fix_swapped_coordinates(long_df)
        long_df, removed_count = remove_extreme_outliers(long_df)

        long_df["Delta_Lat"] = long_df["Lat_true"] - long_df["Lat_test"]
        long_df["Delta_Lon"] = long_df["Lon_true"] - long_df["Lon_test"]

        st.info(
            f"Coordinate check completed. Swapped coordinate rows corrected: {swapped_count}. "
            f"Invalid rows removed: {removed_count}."
        )

    return long_df


def reshape_new_points_file(df):
    if "Lat_test" in df.columns and "Lon_test" in df.columns:
        new_df = df.copy()
        new_df, swapped_count = fix_swapped_coordinates(new_df)
        st.info(f"Coordinate check completed. Swapped coordinate rows corrected: {swapped_count}.")
        return new_df

    long_rows = []

    for idx, row in df.iterrows():
        for i in range(1, 11):
            lat_col = f"Lat_test_{i}"
            lon_col = f"Lon_test_{i}"

            if lat_col in df.columns and lon_col in df.columns:
                if pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                    new_row = {
                        "Original_Row": idx + 1,
                        "Trial": i,
                        "Lat_test": row[lat_col],
                        "Lon_test": row[lon_col],
                    }

                    if "Point-ID" in df.columns:
                        new_row["Point_ID"] = row["Point-ID"]

                    if "Path-Points" in df.columns:
                        new_row["Path_Points"] = row["Path-Points"]

                    long_rows.append(new_row)

    new_df = pd.DataFrame(long_rows)

    if len(new_df) > 0:
        new_df, swapped_count = fix_swapped_coordinates(new_df)
        st.info(f"Coordinate check completed. Swapped coordinate rows corrected: {swapped_count}.")

    return new_df


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


def train_models_on_all_data(training_data):
    X = training_data[["Lat_test", "Lon_test"]]
    y_lat = training_data["Delta_Lat"]
    y_lon = training_data["Delta_Lon"]

    trained_models = {}

    for model_name, (lat_model, lon_model) in get_models().items():
        lat_model.fit(X, y_lat)
        lon_model.fit(X, y_lon)

        trained_models[model_name] = {
            "lat_model": lat_model,
            "lon_model": lon_model
        }

    return trained_models


def train_and_evaluate_models(training_data):
    X = training_data[["Lat_test", "Lon_test"]]
    y_lat = training_data["Delta_Lat"]
    y_lon = training_data["Delta_Lon"]

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
        training_data.index,
        test_size=0.2,
        random_state=42
    )

    test_data = training_data.loc[idx_test].copy()

    summaries = []
    calibrated_outputs = {}

    for model_name, (lat_model, lon_model) in get_models().items():

        lat_model.fit(X_train, y_lat_train)
        lon_model.fit(X_train, y_lon_train)

        pred_delta_lat = lat_model.predict(X_test)
        pred_delta_lon = lon_model.predict(X_test)

        result = test_data.copy()

        result["Pred_Delta_Lat"] = pred_delta_lat
        result["Pred_Delta_Lon"] = pred_delta_lon

        result["Lat_calibrated"] = result["Lat_test"] + result["Pred_Delta_Lat"]
        result["Lon_calibrated"] = result["Lon_test"] + result["Pred_Delta_Lon"]

        result["Error_Before_m"] = haversine(
            result["Lat_test"],
            result["Lon_test"],
            result["Lat_true"],
            result["Lon_true"]
        )

        result["Error_After_m"] = haversine(
            result["Lat_calibrated"],
            result["Lon_calibrated"],
            result["Lat_true"],
            result["Lon_true"]
        )

        result["Improvement_m"] = result["Error_Before_m"] - result["Error_After_m"]

        calibrated_outputs[model_name] = result

        summaries.append({
            "Model": model_name,
            "Mean Error": round(result["Error_After_m"].mean(), 2),
            "Median Error": round(result["Error_After_m"].median(), 2),
            "P90 Reliability": round(np.percentile(result["Error_After_m"], 90), 2),
            "Std Dev": round(result["Error_After_m"].std(), 2),
            "Mean Error Before": round(result["Error_Before_m"].mean(), 2),
            "Improvement (%)": round(
                (
                    result["Error_Before_m"].mean()
                    - result["Error_After_m"].mean()
                )
                / result["Error_Before_m"].mean()
                * 100,
                2
            )
        })

    comparison_df = pd.DataFrame(summaries)

    return comparison_df, calibrated_outputs


def set_equal_zoom(ax, lon_values, lat_values, padding_ratio=0.15):
    lon_values = pd.Series(lon_values).dropna()
    lat_values = pd.Series(lat_values).dropna()

    lon_min, lon_max = lon_values.min(), lon_values.max()
    lat_min, lat_max = lat_values.min(), lat_values.max()

    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    if lon_range == 0:
        lon_range = 0.0001
    if lat_range == 0:
        lat_range = 0.0001

    lon_pad = lon_range * padding_ratio
    lat_pad = lat_range * padding_ratio

    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)


def plot_points(df_model, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df_model["Lon_true"],
        df_model["Lat_true"],
        label="Ground-Truth Points",
        marker="x",
        s=75
    )

    ax.scatter(
        df_model["Lon_test"],
        df_model["Lat_test"],
        label="Original Test Points",
        alpha=0.35,
        s=40
    )

    ax.scatter(
        df_model["Lon_calibrated"],
        df_model["Lat_calibrated"],
        label="Calibrated Points",
        alpha=0.8,
        s=45
    )

    all_lons = pd.concat([
        df_model["Lon_true"],
        df_model["Lon_test"],
        df_model["Lon_calibrated"]
    ])

    all_lats = pd.concat([
        df_model["Lat_true"],
        df_model["Lat_test"],
        df_model["Lat_calibrated"]
    ])

    set_equal_zoom(ax, all_lons, all_lats)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def show_comparison_results(comparison_df, calibrated_outputs):
    st.header("Calibration Model Comparison")

    best_model = comparison_df.sort_values("Mean Error").iloc[0]["Model"]
    best_error = comparison_df.sort_values("Mean Error").iloc[0]["Mean Error"]
    best_p90 = comparison_df.sort_values("P90 Reliability").iloc[0]["P90 Reliability"]

    c1, c2, c3 = st.columns(3)

    c1.metric("Best Model", best_model)
    c2.metric("Lowest Mean Error", f"{best_error} m")
    c3.metric("Best P90 Reliability", f"{best_p90} m")

    display_df = comparison_df.copy()

    for col in ["Mean Error", "Median Error", "P90 Reliability", "Std Dev", "Mean Error Before"]:
        display_df[col] = display_df[col].astype(str) + " m"

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("CDF of Positioning Error")

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

    st.subheader("Boxplot of Residual Positional Errors")

    fig2, ax2 = plt.subplots(figsize=(9, 6))

    ax2.boxplot(
        [calibrated_outputs[m]["Error_After_m"] for m in calibrated_outputs.keys()],
        labels=list(calibrated_outputs.keys())
    )

    ax2.set_ylabel("Positioning Error After Calibration (meters)")
    ax2.set_title("Residual Error Distribution")
    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)

    st.header("Results by Calibration Method")

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
            row = comparison_df[comparison_df["Model"] == model_name].iloc[0]

            st.subheader(f"{model_name} Results")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Mean Error", f"{row['Mean Error']} m")
            m2.metric("Median Error", f"{row['Median Error']} m")
            m3.metric("P90 Reliability", f"{row['P90 Reliability']} m")
            m4.metric("Std Dev", f"{row['Std Dev']} m")

            fig3 = plot_points(df_model, f"{model_name} Calibrated Points")
            st.pyplot(fig3)

            st.dataframe(df_model, use_container_width=True, hide_index=True)

            csv = df_model.to_csv(index=False).encode("utf-8")

            st.download_button(
                label=f"Download {model_name} Results",
                data=csv,
                file_name=f"{model_name.replace(' ', '_')}_results.csv",
                mime="text/csv"
            )


# ======================================================
# MAIN TABS
# ======================================================

tab1, tab2 = st.tabs([
    "1. Calibrate with Reference Points",
    "2. Calibrate New Points Without Reference Locations"
])


# ======================================================
# TAB 1
# ======================================================

with tab1:

    st.markdown("""
    <div class="info-card">
        <h3>Calibrate with Reference Points</h3>
        <p>
        Use this tab to upload a calibration dataset that includes both observed indoor positioning points
        and known reference points. The app compares observed locations with reference locations,
        calculates spatial offsets, and evaluates four calibration methods.
        </p>
        <p>
        This option is recommended when users have collected ground-control points or other trusted
        reference locations for checking calibration accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Required Excel Structure for This Tab"):
        example = pd.DataFrame({
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

        st.write(
            "Each row should represent one reference point. "
            "`Lat_true` and `Lon_true` are the reference coordinates. "
            "`Lat_test_1`, `Lon_test_1` through `Lat_test_10`, `Lon_test_10` are repeated observed points."
        )

        st.dataframe(example, use_container_width=True, hide_index=True)

    uploaded_training_file = st.file_uploader(
        "Upload calibration Excel file with reference points",
        type=["xlsx", "csv"],
        key="tab1_training_file"
    )

    if uploaded_training_file is not None:

        try:
            if uploaded_training_file.name.endswith(".csv"):
                raw_training_df = pd.read_csv(uploaded_training_file)
            else:
                raw_training_df = pd.read_excel(uploaded_training_file, sheet_name="CM")

            training_data = reshape_training_excel(raw_training_df)

            if len(training_data) == 0:
                st.error("No valid calibration observations were found. Please check the file structure.")
            else:
                st.success(f"{len(training_data)} valid calibration observations detected.")

                if st.button("Run Four Calibration Methods", key="run_tab1"):
                    with st.spinner("Running SVM, Random Forest, XGBoost, and GPR..."):
                        comparison_df, calibrated_outputs = train_and_evaluate_models(training_data)

                    st.session_state["tab1_comparison_df"] = comparison_df
                    st.session_state["tab1_calibrated_outputs"] = calibrated_outputs

                    st.success("Calibration completed successfully.")

        except Exception as e:
            st.error("Could not read the uploaded file.")
            st.write(e)

    if "tab1_comparison_df" in st.session_state:
        show_comparison_results(
            st.session_state["tab1_comparison_df"],
            st.session_state["tab1_calibrated_outputs"]
        )


# ======================================================
# TAB 2
# ======================================================

with tab2:

    st.markdown("""
    <div class="info-card">
        <h3>Calibrate New Points Without Reference Locations</h3>
        <p>
        Use this tab to upload new observed indoor positioning points when reference locations are not available.
        The app applies a pre-trained calibration model based on the sample calibration dataset included with
        this application.
        </p>
        <p>
        Because the uploaded file does not include true/reference coordinates, the app cannot calculate actual
        positional error for these new points. The results should be interpreted as estimated corrected locations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-card">
        <b>Important note:</b><br>
        Calibration models are influenced by the environment where the training data were collected.
        Results may vary in buildings with different layouts, wall materials, signal conditions,
        device settings, or sources of interference. For higher accuracy, users are encouraged to
        collect a small number of reference points in the new location when possible.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Required Excel Structure for This Tab"):
        example2 = pd.DataFrame({
            "Point_ID": [1, 2, 3],
            "Lat_test": [42.407100, 42.407090, 42.407080],
            "Lon_test": [-71.119430, -71.119420, -71.119410]
        })

        st.write(
            "The new points file only needs observed coordinates. "
            "The easiest format is `Lat_test` and `Lon_test`. "
            "The app can also read a wide file with `Lat_test_1`, `Lon_test_1`, etc."
        )

        st.dataframe(example2, use_container_width=True, hide_index=True)

    uploaded_new_points = st.file_uploader(
        "Upload new observed points without reference locations",
        type=["xlsx", "csv"],
        key="tab2_new_points"
    )

    model_choice = st.selectbox(
        "Choose calibration method to apply",
        ["Random Forest", "GPR", "XGBoost", "SVM"],
        key="tab2_model_choice"
    )

    if uploaded_new_points is not None:

        try:
            if uploaded_new_points.name.endswith(".csv"):
                raw_new_df = pd.read_csv(uploaded_new_points)
            else:
                raw_new_df = pd.read_excel(uploaded_new_points)

            new_points = reshape_new_points_file(raw_new_df)

            if len(new_points) == 0 or "Lat_test" not in new_points.columns or "Lon_test" not in new_points.columns:
                st.error("No valid new points were detected. Please include `Lat_test` and `Lon_test`.")
            else:
                st.success(f"{len(new_points)} new observed points detected.")

                if st.button("Calibrate New Points", key="run_tab2"):

                    with st.spinner("Loading calibration reference file and training model..."):
                        github_df = pd.read_excel(GITHUB_CALIBRATION_FILE, sheet_name="CM")
                        github_training_data = reshape_training_excel(github_df)

                        if len(github_training_data) == 0:
                            st.error("No valid training data was found in the built-in calibration file.")
                            st.stop()

                        trained_models = train_models_on_all_data(github_training_data)

                    selected_lat_model = trained_models[model_choice]["lat_model"]
                    selected_lon_model = trained_models[model_choice]["lon_model"]

                    X_new = new_points[["Lat_test", "Lon_test"]]

                    pred_delta_lat = selected_lat_model.predict(X_new)
                    pred_delta_lon = selected_lon_model.predict(X_new)

                    output = new_points.copy()

                    output["Pred_Delta_Lat"] = pred_delta_lat
                    output["Pred_Delta_Lon"] = pred_delta_lon
                    output["Lat_calibrated"] = output["Lat_test"] + output["Pred_Delta_Lat"]
                    output["Lon_calibrated"] = output["Lon_test"] + output["Pred_Delta_Lon"]
                    output["Calibration_Method"] = model_choice
                    output["Calibration_Type"] = "Estimated calibration without reference locations"

                    st.session_state["tab2_output"] = output

                    st.success("New points were calibrated successfully.")

        except Exception as e:
            st.error("Could not process the file or calibration reference dataset.")
            st.write(e)

    if "tab2_output" in st.session_state:

        output = st.session_state["tab2_output"]

        st.subheader("Estimated Calibrated Points")

        st.dataframe(output, use_container_width=True, hide_index=True)

        st.subheader("Original vs Estimated Calibrated Points")

        fig4, ax4 = plt.subplots(figsize=(8, 6))

        ax4.scatter(
            output["Lon_test"],
            output["Lat_test"],
            label="Original Points",
            alpha=0.45,
            s=45
        )

        ax4.scatter(
            output["Lon_calibrated"],
            output["Lat_calibrated"],
            label="Estimated Calibrated Points",
            alpha=0.8,
            s=45
        )

        all_lons = pd.concat([output["Lon_test"], output["Lon_calibrated"]])
        all_lats = pd.concat([output["Lat_test"], output["Lat_calibrated"]])

        set_equal_zoom(ax4, all_lons, all_lats)

        ax4.set_xlabel("Longitude")
        ax4.set_ylabel("Latitude")
        ax4.set_title(f"Estimated Calibration Using {output['Calibration_Method'].iloc[0]}")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        st.pyplot(fig4)

        csv = output.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Estimated Calibrated Points",
            data=csv,
            file_name="estimated_calibrated_points_without_reference_locations.csv",
            mime="text/csv"
        )
