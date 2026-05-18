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


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Indoor Positioning Calibration App",
    layout="wide"
)

# ======================================================
# IMPORTANT
# Replace with your REAL raw github excel link
# ======================================================

GITHUB_CALIBRATION_FILE = (
    "https://raw.githubusercontent.com/YOUR_USERNAME/"
    "YOUR_REPOSITORY/main/Callibration_Model.xlsx"
)

# ======================================================
# STYLE
# ======================================================

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
}

.info-card {
    background-color: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0px 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 20px;
}

.warning-card {
    background-color: #fff8e6;
    padding: 20px;
    border-left: 5px solid #f0a500;
    border-radius: 14px;
    margin-bottom: 18px;
}

.method-card {
    background-color: #f8fbff;
    padding: 20px;
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

# ======================================================
# HEADER
# ======================================================

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


# ======================================================
# FIX SWAPPED COORDINATES
# ======================================================

def fix_swapped_coordinates(df, lat_col="Lat_test", lon_col="Lon_test"):

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


# ======================================================
# REMOVE EXTREME INVALID VALUES
# ======================================================

def remove_extreme_outliers(df):

    df = df.copy()

    valid_mask = (
        df["Lat_test"].between(-90, 90) &
        df["Lon_test"].between(-180, 180)
    )

    if "Lat_true" in df.columns:
        valid_mask &= df["Lat_true"].between(-90, 90)

    if "Lon_true" in df.columns:
        valid_mask &= df["Lon_true"].between(-180, 180)

    removed_count = len(df) - valid_mask.sum()

    return df[valid_mask].copy(), removed_count


# ======================================================
# RESHAPE TRAINING FILE
# ======================================================

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

        long_df["Delta_Lat"] = (
            long_df["Lat_true"] - long_df["Lat_test"]
        )

        long_df["Delta_Lon"] = (
            long_df["Lon_true"] - long_df["Lon_test"]
        )

        st.info(
            f"Coordinate check completed. "
            f"Swapped rows corrected: {swapped_count}. "
            f"Invalid rows removed: {removed_count}."
        )

    return long_df


# ======================================================
# RESHAPE NEW POINTS
# ======================================================

def reshape_new_points_file(df):

    if "Lat_test" in df.columns and "Lon_test" in df.columns:

        new_df = df.copy()

        new_df, swapped_count = fix_swapped_coordinates(new_df)

        new_df, removed_count = remove_extreme_outliers(new_df)

        st.info(
            f"Coordinate check completed. "
            f"Swapped rows corrected: {swapped_count}. "
            f"Invalid rows removed: {removed_count}."
        )

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
                        "Lon_test": row[lon_col]
                    }

                    if "Point-ID" in df.columns:
                        new_row["Point_ID"] = row["Point-ID"]

                    if "Path-Points" in df.columns:
                        new_row["Path_Points"] = row["Path-Points"]

                    long_rows.append(new_row)

    new_df = pd.DataFrame(long_rows)

    if len(new_df) > 0:

        new_df, swapped_count = fix_swapped_coordinates(new_df)

        new_df, removed_count = remove_extreme_outliers(new_df)

        st.info(
            f"Coordinate check completed. "
            f"Swapped rows corrected: {swapped_count}. "
            f"Invalid rows removed: {removed_count}."
        )

    return new_df


# ======================================================
# MODELS
# ======================================================

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


# ======================================================
# TRAIN ALL DATA
# ======================================================

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


# ======================================================
# TRAIN AND EVALUATE
# ======================================================

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

        result["Lat_calibrated"] = (
            result["Lat_test"] + result["Pred_Delta_Lat"]
        )

        result["Lon_calibrated"] = (
            result["Lon_test"] + result["Pred_Delta_Lon"]
        )

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

        result["Improvement_m"] = (
            result["Error_Before_m"] -
            result["Error_After_m"]
        )

        calibrated_outputs[model_name] = result

        summaries.append({
            "Model": model_name,
            "Mean Error": round(
                result["Error_After_m"].mean(),
                2
            ),
            "Median Error": round(
                result["Error_After_m"].median(),
                2
            ),
            "P90 Reliability": round(
                np.percentile(
                    result["Error_After_m"],
                    90
                ),
                2
            ),
            "Std Dev": round(
                result["Error_After_m"].std(),
                2
            ),
            "Mean Error Before": round(
                result["Error_Before_m"].mean(),
                2
            ),
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


# ======================================================
# PLOT LIMIT
# ======================================================

def get_plot_limit(calibrated_outputs, percentile=95):

    all_errors = []

    for _, df_model in calibrated_outputs.items():

        errors = df_model["Error_After_m"].dropna()

        all_errors.extend(errors.tolist())

    if len(all_errors) == 0:
        return 50

    return np.percentile(all_errors, percentile)


# ======================================================
# POINT MAP
# ======================================================

def plot_points(df_model, title):

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        df_model["Lon_true"],
        df_model["Lat_true"],
        label="Reference Points",
        marker="x",
        s=80
    )

    ax.scatter(
        df_model["Lon_test"],
        df_model["Lat_test"],
        label="Original Test Points",
        alpha=0.45,
        s=45
    )

    ax.scatter(
        df_model["Lon_calibrated"],
        df_model["Lat_calibrated"],
        label="Calibrated Points",
        alpha=0.75,
        s=45
    )

    # Zoom based only on true + test points
    # prevents one crazy outlier from ruining scale

    zoom_lons = pd.concat([
        df_model["Lon_true"],
        df_model["Lon_test"]
    ]).dropna()

    zoom_lats = pd.concat([
        df_model["Lat_true"],
        df_model["Lat_test"]
    ]).dropna()

    lon_min, lon_max = zoom_lons.min(), zoom_lons.max()
    lat_min, lat_max = zoom_lats.min(), zoom_lats.max()

    lon_pad = max(
        (lon_max - lon_min) * 0.2,
        0.0005
    )

    lat_pad = max(
        (lat_max - lat_min) * 0.2,
        0.0005
    )

    ax.set_xlim(
        lon_min - lon_pad,
        lon_max + lon_pad
    )

    ax.set_ylim(
        lat_min - lat_pad,
        lat_max + lat_pad
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


# ======================================================
# SHOW RESULTS
# ======================================================

def show_comparison_results(
    comparison_df,
    calibrated_outputs
):

    st.header("Calibration Model Comparison")

    best_model = comparison_df.sort_values(
        "Mean Error"
    ).iloc[0]["Model"]

    best_error = comparison_df.sort_values(
        "Mean Error"
    ).iloc[0]["Mean Error"]

    best_p90 = comparison_df.sort_values(
        "P90 Reliability"
    ).iloc[0]["P90 Reliability"]

    c1, c2, c3 = st.columns(3)

    c1.metric("Best Model", best_model)
    c2.metric("Lowest Mean Error", f"{best_error} m")
    c3.metric("Best P90 Reliability", f"{best_p90} m")

    # ==================================================
    # SUMMARY TABLE
    # ==================================================

    st.subheader("Model Accuracy Summary")

    display_df = comparison_df.copy()

    for col in [
        "Mean Error",
        "Median Error",
        "P90 Reliability",
        "Std Dev",
        "Mean Error Before"
    ]:
        display_df[col] = (
            display_df[col].astype(str) + " m"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # ==================================================
    # OUTLIER CHECK
    # ==================================================

    st.subheader("Outlier Check")

    outlier_summary = []

    for model_name, df_model in calibrated_outputs.items():

        errors = df_model["Error_After_m"].dropna()

        outlier_summary.append({
            "Model": model_name,
            "Minimum Error (m)": round(errors.min(), 2),
            "Median Error (m)": round(errors.median(), 2),
            "P90 Error (m)": round(np.percentile(errors, 90), 2),
            "P95 Error (m)": round(np.percentile(errors, 95), 2),
            "Maximum Error (m)": round(errors.max(), 2)
        })

    outlier_df = pd.DataFrame(outlier_summary)

    st.dataframe(
        outlier_df,
        use_container_width=True,
        hide_index=True
    )

    plot_limit = get_plot_limit(
        calibrated_outputs,
        percentile=95
    )

    # ==================================================
    # CDF
    # ==================================================

    st.subheader("CDF of Positioning Error")

    st.write(
        "This chart compares the cumulative error distribution "
        "across calibration methods. "
        "The display is limited to the 95th percentile "
        "to prevent extreme outliers from hiding the main pattern."
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, df_model in calibrated_outputs.items():

        errors = np.sort(
            df_model["Error_After_m"].dropna()
        )

        cdf = (
            np.arange(1, len(errors) + 1)
            / len(errors)
        )

        ax.plot(
            errors,
            cdf,
            label=model_name,
            linewidth=2
        )

    ax.axhline(
        y=0.9,
        linestyle="--",
        label="P90 Reliability"
    )

    ax.set_xlim(0, plot_limit)

    ax.set_xlabel(
        "Positioning Error After Calibration (meters)"
    )

    ax.set_ylabel("Cumulative Probability")

    ax.set_title(
        "CDF of Positioning Error by Calibration Model"
    )

    ax.legend()

    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

    # ==================================================
    # BOXPLOT
    # ==================================================

    st.subheader("Residual Error Distribution")

    st.write(
        "This boxplot compares residual positioning errors "
        "across calibration methods. "
        "Extreme outliers are hidden from the visualization "
        "but still reported in the outlier table above."
    )

    fig2, ax2 = plt.subplots(figsize=(9, 6))

    boxplot_data = []
    labels = []

    for model_name, df_model in calibrated_outputs.items():

        errors = df_model["Error_After_m"].dropna()

        errors_for_plot = errors[
            errors <= plot_limit
        ]

        boxplot_data.append(errors_for_plot)

        labels.append(model_name)

    ax2.boxplot(
        boxplot_data,
        labels=labels,
        showfliers=False
    )

    ax2.set_ylim(0, plot_limit)

    ax2.set_ylabel(
        "Positioning Error After Calibration (meters)"
    )

    ax2.set_title(
        "Residual Error Distribution by Calibration Model"
    )

    ax2.grid(True, alpha=0.3)

    st.pyplot(fig2)

    # ==================================================
    # METHOD TABS
    # ==================================================

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

    descriptions = {

        "Random Forest":
        (
            "Random Forest uses many decision trees "
            "to learn nonlinear spatial correction patterns."
        ),

        "GPR":
        (
            "Gaussian Process Regression estimates smooth "
            "probabilistic spatial correction surfaces."
        ),

        "XGBoost":
        (
            "XGBoost applies sequential gradient boosting "
            "to improve spatial error prediction."
        ),

        "SVM":
        (
            "Support Vector Machine regression uses "
            "kernel-based nonlinear calibration."
        )
    }

    for model_name, tab in tabs.items():

        with tab:

            df_model = calibrated_outputs[model_name]

            row = comparison_df[
                comparison_df["Model"] == model_name
            ].iloc[0]

            st.subheader(f"{model_name} Results")

            st.markdown(f"""
            <div class="method-card">
                {descriptions[model_name]}
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Mean Error",
                f"{row['Mean Error']} m"
            )

            m2.metric(
                "Median Error",
                f"{row['Median Error']} m"
            )

            m3.metric(
                "P90 Reliability",
                f"{row['P90 Reliability']} m"
            )

            m4.metric(
                "Std Dev",
                f"{row['Std Dev']} m"
            )

            if (
                df_model["Error_After_m"].max()
                > plot_limit
            ):
                st.warning(
                    f"This model contains extreme outliers. "
                    f"Maximum error: "
                    f"{round(df_model['Error_After_m'].max(), 2)} m."
                )

            fig3 = plot_points(
                df_model,
                f"{model_name} Calibrated Points"
            )

            st.pyplot(fig3)

            st.dataframe(
                df_model,
                use_container_width=True,
                hide_index=True
            )

            csv = df_model.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label=f"Download {model_name} Results",
                data=csv,
                file_name=(
                    f"{model_name.replace(' ', '_')}"
                    "_results.csv"
                ),
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
        Upload a calibration dataset that contains both
        observed indoor positioning points and known
        reference locations.
        </p>

        <p>
        The app compares the observed locations with the
        reference coordinates, calculates spatial offsets,
        and evaluates four regression-based calibration methods:
        Random Forest, GPR, XGBoost, and SVM.
        </p>

        <p>
        The results include error metrics,
        spatial correction maps,
        cumulative error distributions,
        and residual error comparison plots.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander(
        "Required Excel Structure"
    ):

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

            "Lon_test_10": [-71.119410]
        })

        st.write(
            "Each row should represent one reference point. "
            "The app expects true coordinates "
            "(`Lat_true`, `Lon_true`) and repeated "
            "observed coordinates "
            "(`Lat_test_1`, `Lon_test_1`, etc.)."
        )

        st.dataframe(
            example,
            use_container_width=True,
            hide_index=True
        )

    uploaded_training_file = st.file_uploader(
        "Upload calibration Excel file",
        type=["xlsx", "csv"],
        key="tab1_training_file"
    )

    if uploaded_training_file is not None:

        try:

            if uploaded_training_file.name.endswith(".csv"):

                raw_training_df = pd.read_csv(
                    uploaded_training_file
                )

            else:

                raw_training_df = pd.read_excel(
                    uploaded_training_file,
                    sheet_name="CM"
                )

            training_data = reshape_training_excel(
                raw_training_df
            )

            if len(training_data) == 0:

                st.error(
                    "No valid calibration observations "
                    "were found."
                )

            else:

                st.success(
                    f"{len(training_data)} valid calibration "
                    f"observations detected."
                )

                if st.button(
                    "Run Four Calibration Methods",
                    key="run_tab1"
                ):

                    with st.spinner(
                        "Running calibration models..."
                    ):

                        (
                            comparison_df,
                            calibrated_outputs

                        ) = train_and_evaluate_models(
                            training_data
                        )

                    st.session_state[
                        "tab1_comparison_df"
                    ] = comparison_df

                    st.session_state[
                        "tab1_calibrated_outputs"
                    ] = calibrated_outputs

                    st.success(
                        "Calibration completed successfully."
                    )

        except Exception as e:

            st.error(
                "Could not read the uploaded file."
            )

            st.write(e)

    if "tab1_comparison_df" in st.session_state:

        show_comparison_results(
            st.session_state[
                "tab1_comparison_df"
            ],
            st.session_state[
                "tab1_calibrated_outputs"
            ]
        )

# ======================================================
# TAB 2
# ======================================================

with tab2:

    st.markdown("""
    <div class="info-card">
        <h3>
        Calibrate New Points Without Reference Locations
        </h3>

        <p>
        Upload new observed indoor positioning points
        when reference locations are not available.
        </p>

        <p>
        The app uses the built-in calibration dataset
        stored in the GitHub repository as a training reference,
        trains the selected calibration model,
        and estimates corrected locations for the new points.
        </p>

        <p>
        Because the uploaded points do not include
        true coordinates,
        the app cannot calculate real positioning error
        for these new observations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-card">
        <b>Important note:</b><br>

        Calibration performance depends on the environment
        where the training data were collected.

        Results may vary across buildings with different
        layouts, wall materials, signal conditions,
        device settings, and interference patterns.

        Users are encouraged to collect at least a few
        reference points in the new environment whenever possible.
    </div>
    """, unsafe_allow_html=True)

    with st.expander(
        "Required Excel Structure"
    ):

        example2 = pd.DataFrame({

            "Point_ID": [1, 2, 3],

            "Lat_test": [
                42.407100,
                42.407090,
                42.407080
            ],

            "Lon_test": [
                -71.119430,
                -71.119420,
                -71.119410
            ]
        })

        st.write(
            "The uploaded file only needs observed "
            "coordinates (`Lat_test`, `Lon_test`). "
            "The app can also read wide-format files "
            "with `Lat_test_1`, `Lon_test_1`, etc."
        )

        st.dataframe(
            example2,
            use_container_width=True,
            hide_index=True
        )

    uploaded_new_points = st.file_uploader(
        "Upload new observed points",
        type=["xlsx", "csv"],
        key="tab2_new_points"
    )

    model_choice = st.selectbox(
        "Choose calibration method",
        [
            "Random Forest",
            "GPR",
            "XGBoost",
            "SVM"
        ],
        key="tab2_model_choice"
    )

    if uploaded_new_points is not None:

        try:

            if uploaded_new_points.name.endswith(".csv"):

                raw_new_df = pd.read_csv(
                    uploaded_new_points
                )

            else:

                raw_new_df = pd.read_excel(
                    uploaded_new_points
                )

            new_points = reshape_new_points_file(
                raw_new_df
            )

            if (
                len(new_points) == 0
                or "Lat_test" not in new_points.columns
                or "Lon_test" not in new_points.columns
            ):

                st.error(
                    "No valid points were detected."
                )

            else:

                st.success(
                    f"{len(new_points)} valid "
                    f"observed points detected."
                )

                if st.button(
                    "Calibrate New Points",
                    key="run_tab2"
                ):

                    with st.spinner(
                        "Loading reference calibration file..."
                    ):

                        github_df = pd.read_excel(
                            GITHUB_CALIBRATION_FILE,
                            sheet_name="CM"
                        )

                        github_training_data = (
                            reshape_training_excel(
                                github_df
                            )
                        )

                        if len(github_training_data) == 0:

                            st.error(
                                "No valid training data "
                                "was found in the "
                                "GitHub calibration file."
                            )

                            st.stop()

                        trained_models = (
                            train_models_on_all_data(
                                github_training_data
                            )
                        )

                    selected_lat_model = (
                        trained_models[
                            model_choice
                        ]["lat_model"]
                    )

                    selected_lon_model = (
                        trained_models[
                            model_choice
                        ]["lon_model"]
                    )

                    X_new = new_points[
                        ["Lat_test", "Lon_test"]
                    ]

                    pred_delta_lat = (
                        selected_lat_model.predict(
                            X_new
                        )
                    )

                    pred_delta_lon = (
                        selected_lon_model.predict(
                            X_new
                        )
                    )

                    output = new_points.copy()

                    output["Pred_Delta_Lat"] = (
                        pred_delta_lat
                    )

                    output["Pred_Delta_Lon"] = (
                        pred_delta_lon
                    )

                    output["Lat_calibrated"] = (
                        output["Lat_test"]
                        + output["Pred_Delta_Lat"]
                    )

                    output["Lon_calibrated"] = (
                        output["Lon_test"]
                        + output["Pred_Delta_Lon"]
                    )

                    output["Calibration_Method"] = (
                        model_choice
                    )

                    output["Calibration_Type"] = (
                        "Estimated calibration "
                        "without reference locations"
                    )

                    st.session_state[
                        "tab2_output"
                    ] = output

                    st.success(
                        "New points calibrated successfully."
                    )

        except Exception as e:

            st.error(
                "Could not process the file."
            )

            st.write(e)

    if "tab2_output" in st.session_state:

        output = st.session_state["tab2_output"]

        st.subheader(
            "Estimated Calibrated Points"
        )

        st.dataframe(
            output,
            use_container_width=True,
            hide_index=True
        )

        st.subheader(
            "Original vs Estimated Calibrated Points"
        )

        fig4, ax4 = plt.subplots(
            figsize=(8, 6)
        )

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

        zoom_lons = pd.concat([
            output["Lon_test"],
            output["Lon_calibrated"]
        ]).dropna()

        zoom_lats = pd.concat([
            output["Lat_test"],
            output["Lat_calibrated"]
        ]).dropna()

        lon_min, lon_max = (
            zoom_lons.min(),
            zoom_lons.max()
        )

        lat_min, lat_max = (
            zoom_lats.min(),
            zoom_lats.max()
        )

        lon_pad = max(
            (lon_max - lon_min) * 0.2,
            0.0005
        )

        lat_pad = max(
            (lat_max - lat_min) * 0.2,
            0.0005
        )

        ax4.set_xlim(
            lon_min - lon_pad,
            lon_max + lon_pad
        )

        ax4.set_ylim(
            lat_min - lat_pad,
            lat_max + lat_pad
        )

        ax4.set_xlabel("Longitude")

        ax4.set_ylabel("Latitude")

        ax4.set_title(
            f"Estimated Calibration Using "
            f"{output['Calibration_Method'].iloc[0]}"
        )

        ax4.legend()

        ax4.grid(True, alpha=0.3)

        st.pyplot(fig4)

        csv = output.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Estimated Calibrated Points",
            data=csv,
            file_name=(
                "estimated_calibrated_points.csv"
            ),
            mime="text/csv"
        )
