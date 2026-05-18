import streamlit as st
import pandas as pd
import numpy as np

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


st.set_page_config(page_title="Indoor Point Calibration App", layout="wide")

st.title("Indoor Path Tracking Calibration App")
st.write(
    "This app calibrates indoor tracking points using SVM, Random Forest, "
    "Gradient Boosting/XGBoost, and Gaussian Process Regression."
)


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


uploaded_file = st.file_uploader("Upload your calibration Excel file", type=["xlsx", "csv"])

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file, sheet_name="CM")

    st.subheader("Original Uploaded Data")
    st.dataframe(raw_df.head())

    data = reshape_calibration_excel(raw_df)

    st.subheader("Reshaped Calibration Data")
    st.write(f"Total calibration observations: {len(data)}")
    st.dataframe(data.head())

    if st.button("Run Four Calibration Models"):

        X = data[["Lat_test", "Lon_test"]]
        y_lat = data["Delta_Lat"]
        y_lon = data["Delta_Lon"]

        X_train, X_test, y_lat_train, y_lat_test, y_lon_train, y_lon_test, idx_train, idx_test = train_test_split(
            X,
            y_lat,
            y_lon,
            data.index,
            test_size=0.2,
            random_state=42
        )

        test_data = data.loc[idx_test].copy()

        models = {
            "SVM": (
                make_pipeline(
                    StandardScaler(),
                    SVR(kernel="rbf", C=10, gamma=0.5, epsilon=1e-5)
                ),
                make_pipeline(
                    StandardScaler(),
                    SVR(kernel="rbf", C=10, gamma=0.5, epsilon=1e-5)
                )
            ),

            "Random Forest": (
                RandomForestRegressor(n_estimators=400, random_state=42),
                RandomForestRegressor(n_estimators=400, random_state=42)
            ),

            "Gradient Boosting / XGBoost": (
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
            )
        }

        results = []
        calibrated_files = {}

        error_before = haversine(
            test_data["Lat_test"],
            test_data["Lon_test"],
            test_data["Lat_true"],
            test_data["Lon_true"]
        )

        for model_name, (lat_model, lon_model) in models.items():

            lat_model.fit(X_train, y_lat_train)
            lon_model.fit(X_train, y_lon_train)

            pred_delta_lat = lat_model.predict(X_test)
            pred_delta_lon = lon_model.predict(X_test)

            calibrated = test_data.copy()
            calibrated["Pred_Delta_Lat"] = pred_delta_lat
            calibrated["Pred_Delta_Lon"] = pred_delta_lon

            calibrated["Lat_calibrated"] = calibrated["Lat_test"] + calibrated["Pred_Delta_Lat"]
            calibrated["Lon_calibrated"] = calibrated["Lon_test"] + calibrated["Pred_Delta_Lon"]

            error_after = haversine(
                calibrated["Lat_calibrated"],
                calibrated["Lon_calibrated"],
                calibrated["Lat_true"],
                calibrated["Lon_true"]
            )

            calibrated["Error_Before_m"] = error_before
            calibrated["Error_After_m"] = error_after
            calibrated["Improvement_m"] = calibrated["Error_Before_m"] - calibrated["Error_After_m"]

            mean_error = np.mean(error_after)
            median_error = np.median(error_after)
            p90_error = np.percentile(error_after, 90)
            std_error = np.std(error_after)

            results.append({
                "Model": model_name,
                "Mean Error (m)": round(mean_error, 2),
                "Median Error (m)": round(median_error, 2),
                "P90 Reliability (m)": round(p90_error, 2),
                "Std Dev (m)": round(std_error, 2),
                "Mean Error Before (m)": round(np.mean(error_before), 2),
                "Improvement (%)": round(
                    ((np.mean(error_before) - mean_error) / np.mean(error_before)) * 100,
                    2
                )
            })

            calibrated_files[model_name] = calibrated

        results_df = pd.DataFrame(results).sort_values("Mean Error (m)")

        st.subheader("Model Performance Comparison")
        st.dataframe(results_df)

        best_model = results_df.iloc[0]["Model"]
        st.success(f"Best calibration model: {best_model}")

        st.subheader("Error Comparison Chart")
        chart_df = results_df.set_index("Model")[["Mean Error Before (m)", "Mean Error (m)"]]
        st.bar_chart(chart_df)

        st.subheader("View and Download Calibrated Points")

        selected_model = st.selectbox(
            "Select calibration method",
            list(calibrated_files.keys())
        )

        selected_df = calibrated_files[selected_model]
        st.dataframe(selected_df)

        csv = selected_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label=f"Download {selected_model} Calibrated Points",
            data=csv,
            file_name=f"{selected_model.replace(' ', '_')}_calibrated_points.csv",
            mime="text/csv"
        )

        all_results_csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Model Comparison Results",
            data=all_results_csv,
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )
