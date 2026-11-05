

import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt

from src.explainability import explain_random_forest
from src.analysis import DataAnalyzer
from src.ml_engine import MLModelTrainer

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI DataLab",
    page_icon="🧠",
    layout="wide"
)


# =====================================
# HEADER
# =====================================

st.title("🧠 AI DataLab")
st.caption("Intelligent Data Analysis & Machine Learning Platform")

# Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose section",
    ["Dashboard", "Data Analysis", "Machine Learning"]
)


# =====================================
# DATA UPLOAD
# =====================================

st.sidebar.divider()
st.sidebar.subheader("Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"],
    key="dataset_upload"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["dataset"] = df
        st.session_state["dataset_name"] = uploaded_file.name
    except Exception as error:
        st.error(f"Could not load dataset: {error}")

df = st.session_state.get("dataset")

if df is None:
    st.info("Upload a CSV dataset to start working.")
    st.stop()

analyzer = DataAnalyzer(df)

# Dataset overview
overview = analyzer.get_overview()


# =====================================
# DASHBOARD
# =====================================

if page == "Dashboard":

    st.header("Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{overview['rows']:,}")
    col2.metric("Columns", overview["columns"])
    col3.metric("Missing Values", overview["missing"])
    col4.metric("Duplicates", overview["duplicates"])

    st.divider()

    st.subheader("Dataset Preview")

    st.caption(
        f"Currently analyzing: "
        f"{st.session_state.get('dataset_name', 'Dataset')}"
    )

    st.dataframe(
        df.head(10),
        width="stretch"
    )


# =====================================
# DATA ANALYSIS
# =====================================

elif page == "Data Analysis":

    st.header("Exploratory Data Analysis")

    # Statistical summary
    st.subheader("Statistical Summary")

    st.dataframe(
        analyzer.get_statistics(),
        width="stretch"
    )

    st.divider()

    # Missing values
    st.subheader("Missing Values Analysis")

    missing = analyzer.get_missing_values()

    if not missing.empty:

        missing_df = missing.rename(
            "Missing Values"
        ).to_frame()

        missing_df["Missing %"] = (
            missing_df["Missing Values"] / len(df) * 100
        ).round(2)

        st.dataframe(
            missing_df,
            width="stretch"
        )

    else:
        st.success("No missing values detected!")

    st.divider()

    # Correlation heatmap
    st.subheader("Correlation Matrix")

    correlation = analyzer.get_correlations()

    if not correlation.empty:

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Feature Correlation Heatmap"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # Distribution analysis
    st.divider()

    st.subheader("Distribution Analysis")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    selected_column = st.selectbox(
        "Select a numerical feature",
        numeric_columns
    )

    fig = px.histogram(
        df,
        x=selected_column,
        nbins=40,
        title=f"Distribution of {selected_column}"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =====================================
# MACHINE LEARNING
# =====================================

elif page == "Machine Learning":

    st.header("Machine Learning")
    st.caption("Train, compare and evaluate machine learning models.")

    target_options = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if not target_options:

        st.warning("No numerical target columns found.")

    else:

        default_index = (
            target_options.index("median_house_value")
            if "median_house_value" in target_options
            else len(target_options) - 1
        )

        target_column = st.selectbox(
            "Select target variable",
            target_options,
            index=default_index
        )

        st.write("Selected target:", target_column)

        if st.button("🚀 Train Models", type="primary"):

            try:
                with st.spinner("Training machine learning models..."):

                    trainer = MLModelTrainer(
                        dataframe=df,
                        target_column=target_column
                    )

                    ml_results = trainer.train_and_evaluate()

                    st.session_state["ml_results"] = ml_results
                    st.session_state["ml_target"] = target_column
                    st.session_state.pop("shap_explanation", None)
                    st.session_state.pop("shap_sample", None)

                st.success("Models trained successfully!")

            except Exception as error:

                st.error(f"Training failed: {error}")

        # Display results
        if (
                "ml_results" in st.session_state
                and st.session_state.get("ml_target") == target_column
        ):

            results = st.session_state["ml_results"]

            # Best Hyperparameters
            st.divider()
            st.subheader("Best Random Forest Parameters")

            best_params = results.get("best_params", {})
            rf_params = best_params.get("Random Forest")

            if rf_params:
                st.json(rf_params)
            else:
                st.warning("No hyperparameters found. Train models again.")

            st.divider()

            st.subheader("Model Performance")

            metrics = results["metrics"]

            st.dataframe(
                metrics.style.format({
                    "MAE": "{:,.2f}",
                    "RMSE": "{:,.2f}",
                    "Train R2": "{:.4f}",
                    "Test R2": "{:.4f}",
                    "Generalization Gap": "{:.4f}",
                    "CV R2 Mean": "{:.4f}",
                    "CV R2 Std": "{:.4f}",
                    "CV RMSE Mean": "{:,.2f}"
                }),
                width="stretch"
            )

            st.caption(
                "Cross-validation uses 5 folds on the training set. "
                "The test set remains untouched."
            )

            # R2 comparison
            st.subheader("Model Comparison")

            comparison_fig = px.bar(
                metrics.reset_index(),
                x="Model",
                y="Test R2",
                color="Model",
                title="Test R² Score Comparison",
                text="Test R2"
            )
            cv_fig = px.bar(
                metrics.reset_index(),
                x="Model",
                y="CV R2 Mean",
                color="Model",
                title="5-Fold Cross-Validation R²"
            )

            st.plotly_chart(
                cv_fig,
                width="stretch"
            )

            st.plotly_chart(
                comparison_fig,
                width="stretch"
            )

            # Prediction visualization
            st.subheader("Actual vs Predicted")

            selected_model = st.selectbox(
                "Choose model",
                list(results["predictions"].keys())
            )

            prediction_df = results["predictions"][selected_model]

            fig = px.scatter(
                prediction_df,
                x="Actual",
                y="Predicted",
                title=f"{selected_model}: Predictions",
                opacity=0.5
            )

            min_value = min(
                prediction_df["Actual"].min(),
                prediction_df["Predicted"].min()
            )

            max_value = max(
                prediction_df["Actual"].max(),
                prediction_df["Predicted"].max()
            )

            fig.add_shape(
                type="line",
                x0=min_value,
                y0=min_value,
                x1=max_value,
                y1=max_value,
                line=dict(
                    color="red",
                    dash="dash"
                )
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

            # Feature importance
            if "Random Forest" in results["feature_importance"]:
                st.subheader("Feature Importance")

                importance_df = (
                    results["feature_importance"]["Random Forest"]
                    .head(10)
                    .sort_values("Importance")
                )

                importance_fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 10 Important Features"
                )

                st.plotly_chart(
                    importance_fig,
                    width="stretch"
                )

                # =====================================
                # SHAP EXPLAINABILITY
                # =====================================

                st.divider()
                st.header("SHAP Explainability")

                st.write(
                    "Understand how individual features influence "
                    "machine learning predictions."
                )

                if st.button("🧠 Generate SHAP Explanations"):

                    try:
                        with st.spinner("Calculating SHAP values..."):

                            rf_pipeline = results["models"]["Random Forest"]

                            explanation, X_sample = explain_random_forest(
                                rf_pipeline,
                                results["X_test"],
                                max_samples=200
                            )

                            st.session_state["shap_explanation"] = explanation
                            st.session_state["shap_sample"] = X_sample

                        st.success("SHAP explanations generated!")

                    except Exception as error:
                        st.error(f"SHAP calculation failed: {error}")

                # Display SHAP results
                if "shap_explanation" in st.session_state:
                    explanation = st.session_state["shap_explanation"]
                    X_sample = st.session_state["shap_sample"]

                    # Global explanation
                    st.subheader("Global Feature Impact")

                    shap.plots.beeswarm(
                        explanation,
                        max_display=10,
                        show=False
                    )

                    st.pyplot(
                        plt.gcf(),
                        clear_figure=True
                    )

                    # Local explanation
                    st.subheader("Individual Prediction Explanation")

                    sample_index = st.selectbox(
                        "Select a test observation",
                        range(len(X_sample)),
                        format_func=lambda i: str(X_sample.index[i])
                    )

                    selected_row = X_sample.iloc[[sample_index]]

                    prediction = results["models"]["Random Forest"].predict(
                        selected_row
                    )[0]

                    actual = results["y_test"].loc[selected_row.index[0]]

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Predicted Value",
                        f"${prediction:,.0f}"
                    )

                    col2.metric(
                        "Actual Value",
                        f"${actual:,.0f}"
                    )

                    shap.plots.waterfall(
                        explanation[sample_index],
                        max_display=10,
                        show=False
                    )

                    st.pyplot(
                        plt.gcf(),
                        clear_figure=True
                    )

    st.write("Dataset shape:", df.shape)

    st.write("Available numerical features:")

    st.write(numeric_columns if "numeric_columns" in locals()
             else df.select_dtypes(include="number").columns.tolist())
