
import shap


def explain_random_forest(pipeline, X, max_samples=200):

    if X.empty:
        raise ValueError("No data available for explanation.")

    # Select a representative sample
    X_sample = X.sample(
        n=min(max_samples, len(X)),
        random_state=42
    )

    # Extract trained components
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Apply the already fitted preprocessing
    transformed_data = preprocessor.transform(X_sample)

    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # Create SHAP explainer
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )

    raw_explanation = explainer(transformed_data)

    explanation = shap.Explanation(
        values=raw_explanation.values,
        base_values=raw_explanation.base_values,
        data=transformed_data,
        feature_names=feature_names
    )

    return explanation, X_sample
