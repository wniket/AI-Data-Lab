
import numpy as np
import pandas as pd
import pytest

from src.ml_engine import MLModelTrainer


@pytest.fixture
def sample_dataframe():

    rng = np.random.default_rng(42)
    n = 80

    area = rng.uniform(40, 200, n)
    bedrooms = rng.integers(1, 6, n)
    district = rng.choice(
        ["Center", "Suburb", "Coast"],
        n
    )

    district_effect = {
        "Center": 50000,
        "Suburb": 20000,
        "Coast": 80000
    }

    price = (
        30000
        + area * 1200
        + bedrooms * 15000
        + np.array([district_effect[x] for x in district])
        + rng.normal(0, 10000, n)
    )

    df = pd.DataFrame({
        "area": area,
        "bedrooms": bedrooms.astype(float),
        "district": district,
        "price": price
    })

    # Add a missing value
    df.loc[5, "bedrooms"] = np.nan

    return df


def test_missing_target_column(sample_dataframe):

    with pytest.raises(ValueError):
        MLModelTrainer(
            sample_dataframe,
            "unknown_target"
        )


def test_non_numeric_target(sample_dataframe):

    df = sample_dataframe.copy()
    df["category_target"] = "expensive"

    with pytest.raises(ValueError):
        MLModelTrainer(
            df,
            "category_target"
        )


def test_preprocessor_handles_missing_values(sample_dataframe):

    trainer = MLModelTrainer(
        sample_dataframe,
        "price"
    )

    X = sample_dataframe.drop(columns=["price"])

    preprocessor = trainer.build_preprocessor(X)

    transformed = preprocessor.fit_transform(X)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    assert transformed.shape[0] == len(X)
    assert np.isfinite(transformed).all()


def test_train_and_evaluate(sample_dataframe):

    trainer = MLModelTrainer(
        sample_dataframe,
        "price"
    )

    result = trainer.train_and_evaluate()

    # Check returned models
    assert "Linear Regression" in result["models"]
    assert "Random Forest" in result["models"]

    # Check metrics
    expected_metrics = [
        "MAE",
        "RMSE",
        "Train R2",
        "Test R2",
        "Generalization Gap",
        "CV R2 Mean",
        "CV R2 Std",
        "CV RMSE Mean"
    ]

    for metric in expected_metrics:
        assert metric in result["metrics"].columns

    # Check finite metric values
    assert np.isfinite(
        result["metrics"][expected_metrics].to_numpy()
    ).all()

    # Check predictions
    assert len(result["predictions"]["Random Forest"]) == 16

    # Check hyperparameter tuning
    assert "Random Forest" in result["best_params"]

    # Check feature importance
    assert "Random Forest" in result["feature_importance"]
