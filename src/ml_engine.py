
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_validate,
    GridSearchCV
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class MLModelTrainer:

    def __init__(self, dataframe, target_column):

        self.df = dataframe.copy()
        self.target_column = target_column

        if target_column not in self.df.columns:
            raise ValueError("Target column not found!")

        if not pd.api.types.is_numeric_dtype(self.df[target_column]):
            raise ValueError("Target must be numeric!")

    def build_preprocessor(self, X):

        numeric_features = X.select_dtypes(
            include="number"
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            exclude="number"
        ).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features)
            ]
        )

        return preprocessor

    def train_and_evaluate(self):

        # Separate features and target
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Define models
        models = {
            "Linear Regression": LinearRegression(),

            "Random Forest": RandomForestRegressor(
                n_estimators=150,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        }

        results = {}
        # Cross-validation configuration
        cv_strategy = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
        metric_rows = []
        fitted_models = {}
        feature_importance = {}
        best_params = {}

        # Train and evaluate

        for name, model in models.items():

            pipeline = Pipeline([
                ("preprocessor", self.build_preprocessor(X_train)),
                ("model", model)
            ])

            # =====================================
            # HYPERPARAMETER TUNING
            # =====================================

            if name == "Random Forest":

                param_grid = {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [12, 18],
                    "model__min_samples_leaf": [1, 2]
                }

                grid_search = GridSearchCV(
                    estimator=pipeline,
                    param_grid=param_grid,
                    scoring={
                        "r2": "r2",
                        "rmse": "neg_root_mean_squared_error"
                    },
                    refit="r2",
                    cv=cv_strategy,
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit(X_train, y_train)

                pipeline = grid_search.best_estimator_

                best_params[name] = grid_search.best_params_

                best_index = grid_search.best_index_

                cv_r2_mean = grid_search.cv_results_[
                    "mean_test_r2"
                ][best_index]

                cv_r2_std = grid_search.cv_results_[
                    "std_test_r2"
                ][best_index]

                cv_rmse_mean = -grid_search.cv_results_[
                    "mean_test_rmse"
                ][best_index]

            else:

                # Cross-validation for Linear Regression
                cv_scores = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv_strategy,
                    scoring={
                        "r2": "r2",
                        "rmse": "neg_root_mean_squared_error"
                    },
                    n_jobs=1
                )

                cv_r2_mean = cv_scores["test_r2"].mean()
                cv_r2_std = cv_scores["test_r2"].std()
                cv_rmse_mean = -cv_scores["test_rmse"].mean()

                pipeline.fit(X_train, y_train)

            # =====================================
            # MODEL EVALUATION
            # =====================================

            predictions = pipeline.predict(X_test)
            train_predictions = pipeline.predict(X_train)

            train_r2 = r2_score(y_train, train_predictions)
            test_r2 = r2_score(y_test, predictions)

            generalization_gap = train_r2 - test_r2

            mae = mean_absolute_error(y_test, predictions)

            rmse = mean_squared_error(
                y_test,
                predictions
            ) ** 0.5

            metric_rows.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "Train R2": train_r2,
                "Test R2": test_r2,
                "Generalization Gap": generalization_gap,
                "CV R2 Mean": cv_r2_mean,
                "CV R2 Std": cv_r2_std,
                "CV RMSE Mean": cv_rmse_mean
            })

            # =====================================
            # SAVE PREDICTIONS AND MODELS
            # =====================================

            results[name] = pd.DataFrame({
                "Actual": y_test,
                "Predicted": predictions
            })

            fitted_models[name] = pipeline

            # =====================================
            # FEATURE IMPORTANCE
            # =====================================

            fitted_model = pipeline.named_steps["model"]

            if hasattr(fitted_model, "feature_importances_"):
                feature_names = (
                    pipeline.named_steps["preprocessor"]
                    .get_feature_names_out()
                )

                importance = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": fitted_model.feature_importances_
                })

                feature_importance[name] = importance.sort_values(
                    by="Importance",
                    ascending=False
                )

        metrics = pd.DataFrame(metric_rows).set_index("Model")

        return {
            "metrics": metrics,
            "predictions": results,
            "models": fitted_models,
            "feature_importance": feature_importance,
            "best_params": best_params,
            "X_test": X_test.copy(),
            "y_test": y_test.copy()
        }
