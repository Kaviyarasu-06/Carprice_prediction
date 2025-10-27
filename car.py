def run_models(models, X_train, y_train, X_test, y_test):
    import mlflow
    import mlflow.sklearn
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"\n{name}")

            # Fit model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute metrics
            n = X_test.shape[0]
            p = X_train.shape[1]

            r2 = r2_score(y_test, y_pred)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            # Print results
            print(f"R² Score: {r2:.4f}")
            print(f"Adjusted R² Score: {adj_r2:.4f}")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")

            # Log metrics to MLflow
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("adjusted_r2", adj_r2)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)

            # Optionally log model (uncomment if needed)
            # mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"Run completed. Run ID: {run.info.run_id}")
