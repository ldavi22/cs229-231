import dagshub
import mlflow
from sklearn.metrics import root_mean_squared_error


def log(x_train, y_train, x_test, y_test, result, name):

    dagshub.init(repo_owner='ldavi22', repo_name='cs229-231', mlflow=True)

    mlflow.set_tracking_uri("https://dagshub.com/ldavi22/cs229-231.mlflow")
    mlflow.set_experiment('house-prices-linear-regression_')

    with mlflow.start_run(run_name=name):

        best_idx = result.best_index_
        train_rmse = -result.cv_results_['mean_train_score'][best_idx]
        val_rmse = -result.best_score_
        val_std = result.cv_results_['std_test_score'][best_idx]
        test_pred = result.best_estimator_.predict(x_test)
        test_rmse = root_mean_squared_error(y_test, test_pred)

        mlflow.log_param("model", "RandomForrestRegressor")
        mlflow.log_param("n_features", x_train.shape[1])
        mlflow.log_param("n_samples", x_train.shape[0])
        mlflow.log_param("feature_selection", name)

        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_std", val_std)
        mlflow.log_metric("overfit_gap", val_rmse - train_rmse)

        for key, value in result.best_params_.items():
            mlflow.log_param(key, value)