import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_model():
    processed_folder_path = "data/processed"
    model_path = "models/random_forest_optimized_v1.joblib"

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    try:
        x_train = pd.read_csv(os.path.join(processed_folder_path, "x_train.csv"))
        y_train = pd.read_csv(os.path.join(processed_folder_path, "y_train.csv")).values.ravel()
        x_test = pd.read_csv(os.path.join(processed_folder_path, "x_test.csv"))
        y_test = pd.read_csv(os.path.join(processed_folder_path, "y_test.csv")).values.ravel()
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file dữ liệu.")
        print(e)
        return
    rf_model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    joblib.dump(best_model, model_path)

if __name__ == '__main__':
    train_model()