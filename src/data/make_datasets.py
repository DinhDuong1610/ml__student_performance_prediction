import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

def main():
    raw_data_path = "data/raw/StudentScore.xls"
    processed_data_path = "data/processed"
    preprocessor_path = "models/preprocessor.joblib"

    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

    data = pd.read_csv(raw_data_path)

    result = data.describe()
    print(result)

    info = data.info()
    print(info)

    # profile = ProfileReport(data, title="Student score report", explorative=True)
    # profile.to_file("reports/report.html")

    target = "writing score"
    x = data.drop(target, axis=1)
    y = data[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    numeric_features = ["math score", "reading score"]
    ordinal_features = ["parental level of education", "gender", "lunch", "test preparation course"]
    nominal_features = ["race/ethnicity"]

    education_categories = ["some high school", "high school", "some college", "associate's degree",
                            "bachelor's degree", "master's degree"]
    gender_categories = ["male", "female"]
    lunch_categories = list(x_train["lunch"].unique())
    test_prep_categories = list(x_train["test preparation course"].unique())

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


    ord_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[education_categories, gender_categories, lunch_categories, test_prep_categories], handle_unknown='use_encoded_value',
                                   unknown_value=-1))
    ])

    nom_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("ord", ord_transformer, ordinal_features),
            ("nom", nom_transformer, nominal_features)
        ],
        remainder='passthrough'
    )

    X_train_processed = preprocessor.fit_transform(x_train)
    X_test_processed = preprocessor.transform(x_test)

    processed_cols = (
            numeric_features +
            ordinal_features +
            list(preprocessor.named_transformers_['nom'].named_steps['encoder'].get_feature_names_out(nominal_features))
    )

    x_train_processed = pd.DataFrame(X_train_processed, columns=processed_cols)
    x_test_processed = pd.DataFrame(X_test_processed, columns=processed_cols)

    print(x_train_processed.head())

    x_train_path = os.path.join(processed_data_path, "x_train.csv")
    x_test_path = os.path.join(processed_data_path, "x_test.csv")
    y_train_path = os.path.join(processed_data_path, "y_train.csv")
    y_test_path = os.path.join(processed_data_path, "y_test.csv")

    x_train_processed.to_csv(x_train_path, index=False)
    x_test_processed.to_csv(x_test_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    y_test.to_csv(y_test_path, index=False)


if __name__ == '__main__':
    main()