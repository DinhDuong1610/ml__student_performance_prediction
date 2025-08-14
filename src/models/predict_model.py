import pandas as pd
import joblib
import os

def predict_score(new_student_data):
    model_path = "models/random_forest_optimized_v1.joblib"
    preprocessor_path = "models/preprocessor.joblib"

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("Lỗi: Không tìm thấy file model hoặc preprocessor.")
        return None

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        print(f"Lỗi khi tải file: {e}")
        return None

    processed_new_data = preprocessor.transform(new_student_data)

    predicted_scores = model.predict(processed_new_data)

    return predicted_scores

if __name__ == '__main__':
    sample_data = pd.DataFrame({
        'gender': ['female', 'male'],
        'race/ethnicity': ['group C', 'group A'],
        'parental level of education': ["master's degree", 'high school'],
        'lunch': ['standard', 'free/reduced'],
        'test preparation course': ['completed', 'none'],
        'math score': [85, 52],
        'reading score': [92, 55]
    })

    predictions = predict_score(sample_data)

    if predictions is not None:
        sample_data['predicted_writing_score'] = predictions
        print(sample_data[['gender', 'math score', 'reading score', 'predicted_writing_score']].round(2))