import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import logging

# Load the processed data
X_train_vec, X_test_vec, y_train, y_test, vectorizer = pickle.load(open('data/processed_data.pkl', 'rb'))

# Initialize MLflow
mlflow.set_experiment("Sentiment Analysis")

with mlflow.start_run():
    # Train the model
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_text(report, "classification_report.txt")

    # Log the model
    mlflow.sklearn.log_model(model, "sentiment_model")

    # Log parameters (if any)
    mlflow.log_param("max_features", 5000)

    # Log artifacts like the vectorizer
    vectorizer_path = "vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    mlflow.log_artifact(vectorizer_path)

# Monitor model performance
logging.basicConfig(filename='model_monitor.log', level=logging.INFO)
logging.info(f'Accuracy: {accuracy}')
logging.info(f'Classification Report: \n{report}')