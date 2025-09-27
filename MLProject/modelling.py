import mlflow
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __name__ == "__main__":
    print("--- Memulai Training Model untuk CI/CD ---", file=sys.stderr)

    # Set MLflow tracking URI untuk local
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set experiment name
    experiment_name = "CI_CD_Credit_Scoring"
    mlflow.set_experiment(experiment_name)

    # 1. Muat Dataset
    try:
        data = pd.read_csv("dataset_preprocessing/creditcard_processed.csv")
        X = data.drop('Class', axis=1)
        y = data['Class']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Dataset berhasil dimuat dan dibagi.", file=sys.stderr)
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}", file=sys.stderr)
    except FileNotFoundError:
        print("Error: Dataset 'creditcard_processed.csv' tidak ditemukan.", file=sys.stderr)
        sys.exit(1)

    # 2. Parameter model yang sudah dioptimasi
    best_params = {
        'n_estimators': 100,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
    
    # 3. Training Model dengan MLflow tracking
    with mlflow.start_run() as run:
        print(f"Memulai training dalam run ID: {run.info.run_id}", file=sys.stderr)
        
        # Log parameters
        mlflow.log_params(best_params)
        
        # Training model
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))
        
        print(f"Akurasi model: {accuracy:.4f}", file=sys.stderr)
        
        # Log classification report sebagai artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("precision_class_0", report['0']['precision'])
        mlflow.log_metric("recall_class_0", report['0']['recall'])
        mlflow.log_metric("f1_class_0", report['0']['f1-score'])
        mlflow.log_metric("precision_class_1", report['1']['precision'])
        mlflow.log_metric("recall_class_1", report['1']['recall'])
        mlflow.log_metric("f1_class_1", report['1']['f1-score'])
        
        # Log model sebagai artifact
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="CreditScoringModel"
        )
        print("Model berhasil di-log sebagai artifact.", file=sys.stderr)
        
        # Log feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
            mlflow.log_metric(f"feature_importance_{feature}", importance)

    print("--- Training Model Selesai ---", file=sys.stderr)
    print(f"Model artifacts tersimpan di: mlruns/", file=sys.stderr)