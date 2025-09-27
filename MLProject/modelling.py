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
    
    # Jangan set experiment name karena sudah diatur oleh mlflow run
    # experiment_name = "CI_CD_Credit_Scoring"
    # mlflow.set_experiment(experiment_name)

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
    # Menggunakan run yang sudah dibuat oleh mlflow run command
    print("Memulai training model...", file=sys.stderr)
    
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
    mlflow.log_metric("n_estimators", best_params['n_estimators'])
    mlflow.log_metric("max_depth", best_params['max_depth'])
    
    # Log classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metric("precision_class_0", report['0']['precision'])
    mlflow.log_metric("recall_class_0", report['0']['recall'])
    mlflow.log_metric("f1_score_class_0", report['0']['f1-score'])
    mlflow.log_metric("precision_class_1", report['1']['precision'])
    mlflow.log_metric("recall_class_1", report['1']['recall'])
    mlflow.log_metric("f1_score_class_1", report['1']['f1-score'])
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance as artifact
    feature_importance.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
    
    # Log model
    mlflow.sklearn.log_model(
        model, 
        "model",
        registered_model_name="CreditScoringModel"
    )
    
    print(f"Model berhasil dilatih dengan akurasi: {accuracy:.4f}", file=sys.stderr)
    print(f"Model disimpan sebagai artefak MLflow", file=sys.stderr)

    print("--- Training Model Selesai ---", file=sys.stderr)
    print(f"Model artifacts tersimpan di: mlruns/", file=sys.stderr)