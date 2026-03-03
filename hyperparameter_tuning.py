import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    print("--- 1. Veri Yükleniyor ve Ön İşlem Yapılıyor ---")
    try:
        df = pd.read_csv('Loan_default.csv')
    except FileNotFoundError:
        print("Hata: 'Loan_default.csv' dosyası bulunamadı.")
        return

    # Gereksiz kolon
    if 'LoanID' in df.columns:
        df = df.drop('LoanID', axis=1)
        
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if 'Default' in categorical_cols:
        categorical_cols.remove('Default')
        
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    y = df_encoded['Default']
    X = df_encoded.drop('Default', axis=1)
    
    # Train / Test bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Ölçeklendirme (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- 2. Hiperparametre Optimizasyonu İçin Hazırlık ---")
    class_counts = y_train.value_counts()
    scale_pos_weight_val = class_counts[0] / class_counts[1]
    print(f"Sabit bırakılan scale_pos_weight: {scale_pos_weight_val:.2f}")

    # XGBoost modeli (Sabit parametreler)
    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight_val,
        random_state=42,
        eval_metric='logloss'
    )

    # Aranacak parametre ızgarası (Grid)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 300, 500],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # RandomizedSearchCV (GridSearchCV çok daha uzun süreceği için bunu tercih ediyoruz)
    print("\n--- 3. Optimizasyon Başlıyor (RandomizedSearchCV) ---")
    print("Not: Bu işlem donanım gücüne bağlı olarak uzun sürebilir. Lütfen bekleyin...")
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20, # Toplamda 20 farklı kombinasyon denenecek
        scoring='recall', # En önemli özellik! Recall'a göre optimize ediliyor.
        cv=5,      # 5 katlı çapraz doğrulama
        n_jobs=-1, # Tüm işlemci çekirdeklerini kullan
        random_state=42,
        verbose=2
    )

    # Optimizasyonu çalıştır
    random_search.fit(X_train_scaled, y_train)

    print("\n--- 4. Optimizasyon Sonuçları ---")
    best_model = random_search.best_estimator_
    print("En İyi Parametreler:")
    for param, value in random_search.best_params_.items():
        print(f"  - {param}: {value}")

    print("\n--- 5. Optimize Edilmiş Model ile Test Setinde Tahmin ---")
    y_pred = best_model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Yeni Accuracy : {accuracy:.4f}")
    print(f"Yeni Precision: {precision:.4f}")
    print(f"Yeni Recall   : {recall:.4f}")
    print(f"Yeni F1-Score : {f1:.4f}")

    print("\n--- 6. Optimize Modeli Kaydetme ---")
    joblib.dump(best_model, 'xgboost_optimized.pkl')
    print("✅ Optimize edilmiş model 'xgboost_optimized.pkl' olarak başarıyla kaydedildi!")

if __name__ == "__main__":
    main()
