import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def main():
    print("--- 1. Veri Yükleniyor ve Ön İşlem Yapılıyor ---")
    df = pd.read_csv('Loan_default.csv')
    
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
    
    # Feature isimlerini (kolonları) SHAP açıklanabilirliği için bir listeye alalım
    feature_names = X.columns.tolist()
    
    # Train / Test bölme
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Ölçeklendirme (Scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("--- 2. XGBoost Modeli Eğitiliyor ---")
    scale_pos_weight_val = y_train.value_counts()[0] / y_train.value_counts()[1]
    
    xgb_clf = XGBClassifier(scale_pos_weight=scale_pos_weight_val, random_state=42, n_jobs=-1, eval_metric='logloss')
    xgb_clf.fit(X_train_scaled, y_train)
    print("XGBoost eğitimi tamamlandı!")

    print("\n--- 3. Model ve Scaler Kaydediliyor (joblib) ---")
    # Dosyaları kaydetme işlemi (Web projesinde kullanabilmek için)
    joblib.dump(xgb_clf, 'xgboost_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Web uygulamasında formu modelin anlayacağı formata (One-Hot) çevirirken
    # oluşturduğumuz kolonların birebir aynı sırasını bilmemiz gerekir. Bunu da kaydedelim!
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("✅ Dosyalar başarıyla kaydedildi: 'xgboost_model.pkl', 'scaler.pkl', 'feature_names.pkl'")
    
    print("\n--- 4. SHAP ile Model Açıklanabilirliği (Explainability) ---")
    # XGBoost'u açıklamak için TreeExplainer kullanıyoruz
    explainer = shap.TreeExplainer(xgb_clf)
    
    # Tüm test setiyle (51000 satır) SHAP matrisi çizmek bilgisayarınızı çok yorabilir.
    # O yüzden test setinden hem performansı hızlandıracak hem de yeterince temsil gücü olacak rasgele 1000 örnek alıyoruz:
    X_test_sample = shap.sample(X_test_scaled, 1000)
    
    print("SHAP değerleri hesaplanıyor... (Lütfen bekleyin)")
    shap_values = explainer.shap_values(X_test_sample)
    
    print("\n--- 5. SHAP Grafikleri Oluşturuluyor ---")
    
    # 1inci Grafik: SHAP Bar Grafiği (Hangi özellik kararı NE KADAR etkiliyor?)
    # Özellik (Feature) Önemi
    plt.figure()
    plt.title("SHAP Bar Grafiği (Değişkenlerin Etki Gücü)", fontsize=14)
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.show()

    # 2nci Grafik: SHAP Nokta (Summary) Grafiği (Özelliğin DEĞERİ, kararı OLUMLU mu OLUMSUZ mu etkiliyor?)
    plt.figure()
    plt.title("SHAP Nokta Grafiği (Karar Yönü ve Yoğunluk)", fontsize=14)
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
