import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def load_and_preprocess_data(filepath='Loan_default.csv'):
    """Aşama 2'deki ön işleme adımlarını uygular ve veriyi hazır döndürür."""
    # 1. Veri Yükleme
    df = pd.read_csv(filepath)
    
    # Gereksiz kolonları çıkarma
    if 'LoanID' in df.columns:
        df = df.drop('LoanID', axis=1)
        
    # 2. Kategorik değişkenleri One-Hot Encoding ile dönüştürme
    # Pandas >= 3.0 uyumluluğu için veri tiplerini kontrol edelim
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if 'Default' in categorical_cols:
        categorical_cols.remove('Default')
        
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 3. Bağımlı (y) ve Bağımsız (X) Değişkenleri Ayırma
    y = df_encoded['Default']
    X = df_encoded.drop('Default', axis=1)
    
    # 4. Train/Test Bölme (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # 5. StandardScaler ile Ölçeklendirme
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(y_true, y_pred, model_name):
    """Modelin başarı metriklerini hesaplar ve ekrana yazdırır."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"--- {model_name} Sonuçları ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}  <-- Riskli müşteriyi yakalama gücü (Önemli!)")
    print(f"F1-Score : {f1:.4f}\n")
    return [accuracy, precision, recall, f1]

def plot_confusion_matrix(y_true, y_pred, model_name, ax):
    """Confusion Matrix'i heatmap olarak çizer."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_xlabel('Tahmin Edilen (Predicted)')
    ax.set_ylabel('Gerçek (Actual)')
    
def main():
    print("Veri yükleniyor ve ön işlemler (Aşama 2) tekrarlanıyor...")
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
    except FileNotFoundError:
        print("Hata: 'Loan_default.csv' dosyası bulunamadı.")
        return

    # Sınıf dengesizliğini kontrol etme (y_train üzerinden)
    # Temerrüde Düşmeyen : 0, Temerrüde Düşen : 1
    class_counts = y_train.value_counts()
    neg_class = class_counts[0]
    pos_class = class_counts[1]
    
    # XGBoost için scale_pos_weight parametresi (Negatif Sınıf Sayısı / Pozitif Sınıf Sayısı)
    scale_pos_weight_val = neg_class / pos_class
    print(f"\nSınıf Dağılımı: 0 (Ödedi): {neg_class}, 1 (Temerrüt): {pos_class}")
    print(f"XGBoost için hesaplanan scale_pos_weight: {scale_pos_weight_val:.2f}\n")

    print("Modeller tanımlanıyor ve class_weight='balanced' parametreleri ekleniyor...\n")
    
    # 1. Logistic Regression Model (Dengesiz veri çözümü)
    log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    
    # 2. Random Forest Model (Dengesiz veri çözümü)
    # n_jobs=-1 ile tüm işlemci çekirdeklerini kullanarak eğitimi hızlandırıyoruz
    rf_clf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1, n_estimators=100)
    
    # 3. XGBoost Model (Dengesiz veri çözümü)
    xgb_clf = XGBClassifier(scale_pos_weight=scale_pos_weight_val, random_state=42, n_jobs=-1, eval_metric='logloss')

    # Modelleri tek bir sözlükte tutalım
    models = {
        'Logistic Regression': log_reg,
        'Random Forest': rf_clf,
        'XGBoost': xgb_clf
    }

    results = {}
    
    # Grafikler için Matplotlib figürü hazırlayalım (1 satır, 3 sütun)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Her bir modeli eğitme, tahmin etme ve sonuçları değerlendirme döngüsü
    for idx, (model_name, model) in enumerate(models.items()):
        print(f">>> {model_name} eğitiliyor... Lütfen bekleyin.")
        # Eğit
        model.fit(X_train, y_train)
        
        # Tahmin yap
        y_pred = model.predict(X_test)
        
        # Metrikleri hesapla
        results[model_name] = evaluate_model(y_test, y_pred, model_name)
        
        # Confusion Matrix grafiğini çiz (ax nesnesi ile)
        plot_confusion_matrix(y_test, y_pred, model_name, axes[idx])

    # Grafikleri göster (Bloke eden komut, kapatana kadar terminal bekler)
    plt.tight_layout()
    plt.show()
    
    # Tüm modellerin RECALL sonuçlarına göre genel bir özet
    print("--- Özet Karşılaştırma (Recall Odaklı) ---")
    for model_name, metrics in results.items():
        print(f"{model_name:<20} | Recall: {metrics[2]:.4f} | F1-Score: {metrics[3]:.4f}")

if __name__ == "__main__":
    main()
