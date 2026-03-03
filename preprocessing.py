import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    print("--- 1. Veri Yükleniyor ---")
    try:
        df = pd.read_csv('Loan_default.csv')
    except FileNotFoundError:
        print("Hata: 'Loan_default.csv' dosyası bulunamadı.")
        return

    # Eğer gereksiz bir ID kolonu varsa (örn: LoanID), modeli yanıltmaması için çıkartıyoruz
    if 'LoanID' in df.columns:
        df = df.drop('LoanID', axis=1)

    print("\n--- 2. Kategorik Değişken Tespiti ve One-Hot Encoding ---")
    # Tipi 'object' (metin/kategorik) olan kolonları tespit edelim
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Eğer 'Default' hedef değişkeni kazara objeyse, içinden çıkartalım ki onu dönüştürmesin
    if 'Default' in categorical_cols:
        categorical_cols.remove('Default')
    
    print(f"Tespit edilen kategorik kolonlar: {categorical_cols}")

    # get_dummies komutuyla 'Object' olan verileri One-Hot Encoding'e çeviriyoruz.
    # drop_first=True kullanarak "dummy trap" (çoklu doğrusallık problemi) sorununu engelliyoruz.
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"Encoding sonrası veri setinin yeni boyutu: {df_encoded.shape}")

    print("\n--- 3. Hedef Değişken (y) ve Bağımsız Değişken (X) Ayrımı ---")
    if 'Default' not in df_encoded.columns:
        print("Hata: 'Default' kolonu veri setinde bulunamadı!")
        return
        
    y = df_encoded['Default']
    X = df_encoded.drop('Default', axis=1)
    
    print(f"X (Bağımsız Değişkenler) boyutu: {X.shape}")
    print(f"y (Hedef Değişken - Default) boyutu: {y.shape}")

    print("\n--- 4. Train/Test Ayırma (%80 Train - %20 Test) ---")
    # stratify=y kullanılarak dengesiz sınıflar test ve train setine eşit oranda yansıtılıyor
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    print(f"Eğitim Seti (Train): X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Test Seti (Test) : X_test {X_test.shape}, y_test {y_test.shape}")

    print("\n--- 5. StandardScaler ile Ölçeklendirme (Scaling) ---")
    scaler = StandardScaler()

    # Eğitim setine verilerin ortalama/varyans değerlerini 'fit' ediyor ve aynı anda değiştiriyoruz
    # Not: Makine öğrenmesi ve yapay zeka alanında, one-hot edilip edilmediğine bakılmaksızın bütün özellikler standartlaştırılabilir
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Test setinde sadece transform kullanılır (Eğitim setinden çıkardığımız değerleri test setine uygularız)
    # Nedeni: Gelecekte gelecek (test) verilerin istatistiğini önceden bilemeyiz (Veri Sızıntısı / Data Leakage engellemesi)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Ölçeklenmiş Eğitim Seti (X_train_scaled) boyutu: {X_train_scaled.shape}")
    print(f"Ölçeklenmiş Test Seti (X_test_scaled) boyutu: {X_test_scaled.shape}")
    
    print("\n✅ Veri Ön İşleme (Preprocessing) başarıyla tamamlandı!")

if __name__ == "__main__":
    main()
