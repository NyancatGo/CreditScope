import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Veri setini yükle
    print("--- Veri Seti Yükleniyor ---")
    try:
        df = pd.read_csv('Loan_default.csv')
    except FileNotFoundError:
        print("Hata: 'Loan_default.csv' dosyası bulunamadı. Lütfen dosyanın bu script ile aynı dizinde olduğundan emin olun.")
        return

    # İlk 5 satırını yazdır
    print("\n--- İlk 5 Satır ---")
    print(df.head())

    # Veri tiplerini yazdır
    print("\n--- Veri Tipleri ve Bilgiler ---")
    print(df.info())

    # Temel istatistiksel özetini yazdır
    print("\n--- Temel İstatistiksel Özet ---")
    print(df.describe())

    # 2. Veride eksik değer (missing value) olup olmadığını kontrol et
    print("\n--- Eksik Değer Kontrolü ---")
    print(df.isnull().sum())

    # Grafikler için genel arka plan stili belirle
    sns.set_theme(style="whitegrid")

    # 3. Hedef değişken olan 'Default' kolonunun dağılımını gösteren countplot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Default', data=df, palette='Set2')
    plt.title('Hedef Değişken (Default) Dağılımı', fontsize=14)
    plt.xlabel('Default (0: Ödedi, 1: Temerrüt)', fontsize=12)
    plt.ylabel('Frekans', fontsize=12)
    plt.show()

    # 4. 'Age', 'Income', 'LoanAmount', 'CreditScore' histogramları
    num_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore']
    
    # Tüm kolonların veride olup olmadığını kontrol et
    missing_cols = [col for col in num_cols if col not in df.columns]
    if missing_cols:
        print(f"\nUyarı: Şu kolonlar veri setinde bulunamadı ve histogramları çizilemeyecek: {missing_cols}")
        num_cols = [col for col in num_cols if col in df.columns]

    if num_cols:
        plt.figure(figsize=(14, 10))
        for i, col in enumerate(num_cols, 1):
            plt.subplot(2, 2, i)
            sns.histplot(df[col], kde=True, bins=30, color='skyblue')
            plt.title(f'{col} Dağılımı', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Frekans', fontsize=12)
        
        plt.tight_layout()
        plt.show()

    # 5. Sayısal değişkenler arasındaki ilişkiyi görmek için Korelasyon Matrisi (Heatmap)
    plt.figure(figsize=(12, 10))
    # Sadece sayısal kolonları seçerek korelasyon hesaplayalım
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numerical_df.corr()

    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Sayısal Değişkenler İçin Korelasyon Matrisi', fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()
