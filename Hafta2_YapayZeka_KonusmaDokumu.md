# Hafta 2: Keşifsel Veri Analizi (EDA) - Yapay Zeka Konuşma Dökümü

**Baran Atıcı:** selam projeye başlıyoruz Loan_default.csv diye bi data var 255k küsur satır. bu hafta EDA (veri analizi) yapmamız lazım neresinden başlayalım?

**Yapay Zeka:** Merhaba Baran! 255 bin satırlık büyük bir veriniz olduğu için Keşifsel Veri Analizi (EDA) aşamasını titiz yapmanız çok önemli. İlk olarak Python'da kütüphaneleri dahil edip verinin eksik değerlerine (null) ve genel veri tiplerine bakarak başlamalısın. Şu temel kodu kullanabilirsin:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Loan_default.csv')
print(df.info())
print("\nEksik Veri Sayısı:\n", df.isnull().sum())
```

**Baran Atıcı:** kodu çalıştırdım eksik veri yokmuş temiz. hedef kolonumuz Default diye geçiyo. bunun ve diğer özelliklerin dağılımını grafiğe döksek kodları nasıl olur?

**Yapay Zeka:** Eksik veri çıkmaması veri temizliği için büyük avantaj! Hedef değişkeninin (`Default`) oranlarını görmek projenin en kritik kısmıdır. Dağılımları ve histogramları çizdirmek için şu kod bloğunu kullanabiliriz:

```python
# Kredi Temerrüt Dağılımı (Sınıf dengesine bakmak için)
sns.countplot(data=df, x='Default', palette='Set2')
plt.title('Hedef Değişken (Default) Dağılımı')
plt.show()

oran = df['Default'].value_counts(normalize=True) * 100
print(f"Yüzdelik Dağılım:\n{oran}")

# Sayısal bir kolonun dağılımı (Age, Income vb.)
sns.histplot(df['Income'], kde=True, color='blue')
plt.title('Gelir Dağılımı (Histogram)')
plt.show()
```

**Baran Atıcı:** knk oranlara baktım da sınıf 0 (güvenilir olanlar) %79.6, sınıf 1 (riskli) %20.4 çıktı. ikisinin arasında çok fark var bu ilerde modelleri eğitirken sorun çıkarır mı napıcaz buna?

**Yapay Zeka:** Evet Baran, çok doğru bir noktaya değindin. Bu tablo bize çok net bir **"Class Imbalance" (Sınıf Dengesizliği)** problemi olduğunu gösteriyor. Modeller çoğunluk olan %79'luk kısmı tahmine yönelir, riskli hastayı atlar. 
Bu bulguyu mutlaka raporuna "Önemli Bulgu" diye not et. İleride makine öğrenmesine geçerken bunu mecburen **SMOTE/ADASYN** gibi veri artırma (oversampling) teknikleriyle ya da modele ağırlık (scale_pos_weight vb.) vererek çözeceğiz.

**Baran Atıcı:** anladım mantıklı. son bişe bu 18 özelliğin bibiriyle ilişkisini tabloya/matris falan nasıl dökerim korelasyon için?

**Yapay Zeka:** Korelasyon matrisini bir ısı haritası (heatmap) ile çok net görselleştirebilirsiniz. Hangi değişkenlerin doğrudan Default skorunu etkilediğini görmek için şu kodu kullanabilirsin:

```python
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.show()
```

**Baran Atıcı:** tamamdır harika oldu hepsi çalıştı eyvallah sağol

**Yapay Zeka:** Rica ederim Baran! EDA adımını, bulduğun sınıf dengesizliğini (imbalance) ve çizdiğin korelasyonları raporuna ekleyebilirsin. Bir sonraki hafta için veri temizliği ve hazırlığı kodlarında görüşmek üzere, ekibe kolay gelsin!
