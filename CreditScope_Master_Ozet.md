# CreditScope - Master Proje Özeti (AI Context Belgesi)

> **Not:** Bu belge, CreditScope projesinin ne olduğunu, teknik altyapısını, şu ana kadar tamamlanan aşamaları ve mevcut durumunu yeni bir Yapay Zeka asistanına (ChatGPT, Claude vb.) eksiksiz şekilde aktarmak için "Sistem Bağlamı (Context)" olarak tasarlanmıştır.

---

## 1. Projenin Amacı ve Genel Çerçeve
**Proje Adı:** CreditScope
**Konu:** Yapay Zeka Tabanlı Kredi Risk Analizi ve Karar Destek Sistemi
**Amaç:** Bankalara gelen kredi başvurularındaki temerrüt (default - krediyi ödeyememe) riskini, makine öğrenmesi algoritmaları kullanarak başvuru aşamasında tespit eden uçtan uca (end-to-end) bir Full-Stack sistem geliştirmek.
**Veri Seti:** Yaklaşık 255.000 satırdan ve 18 özellikten (feature) oluşan kapsamlı bir `Loan_default.csv` bankacılık veri seti. 
**Temel Optimizasyon Hedefi:** Modelde doğruluğu (Accuracy) maksimize etmek yerine, riskli müşteriyi kaçırmamak adına **Duyarlılık (Recall)** skorunu maksimize etmek.

## 2. Teknik Mimari (Tech Stack)
* **Veri Bilimi ve Makine Öğrenmesi (Python):** Pandas, Scikit-learn, XGBoost, Joblib, SHAP (Model Explainability).
* **Veri Optimizasyonu:** `RandomizedSearchCV` (Hiperparametre optimizasyonu), `StandardScaler` (Uzaklık ölçümü), Sınıf Ağırlıklandırması (`scale_pos_weight`).
* **Backend:** FastAPI (Python), REST API sunucusu (`uvicorn`).
* **Frontend:** Özelleştirilmiş HTML5, Vanilla CSS3 (Glassmorphism ve modern UI), JavaScript (Backend'e Fetch API ile bağlanan dinamik yapı).

## 3. Şu Ana Kadar Neler Yapıldı? (Teknik Geliştirme Süreci)
Proje şu ana kadar 15 haftalık çalışma planının **ilk 9 haftasını (Tam Kodlama ve Entegrasyon)** kusursuz bir şekilde tamamlamıştır. Aşamalar şunlardır:

### Adım 1: Veri Ön İşleme ve Keşifsel Veri Analizi (Hafta 1-4)
* **EDA:** Hedef değişkende (`Default`) ciddi bir imbalanced (dengesiz sınıf) problemi tespit edildi: Veri setinin **%79.6'sı güvenilir, %20.4'ü riskli**.
* **Preprocessing:** `LoanID` gibi gereksiz veri sütunları atıldı. Kategorik veriler One-Hot Encoding yöntemi (`get_dummies`) ile sayısal forma dönüştürüldü.
* **Feature Engineering:** Finansal mantığa uyması adına modele `Income_minus_LoanAmount` (Kalan Gelir), `CreditScore_x_DTIRatio` (Risk Çarpanı) ve `Employment_per_CreditLine` (Borç/İstikrar Oranı) gibi güçlü yapay özellikler eklendi.
* **Scaling:** Tüm veri seti StandardScaler ile ölçeklendirildi ve `scaler.pkl` olarak kaydedildi.

### Adım 2: Modelleme, Tuning ve Analiz (Hafta 5-7)
* **Baseline Modeller:** Logistic Regression, Random Forest ve XGBoost eğitimleri yapıldı.
* **Model Seçimi:** Bankacılık senaryosunda "Riskli" müşteriyi yakalamak daha önemli olduğu için ağırlığı XGBoost ve Recall (Duyarlılık) üzerine verdik.
* **Imbalance Çözümü:** SMOTE yerine daha verimli olan algoritmik `scale_pos_weight` metodu kullanıldı.
* **Optimizasyon:** `RandomizedSearchCV` ile XGBoost paramatleri **"Recall odaklı"** optimize edildi. Model %69.23 Recall skoru (test verisinde) yakaladı ve `xgboost_optimized.pkl` adıyla kaydedildi.
* **Açıklanabilirlik:** SHAP (`TreeExplainer`) kütüphanesi kullanılarak modelin hangi özellikleri baz alarak (Faiz oranı, Kredi skoru vs.) karar verdiği tespit edilip bar ve dot plot grafikleri otomatik olarak `outputs/` klasörüne aktarıldı.

### Adım 3: Uçtan Uca Entegrasyon (Hafta 9)
* **API Gelişimi:** `api.py` dosyası üzerinden FastAPI ile modeli dinleyen `/predict` endpoint'i yazıldı. Canlıdan gelen JSON verisi otomatik StandardScaler'dan ve Feature match işleminden geçer duruma getirildi.
* **Web Arayüzü:** `index.html` ve JavaScript kodları (`app.js`) tamamlandı. Kullanıcı ekrandaki formu doldurup "Hesapla" butonuna bastığında API'ye HTTP POST isteği gider ve arayüzde anında dinamil bir Risk/Güvenilir kartı çizilir. Sistem şu anda tamamen çalışmaktadır.

## 4. Mevcut Durum (Biz Şu An Neredeyiz?)
Teknik kodlama (Python + AI + API + Web) süreci %100 oranında tamamlanmış, sistem canlıya veya projenin son teslimine hazır bir ürüne (MVP) dönüşmüştür. Ekipler şu anda **Hafta 10, 11 ve 12** adımdalarındadır.

## 5. Bundan Sonra Ne Yapmak İstiyoruz? (Gelecek Hedefleri)
Şu anki geliştirme odağımız kod yazmaktan ziyade projenin **Dokümantasyon, Analiz Raporlaması ve Sunumuna** dönmüştür:
1. **Error Analysis & Results (Hata Analizi ve Sonuçlar):** Model hatasını detaylı inceleyecek makale kıvamında Sonuç ve Tartışma (Results and Discussion) yazımı (Hafta 10).
2. **Grafik Revizyonları:** Çıkarılan SHAP, Confusion Matrix ve Correlation matrix'lerin akademik bir rapora yerleştirebilecek kalitede hazırlanması.
3. **Dokümantasyon ve Final:** "Future Works" (Gelecekte Eklenecek Özellikler) taslaklarının çıkarılması, final sunumu PPTX dosyası veya PDF raporunun toparlanması.
4. **Kod Temizliği:** Github'a yüklenecek olan nihai formülün temizlenmesi ve README belgesinin mükemmelleştirilmesi.

> **Yapay Zeka Asistanı İçin Talimat:**
> Sana bu dosyayla gelen kullanıcı, "CreditScope" bitirme projesinde yer alan bir öğrencidir (Baran Atıcı). Amacı; projenin raporlama, error analizi (Week 7-10 arası scriptler veya dokümanlar), sonuç/tartışma kısımlarının yazılması, akademik/resmi formata dökülmesine olan ihtiyacını gidermektir. Kodu bozacak yeni modeller eğitmek yerine yukarıda açıklanan mevcut mükemmel yapı üzerinden raporlar, slayt taslakları ve README içerikleri geliştirmen, ve kullanıcının sorduğu spesifik işlemlere akademik/teknik cevaplar vermen beklenmektedir.
