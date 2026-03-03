# 🏦 CreditScope  
## Yapay Zeka Tabanlı Kredi Risk Analizi ve Karar Destek Sistemi

CreditScope, bankacılık verileri üzerinde eğitilmiş gelişmiş makine öğrenmesi algoritmalarını kullanarak kredi başvurularının temerrüt (default) riskini öngören uçtan uca bir **Full-Stack Karar Destek Sistemi**dir.

> 🎯 Amaç: Bankaların finansal riskini minimize etmek için riskli müşterileri başvuru aşamasında tespit etmek.

---

## 🚀 Proje Özeti

Bu proje, yüksek sınıf dengesizliği (imbalanced data) içeren **255.000+ satırlık gerçekçi bankacılık verisi** üzerinde geliştirilmiştir.

Modelin temel optimizasyon hedefi:

- 📌 **Recall (Duyarlılık) skorunu maksimize etmek**
- 📌 Riskli müşterileri mümkün olduğunca erken tespit etmek
- 📌 Bankanın temerrüt kaynaklı finansal zararını azaltmak

---

## 🛠️ Teknik Yığın (Tech Stack)

### 🤖 Model & Veri Bilimi
- **Model:** XGBoost (Hyperparameter Optimized)
- **Optimizasyon:** RandomizedSearchCV
- **Kütüphaneler:**
  - Pandas
  - Scikit-learn
  - Joblib
  - Matplotlib / Seaborn

### ⚙️ Backend
- FastAPI (Python)
- RESTful API mimarisi
- Gerçek zamanlı tahmin üretimi

### 🎨 Frontend
- HTML5
- CSS3 (Modern UI)
- JavaScript (Fetch API)

---

## 📊 Model Performansı

Model, `RandomizedSearchCV` ile hiperparametre optimizasyonuna tabi tutulmuş ve bankacılık risk yönetimi için en uygun denge noktasına getirilmiştir.

| Metrik | Skor | Açıklama |
|--------|------|----------|
| **Recall** | **%69.23** | Riskli müşterilerin yakalanma oranı (Temel odak) |
| **Accuracy** | **%68.80** | Genel doğru tahmin oranı |
| **Precision** | **%22.54** | Riskli denilenlerin gerçekten riskli çıkma oranı |
| **F1-Score** | **%34.01** | Precision & Recall dengesi |

> ⚠️ Not: Model özellikle **Recall optimizasyonuna** odaklanmıştır. Bankacılık senaryosunda riskli müşteriyi kaçırmak, yanlış pozitif üretmekten daha maliyetlidir.

---

## 🏗️ Sistem Mimarisi

Proje iki ana katmandan oluşmaktadır:

### 1️⃣ AI Service (Backend)
- Kaydedilmiş `.pkl` modeli kullanır
- Gerçek zamanlı tahmin üretir
- REST API üzerinden JSON formatında sonuç döner

### 2️⃣ Web Interface (Frontend)
- Kullanıcıdan kredi başvuru bilgilerini alır
- Backend API'ye gönderir
- Sonucu dinamik olarak arayüzde gösterir

---

## 📁 Proje Yapısı (Örnek)
CreditScope/
│
├── api.py
├── model.pkl
├── scaler.pkl
├── index.html
├── static/
│ ├── style.css
│ └── script.js
│
├── notebooks/
│ └── model_training.ipynb
│
└── README.md


---

## ⚙️ Kurulum ve Çalıştırma

### 1️⃣ Gereksinimlerin Yüklenmesi

```bash
pip install fastapi uvicorn pandas scikit-learn xgboost joblib
uvicorn api:app --reload

```
### 2️⃣ API'nin Başlatılması
```bash
uvicorn api:app --reload
API varsayılan olarak şu adreste çalışacaktır: http://localhost:8000
```

```bash
### 3️⃣ Arayüzün Açılması

index.html dosyasını herhangi bir modern tarayıcıda açarak sistemi kullanmaya başlayabilirsiniz.
```

Bu proje BTS324 Bitirme Projesi kapsamında geliştirilmiştir.
