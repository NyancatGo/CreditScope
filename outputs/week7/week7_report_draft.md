# CreditScope: Proje Raporu Taslağı (Hafta 7)

## 1. Giriş
Bu analizde, CreditScope XGBoost kredi risk değerlendirme modelimizin detaylı çıktıları incelenmiştir. Ekibimiz (Baran, Arda) özellikle modelin başarısız olduğu ve hata yaptığı durumları analiz ederek SHAP (SHapley Additive exPlanations) tabanlı öngörüler çıkarmıştır.

## 2. Model Hata Matrisi ve Metrikleri
Test seti üzerinde gerçekleştirilen değerlendirmede, modelimizin vakaları aşağıdaki şekilde ayırdığı görülmektedir:

- **True Positive (TP) - Doğru Bildirilen Riskler:** 4100
- **True Negative (TN) - Doğru Bildirilen Güvenilirler:** 31225
- **False Positive (FP) - Yanlış Alarm (Riskli Denilen Ama Değil):** 13914
- **False Negative (FN) - Kaçırılan Riskler (Güvenilir Denilen Ama Riskli):** 1831

Modelin bankacılık bağlamındaki temel hedefi riskli müşteriyi kaçırmamaktır (Recall Maksimizasyonu). Mevcut Recall değeri **%69.13**'dir. Ayrıca kesinlik (Precision) ise **%22.76** seviyesindedir. Yanlış alarm (FP) sayısının yüksek olması, sınıf dengesizliği optimizasyonlarında beklenen bir durumdur.

## 3. Hata Analizi Zıtlaşmaları
- **False Negative (FN) Profili:** Modelin güvenilir bulup gerçekte temerrüde düşen vakalar. Ortalama özellik dağılımları `error_analysis_feature_means.csv` içerisinde incelenmiş ve gözlemlenmiştir ki bu profildeki müşterilerin özellikleri... *(Arda: Buraya CSV'ye bakarak yorum ekle)*.
- **False Positive (FP) Profili:** Modelin riskli bulup gerçekte ödeyen vakalar. Bu durum, modelin katı davrandığı kesimi temsil eder. *(Baran: Buraya Boxplot görsellerine göre yorum ekle).*

## 4. SHAP Odaklı Değerlendirme

Modelin kararlarını neyin yönlendirdiğini anlamak için, yanlılık yaptığı FP ve FN gruplarına özel SHAP grafikleri oluşturulmuştur.

### Kaçırılan Vakalar (False Negatives - SHAP)
Modelin FN tahminlerinde, modeli kararı "0" (Güvenilir) vermeye iten en önemli faktörler şunlardır:
*(Şu görsele bakarak yorumlanacak: `shap_summary_False_Negatives.png`)*

### Yanlış Alarmlar (False Positives - SHAP)
Modelin FP tahminlerinde, karar vericiyi "1" (Riskli) yapmaya iten hatalı ağırlıklandırmaların faktörleri:
*(Şu görsele bakarak yorumlanacak: `shap_summary_False_Positives.png`)*

## 5. Sonuç ve Öneriler
Bu hata analizi ışığında modelin zayıf noktaları tespit edilmiştir. Sonraki aşamalarda (Hafta 9-12 arası) bu zafiyetlerin arayüz üzerinden iş kuralları (business rules) girilerek kompanse edilmesi değerlendirilecektir.

---
*Otomatik Oluşturulma Tarihi: 2026-03-23 13:03:10*
