# Hafta 11 - Final Model Seçimi ve Doğrulama Özeti

## Final Karar

CreditScope'un final teslim modeli `SMOTE + tuned XGBoost + calibrated threshold + mevcut business rules` hattıdır. Hafta 11'de yeni bir model arayışı yerine, Hafta 10 sonunda çalışan sistemin teknik olarak dondurulması ve teslim kalitesinin doğrulanması tercih edilmiştir.

## Dondurulan Teknik Baseline

| Bileşen | Final seçim |
| --- | --- |
| Ana model | XGBoost |
| Sınıf dengesizliği yaklaşımı | SMOTE |
| Karar eşiği | `0.231` |
| Hedef metrik | Recall odaklı değerlendirme |
| Feature engineering | `DTIRatio`, `Age_Income_Interaction` |
| Business rule sürümü | `default-review-2026-04-19` |

### XGBoost hiperparametreleri

| Parametre | Değer |
| --- | ---: |
| `learning_rate` | `0.08` |
| `max_depth` | `3` |
| `n_estimators` | `220` |
| `subsample` | `0.7` |
| `colsample_bytree` | `0.85` |
| `gamma` | `1.0` |

## Neden Final Model XGBoost Olarak Sabitlendi?

Model karşılaştırmasında Logistic Regression'ın recall değeri `0.6962`, final XGBoost hattının recall değeri ise `0.6901` olarak görülmektedir. Buna rağmen final teslim modeli XGBoost olarak korunmuştur; çünkü CreditScope'un deploy edilen gerçek hattı yalnızca ham sınıflandırma performansına göre değil, uçtan uca ürünleşme ve doğrulanabilirlik açısından değerlendirilmiştir.

XGBoost'un final model olarak korunma gerekçeleri:

- SMOTE, optimize hiperparametreler ve recall hedefli threshold kalibrasyonu yalnızca bu hatta tam olarak uygulanmıştır.
- SHAP tabanlı `False Positive` ve `False Negative` analizleri mevcut XGBoost artifact'leriyle üretilmiş ve yorumlanmıştır.
- API, UI ve business-rule katmanı halihazırda bu model artifact'leri etrafında entegre edilmiştir.
- `0.6901` recall değeri proje için belirlenen `0.69` hedef bandını pratik olarak karşılamaktadır.
- Hafta 11 amacı yeni araştırma yapmak değil, teslime hazır, savunulabilir ve tekrarlanabilir hattı resmileştirmektir.

Bu nedenle Logistic Regression final benchmark olarak korunmuş, fakat son teslim sistemine geçilmemiştir. Logistic Regression ileride alternatif baseline olarak yeniden değerlendirilebilir.

## Final Doğrulama Sonuçları

19 Nisan 2026 tarihinde final doğrulama hattı yeniden çalıştırılmıştır.

### Eğitim sonuçları

| Metrik | Sonuç | Kabul durumu |
| --- | ---: | --- |
| Accuracy | `0.6827` | Kabul |
| Precision | `0.2217` | Bilgi amaçlı |
| Recall | `0.6901` | Hedef bandında |
| F1-score | `0.3356` | Kabul |
| Decision Threshold | `0.231` | Stabil |

Kabul kriterleri karşılanmıştır:

- Recall değeri `0.67` alt sınırının üzerinde kalmış ve hedeflenen `0.69` bandına ulaşmıştır.
- Decision threshold `0.231` olarak korunmuş, önceki davranıştan sapmamıştır.
- Model artifact'leri yeniden üretilmiş ve API tarafından kullanılabilir durumda kalmıştır.

### Model karşılaştırması

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | `0.6885` | `0.2265` | `0.6962` | `0.3417` |
| Random Forest | `0.8117` | `0.2968` | `0.4539` | `0.3589` |
| XGBoost | `0.6827` | `0.2217` | `0.6901` | `0.3356` |

### Hata analizi özeti

| Grup | Adet |
| --- | ---: |
| True Negative | `30773` |
| False Positive | `14366` |
| True Positive | `4093` |
| False Negative | `1838` |

Bu dağılım, sistemin recall odaklı kurgusuyla uyumludur. False Positive vakaları manuel inceleme ile elenebilirken, asıl amaç riskli müşteriyi kaçırmama ilkesini korumaktır.

SHAP doğrulaması başarıyla tamamlanmıştır:

- `outputs/shap/shap_summary_False_Positives.png`
- `outputs/shap/shap_summary_False_Negatives.png`
- `outputs/shap/shap_error_analysis_summary.json`

### API, demo ve edge-case doğrulaması

Final demo hattında dört senaryo çalıştırılmış ve tamamı beklenen kararı üretmiştir.

| Senaryo | Beklenen karar | Gerçek karar | Model skoru | Düzeltilmiş skor |
| --- | --- | --- | ---: | ---: |
| Düşük Risk | Onaylanabilir Profil | Onaylanabilir Profil | `%8.24` | `%4.81` |
| Manuel İnceleme | Manuel İnceleme | Manuel İnceleme | `%40.01` | `%48.02` |
| Yüksek Risk | Manuel İnceleme | Manuel İnceleme | `%58.04` | `%92.11` |
| Güçlü Skor, Zayıf İstihdam | Manuel İnceleme | Manuel İnceleme | `%26.68` | `%24.54` |

Özellikle edge-case senaryosu, business-rule katmanının beklenen mantıkla çalıştığını göstermiştir: güçlü kredi skoru tek başına yeterli görülmemiş, kısa istihdam geçmişi profili manuel inceleme bandında tutmuştur.

API smoke test sonuçları:

- `GET /` -> `200`
- `GET /static/style.css` -> `200`
- `GET /static/script.js` -> `200`
- `POST /predict` düşük risk senaryosu -> `200`
- `POST /predict` edge-case senaryosu -> `200`

## Business Rules Sabitlenmiş Hali

Hafta 11'de business rules yeniden tasarlanmamış, mevcut sürüm aynen korunmuştur.

| Kural | Etki |
| --- | --- |
| `CreditScore >= 750` | Temerrüt olasılığını `%20` azalt |
| `DTIRatio <= 0.35` | Temerrüt olasılığını `%10` azalt |
| En az `36` ay tam zamanlı çalışma | Temerrüt olasılığını `%10` azalt |
| Kefil var | Temerrüt olasılığını `%10` azalt |
| `DTIRatio >= 0.50` | Temerrüt olasılığını `%20` artır |
| `CreditScore < 600` | Temerrüt olasılığını `%15` artır |
| İşsiz veya `6` aydan az çalışma süresi | Temerrüt olasılığını `%15` artır |

Bu katman `/predict` yanıtında hem ham model skorunu hem de düzeltilmiş skoru ayrı sunduğu için denetlenebilir kalmaktadır.
