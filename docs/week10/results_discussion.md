# Hafta 10 - Results ve Discussion

## Model Sonuçları

| Metrik | Değer |
| --- | ---: |
| Accuracy | 0.6827 |
| Precision | 0.2217 |
| Recall | 0.6901 |
| F1-score | 0.3356 |
| Decision Threshold | 0.231 |

CreditScope modeli recall odaklı kalibre edilmiştir. Bankacılık senaryosunda temerrüde düşebilecek bir müşteriyi kaçırmak, güvenilir bir müşteriyi manuel incelemeye göndermekten daha maliyetli kabul edildiği için karar eşiği klasik `0.50` yerine `0.231` olarak belirlenmiştir.

## Model Karşılaştırması

| Model | Accuracy | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6885 | 0.2265 | 0.6962 | 0.3417 |
| Random Forest | 0.8117 | 0.2968 | 0.4539 | 0.3589 |
| XGBoost | 0.6827 | 0.2217 | 0.6901 | 0.3356 |

Logistic Regression ve Random Forest karşılaştırmaları aynı test seti üzerinde hızlı değerlendirme amacıyla stratified eğitim örneklemiyle çalıştırılmıştır. XGBoost satırı ise SMOTE, optimize hiperparametreler ve kalibre karar eşiği ile elde edilen ana CreditScope modelidir. Grafik, modeller arasındaki accuracy/precision/recall/F1 dengesini gösterir; CreditScope demosunda XGBoost seçimi yalnızca tek metrik üstünlüğüne değil, tuning süreci, SHAP uyumu ve API'ye alınmış üretim artifact'lerine dayanır.

## Hata Analizi

| Grup | Adet |
| --- | ---: |
| True Negative | 30773 |
| False Positive | 14366 |
| True Positive | 4093 |
| False Negative | 1838 |

False Positive vakaları modelin temkinli davrandığı başvuruları gösterir. Bu vakalar kredi uzmanı tarafından manuel incelemeye alınabilir ve iş kuralları bu gruptaki gereksiz alarmları azaltmak için kullanılır. False Negative vakaları ise modelin kaçırdığı riskli profilleri temsil eder; bu nedenle SHAP analizi özellikle bu gruptaki karar sinyallerini anlamak için önemlidir.

## Revize Grafikler

- UI kokpit ekran görüntüsü: `figures/ui_cockpit_week10.png`
- Model karşılaştırma grafiği: `figures/model_comparison_grouped_bar_week10.png`
- Karışıklık matrisi: `figures/confusion_matrix_week10.png`
- False Positive SHAP: `figures/shap_false_positive_week10.png`
- False Negative SHAP: `figures/shap_false_negative_week10.png`

![Hafta 10 Risk Kokpiti](figures/ui_cockpit_week10.png)

![Model Karşılaştırması](figures/model_comparison_grouped_bar_week10.png)

![Karışıklık Matrisi](figures/confusion_matrix_week10.png)

## Business Rules Değerlendirmesi

API katmanındaki business rules modeli değiştirmeden karar olasılığını şeffaf şekilde düzeltir. Örneğin güçlü kredi notu, düşük DTI, uzun süreli tam zamanlı istihdam ve kefil varlığı riski düşürürken; yüksek DTI, düşük kredi notu ve istikrarsız istihdam riski artırır. `/predict` yanıtında hem ham model skoru hem de düzeltilmiş skor ayrı verildiği için bu katman denetlenebilir kalır.

## UI Demo Değerlendirmesi

Hafta 9 arayüzü Hafta 10 demo testinde risk kokpiti olarak kullanılmıştır. Arayüz, üç senaryo butonu ile formu doldurur, DTI ve ödeme önizlemelerini canlı hesaplar, API sonucunu karar paneline yansıtır ve business rule etkilerini kullanıcıya açıkça gösterir.
