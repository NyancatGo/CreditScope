# Hafta 13 - Stress Test Raporu

## Genel Sonuç

| Kontrol | Sonuç |
| --- | --- |
| Genel durum | Başarılı |
| Çalıştırma zamanı | 2026-04-20T11:33:12 |
| Test edilen UI route sayısı | 5 |
| Test edilen static asset sayısı | 6 |
| Test edilen predict senaryosu | 4 |
| Ardışık predict isteği | 100 |

## Route Kontrolleri

| Endpoint | Status | Süre (ms) | Durum |
| --- | ---: | ---: | --- |
| GET / | 200 | 5.42 | Geçti |
| GET /genel-bakis | 200 | 15.86 | Geçti |
| GET /demo-senaryolari | 200 | 39.07 | Geçti |
| GET /model-izleme | 200 | 18.86 | Geçti |
| GET /kurallar | 200 | 15.41 | Geçti |

## Static Asset Kontrolleri

| Asset | Status | Boyut | Durum |
| --- | ---: | ---: | --- |
| GET /static/style.css?v=20260420-week13 | 200 | 28058 | Geçti |
| GET /static/script.js?v=20260420-week13 | 200 | 11122 | Geçti |
| GET /static/figures/model_comparison_grouped_bar_week10.png?v=20260420-week13 | 200 | 57957 | Geçti |
| GET /static/figures/confusion_matrix_week10.png?v=20260420-week13 | 200 | 60427 | Geçti |
| GET /static/figures/shap_false_positive_week10.png?v=20260420-week13 | 200 | 162960 | Geçti |
| GET /static/figures/shap_false_negative_week10.png?v=20260420-week13 | 200 | 164597 | Geçti |

## Predict Senaryoları

| Senaryo | Beklenen | Gerçek | Model Skoru | Düzeltilmiş Skor | Durum |
| --- | --- | --- | ---: | ---: | --- |
| Düşük Risk | Onaylanabilir Profil | Onaylanabilir Profil | %8.24 | %4.81 | Geçti |
| Manuel İnceleme | Manuel İnceleme | Manuel İnceleme | %40.01 | %48.02 | Geçti |
| Yüksek Risk | Manuel İnceleme | Manuel İnceleme | %58.04 | %92.11 | Geçti |
| Güçlü Skor, Zayıf İstihdam | Manuel İnceleme | Manuel İnceleme | %26.68 | %24.54 | Geçti |

## Invalid Payload Kontrolü

| Endpoint | Beklenen | Gerçek | Durum |
| --- | ---: | ---: | --- |
| POST /predict | 422 | 422 | Geçti |

## Ardışık Predict Stres Kontrolü

| Metrik | Değer |
| --- | ---: |
| İstek sayısı | 100 |
| Hata sayısı | 0 |
| Ortalama süre | 34.77 ms |
| Minimum süre | 31.4 ms |
| Maksimum süre | 55.85 ms |
| P95 süre | 41.0 ms |

## Yorum

Hafta 13 stress testi, CreditScope'un final demo öncesinde route, static asset, API davranışı, edge-case karar mantığı ve ardışık tahmin dayanıklılığı açısından kontrol edildiğini gösterir. Bu test gerçek production load test değildir; akademik demo güvenilirliği için dengeli bir sağlamlaştırma kontrolüdür.
