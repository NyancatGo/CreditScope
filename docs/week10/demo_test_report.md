# Hafta 10 - Demo Test Raporu

## Senaryo Sonuçları

| Senaryo | Beklenen Karar | Gerçek Karar | Model Skoru | Düzeltilmiş Skor | Durum |
| --- | --- | --- | ---: | ---: | --- |
| Düşük Risk | Onaylanabilir Profil | Onaylanabilir Profil | %8.24 | %4.81 | Geçti |
| Manuel İnceleme | Manuel İnceleme | Manuel İnceleme | %40.01 | %48.02 | Geçti |
| Yüksek Risk | Manuel İnceleme | Manuel İnceleme | %58.04 | %92.11 | Geçti |
| Güçlü Skor, Zayıf İstihdam | Manuel İnceleme | Manuel İnceleme | %26.68 | %24.54 | Geçti |

## Senaryo Yorumları

### Düşük Risk

- Gelir: `92000`, kredi tutarı: `18000`, kredi notu: `790`
- İstihdam: `Full-time`, çalışma süresi: `84` ay, kefil: `Yes`
- API kararı: **Onaylanabilir Profil**
- Uygulanan kurallar: strong_credit_score, low_dti, stable_full_time_employment, cosigner_present
- Yorum: Model skoru %8.24, iş kuralları sonrası %4.81. DTI 0.1957 ve 4 kural etkisi düşük risk kararını destekliyor.

### Manuel İnceleme

- Gelir: `54000`, kredi tutarı: `29500`, kredi notu: `665`
- İstihdam: `Self-employed`, çalışma süresi: `22` ay, kefil: `No`
- API kararı: **Manuel İnceleme**
- Uygulanan kurallar: high_dti
- Yorum: Model skoru %40.01, iş kuralları sonrası %48.02. Profil orta bantta kaldığı için karar manuel inceleme ekranında tartışılabilir.

### Yüksek Risk

- Gelir: `36000`, kredi tutarı: `34000`, kredi notu: `560`
- İstihdam: `Unemployed`, çalışma süresi: `4` ay, kefil: `No`
- API kararı: **Manuel İnceleme**
- Uygulanan kurallar: high_dti, weak_credit_score, unstable_employment
- Yorum: Model skoru %58.04, iş kuralları sonrası %92.11. DTI 0.9444, düşük kredi notu ve istihdam sinyali yüksek risk davranışını gösteriyor.

### Güçlü Skor, Zayıf İstihdam

- Gelir: `68000`, kredi tutarı: `30000`, kredi notu: `780`
- İstihdam: `Full-time`, çalışma süresi: `3` ay, kefil: `No`
- API kararı: **Manuel İnceleme**
- Uygulanan kurallar: strong_credit_score, unstable_employment
- Yorum: Model skoru %26.68, iş kuralları sonrası %24.54. Güçlü kredi notuna rağmen kısa çalışma geçmişi profili manuel inceleme bandında tutuyor.


## API ve Statik Dosya Kontrolleri

| Kontrol | Endpoint | Status | Durum |
| --- | --- | ---: | --- |
| index | GET / | 200 | Geçti |
| css | GET /static/style.css?v=20260419-week10 | 200 | Geçti |
| js | GET /static/script.js?v=20260419-week10 | 200 | Geçti |
| predict_low_risk | POST /predict | 200 | Geçti |
| predict_edge_case | POST /predict | 200 | Geçti |

## UI Kabul Kriterleri

 - Düşük risk, manuel inceleme, yüksek risk ve edge-case senaryoları formu otomatik doldurur.
- DTI, kredi/gelir oranı, aylık taksit ve kredi segmenti anlık hesaplanır.
- `/predict` sonucu sağ karar paneline yansır.
- Model skoru, düzeltilmiş skor, eşik ve DTI ayrı gösterilir.
- Business rule etkileri listelenir.

## Kanıt Dosyaları

- Demo tahminleri: `../../outputs/week10/demo_predictions.json`
- API smoke test: `../../outputs/week10/api_smoke_test.json`
- Model metrikleri: `../../outputs/week10/model_metrics.json`
- Figürler: `docs\week10\figures\ui_cockpit_week10.png`, `docs\week10\figures\model_comparison_grouped_bar_week10.png`, `docs\week10\figures\confusion_matrix_week10.png`, `docs\week10\figures\shap_false_positive_week10.png`, `docs\week10\figures\shap_false_negative_week10.png`
