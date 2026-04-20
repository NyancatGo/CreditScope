# Hafta 14 - Yedekleme Manifesti

Bu dosya final teslim öncesi hangi dosyaların yedekleneceğini ve hangi dosyaların teslim paketinde kritik olduğunu açıklar.

## Mutlaka Yedeklenecek Dosyalar

| Grup | Yol | Açıklama |
| --- | --- | --- |
| Kaynak kod | `api.py`, `preprocessing.py`, `training.py`, `shap_analysis.py` | Model, API ve analiz hattı |
| UI | `templates/`, `static/` | Çok sayfalı FastAPI/Jinja2 arayüzü |
| Model artifact | `xgboost_optimized.pkl`, `scaler.pkl`, `feature_names.pkl`, `decision_threshold.pkl` | Demo için gerekli model dosyaları |
| Veri snapshot | `data/final_validation_snapshot.json` | UI metrik ve demo senaryolarının tek kaynak dosyası |
| Dokümantasyon | `README.md`, `docs/` | Raporlar, wiki HTML, demo ve jüri hazırlığı |
| Sunum | `presentations/` | Final sunum dosyası ve anlatı planı |
| Test araçları | `tools/` | Demo validation ve stress test scriptleri |
| Bağımlılıklar | `requirements.txt` | Kurulum için Python paket listesi |

## Yedekleme Önerisi

Final teslim için iki yedek tutulması önerilir:

- GitHub deposu: Kod geçmişi ve final sürüm için ana kaynak.
- Harici zip/Drive yedeği: Sunum günü internet veya GitHub erişimi problemine karşı güvenli kopya.

## Zip İçeriği Önerisi

```text
CreditScope/
  api.py
  preprocessing.py
  training.py
  shap_analysis.py
  requirements.txt
  README.md
  data/
  docs/
  presentations/
  static/
  templates/
  tools/
  *.pkl
```

## Yedek Dışı Kalabilecekler

- `__pycache__/`
- `.pytest_cache/`
- Geçici `tmp/` çıktıları
- Çok büyük ara çıktı dosyaları, eğer teslim için şart değilse

## Son Kontrol Komutu

```powershell
py tools\week13_stress_test.py
```

Bu komut başarılıysa demo paketi teknik olarak çalışır durumdadır.
