# Hafta 14 - Sunum Son Prova Planı

Bu prova planı, final demo sırasında sıranın bozulmaması ve anlatının kısa kalması için hazırlanmıştır.

## 1. Başlangıç

```powershell
cd C:\Users\baran\GitHub\CreditScope
py -m uvicorn api:app --reload
```

Tarayıcıda şu adres açılır:

```text
http://localhost:8000
```

## 2. Demo Akışı

| Sıra | Ekran | Anlatılacak Ana Mesaj |
| ---: | --- | --- |
| 1 | Genel Bakış | CreditScope recall odaklı bir kredi risk karar destek prototipidir |
| 2 | Risk Kokpiti | Başvuru verisi modele gider, model skoru ve kural etkileri görünür |
| 3 | Düşük Risk Senaryosu | Güçlü kredi notu, düşük DTI ve kefil riski düşürür |
| 4 | Yüksek Risk Senaryosu | Yüksek DTI, düşük kredi notu ve zayıf istihdam manuel incelemeye taşır |
| 5 | Edge-case Senaryosu | Güçlü skor her zaman otomatik güven anlamına gelmez |
| 6 | Model İzleme | Recall, threshold, confusion matrix ve SHAP ile model davranışı açıklanır |
| 7 | Kurallar | Business rule engine modeli değiştirmez, karar skorunu denetlenebilir biçimde düzeltir |
| 8 | Kapanış | Sistem final demo öncesi stress testten geçmiş ve teslim seviyesine getirilmiştir |

## 3. Kısa Kapanış Cümlesi

CreditScope, yalnızca tahmin yapan bir model değil; recall odaklı, açıklanabilir, iş kurallarıyla desteklenen ve stress testten geçirilmiş bir kredi karar destek prototipidir.

## 4. Sunum Günü Kontrolü

- Sunum dosyası açılıyor mu?
- FastAPI sunucusu başlıyor mu?
- Ana sayfa ve diğer route'lar yükleniyor mu?
- `py tools\week13_stress_test.py` başarılı mı?
- Demo sırasında kullanılacak senaryolar önceden bir kez çalıştırıldı mı?
- Jüri Q&A dosyası erişilebilir mi?
