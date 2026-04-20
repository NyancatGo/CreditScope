# Hafta 13 - Kusursuz Demo Runbook

Bu runbook, CreditScope final demosunun aynı sırayla, kısa ve savunulabilir biçimde gösterilmesi için hazırlandı.

## Demo Öncesi Kontrol

1. Terminalde proje klasörüne geç:

```bash
cd C:\Users\baran\GitHub\CreditScope
```

2. Uygulamayı başlat:

```bash
py -m uvicorn api:app --reload
```

3. Tarayıcıda ana ekranı aç:

```text
http://127.0.0.1:8000/
```

4. Demo öncesi hızlı kontrol:

```bash
py tools/week13_stress_test.py
```

## Demo Akışı

### 1. Genel Bakış

Route:

```text
/genel-bakis
```

Anlatılacak ana mesaj:

CreditScope, kredi temerrüt riskini recall odaklı tahmin eden ve model skorunu business rules ile açıklanabilir hale getiren hibrit bir karar destek sistemidir.

Vurgu:

- Sistem sadece model skoru üretmez.
- Ham model skoru, düzeltilmiş skor ve uygulanan kurallar ayrı görünür.
- Nihai karar otomatik onay değil, kredi uzmanına destek mantığıdır.

### 2. Risk Kokpiti - Düşük Risk Senaryosu

Route:

```text
/
```

Yapılacak işlem:

- `Düşük Risk` preset butonuna bas.
- `Karar Skorunu Çalıştır` butonuna bas.

Beklenen sonuç:

- Karar: `Onaylanabilir Profil`
- Model skoru yaklaşık `%8.24`
- Düzeltilmiş skor yaklaşık `%4.81`
- Uygulanan kurallar: güçlü kredi notu, düşük DTI, stabil istihdam, kefil

Anlatılacak ana mesaj:

Bu profil hem model hem business rules açısından düşük risk sinyali taşır.

### 3. Risk Kokpiti - Yüksek Risk Senaryosu

Yapılacak işlem:

- `Yüksek Risk` preset butonuna bas.
- `Karar Skorunu Çalıştır` butonuna bas.

Beklenen sonuç:

- Karar: `Manuel İnceleme`
- Model skoru yaklaşık `%58.04`
- Düzeltilmiş skor yaklaşık `%92.11`
- Uygulanan kurallar: yüksek DTI, zayıf kredi notu, dengesiz istihdam

Anlatılacak ana mesaj:

Bu profil, model ve rule engine tarafından yüksek riskli görülür; sistem başvuruyu otomatik reddetmez, manuel inceleme bandına alır.

### 4. Edge-Case Anlatımı

Route:

```text
/demo-senaryolari
```

Gösterilecek senaryo:

`Güçlü Skor, Zayıf İstihdam`

Ana mesaj:

Güçlü kredi skoru tek başına yeterli değildir. Kısa çalışma geçmişi profili manuel inceleme bandında tutar. Bu senaryo, rule engine katmanının sadece skora bakmadığını gösterir.

### 5. Model İzleme

Route:

```text
/model-izleme
```

Vurgulanacak noktalar:

- Recall: `0.6901`
- Threshold: `0.231`
- Logistic Regression recall olarak güçlü benchmark olsa da deploy hattı XGBoost üzerinde tamamlandı.
- Confusion matrix recall odaklı stratejiyle uyumludur.
- SHAP FP/FN grafikleri modelin hata noktalarını yorumlamak için kullanıldı.

Ana mesaj:

Final model seçimi tek metrik yarışı değildir; explainability, API entegrasyonu, threshold tuning ve demo readiness birlikte değerlendirilmiştir.

### 6. Kurallar

Route:

```text
/kurallar
```

Vurgulanacak noktalar:

- Riski azaltan 4 kural
- Riski artıran 3 kural
- Kural sürümü: `default-review-2026-04-19`
- Ham model skoru ve düzeltilmiş skor ayrı döner

Ana mesaj:

Business rules modeli gizlice değiştiren bir kara kutu değildir; her etki `/predict` yanıtında açıkça listelenir.

## Kapanış Cümlesi

CreditScope, final aşamada yalnızca çalışan bir sınıflandırıcı değil; recall odaklı, açıklanabilir, rule-supported ve stres testinden geçirilmiş bir kredi karar destek prototipi haline gelmiştir.

## Demo Sırasında Kaçınılacak Şeyler

- Sistemi gerçek banka ürünü gibi pazarlama.
- False Positive sayısını saklamaya çalışma.
- Logistic Regression sonucunu yok sayma.
- Business rules katmanını modelin yerine geçen yapı gibi anlatma.
- SMOTE'u test veya inference tarafında kullanılmış gibi ifade etme.
