# Hafta 13 - Jüri Soru Cevap Hazırlığı

Bu doküman, final demo veya jüri değerlendirmesinde gelebilecek temel sorulara kısa ve savunulabilir cevaplar hazırlamak için oluşturuldu.

## 1. Neden accuracy yerine recall odaklı ilerlediniz?

Kredi temerrüt probleminde en kritik hata, gerçekten riskli bir müşteriyi güvenilir kabul etmektir. Bu hata bankacılık açısından daha maliyetli olduğu için model seçiminde recall ana öncelik oldu. Daha fazla başvurunun manuel incelemeye düşmesi kabul edilebilir; fakat riskli müşteriyi kaçırmak daha büyük problemdir.

## 2. XGBoost neden final model seçildi?

XGBoost, final deploy hattında SMOTE, optimize hiperparametreler, threshold calibration, SHAP analizi, API entegrasyonu ve business rules katmanıyla birlikte tamamlanmış modeldir. Bu yüzden seçim sadece tek metrik üstünlüğüne değil, uçtan uca sistem olgunluğuna dayanır.

## 3. Logistic Regression recall değeri biraz daha yüksekken neden deploy edilmedi?

Logistic Regression güçlü bir benchmark olarak korunmuştur; ancak final sistem XGBoost artifact'leri etrafında ürünleşmiştir. XGBoost hattı SHAP analizleri, threshold tuning, API ve UI entegrasyonu ile daha bütünlüklü teslim seviyesine gelmiştir. Bu nedenle final model XGBoost olarak sabitlenmiştir.

## 4. SMOTE veri sızıntısı yaratıyor mu?

Hayır. SMOTE yalnızca eğitim setinde uygulanır. Test seti ve API inference hattı sentetik örneklerle karıştırılmaz. Böylece model değerlendirmesinde ve canlı tahminde veri sızıntısı oluşmaz.

## 5. Threshold neden 0.231?

Klasik 0.50 threshold, riskli sınıfı yakalamada yeterli değildi. Bu nedenle karar eşiği recall hedefi etrafında kalibre edildi ve final threshold 0.231 olarak seçildi. Amaç, temerrüt riski taşıyan başvuruları daha iyi yakalamaktır.

## 6. False Positive sayısı neden yüksek?

Sistem bilinçli olarak temkinli davranır. False Positive başvurular manuel inceleme ile elenebilir; fakat False Negative, yani riskli müşteriyi kaçırmak daha kritik kabul edilir. Bu nedenle modelin riskli profilleri daha fazla yakalaması tercih edilmiştir.

## 7. Business rules modeli manipüle ediyor mu?

Business rules modeli değiştirmez; modelin ürettiği olasılığı şeffaf çarpanlarla düzeltir. `/predict` yanıtında ham model skoru, düzeltilmiş skor ve uygulanan her kural ayrı ayrı döner. Bu yüzden kural etkisi denetlenebilir kalır.

## 8. Sistem gerçek bir bankada doğrudan kullanılabilir mi?

Hayır, doğrudan production kullanımı için hazır değildir. Bu proje akademik ve prototip seviyesinde bir karar destek sistemidir. Gerçek kullanım için daha geniş veri, monitoring, fairness analizi, regülasyon kontrolleri, güvenlik, audit logging ve model drift takibi gerekir.

## 9. Fairness veya bias kontrolü yapıldı mı?

Bu aşamada kapsamlı fairness/bias validasyonu yapılmadı. Bu durum proje sınırlılığı olarak açıkça belirtilmiştir. Gelecek çalışmalarda farklı gruplar üzerinde hata oranı, eşik etkisi ve karar dağılımı analiz edilmelidir.

## 10. Neden çok sayfalı UI eklediniz?

Hafta 12 öncesinde sidebar menüleri demo görünümü veriyordu ama gerçek route'lara bağlı değildi. Çok sayfalı yapı ile proje daha gerçekçi bir karar destek prototipine dönüştürüldü. Genel Bakış, Demo Senaryoları, Model İzleme ve Kurallar sayfaları ayrı bağlamlar sundu.

## 11. Stress test neyi kanıtlıyor?

Hafta 13 stress test, sistemin final demo öncesinde tüm route'larda, static asset yüklemede, dört predict senaryosunda, invalid payload davranışında ve 100 ardışık tahminde hata vermediğini kontrol eder. Bu production load test değildir; akademik demo güvenilirliği kanıtıdır.

## 12. Projenin en güçlü tarafı nedir?

Projenin en güçlü tarafı model, rule engine, explainability, API ve UI katmanlarını tek bir karar akışında birleştirmesidir. Sistem sadece tahmin üretmez; kararın neden oluştuğunu da görünür hale getirir.

## 13. Projenin en zayıf veya geliştirilebilir tarafı nedir?

Mevcut rule ağırlıkları uzman sezgisine dayanır; veriyle kalibre edilmesi gerekir. Ayrıca gerçek production kullanımı için drift monitoring, fairness kontrolleri, audit logging ve daha geniş validasyon yapılmalıdır.

## 14. Bundan sonra ne geliştirilecek?

Öncelikli geliştirmeler:

- Business rule etkilerini veriyle kalibre etmek
- Week 12 route'larını otomatik validation kapsamına almak
- Snapshot verilerini tek kaynaklı hale getirmek
- Monitoring ve drift takibi eklemek
- Fairness, bias ve regülasyon kontrollerini genişletmek

## 15. Kısa final savunma cümlesi nedir?

CreditScope, recall odaklı modelleme stratejisini, açıklanabilir hata analizini, şeffaf business rules katmanını ve çok sayfalı bir demo arayüzünü birleştirerek final teslim öncesi stres testinden geçirilmiş bir kredi karar destek prototipi haline gelmiştir.
