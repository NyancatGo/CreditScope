# Hafta 11 - Future Work

## Genel Yön

CreditScope'un mevcut sürümü, recall odaklı hibrit karar destek sistemi olarak teslime hazırdır. Gelecek çalışmaların amacı modeli tamamen yeniden kurmak değil, sistemin karar kalitesini, izlenebilirliğini ve operasyonel güvenilirliğini artırmaktır.

## Öncelikli Gelişim Alanları

### 1. Business rule katmanını veriyle kalibre etmek

Mevcut kurallar uzman sezgisine ve ürün mantığına uygundur; ancak kural çarpanları henüz veriyle sistematik olarak optimize edilmemiştir. Gelecek adım, rule etkilerini tarihsel performans üzerinden kalibre etmek ve yanlış pozitifleri daha kontrollü azaltmaktır.

Beklenen çıktı:

- rule bazlı etki analizi
- kural ağırlıkları için deney tablosu
- FP azaltımı ile recall kaybı arasındaki dengenin ölçülmesi

### 2. Maliyet-duyarlı öğrenme ve precision-recall optimizasyonu

Proje bilerek recall tarafına yaslanmaktadır. Sonraki aşamada, yanlış negatif ve yanlış pozitif maliyetleri açıkça tanımlanarak cost-sensitive learning, threshold sweep ve precision-recall eğrisi üzerinden daha kurumsal bir karar politikası kurulabilir.

Beklenen çıktı:

- maliyet matrisi tanımı
- threshold seçimi için daha açık karar kuralı
- recall korunurken precision artışı hedefi

### 3. Monitoring ve drift takibi

Model şu anda demo ve akademik teslim seviyesinde doğrulanmıştır. Gerçek kullanım senaryosunda veri dağılımı değişebilir; bu nedenle prediction drift, feature drift ve rule kullanım sıklığı izlenmelidir.

Beklenen çıktı:

- periyodik metrik raporu
- veri kayması alarm eşikleri
- kural bazlı kullanım ve override istatistikleri

### 4. Daha güçlü açıklanabilirlik ve karar gerekçesi ekranı

SHAP çıktıları hata analizi için yeterlidir; fakat son kullanıcı veya kredi uzmanı tarafında daha okunabilir bir karar gerekçesi ekranı geliştirilebilir. Böylece model sonucu, uygulanan kurallar ve kritik risk sinyalleri aynı anlatı içinde sunulur.

Beklenen çıktı:

- müşteri bazlı karar özeti
- en etkili risk sinyalleri paneli
- manuel inceleme için kısa karar notu üretimi

### 5. Veri kapsamını ve düzenleyici doğrulamayı genişletmek

Mevcut veri seti akademik bir prototip için yeterlidir, ancak daha geniş ve daha gerçekçi veri ile model daha güvenilir hale getirilebilir. Buna ek olarak, açıklanabilirlik, adalet ve düzenleyici uygunluk başlıkları daha derin incelenmelidir.

Beklenen çıktı:

- daha çeşitli örnekler içeren genişletilmiş veri
- fairness ve bias kontrolleri
- regülasyon ve etik uygunluk değerlendirmesi

## Sonuç

Hafta 11 sonrasında CreditScope için en doğru yol, sistemi sıfırdan yeniden yazmak değil; mevcut hibrit yapıyı veriyle kalibre edilmiş, daha ölçülebilir ve daha operasyonel bir karar destek sistemine dönüştürmektir.
