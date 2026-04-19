# Hafta 11 - Presentation Storyboard

Bu dosya, `CreditScope_Week11_Final.pptx` sunumunun slayt slayt anlatım akışını, ana mesajını ve konuşma vurgularını sabitler. Özellikle `SMOTE teknik zorluğu`nun hangi noktada öne çıkarılacağı burada netleştirilmiştir.

## Genel Sunum Hedefi

- CreditScope'u sadece bir model deneyi olarak değil, recall odaklı hibrit karar destek sistemi olarak anlatmak
- Final model seçiminin neden XGBoost olarak sabitlendiğini savunmak
- Hafta 11'in yeniden geliştirme değil, teslim kalitesini mühürleme haftası olduğunu göstermek

## Sunum Tonu

- Akademik ama ürünleşme farkındalığı yüksek
- Kısa, net ve savunulabilir
- Skor yarışından çok karar mantığına odaklı

## Slayt 1 - Kapak ve Final Çerçeve

**Amaç**

Sunuma güçlü bir çerçeveyle başlamak ve projenin artık son teslim seviyesinde olduğunu göstermek.

**Ekranda Ne Var**

- Proje adı
- Finalizasyon vurgusu
- Recall, threshold, demo ve smoke test kartları
- UI cockpit görseli

**Ana Mesaj**

Bu proje artık sadece kredi riski tahmini yapan bir model değil; açıklanabilir, denetlenebilir ve manuel incelemeyi destekleyen hibrit bir karar destek sistemidir.

**Konuşma Notu**

- "Hafta 11'de yeni bir model peşine düşmek yerine mevcut çalışan hattı teslim seviyesine sabitledik."
- "Final kararımızı teknik metrikler, explainability ve sistem entegrasyonu birlikte belirledi."

## Slayt 2 - Problem, Veri ve Recall Mantığı

**Amaç**

Neden recall odaklı bir yaklaşım seçildiğini ve veri dengesizliğinin projeyi nasıl şekillendirdiğini açıklamak.

**Ekranda Ne Var**

- Problem tanımı
- Sınıf dağılımı grafiği
- Güvenilir/riskli oran kartları

**Ana Mesaj**

Yanlış negatifler kredi senaryosunda en pahalı hatadır; bu yüzden sistem daha temkinli davranacak şekilde kurulmuştur.

**Konuşma Notu**

- "Yaklaşık `%79.6` güvenilir, `%20.4` riskli dağılım var."
- "Bu yapı bize klasik accuracy odaklı değil, recall öncelikli bir karar politikası kurdurdu."

## Slayt 3 - Preprocessing ve Feature Engineering

**Amaç**

Modelin başarısının yalnızca algoritmadan gelmediğini, veri hattının da kritik olduğunu göstermek.

**Ekranda Ne Var**

- Temizleme, encoding, scaling, feature engineering akışı
- `DTIRatio` ve `Age_Income_Interaction`

**Ana Mesaj**

CreditScope'un gücü yalnızca model seçiminde değil, eğitim ve inference tarafında aynı preprocessing mantığını korumasındadır.

**Konuşma Notu**

- "`DTIRatio` kredi yükünü gelire bağlayan en önemli yorumlanabilir özelliklerden biri oldu."
- "API tarafında bu özellikler tekrar hesaplandığı için frontend'e körü körüne güvenilmiyor."

## Slayt 4 - Model Karşılaştırması ve Final Seçim

**Amaç**

Neden final modelin XGBoost olarak sabitlendiğini, buna rağmen Logistic Regression'ın neden tamamen reddedilmediğini anlatmak.

**Ekranda Ne Var**

- Model karşılaştırma grafiği
- XGBoost seçimi için gerekçe kartı

**Ana Mesaj**

Final seçim, tek metrik üstünlüğüyle değil; tuning süreci, açıklanabilirlik, entegrasyon ve teslim güveniyle yapıldı.

**Konuşma Notu**

- "Logistic Regression recall tarafında güçlü bir benchmark verdi; ama deploy hattımız XGBoost etrafında olgunlaştı."
- "Bu slaytta model seçim kararının sadece tablo değil, sistem kararı olduğunu vurguluyoruz."

## Slayt 5 - Final Teknik Baseline ve SMOTE Teknik Zorluğu

**Amaç**

Final hattın teknik olarak nasıl dondurulduğunu ve burada SMOTE kullanımının neden önemli ama dikkat gerektiren bir karar olduğunu açıklamak.

**Ekranda Ne Var**

- Shared preprocessing -> SMOTE -> tuned XGBoost -> threshold -> rules akışı
- Recall, threshold, false negative ve feature count kartları

**Ana Mesaj**

Final baseline bilinçli olarak `SMOTE + tuned XGBoost + threshold calibration + business rules` kombinasyonudur.

**SMOTE Teknik Zorluğu Vurgusu**

Bu sunumda `SMOTE teknik zorluğu` en net burada anlatılmalıdır.

**Vurgu Metni**

- "Veri seti dengesiz olduğu için SMOTE kullandık; ancak bu adımı sadece eğitim tarafında ve veri sızıntısı yaratmayacak şekilde uygulamak kritik bir teknik zorluktu."
- "SMOTE tek başına yeterli değildi; oversampling sonrası modelin davranışını threshold kalibrasyonu ile yeniden dengelememiz gerekti."
- "Yani teknik zorluk sadece sınıf çoğaltmak değil, bu çoğaltmanın deploy edilen gerçek karar hattıyla uyumlu kalmasını sağlamaktı."

**Konuşma Notu**

- "Burada özellikle söylememiz gereken şey şu: SMOTE bir sihirli çözüm değil, doğru yerde uygulanmazsa yanıltıcı sonuç üretir."
- "Biz bu zorluğu shared preprocessing, train-only oversampling ve final threshold ayarı ile yönettik."

## Slayt 6 - Business Rules ve Hibrit Karar Destek Mantığı

**Amaç**

Projenin modelden ürüne dönüşmesini sağlayan business-rule katmanını anlatmak.

**Ekranda Ne Var**

- Karar akışı
- Riski azaltan ve artıran kurallar

**Ana Mesaj**

CreditScope bir black-box model değil; model skoru üzerine uzman mantığı ekleyen hibrit sistemdir.

**Konuşma Notu**

- "Ham model skoru ile düzeltilmiş skorun ayrı verilmesi, sistemin denetlenebilir kalmasını sağlıyor."
- "Bu katman özellikle yanlış pozitifleri daha yönetilebilir hale getirmek için önemli."

## Slayt 7 - SHAP ve FP/FN Analizi

**Amaç**

Modelin neden yanıldığını anlamak ve final seçim kararını explainability ile desteklemek.

**Ekranda Ne Var**

- False Positive SHAP
- False Negative SHAP
- Confusion matrix kartları

**Ana Mesaj**

Final model, sadece çalışan model değil; aynı zamanda neden böyle karar verdiği yorumlanabilen modeldir.

**Konuşma Notu**

- "False Positive tarafı sistemin temkinli davranışını gösteriyor."
- "False Negative tarafı ise kaçırmak istemediğimiz kritik riskli profilleri temsil ediyor."

## Slayt 8 - Final Doğrulama, Demo ve Edge-Case

**Amaç**

Sistemin sadece eğitim çıktısıyla değil, uçtan uca kullanım senaryolarıyla doğrulandığını göstermek.

**Ekranda Ne Var**

- UI cockpit görseli
- Dört demo senaryosu
- Smoke test özeti

**Ana Mesaj**

Final teslim hattı eğitim, API, UI ve senaryo bazlı kararlar birlikte düşünülerek doğrulanmıştır.

**Konuşma Notu**

- "4/4 demo senaryosu geçti."
- "Özellikle güçlü kredi skoru ama zayıf istihdam edge-case'i, business-rule mantığının doğru çalıştığını gösterdi."

## Slayt 9 - Future Work ve Kapanış

**Amaç**

Projeyi doğru yerde kapatmak: bitmiş gibi değil, bir sonraki olgunlaşma fazına hazır gibi anlatmak.

**Ekranda Ne Var**

- Beş future work kartı
- Kapanış cümlesi

**Ana Mesaj**

Bir sonraki adım projeyi yeniden kurmak değil; mevcut hibrit hattı daha ölçülebilir, daha izlenebilir ve daha kurumsal hale getirmektir.

**Konuşma Notu**

- "Hafta 11 sonunda elde ettiğimiz şey, salt model performansı değil; teslim edilebilir bir karar destek prototipi."
- "Buradan sonrası research değil, olgunlaştırma işi."

## Sunumda Özellikle Vurgulanacak 3 Cümle

1. "CreditScope'un başarısı tek bir model skorundan değil, recall mantığı, threshold ayarı, explainability ve business rules birleşiminden geliyor."
2. "SMOTE burada basit bir dengeleme tekniği değil, veri sızıntısı ve karar dengesi açısından dikkatle yönetilmiş bir teknik zorluktu."
3. "Hafta 11'in değeri yeni bir şey icat etmek değil, çalışan sistemi akademik ve teknik olarak savunulabilir biçimde mühürlemektir."
