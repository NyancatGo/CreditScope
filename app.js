document.getElementById("loanForm").addEventListener("submit", async function(e) {
    e.preventDefault(); // Sayfa yenilenmesini engelle

    // UI Elementlerini Seçme
    const submitBtn = document.getElementById("submitBtn");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");
    
    const resultCard = document.getElementById("resultDashboardCard");
    const resultTitle = document.getElementById("resultTitle");
    const resultIcon = document.getElementById("resultIcon");
    const resultScore = document.getElementById("resultScore");
    const progressBar = document.getElementById("resultProgressBar");

    // Yükleniyor Durumuna Geçiş
    submitBtn.disabled = true;
    if(btnText) btnText.classList.add("hidden");
    if(btnSpinner) btnSpinner.classList.remove("hidden");
    
    // Önceki sonuç kartını temizle ve gizle
    if(resultCard) {
        resultCard.classList.add("hidden");
        resultCard.classList.remove("status-safe", "status-danger");
    }
    if(progressBar) progressBar.style.width = "0%"; // Çubuğu sıfırla

    // Formdaki verileri JSON objesine çevir
    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        // Rakam girilen alanları Integer/Float'a çeviriyoruz
        if (!isNaN(value) && value.trim() !== '') {
            data[key] = Number(value);
        } else {
            data[key] = value;
        }
    });

    try {
        // FastAPI Backend'ine POST İsteği
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error(`API Hatası! Durum Kodu: ${response.status}`);
        }

        const result = await response.json();
        
        if(resultCard) {
            // Sonuç Kartını Görünür Yap
            resultCard.classList.remove("hidden");
            
            // Gelen API Verisi: result.risk_durumu (0 veya 1) ve result.temerrut_olasiligi (% skor)
            const riskSkoru = parseFloat(result.temerrut_olasiligi);
            
            // Ekrana Yüzdeyi Bas ve Progress Çubuğunu İlerlet
            if(resultScore) resultScore.innerText = `%${riskSkoru.toFixed(2)}`;
            
            // Tarayıcının çubuk animasyonunu rahat algılaması için çok kısa bir gecikme ekliyoruz
            setTimeout(() => {
                if(progressBar) progressBar.style.width = `${riskSkoru}%`;
            }, 100);

            // Duruma Göre Renk ve Metin Değişimi
            if (result.risk_durumu === 0) {
                // Güvenilir - YEŞİL Tema
                resultCard.classList.add("status-safe");
                if(resultIcon) resultIcon.className = "fa-solid fa-shield-check"; // Onay ikonu
                if(resultTitle) resultTitle.innerText = "Güvenilir Kredi Profili";
            } else {
                // Riskli - KIRMIZI Tema
                resultCard.classList.add("status-danger");
                if(resultIcon) resultIcon.className = "fa-solid fa-triangle-exclamation"; // Uyarı ikonu
                if(resultTitle) resultTitle.innerText = "Yüksek Temerrüt Riski!";
            }
        }
        
    } catch (error) {
        alert("Bağlantı hatası oluştu: " + error.message);
    } finally {
        // İşlem bitince Butonu Eski Haline Getir
        submitBtn.disabled = false;
        if(btnText) btnText.classList.remove("hidden");
        if(btnSpinner) btnSpinner.classList.add("hidden");
    }
});
