document.getElementById("loanForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const submitBtn = document.getElementById("submitBtn");
    const resultBox = document.getElementById("resultBox");
    const resultTitle = document.getElementById("resultTitle");
    const resultDesc = document.getElementById("resultDesc");
    
    // Change Button State to Loading
    const originalBtnText = submitBtn.innerText;
    submitBtn.innerText = "Risk Analiz Ediliyor...";
    submitBtn.disabled = true;
    
    // Hide previous results
    resultBox.classList.add("hidden");
    resultBox.className = "hidden"; // Clears success or danger tags completely

    // Prepare JSON from Form
    const formData = new FormData(this);
    const data = {};
    
    formData.forEach((value, key) => {
        // Convert input numbers to JS numbers, keep strings (categories) as string
        if (!isNaN(value) && value.trim() !== '') {
            data[key] = Number(value);
        } else {
            data[key] = value;
        }
    });

    try {
        // Send POST request to FastAPI backend
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        // Error Handling
        if (!response.ok) {
            throw new Error(`Sunucu Hatası (API'nin çalıştığından emin olun!). Durum Kodu: ${response.status}`);
        }

        const result = await response.json();
        
        // Show Result Container
        resultBox.classList.remove("hidden");
        
        // Logical check from prediction (0 = Safe, 1 = Default)
        if (result.risk_durumu === 0) {
            resultBox.classList.add("success-result");
            resultTitle.innerText = "✅ Kredi Onaylanabilir - Güvenilir Profil";
            resultDesc.innerText = `Tahmin Edilen Temerrüt (Ödememe) İhtimali: %${result.temerrut_olasiligi}`;
        } else {
            resultBox.classList.add("danger-result");
            resultTitle.innerText = "⚠️ Yüksek Risk - Temerrüt İhtimali";
            resultDesc.innerText = `Tahmin Edilen Temerrüt (Ödememe) İhtimali: %${result.temerrut_olasiligi}`;
        }
        
    } catch (error) {
        alert("Bir hata oluştu: " + error.message);
    } finally {
        // Restore Button State
        submitBtn.innerText = originalBtnText;
        submitBtn.disabled = false;
    }
});
