const form = document.getElementById("loanForm");
const resetButton = document.getElementById("resetForm");
const presetButtons = document.querySelectorAll("[data-preset]");

const el = {
    submitBtn: document.getElementById("submitBtn"),
    btnText: document.getElementById("btnText"),
    btnSpinner: document.getElementById("btnSpinner"),
    resultCard: document.getElementById("resultDashboardCard"),
    resultTitle: document.getElementById("resultTitle"),
    resultIcon: document.getElementById("resultIcon"),
    resultScore: document.getElementById("resultScore"),
    modelScore: document.getElementById("modelScore"),
    adjustedScore: document.getElementById("adjustedScore"),
    thresholdScore: document.getElementById("thresholdScore"),
    dtiPreview: document.getElementById("dtiPreview"),
    spectrumMarker: document.getElementById("spectrumMarker"),
    ruleList: document.getElementById("ruleList"),
    decisionMemo: document.getElementById("decisionMemo"),
    liveDti: document.getElementById("liveDti"),
    liveLoanIncome: document.getElementById("liveLoanIncome"),
    livePayment: document.getElementById("livePayment"),
    liveCreditBand: document.getElementById("liveCreditBand")
};

const presets = {
    safe: {
        Age: 38,
        Income: 92000,
        LoanAmount: 18000,
        CreditScore: 790,
        MonthsEmployed: 84,
        NumCreditLines: 2,
        InterestRate: 9.8,
        LoanTerm: 36,
        Education: "Master's",
        EmploymentType: "Full-time",
        MaritalStatus: "Married",
        HasMortgage: "No",
        HasDependents: "No",
        LoanPurpose: "Auto",
        HasCoSigner: "Yes"
    },
    review: {
        Age: 31,
        Income: 54000,
        LoanAmount: 29500,
        CreditScore: 665,
        MonthsEmployed: 22,
        NumCreditLines: 4,
        InterestRate: 16.4,
        LoanTerm: 48,
        Education: "Bachelor's",
        EmploymentType: "Self-employed",
        MaritalStatus: "Single",
        HasMortgage: "No",
        HasDependents: "Yes",
        LoanPurpose: "Business",
        HasCoSigner: "No"
    },
    risky: {
        Age: 24,
        Income: 36000,
        LoanAmount: 34000,
        CreditScore: 560,
        MonthsEmployed: 4,
        NumCreditLines: 6,
        InterestRate: 23.5,
        LoanTerm: 60,
        Education: "High School",
        EmploymentType: "Unemployed",
        MaritalStatus: "Single",
        HasMortgage: "Yes",
        HasDependents: "Yes",
        LoanPurpose: "Other",
        HasCoSigner: "No"
    }
};

function money(value) {
    if (!Number.isFinite(value)) return "-";
    return new Intl.NumberFormat("tr-TR", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 0
    }).format(value);
}

function percent(value) {
    if (!Number.isFinite(value)) return "-";
    return `%${value.toFixed(2)}`;
}

function numberValue(id) {
    return Number(document.getElementById(id)?.value);
}

function calculateMonthlyPayment(principal, annualRate, termMonths) {
    if (!principal || !termMonths) return NaN;
    const monthlyRate = annualRate / 100 / 12;
    if (!monthlyRate) return principal / termMonths;
    return principal * (monthlyRate * (1 + monthlyRate) ** termMonths) / ((1 + monthlyRate) ** termMonths - 1);
}

function creditBand(score) {
    if (!Number.isFinite(score) || score <= 0) return "-";
    if (score >= 750) return "Prime";
    if (score >= 670) return "Güçlü";
    if (score >= 600) return "Orta";
    return "Zayıf";
}

function calculateDtiRatio() {
    const income = numberValue("Income");
    const loanAmount = numberValue("LoanAmount");
    const dtiInput = document.getElementById("DTIRatio");

    if (!Number.isFinite(income) || income <= 0 || !Number.isFinite(loanAmount)) {
        dtiInput.value = "";
        updateLivePreview(null);
        return null;
    }

    const dtiRatio = loanAmount / income;
    dtiInput.value = dtiRatio.toFixed(4);
    updateLivePreview(dtiRatio);
    return dtiRatio;
}

function updateLivePreview(dtiRatio = null) {
    const income = numberValue("Income");
    const loanAmount = numberValue("LoanAmount");
    const interestRate = numberValue("InterestRate");
    const loanTerm = numberValue("LoanTerm");
    const creditScore = numberValue("CreditScore");
    const ratio = dtiRatio ?? (income > 0 ? loanAmount / income : NaN);
    const payment = calculateMonthlyPayment(loanAmount, interestRate, loanTerm);

    el.liveDti.innerText = Number.isFinite(ratio) ? ratio.toFixed(4) : "-";
    el.liveLoanIncome.innerText = Number.isFinite(ratio) ? `${(ratio * 100).toFixed(1)}%` : "-";
    el.livePayment.innerText = money(payment);
    el.liveCreditBand.innerText = creditBand(creditScore);
    el.dtiPreview.innerText = Number.isFinite(ratio) ? ratio.toFixed(4) : "-";
}

function formDataToPayload() {
    const formData = new FormData(form);
    const data = {};

    formData.forEach((value, key) => {
        if (!Number.isNaN(Number(value)) && String(value).trim() !== "") {
            data[key] = Number(value);
        } else {
            data[key] = value;
        }
    });

    const computedDtiRatio = calculateDtiRatio();
    if (computedDtiRatio !== null) {
        data.DTIRatio = computedDtiRatio;
    }

    return data;
}

function setLoading(isLoading) {
    el.submitBtn.disabled = isLoading;
    el.btnText.classList.toggle("hidden", isLoading);
    el.btnSpinner.classList.toggle("hidden", !isLoading);
}

function resetResultPanel() {
    el.resultCard.classList.remove("status-safe", "status-danger");
    el.resultCard.style.setProperty("--risk-angle", "0deg");
    el.resultCard.style.setProperty("--risk-left", "0%");
    el.resultTitle.innerText = "Analiz Bekliyor";
    el.resultIcon.innerHTML = '<i class="fa-solid fa-gauge-high"></i>';
    el.resultScore.innerText = "%0.00";
    el.modelScore.innerText = "%0.00";
    el.adjustedScore.innerText = "%0.00";
    el.thresholdScore.innerText = "0.231";
    el.dtiPreview.innerText = "-";
    el.decisionMemo.innerText = "Başvuru çalıştırıldığında model skoru ve iş kurallarıyla birlikte karar notu üretilecek.";
    el.ruleList.innerHTML = '<span class="empty-rule">Henüz kural çalışmadı</span>';
    updateLivePreview(null);
}

function renderRules(rules) {
    if (!Array.isArray(rules) || rules.length === 0) {
        el.ruleList.innerHTML = '<span class="empty-rule">Kural uygulanmadı</span>';
        return;
    }

    el.ruleList.innerHTML = "";
    rules.forEach((rule) => {
        const item = document.createElement("span");
        item.className = "rule-pill";
        item.title = rule.reason || rule.id;
        item.innerHTML = `<span>${rule.id.replaceAll("_", " ")}</span><small>%${Number(rule.after || 0).toFixed(2)}</small>`;
        el.ruleList.appendChild(item);
    });
}

function decisionMemo(result) {
    const risk = Number(result.temerrut_olasiligi || 0);
    const model = Number(result.model_temerrut_olasiligi || 0);
    const dti = Number(result.hesaplanan_dti || 0);
    const rules = Array.isArray(result.business_rule_adjustments) ? result.business_rule_adjustments.length : 0;

    if (result.risk_durumu === 0) {
        return `Başvuru onaylanabilir bölgede. Model skoru ${percent(model)}, iş kuralları sonrası skor ${percent(risk)}. DTI ${dti.toFixed(4)} ve ${rules} kural etkisiyle profil düşük risk bandında kaldı.`;
    }

    return `Başvuru manuel inceleme bandında. Model skoru ${percent(model)}, iş kuralları sonrası skor ${percent(risk)}. DTI ${dti.toFixed(4)}; gelir yükü, kredi notu ve istihdam sinyalleri kredi uzmanı tarafından gözden geçirilmeli.`;
}

function renderResult(result) {
    const riskScore = Number(result.temerrut_olasiligi || 0);
    const modelScore = Number(result.model_temerrut_olasiligi || 0);
    const riskAngle = Math.max(0, Math.min(100, riskScore)) * 3.6;
    const riskLeft = Math.max(0, Math.min(100, riskScore));

    el.resultCard.classList.remove("status-safe", "status-danger");
    el.resultCard.classList.add(result.risk_durumu === 0 ? "status-safe" : "status-danger");
    el.resultCard.style.setProperty("--risk-angle", `${riskAngle}deg`);
    el.resultCard.style.setProperty("--risk-left", `${riskLeft}%`);

    el.resultScore.innerText = percent(riskScore);
    el.modelScore.innerText = percent(modelScore);
    el.adjustedScore.innerText = percent(riskScore);
    el.thresholdScore.innerText = Number(result.decision_threshold || 0).toFixed(3);
    el.dtiPreview.innerText = Number(result.hesaplanan_dti || 0).toFixed(4);

    if (result.risk_durumu === 0) {
        el.resultTitle.innerText = "Onaylanabilir Profil";
        el.resultIcon.innerHTML = '<i class="fa-solid fa-shield-check"></i>';
    } else {
        el.resultTitle.innerText = "Manuel İnceleme";
        el.resultIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
    }

    el.decisionMemo.innerText = decisionMemo(result);
    renderRules(result.business_rule_adjustments);
}

function applyPreset(name) {
    const preset = presets[name];
    if (!preset) return;

    Object.entries(preset).forEach(([key, value]) => {
        const field = document.getElementById(key);
        if (field) field.value = value;
    });

    calculateDtiRatio();
    resetResultPanel();
    calculateDtiRatio();
}

["Age", "Income", "LoanAmount", "CreditScore", "InterestRate", "LoanTerm"].forEach((id) => {
    const field = document.getElementById(id);
    if (field) field.addEventListener("input", calculateDtiRatio);
});

presetButtons.forEach((button) => {
    button.addEventListener("click", () => applyPreset(button.dataset.preset));
});

resetButton.addEventListener("click", () => {
    form.reset();
    resetResultPanel();
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setLoading(true);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(formDataToPayload())
        });

        if (!response.ok) {
            throw new Error(`API yanıtı: ${response.status}`);
        }

        renderResult(await response.json());
    } catch (error) {
        el.resultCard.classList.remove("status-safe");
        el.resultCard.classList.add("status-danger");
        el.resultTitle.innerText = "Bağlantı Hatası";
        el.resultIcon.innerHTML = '<i class="fa-solid fa-plug-circle-xmark"></i>';
        el.decisionMemo.innerText = error.message;
        el.ruleList.innerHTML = '<span class="empty-rule">API bağlantısı kurulamadı</span>';
    } finally {
        setLoading(false);
    }
});

resetResultPanel();
