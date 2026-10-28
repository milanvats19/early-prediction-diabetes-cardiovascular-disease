/*
  Technology: plain JavaScript + the Fetch API (no framework needed).
  This file:
   1. Switches between the "Diabetes" and "Heart disease" tabs
   2. Reads form values, sends them to the FastAPI backend
   3. Renders the returned prediction (risk %, label, top factors)

  CHANGE THIS if your backend runs somewhere else (e.g. after you deploy it):
*/
const API_BASE_URL = "http://localhost:8000";

// =====================================================================
// Auth: login/signup, storing the token, and attaching it to API calls
// =====================================================================
// Technology: the JWT the backend issues on login is stored in the
// browser's localStorage, so the person stays logged in even after
// closing the tab. Every API call below reads it and sends it as an
// "Authorization: Bearer <token>" header.

const authScreen = document.getElementById("authScreen");
const appShell = document.getElementById("appShell");
const authError = document.getElementById("authError");
const userEmailEl = document.getElementById("userEmail");

function getToken() {
  return localStorage.getItem("token");
}

function saveSession(token, user) {
  localStorage.setItem("token", token);
  localStorage.setItem("userEmail", user.email);
}

function clearSession() {
  localStorage.removeItem("token");
  localStorage.removeItem("userEmail");
}

function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function showApp() {
  authScreen.classList.add("hidden");
  appShell.classList.remove("hidden");
  userEmailEl.textContent = localStorage.getItem("userEmail") || "";
  // these are defined further down in this file
  checkApiStatus();
  loadHistory();
}

function showAuthScreen(message) {
  appShell.classList.add("hidden");
  authScreen.classList.remove("hidden");
  if (message) {
    authError.textContent = message;
    authError.classList.remove("hidden");
  } else {
    authError.classList.add("hidden");
  }
}

// ---- Auth tab switching (Log in / Sign up) ----
const authTabs = document.querySelectorAll(".auth-tab");
const loginForm = document.getElementById("loginForm");
const signupForm = document.getElementById("signupForm");

authTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    authTabs.forEach((t) => t.classList.toggle("active", t === tab));
    const showLogin = tab.dataset.form === "login";
    loginForm.classList.toggle("hidden", !showLogin);
    signupForm.classList.toggle("hidden", showLogin);
    authError.classList.add("hidden");
  });
});

loginForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = document.getElementById("login_email").value;
  const password = document.getElementById("login_password").value;
  try {
    const res = await fetch(`${API_BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Login failed");
    saveSession(data.access_token, data.user);
    showApp();
  } catch (err) {
    showAuthScreen(err.message === "Failed to fetch" ? "Can't reach the API — is the backend running?" : err.message);
  }
});

signupForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const email = document.getElementById("signup_email").value;
  const password = document.getElementById("signup_password").value;
  try {
    const res = await fetch(`${API_BASE_URL}/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Signup failed");
    saveSession(data.access_token, data.user);
    showApp();
  } catch (err) {
    showAuthScreen(err.message === "Failed to fetch" ? "Can't reach the API — is the backend running?" : err.message);
  }
});

document.getElementById("logoutBtn").addEventListener("click", () => {
  clearSession();
  showAuthScreen();
});

// ---------- Tab switching ----------
const tabs = document.querySelectorAll(".tab");
const panels = {
  diabetes: document.getElementById("panel-diabetes"),
  heart: document.getElementById("panel-heart"),
};
const pageTitle = document.getElementById("pageTitle");
const pageSub = document.getElementById("pageSub");

const COPY = {
  diabetes: {
    title: "Diabetes risk",
    sub: "Based on the Pima Indians Diabetes study",
    modelName: "Diabetes model",
  },
  heart: {
    title: "Heart disease risk",
    sub: "Based on the UCI Cleveland Heart Disease study",
    modelName: "Heart disease model",
  },
};
COPY.diabetes.sub = "General population risk model";

let activePanel = "diabetes";

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.panel;
    activePanel = target;

    tabs.forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    Object.entries(panels).forEach(([key, el]) => {
      el.classList.toggle("hidden", key !== target);
    });

    pageTitle.textContent = COPY[target].title;
    pageSub.textContent = COPY[target].sub;
    resetResultPanel();
    loadHistory();
  });
});

// ---------- Result panel elements ----------
const resultEmpty = document.getElementById("resultEmpty");
const resultBody = document.getElementById("resultBody");
const resultError = document.getElementById("resultError");
const errorText = document.getElementById("errorText");
const resultModelName = document.getElementById("resultModelName");
const dialFill = document.getElementById("dialFill");
const dialValue = document.getElementById("dialValue");
const riskLabel = document.getElementById("riskLabel");
const riskCopy = document.getElementById("riskCopy");
const factorsList = document.getElementById("factorsList");

const RISK_COPY = {
  "Low Risk": "Indicators are broadly within a typical range.",
  "Moderate Risk": "Some indicators are outside a typical range — worth discussing with a doctor.",
  "High Risk": "Several indicators suggest elevated risk — a clinical check-up is recommended.",
};

const RISK_COLOR_VAR = {
  "Low Risk": "--risk-low",
  "Moderate Risk": "--risk-mod",
  "High Risk": "--risk-high",
};

function resetResultPanel() {
  resultEmpty.classList.remove("hidden");
  resultBody.classList.add("hidden");
  resultError.classList.add("hidden");
}

function showResult(data) {
  resultEmpty.classList.add("hidden");
  resultError.classList.add("hidden");
  resultBody.classList.remove("hidden");

  resultModelName.textContent = COPY[activePanel].modelName;

  const pct = Math.round(data.probability * 100);
  dialValue.textContent = pct + "%";

  // dasharray total length of the arc path is 157 (see CSS/SVG)
  const offset = 157 - (157 * data.probability);
  dialFill.style.strokeDashoffset = offset;

  const colorVar = RISK_COLOR_VAR[data.risk_label] || "--risk-mod";
  dialFill.style.stroke = getComputedStyle(document.documentElement).getPropertyValue(colorVar);

  riskLabel.textContent = data.risk_label;
  riskLabel.style.color = getComputedStyle(document.documentElement).getPropertyValue(colorVar);
  riskCopy.textContent = RISK_COPY[data.risk_label] || "";

  factorsList.innerHTML = "";
  data.top_factors.forEach((f) => {
    const li = document.createElement("li");
    li.textContent = humanizeFactor(f);
    factorsList.appendChild(li);
  });
}

function showError(message) {
  resultEmpty.classList.add("hidden");
  resultBody.classList.add("hidden");
  resultError.classList.remove("hidden");
  errorText.textContent = message;
}

const FACTOR_LABELS = {
  age: "Age",
  bmi: "BMI",
  hypertension: "Hypertension",
  heart_disease: "Existing heart disease",
  HbA1c_level: "HbA1c level",
  blood_glucose_level: "Blood glucose level",
  "gender_Female": "Gender",
  "gender_Male": "Gender",
  "gender_Other": "Gender",
  "smoking_history_never": "Smoking history",
  "smoking_history_current": "Smoking history",
  "smoking_history_former": "Smoking history",
  "smoking_history_not current": "Smoking history",
  "smoking_history_ever": "Smoking history",
  "smoking_history_No Info": "Smoking history",
  cp: "Chest pain type",
  thal: "Thalassemia result",
  oldpeak: "ST depression (oldpeak)",
  thalach: "Max heart rate",
  ca: "Major vessels colored",
  exang: "Exercise-induced angina",
  age: "Age",
  slope: "ST slope",
  chol: "Cholesterol",
  trestbps: "Resting blood pressure",
  sex: "Sex",
  restecg: "Resting ECG result",
  fbs: "Fasting blood sugar",
};
function humanizeFactor(key) {
  return FACTOR_LABELS[key] || key;
}

// ---------- Form submission ----------
async function postPrediction(endpoint, payload) {
  const submitButtons = document.querySelectorAll(".submit-btn");
  submitButtons.forEach((b) => (b.disabled = true));

  try {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(payload),
    });

    if (res.status === 401) {
      clearSession();
      showAuthScreen("Your session expired — please log in again.");
      return;
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    showResult(data);
    loadHistory();
  } catch (err) {
    showError(
      err.message === "Failed to fetch"
        ? "Can't reach the API. Is the backend server running on " + API_BASE_URL + "?"
        : err.message
    );
  } finally {
    submitButtons.forEach((b) => (b.disabled = false));
  }
}

document.getElementById("diabetesForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const payload = {
    gender: document.getElementById("d_gender").value,
    age: Number(document.getElementById("d_age").value),
    hypertension: Number(document.getElementById("d_hypertension").value),
    heart_disease: Number(document.getElementById("d_heart_disease").value),
    smoking_history: document.getElementById("d_smoking").value,
    bmi: Number(document.getElementById("d_bmi").value),
    HbA1c_level: Number(document.getElementById("d_hba1c").value),
    blood_glucose_level: Number(document.getElementById("d_glucose").value),
  };
  postPrediction("/predict/diabetes", payload);
});

document.getElementById("heartForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const payload = {
    age: Number(document.getElementById("h_age").value),
    sex: Number(document.getElementById("h_sex").value),
    cp: Number(document.getElementById("h_cp").value),
    trestbps: Number(document.getElementById("h_trestbps").value),
    chol: Number(document.getElementById("h_chol").value),
    fbs: Number(document.getElementById("h_fbs").value),
    restecg: Number(document.getElementById("h_restecg").value),
    thalach: Number(document.getElementById("h_thalach").value),
    exang: Number(document.getElementById("h_exang").value),
    oldpeak: Number(document.getElementById("h_oldpeak").value),
    slope: Number(document.getElementById("h_slope").value),
    ca: Number(document.getElementById("h_ca").value),
    thal: Number(document.getElementById("h_thal").value),
  };
  postPrediction("/predict/heart", payload);
});

// ---------- History table ----------
const historyEmpty = document.getElementById("historyEmpty");
const historyTable = document.getElementById("historyTable");
const historyHead = document.getElementById("historyHead");
const historyBody = document.getElementById("historyBody");
const refreshHistoryBtn = document.getElementById("refreshHistory");

const HISTORY_COLUMNS = {
  diabetes: [
    { key: "created_at", label: "When" },
    { key: "gender", label: "Gender" },
    { key: "age", label: "Age" },
    { key: "bmi", label: "BMI" },
    { key: "risk_label", label: "Result" },
  ],
  heart: [
    { key: "created_at", label: "When" },
    { key: "age", label: "Age" },
    { key: "chol", label: "Cholesterol" },
    { key: "thalach", label: "Max HR" },
    { key: "risk_label", label: "Result" },
  ],
};

const RISK_PILL_BG = {
  "Low Risk": "#2F8F6B",
  "Moderate Risk": "#C9852E",
  "High Risk": "#B84C3B",
};

function formatWhen(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

async function loadHistory() {
  if (!getToken()) return; // not logged in yet
  const endpoint = activePanel === "diabetes" ? "/history/diabetes" : "/history/heart";
  try {
    const res = await fetch(`${API_BASE_URL}${endpoint}?limit=10`, { headers: authHeaders() });
    if (res.status === 401) {
      clearSession();
      showAuthScreen("Your session expired — please log in again.");
      return;
    }
    if (!res.ok) throw new Error();
    const rows = await res.json();
    renderHistory(rows);
  } catch {
    historyEmpty.textContent = "Couldn't load history — is the backend running?";
    historyEmpty.classList.remove("hidden");
    historyTable.classList.add("hidden");
  }
}

function renderHistory(rows) {
  if (!rows.length) {
    historyEmpty.textContent = "No predictions saved yet.";
    historyEmpty.classList.remove("hidden");
    historyTable.classList.add("hidden");
    return;
  }

  historyEmpty.classList.add("hidden");
  historyTable.classList.remove("hidden");

  const columns = HISTORY_COLUMNS[activePanel];
  historyHead.innerHTML = "<tr>" + columns.map((c) => `<th>${c.label}</th>`).join("") + "</tr>";

  historyBody.innerHTML = rows
    .map((row) => {
      const cells = columns
        .map((c) => {
          if (c.key === "created_at") return `<td>${formatWhen(row.created_at)}</td>`;
          if (c.key === "risk_label") {
            const bg = RISK_PILL_BG[row.risk_label] || "#999";
            return `<td><span class="risk-pill" style="background:${bg}">${row.risk_label}</span></td>`;
          }
          return `<td>${row[c.key]}</td>`;
        })
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
}

refreshHistoryBtn.addEventListener("click", loadHistory);

// ---------- API health check (shows the dot in sidebar) ----------
const apiDot = document.getElementById("apiDot");
const apiStatus = document.getElementById("apiStatus");

function checkApiStatus() {
  fetch(`${API_BASE_URL}/`)
    .then((res) => {
      if (!res.ok) throw new Error();
      apiDot.classList.remove("down");
      apiDot.classList.add("ok");
      apiStatus.textContent = "API connected";
    })
    .catch(() => {
      apiDot.classList.remove("ok");
      apiDot.classList.add("down");
      apiStatus.textContent = "API offline";
    });
}

// ---------- Decide whether to show the app or the login screen ----------
if (getToken()) {
  showApp();
} else {
  showAuthScreen();
}
