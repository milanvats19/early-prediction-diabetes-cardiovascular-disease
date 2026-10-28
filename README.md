# Early Prediction of Diabetes & Cardiovascular Disease

A working starter project: two trained ML models, a backend API serving them,
and a website that talks to that API. Everything here already runs end-to-end —
your team's job now is to improve and extend each piece.

---

## 1. What's already built, and what technology does what

```
diabetes-cvd-prediction/
├── backend/
│   ├── app/
│   │   ├── main.py                  <- FastAPI app: defines the API endpoints
│   │   ├── schemas.py                <- Pydantic: defines/validates request & response shape
│   │   ├── train_diabetes_model.py   <- trains the diabetes model
│   │   ├── train_heart_model.py      <- trains the heart disease model
│   │   └── models/                   <- saved trained models (.pkl files)
│   ├── data/
│   │   ├── diabetes.csv              <- Pima Indians Diabetes dataset
│   │   └── heart.csv                 <- UCI Cleveland Heart Disease dataset
│   └── requirements.txt
└── frontend/
    ├── index.html                    <- page structure, two forms (diabetes / heart)
    ├── style.css                     <- all styling
    └── script.js                     <- talks to the API, renders the result
```

| Layer | Technology | Why |
|---|---|---|
| Data handling | **pandas / numpy** | load, clean, and reshape the CSV datasets |
| Modeling | **scikit-learn** (`RandomForestClassifier`) | robust, interpretable, works well on tabular medical data without huge tuning effort |
| Model storage | **joblib** | saves a trained model to a `.pkl` file so the API loads it instantly instead of retraining every time it starts |
| API | **FastAPI** + **Uvicorn** | FastAPI validates incoming data automatically and generates free interactive docs at `/docs`; Uvicorn is the server that runs it |
| Data validation | **Pydantic** (bundled with FastAPI) | rejects bad input (e.g. missing field, wrong type) before it ever reaches the model |
| Frontend | **plain HTML / CSS / JavaScript** | no build tools, no framework — every teammate can read and edit it directly. `fetch()` is used to call the API |
| Cross-origin access | **CORS middleware** (in `main.py`) | lets the frontend (a different "origin"/port) call the backend API from the browser |

No database yet — predictions aren't saved. That's a good "phase 2" feature (see below).

---

## 2. How to run it yourself

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` — you'll get an interactive page to test both endpoints directly.

**Frontend:**
```bash
cd frontend
python3 -m http.server 5500
```
Visit `http://localhost:5500`. It's already wired to call the backend at `http://localhost:8000`.

**Retraining a model** (after you improve the data/features/algorithm):
```bash
cd backend
python3 -m app.train_diabetes_model
python3 -m app.train_heart_model
```
This overwrites the `.pkl` files in `backend/app/models/`. Restart the API server to pick up the new model.

---

## 3. Current model performance

| Model | Algorithm | Accuracy | Recall | ROC-AUC |
|---|---|---|---|---|
| Diabetes | XGBoost | 92% | 90% | 0.98 |
| Heart disease | Random Forest | 82% | 97% | 0.90 |

**Note on the diabetes dataset:** the diabetes model was originally trained on
the Pima Indians Diabetes Dataset, which only contains data from Pima Indian
women — it had no `gender` field at all and would have been misleading if
used for male users. It's since been swapped for a general-population dataset
(~100,000 patients, both genders: `gender, age, hypertension, heart_disease,
smoking_history, bmi, HbA1c_level, blood_glucose_level`). The old file is kept
at `backend/data/diabetes_pima_female_only_OLD.csv` for reference but is no
longer used by any script.

This general dataset appears to be synthetic/heavily processed (very few
unique lab values across 100k rows) — mention this limitation in your report
rather than presenting the accuracy numbers as if from real clinical data.

**Recall matters most here** — in a medical screening tool, missing a true positive
(telling someone they're low-risk when they're not) is worse than a false alarm.
Keep an eye on recall as you tune the models, not just accuracy.

---

## 3b. Database (PostgreSQL) — storing prediction history

Every prediction is now saved to a PostgreSQL database and can be viewed via
the "Recent predictions" table on the website, or through `GET /history/diabetes`
and `GET /history/heart`.

**One-time setup on your laptop:**
1. Install PostgreSQL from https://www.postgresql.org/download/ (Windows: use the installer, remember the password you set for the `postgres` user)
2. Open **pgAdmin** (installed alongside PostgreSQL) or a terminal, and create the project database:
   ```sql
   CREATE DATABASE diabetes_cvd_db;
   ```
3. Set an environment variable so the backend knows how to connect (replace `yourpassword`):
   - **Windows (Command Prompt, before starting the server):**
     ```
     set DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/diabetes_cvd_db
     ```
   - **Mac/Linux:**
     ```
     export DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/diabetes_cvd_db
     ```
   If you skip this step, it falls back to `postgres:postgres@localhost:5432/diabetes_cvd_db` (see `app/database.py`).
4. Start the backend as usual — it automatically creates the two tables (`diabetes_predictions`, `heart_predictions`) on first run.

**What's new in the code:**
- `app/database.py` — connects to PostgreSQL via SQLAlchemy
- `app/db_models.py` — defines the two database tables
- `app/main.py` — every `/predict/...` call now also saves a row; two new endpoints, `/history/diabetes` and `/history/heart`, return the most recent rows
- Frontend — a "Recent predictions" table below each form, refreshed automatically after every prediction

---

## 3c. Login / signup (user accounts)

Every prediction is now tied to a specific logged-in user. The site shows a
login/signup screen first; after logging in, your own predictions and your
own history are the only ones you see.

**How it works:**
- Passwords are never stored in plain text — they're hashed with **bcrypt** (via `passlib`) before being saved
- On login/signup, the backend issues a **JWT** (a signed token) via `python-jose`
- The frontend stores that token in `localStorage` and sends it as an `Authorization: Bearer <token>` header on every request
- `/predict/diabetes`, `/predict/heart`, `/history/diabetes`, and `/history/heart` all now require a valid token — without one, you get a `401 Unauthorized`
- Each prediction row now has a `user_id`, and history endpoints only return rows belonging to the logged-in user

**New files:**
- `app/auth.py` — password hashing, JWT creation/verification, the `get_current_user` dependency
- `app/db_models.py` — added a `User` table, and `user_id` columns on the prediction tables
- `app/schemas.py` — added `UserSignup`, `UserLogin`, `Token`, `UserOut`
- `app/main.py` — added `/auth/signup`, `/auth/login`, `/auth/me`; predict/history endpoints now require login
- Frontend — a login/signup screen (`authScreen` in `index.html`), shown until you log in; a logout button in the sidebar

**One-time setup on your laptop:** nothing extra beyond what you already did for
the database — just `pip install -r requirements.txt` again to pick up the
new packages (`passlib`, `bcrypt`, `python-jose`, `email-validator`).

**Before you deploy this for real:** set your own `SECRET_KEY` environment
variable (a long random string) — the one in `app/auth.py` has an insecure
default so the project runs out of the box, and anyone who knows that default
could forge login tokens. Treat this the same way you treat `DATABASE_URL`.

**What's NOT included yet** (fine for a college project, but worth knowing):
no "forgot password" flow, no email verification, and the JWT secret defaults
to a hardcoded value until you change it. If your report asks about security
considerations, this section is useful to cite.

---

## 4. Suggested team task split

**ML / Data person**
- Try other algorithms (Logistic Regression, XGBoost, SVM) and compare against the Random Forest baseline in `train_diabetes_model.py` / `train_heart_model.py`
- Do proper hyperparameter tuning (`GridSearchCV` or `RandomizedSearchCV`)
- Add more EDA — correlation heatmaps, distribution plots — for the report
- Investigate combining both conditions into a shared risk score (they share risk factors: BMI, age, blood pressure)

**Backend person**
- Add a database (SQLite or PostgreSQL) to store submitted predictions/history
- Add user accounts (signup/login) if the project needs personalized history
- Add input sanity checks beyond basic range validation (e.g. warn on physiologically implausible combinations)
- Prepare for deployment (Render/Railway/Fly.io)

**Frontend person**
- Turn `index.html` into a multi-page or React-based app if the project outgrows a single page
- Add a results history view (once the backend has a database)
- Improve accessibility and mobile polish
- Add charts (e.g. Chart.js) comparing the user's inputs against healthy reference ranges

**Integration / PM person**
- Own the GitHub repo, branch strategy, and code reviews
- Handle deployment of both frontend and backend so there's a live link to demo
- Write up the project report/documentation, including model limitations
- Coordinate the mobile app phase (see below)

If your team has more or fewer members, this is just a starting split — happy to
adjust it to your actual headcount.

---

## 5. Phase 2: turning this into a mobile app

Because the backend is a standalone API, a mobile app doesn't need a rewrite of
any prediction logic — it just calls the same endpoints:
- **React Native** (reuses JS/React knowledge if your frontend person picks it up) or **Flutter**
- Point it at the same `/predict/diabetes` and `/predict/heart` endpoints
- Once deployed, swap `API_BASE_URL` in the app config from `localhost` to your real backend URL

---

## 6. Important note for your report/demo

Be upfront about limitations: these models are trained on relatively small,
older public datasets (Pima Indians Diabetes study; UCI Cleveland Heart Disease
study) and are **not** validated for real clinical use. Frame this project as an
early-screening decision-support tool, not a diagnostic device — that framing is
both more accurate and stronger for an academic evaluation.
