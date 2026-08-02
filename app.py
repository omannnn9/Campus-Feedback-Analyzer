"""
Sections (top to bottom):
    1. setup      - creates database.db, 5 tables, demo users, loads the CSV
    2. ML         - TF-IDF + Naive Bayes category model (scikit-learn) + accuracy
    3. login      - login / signup (student, lecturer, management)
    4. feedback   - submit feedback, sentiment model, predicted category
    5. dashboard  - counts + model accuracy for lecturer / management dashboards
    6. events     - lecturer creates an event, students join with the code
    7. AI         - Gemini summary (cached in SQLite) and a keyword chatbot
"""

import os
import random
import sqlite3
import string

import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "database.db")
CSV = os.path.join(HERE, "student_feedback_dataset.csv")

# .env  ->  GEMINI_API_KEY=your_key_here   (read without any extra library)
if os.path.exists(os.path.join(HERE, ".env")):
    for line in open(os.path.join(HERE, ".env")):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"').strip("'"))

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "gemini-1.5-flash:generateContent")
AI_LIMIT = 30          # maximum Gemini calls per server run (quota safety)
ai_calls = 0

CATEGORIES = ["Lecturer", "Canteen", "Hostel", "Cleanliness", "Security"]

app = Flask(__name__, static_folder=HERE, static_url_path="")




def db():
    """A new SQLite connection - simple and good enough for a demo."""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


# ------------------------------------------------------------------ 1. setup
def setup():
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE,
            password TEXT, role TEXT, sid TEXT);

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY, user_id INTEGER, category TEXT,
            text TEXT, sentiment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP);

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, title TEXT, date TEXT,
            venue TEXT, code TEXT UNIQUE);

        CREATE TABLE IF NOT EXISTS registrations (
            id INTEGER PRIMARY KEY, event_id INTEGER, user_id INTEGER);

        CREATE TABLE IF NOT EXISTS ai_cache (
            category TEXT PRIMARY KEY, summary TEXT);
        """
    )

    if not conn.execute("SELECT 1 FROM users").fetchone():
        conn.executemany(
            "INSERT INTO users (name, email, password, role, sid) VALUES (?,?,?,?,?)",
            [("Anita Rao", "student@campus.edu", "student123", "student", "SID1001"),
             ("Dr. Mehta", "lecturer@campus.edu", "lecturer123", "lecturer", None),
             ("Admin Office", "management@campus.edu", "manage123", "management", None)],
        )

    # pandas reads the 500-row dataset once
    if not conn.execute("SELECT 1 FROM feedback").fetchone() and os.path.exists(CSV):
        data = pd.read_csv(CSV)
        conn.executemany(
            "INSERT INTO feedback (user_id, category, text, sentiment) VALUES (1,?,?,?)",
            data[["category", "feedback_text", "sentiment"]].values.tolist(),
        )

    conn.commit()
    conn.close()


# --------------------------------------------------------------------- 2. ML
# Two classic models, same workflow:
#   pandas -> TF-IDF -> train_test_split -> fit -> accuracy_score
# 1) category  : TF-IDF + Naive Bayes
# 2) sentiment : TF-IDF + Logistic Regression (replaces the old lexicon approach)
cat_vectorizer = TfidfVectorizer(stop_words="english")
cat_model = MultinomialNB()

sent_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
sent_model = LogisticRegression(max_iter=1000)

ACCURACY = 0.0            # category model accuracy (%)
SENT_ACCURACY = 0.0       # sentiment model accuracy (%)


def train_model():
    """Trains both classifiers once at start-up and stores their accuracy."""
    global ACCURACY, SENT_ACCURACY
    data = pd.read_csv(CSV)

    # --- category model ---
    x_train, x_test, y_train, y_test = train_test_split(
        data["feedback_text"], data["category"], test_size=0.2, random_state=42)
    cat_model.fit(cat_vectorizer.fit_transform(x_train), y_train)
    ACCURACY = round(accuracy_score(
        y_test, cat_model.predict(cat_vectorizer.transform(x_test))) * 100, 2)

    # --- sentiment model ---
    a_train, a_test, b_train, b_test = train_test_split(
        data["feedback_text"], data["sentiment"], test_size=0.2, random_state=42)
    sent_model.fit(sent_vectorizer.fit_transform(a_train), b_train)
    SENT_ACCURACY = round(accuracy_score(
        b_test, sent_model.predict(sent_vectorizer.transform(a_test))) * 100, 2)

    print(f"Category accuracy: {ACCURACY}%  |  Sentiment accuracy: {SENT_ACCURACY}%")


def predict_category(text):
    """The trained model decides the category when the student does not pick one."""
    return cat_model.predict(cat_vectorizer.transform([text]))[0]


# ------------------------------------------------------------------ 3. login
@app.post("/api/login")
def login():
    d = request.get_json() or {}
    conn = db()
    user = conn.execute("SELECT * FROM users WHERE email=? AND password=?",
                        (d.get("email", "").lower(), d.get("password", ""))).fetchone()
    conn.close()
    if not user:
        return jsonify({"error": "Invalid email or password"}), 401
    return jsonify({"id": user["id"], "name": user["name"], "role": user["role"]})


@app.post("/api/signup")
def signup():
    d = request.get_json() or {}
    if d.get("role") not in ("student", "lecturer", "management"):
        return jsonify({"error": "Pick a role"}), 400
    if (d.get("password") or "") != d.get("confirm"):
        return jsonify({"error": "Passwords do not match"}), 400
    conn = db()
    try:
        cur = conn.execute(
            "INSERT INTO users (name, email, password, role, sid) VALUES (?,?,?,?,?)",
            (d.get("name"), d.get("email", "").lower(), d.get("password"),
             d["role"], d.get("sid")))
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered"}), 409
    finally:
        conn.close()
    return jsonify({"id": cur.lastrowid, "name": d.get("name"), "role": d["role"]})


# --------------------------------------------------------------- 4. feedback
def sentiment_of(text):
    """TF-IDF + Logistic Regression predicts Positive / Neutral / Negative."""
    return sent_model.predict(sent_vectorizer.transform([text]))[0]


@app.post("/api/feedback")
def add_feedback():
    d = request.get_json() or {}
    text = (d.get("text") or "").strip()
    if len(text) < 10:
        return jsonify({"error": "Write at least 10 characters"}), 400

    category = d.get("category") if d.get("category") in CATEGORIES else predict_category(text)
    sentiment = sentiment_of(text)
    conn = db()
    conn.execute("INSERT INTO feedback (user_id, category, text, sentiment) VALUES (?,?,?,?)",
                 (d.get("user_id"), category, text, sentiment))
    conn.commit()
    conn.close()
    return jsonify({"message": "Feedback submitted anonymously",
                    "category": category, "sentiment": sentiment})


@app.get("/api/feedback")
def list_feedback():
    """Privacy: name, SID and email are never selected, so staff only ever
    see 'Anonymous Student'."""
    conn = db()
    rows = conn.execute("SELECT id, category, text, sentiment, created_at "
                        "FROM feedback ORDER BY id DESC LIMIT 300").fetchall()
    conn.close()
    return jsonify([{**dict(r), "student": "Anonymous Student"} for r in rows])


# -------------------------------------------------------------- 5. dashboard
@app.get("/api/stats")
def stats():
    conn = db()

    def counts(column):
        rows = conn.execute(
            f"SELECT {column} AS k, COUNT(*) AS n FROM feedback GROUP BY {column}").fetchall()
        return {r["k"]: r["n"] for r in rows}

    sent = counts("sentiment")
    cats = counts("category")
    total = conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]
    conn.close()
    return jsonify({"total": total,
                    "positive": sent.get("Positive", 0),
                    "neutral": sent.get("Neutral", 0),
                    "negative": sent.get("Negative", 0),
                    "accuracy": ACCURACY,
                    "sentiment_accuracy": SENT_ACCURACY,
                    "categories": cats})


# ----------------------------------------------------------------- 6. events
@app.post("/api/events")
def create_event():
    d = request.get_json() or {}
    code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    conn = db()
    conn.execute("INSERT INTO events (title, date, venue, code) VALUES (?,?,?,?)",
                 (d.get("title"), d.get("date"), d.get("venue"), code))
    conn.commit()
    conn.close()
    return jsonify({"code": code})


@app.get("/api/events")
def list_events():
    conn = db()
    rows = conn.execute(
        "SELECT e.*, (SELECT COUNT(*) FROM registrations r WHERE r.event_id=e.id) "
        "AS registered FROM events e ORDER BY e.date").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.post("/api/register")
def register():
    d = request.get_json() or {}
    conn = db()
    event = conn.execute("SELECT id FROM events WHERE code=?",
                         (d.get("code", "").upper(),)).fetchone()
    if not event:
        conn.close()
        return jsonify({"error": "Event code not found"}), 404
    conn.execute("INSERT INTO registrations (event_id, user_id) VALUES (?,?)",
                 (event["id"], d.get("user_id")))
    conn.commit()
    conn.close()
    return jsonify({"message": "Registered for the event"})


# --------------------------------------------------------------------- 7. AI
def ask_gemini(prompt):
    """The ONLY place Gemini is used. Everything else runs locally."""
    global ai_calls
    if not GEMINI_KEY:
        return "Add GEMINI_API_KEY to your .env file to enable the AI features."
    if ai_calls >= AI_LIMIT:
        return "AI request limit reached for this session. Restart the server to reset."
    ai_calls += 1
    reply = requests.post(GEMINI_URL, params={"key": GEMINI_KEY},
                          json={"contents": [{"parts": [{"text": prompt}]}]},
                          timeout=30).json()
    try:
        return reply["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return "The AI service did not return an answer. Try again."


@app.post("/api/summary")
def summary():
    """One button. The answer is cached in ai_cache so Gemini is rarely called."""
    d = request.get_json() or {}
    category = d.get("category")
    conn = db()
    cached = conn.execute("SELECT summary FROM ai_cache WHERE category=?",
                          (category,)).fetchone()
    if cached and not d.get("refresh"):
        conn.close()
        return jsonify({"summary": cached["summary"], "cached": True})

    rows = conn.execute("SELECT text FROM feedback WHERE category=? ORDER BY id DESC LIMIT 40",
                        (category,)).fetchall()
    if not rows:
        conn.close()
        return jsonify({"error": "No feedback in this category yet"}), 404

    text = ask_gemini(
        f"Summarise this anonymous student feedback about {category} in 4 short "
        "bullet points, then one suggested action.\n\n"
        + "\n".join("- " + r["text"] for r in rows))
    conn.execute("INSERT OR REPLACE INTO ai_cache VALUES (?,?)", (category, text))
    conn.commit()
    conn.close()
    return jsonify({"summary": text, "cached": False})


@app.post("/api/chat")
def chat():
    """SQLite keyword search -> top 10 rows -> Gemini -> answer."""
    question = ((request.get_json() or {}).get("question") or "").strip()
    if len(question) < 3:
        return jsonify({"error": "Ask a longer question"}), 400

    words = [w for w in question.lower().split() if len(w) > 3][:4]
    sql = "SELECT text FROM feedback"
    params = []
    if words:
        sql += " WHERE " + " OR ".join("LOWER(text) LIKE ?" for _ in words)
        params = [f"%{w}%" for w in words]
    conn = db()
    rows = conn.execute(sql + " ORDER BY id DESC LIMIT 10", params).fetchall()
    conn.close()

    answer = ask_gemini("Answer the question using only this anonymous student feedback.\n\n"
                        + "\n".join("- " + r["text"] for r in rows)
                        + f"\n\nQuestion: {question}")
    return jsonify({"answer": answer, "used_records": len(rows)})


# --------------------------------------------------------------- the webpage
@app.get("/")
def home():
    return send_from_directory(HERE, "index.html")


setup()
train_model()

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5003)), host="0.0.0.0")
