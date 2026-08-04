"""
=====================================================================
 CAMPUS VOICE AI  -  BACKEND (Single-file Flask Application)
=====================================================================

 PRESENTATION NOTE
 -----------------
 Complete single-file backend providing:
   * Anonymous student feedback collection & Severity scoring
   * Role-based permissions:
       - Students: Give Feedback, View/Join Events ONLY.
       - Lecturers & Management: Dashboard, Feedback Management,
         CSV Explorer, Event Creation, and AI Analytics.
   * Two locally-trained machine learning models (scikit-learn)
   * 100% Offline AI Structured Summary & Grounded Chatbot Engine
=====================================================================
"""

import os          # file paths
import random      # random characters & sampling for summaries
import sqlite3     # database engine
import string      # alphabet for event codes
import time        # timestamp & delay
import logging     # backend console logging

import pandas as pd                                            # CSV -> DataFrame
from flask import Flask, jsonify, request, send_from_directory  # web server
from sklearn.feature_extraction.text import TfidfVectorizer     # text vectorization
from sklearn.linear_model import LogisticRegression             # sentiment classifier
from sklearn.metrics import accuracy_score                      # accuracy evaluation
from sklearn.model_selection import train_test_split            # train/test split
from sklearn.naive_bayes import MultinomialNB                   # category classifier

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# CONFIGURATION / FILE PATHS
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))             # folder containing app.py
DB = os.path.join(HERE, "database.db")                        # SQLite database file
CSV = os.path.join(HERE, "student_feedback_dataset.csv")    # ML dataset

CATEGORIES = ["Lecturer", "Canteen", "Hostel", "Cleanliness", "Security"]

app = Flask(__name__, static_folder=HERE, static_url_path="")


# ---------------------------------------------------------------------------
# DATABASE HELPER & SEVERITY CLASSIFICATION ENGINE
# ---------------------------------------------------------------------------
def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def classify_severity(text, sentiment="Neutral", category=""):
    """Rule-based severity scoring engine: Critical, High, Medium, Low."""
    text_lower = (text or "").lower()

    # 1. CRITICAL SEVERITY
    critical_keywords = [
        "safety", "threat", "harass", "hazard", "fire", "assault", "poison",
        "emergency", "broken lock", "stolen", "security threat", "injury",
        "mold", "electric shock", "attack", "danger", "weapon", "blood", "robbed",
        "dark", "curfew", "police", "light", "lights", "street lights", "night"
    ]
    if (category == "Security" and sentiment == "Negative") or any(k in text_lower for k in critical_keywords):
        return "Critical"

    # 2. HIGH SEVERITY
    high_keywords = [
        "broken", "water leak", "no power", "cancelled", "unfair", "unhygienic",
        "rude", "cheating", "overflow", "infestation", "pest", "rodent", "sick", "raw food", "disconnect"
    ]
    if sentiment == "Negative" and (category in ["Hostel", "Canteen", "Lecturer"] or any(k in text_lower for k in high_keywords)):
        return "High"

    # 3. MEDIUM SEVERITY
    if sentiment == "Negative":
        return "Medium"

    # 4. LOW SEVERITY
    return "Low"


# ------------------------------------------------------------------ 1. setup
def setup():
    """Create schema, apply migrations, and seed demo data idempotently."""
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE,
            password TEXT, role TEXT, sid TEXT);

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY, user_id INTEGER, category TEXT,
            text TEXT, sentiment TEXT, severity TEXT DEFAULT 'Low',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP);

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, title TEXT, date TEXT,
            venue TEXT, code TEXT UNIQUE, creator_id INTEGER);

        CREATE TABLE IF NOT EXISTS event_feedback (
            id INTEGER PRIMARY KEY, event_id INTEGER, user_id INTEGER,
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            comment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(event_id, user_id)
        );

        CREATE TABLE IF NOT EXISTS registrations (
            id INTEGER PRIMARY KEY, event_id INTEGER, user_id INTEGER,
            UNIQUE(event_id, user_id));

        CREATE TABLE IF NOT EXISTS ai_cache (
            category TEXT PRIMARY KEY, summary TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
        """
    )

    # Schema Migrations
    try:
        conn.execute("ALTER TABLE feedback ADD COLUMN severity TEXT DEFAULT 'Low'")
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE ai_cache ADD COLUMN updated_at TEXT DEFAULT CURRENT_TIMESTAMP")
    except sqlite3.OperationalError:
        pass

    # Seed Demo Users
    if not conn.execute("SELECT 1 FROM users").fetchone():
        conn.executemany(
            "INSERT INTO users (name, email, password, role, sid) VALUES (?,?,?,?,?)",
            [("Anita Rao", "student@campus.edu", "student123", "student", "SID1001"),
             ("Dr. Mehta", "lecturer@campus.edu", "lecturer123", "lecturer", None),
             ("Admin Office", "management@campus.edu", "manage123", "management", None)],
        )

    # Seed 500-row Dataset
    if not conn.execute("SELECT 1 FROM feedback").fetchone() and os.path.exists(CSV):
        data = pd.read_csv(CSV)
        rows_to_insert = []
        for _, row in data.iterrows():
            cat = row["category"]
            txt = row["feedback_text"]
            sent = row["sentiment"]
            sev = classify_severity(txt, sent, cat)
            rows_to_insert.append((1, cat, txt, sent, sev))

        conn.executemany(
            "INSERT INTO feedback (user_id, category, text, sentiment, severity) VALUES (?,?,?,?,?)",
            rows_to_insert,
        )

    # Re-classify all feedback rows
    all_rows = conn.execute("SELECT id, text, sentiment, category FROM feedback").fetchall()
    for r in all_rows:
        sev = classify_severity(r["text"], r["sentiment"], r["category"])
        conn.execute("UPDATE feedback SET severity=? WHERE id=?", (sev, r["id"]))

    conn.commit()
    conn.close()


# --------------------------------------------------------------------- 2. ML
cat_vectorizer = TfidfVectorizer(stop_words="english")
cat_model = MultinomialNB()

sent_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
sent_model = LogisticRegression(max_iter=1000)

ACCURACY = 0.0
SENT_ACCURACY = 0.0


def train_model():
    global ACCURACY, SENT_ACCURACY
    if not os.path.exists(CSV):
        return

    data = pd.read_csv(CSV)

    x_train, x_test, y_train, y_test = train_test_split(
        data["feedback_text"], data["category"], test_size=0.2, random_state=42)
    cat_model.fit(cat_vectorizer.fit_transform(x_train), y_train)
    ACCURACY = round(accuracy_score(
        y_test, cat_model.predict(cat_vectorizer.transform(x_test))) * 100, 2)

    a_train, a_test, b_train, b_test = train_test_split(
        data["feedback_text"], data["sentiment"], test_size=0.2, random_state=42)
    sent_model.fit(sent_vectorizer.fit_transform(a_train), b_train)
    SENT_ACCURACY = round(accuracy_score(
        b_test, sent_model.predict(sent_vectorizer.transform(a_test))) * 100, 2)

    logging.info(f"Category ML Accuracy: {ACCURACY}% | Sentiment ML Accuracy: {SENT_ACCURACY}%")


def predict_category(text):
    return cat_model.predict(cat_vectorizer.transform([text]))[0]


def sentiment_of(text):
    return sent_model.predict(sent_vectorizer.transform([text]))[0]


def role_of(user_id):
    if not user_id:
        return None
    conn = db()
    row = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return row["role"] if row else None


# ------------------------------------------------------------------ 3. login
@app.post("/api/login")
def login():
    d = request.get_json() or {}
    email = (d.get("email") or "").strip().lower()
    password = d.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    conn = db()
    user = conn.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password)).fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "Invalid email or password"}), 401
    return jsonify({"id": user["id"], "name": user["name"], "role": user["role"]})


@app.post("/api/signup")
def signup():
    d = request.get_json() or {}
    name = (d.get("name") or "").strip()
    email = (d.get("email") or "").strip().lower()
    password = d.get("password") or ""
    confirm = d.get("confirm") or ""
    role = d.get("role") or ""
    sid = (d.get("sid") or "").strip() or None

    if not name or not email or not password:
        return jsonify({"error": "Please fill in all required fields"}), 400
    if role not in ("student", "lecturer", "management"):
        return jsonify({"error": "Select a valid role (Student, Lecturer, Management)"}), 400
    if password != confirm:
        return jsonify({"error": "Passwords do not match"}), 400

    conn = db()
    try:
        cur = conn.execute(
            "INSERT INTO users (name, email, password, role, sid) VALUES (?,?,?,?,?)",
            (name, email, password, role, sid))
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email address already registered"}), 409
    finally:
        conn.close()
    return jsonify({"id": user_id, "name": name, "role": role})


# --------------------------------------------------------------- 4. feedback
@app.post("/api/feedback")
def add_feedback():
    d = request.get_json() or {}
    text = (d.get("text") or "").strip()
    if len(text) < 10:
        return jsonify({"error": "Please write at least 10 characters of feedback."}), 400

    category = d.get("category") if d.get("category") in CATEGORIES else predict_category(text)
    sentiment = sentiment_of(text)
    severity = classify_severity(text, sentiment, category)

    conn = db()
    conn.execute(
        "INSERT INTO feedback (user_id, category, text, sentiment, severity) VALUES (?,?,?,?,?)",
        (d.get("user_id"), category, text, sentiment, severity)
    )
    conn.commit()
    conn.close()

    return jsonify({
        "message": "Feedback submitted anonymously",
        "category": category,
        "sentiment": sentiment,
        "severity": severity
    })


@app.get("/api/feedback")
def list_feedback():
    user_id = request.args.get("user_id")
    if user_id:
        role = role_of(user_id)
        if role == "student":
            return jsonify({"error": "Access Denied: Students are not authorized to view feedback lists."}), 403

    cat = request.args.get("category")
    sent = request.args.get("sentiment")
    sev = request.args.get("severity")
    q = request.args.get("q", "").strip()
    sort_by = request.args.get("sort", "date_desc")
    page = max(1, int(request.args.get("page", 1)))
    limit = min(300, max(1, int(request.args.get("limit", 50))))
    offset = (page - 1) * limit

    where_clauses = ["1=1"]
    params = []

    if cat and cat != "All":
        where_clauses.append("category = ?")
        params.append(cat)
    if sent and sent != "All":
        where_clauses.append("sentiment = ?")
        params.append(sent)
    if sev and sev != "All":
        where_clauses.append("severity = ?")
        params.append(sev)
    if q:
        where_clauses.append("LOWER(text) LIKE ?")
        params.append(f"%{q.lower()}%")

    where_sql = " AND ".join(where_clauses)

    if sort_by == "severity_desc":
        order_sql = "ORDER BY CASE severity WHEN 'Critical' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 WHEN 'Low' THEN 4 END ASC, id DESC"
    elif sort_by == "date_desc":
        order_sql = "ORDER BY id DESC"
    elif sort_by == "date_asc":
        order_sql = "ORDER BY id ASC"
    else:
        order_sql = "ORDER BY id DESC"

    conn = db()
    count_query = f"SELECT COUNT(*) c FROM feedback WHERE {where_sql}"
    total_matching = conn.execute(count_query, params).fetchone()["c"]

    data_query = f"SELECT id, category, text, sentiment, severity, created_at FROM feedback WHERE {where_sql} {order_sql} LIMIT ? OFFSET ?"
    query_params = params + [limit, offset]

    rows = conn.execute(data_query, query_params).fetchall()
    conn.close()

    result_rows = [{**dict(r), "student": "Anonymous Student"} for r in rows]

    return jsonify({
        "data": result_rows,
        "total": total_matching,
        "page": page,
        "limit": limit,
        "total_pages": max(1, (total_matching + limit - 1) // limit)
    })


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
    sevs = counts("severity")
    total = conn.execute("SELECT COUNT(*) c FROM feedback").fetchone()["c"]

    top_cat = max(cats.items(), key=lambda x: x[1])[0] if cats else "None"

    crit_cats = conn.execute(
        "SELECT category, COUNT(*) AS n FROM feedback WHERE severity IN ('Critical', 'High') "
        "GROUP BY category ORDER BY n DESC LIMIT 1"
    ).fetchone()
    most_critical_cat = crit_cats["category"] if crit_cats else (top_cat if cats else "None")

    conn.close()

    return jsonify({
        "total": total,
        "positive": sent.get("Positive", 0),
        "neutral": sent.get("Neutral", 0),
        "negative": sent.get("Negative", 0),
        "severity_counts": {
            "Critical": sevs.get("Critical", 0),
            "High": sevs.get("High", 0),
            "Medium": sevs.get("Medium", 0),
            "Low": sevs.get("Low", 0)
        },
        "categories": cats,
        "top_category": top_cat,
        "most_critical_category": most_critical_cat,
        "accuracy": ACCURACY,
        "sentiment_accuracy": SENT_ACCURACY
    })


# ----------------------------------------------------------------- 6. events
@app.post("/api/events")
def create_event():
    d = request.get_json() or {}
    user_id = d.get("user_id")
    title = (d.get("title") or "").strip()
    date = (d.get("date") or "").strip()
    venue = (d.get("venue") or "").strip()

    if not user_id:
        return jsonify({"error": "Please log in first"}), 401

    role = role_of(user_id)
    if role not in ("lecturer", "management"):
        return jsonify({"error": "Only lecturers and management can create events"}), 403

    if not title or not date or not venue:
        return jsonify({"error": "Title, Date, and Venue are all required to create an event."}), 400

    code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    conn = db()
    try:
        conn.execute("INSERT INTO events (title, date, venue, code) VALUES (?,?,?,?)",
                     (title, date, venue, code))
        conn.commit()
    except sqlite3.IntegrityError:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        conn.execute("INSERT INTO events (title, date, venue, code) VALUES (?,?,?,?)",
                     (title, date, venue, code))
        conn.commit()
    finally:
        conn.close()

    logging.info(f"[Event Created] '{title}' by User #{user_id} ({role}) - Code: {code}")
    return jsonify({"code": code, "message": "Event created successfully!"})


@app.get("/api/events")
def list_events():
    conn = db()
    rows = conn.execute(
        "SELECT e.*, (SELECT COUNT(*) FROM registrations r WHERE r.event_id=e.id) "
        "AS registered FROM events e ORDER BY e.date DESC, e.id DESC").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.post("/api/register")
def register():
    d = request.get_json() or {}
    code = (d.get("code") or "").strip().upper()
    user_id = d.get("user_id")

    if not code:
        return jsonify({"error": "Please enter a valid 6-character event code"}), 400
    if not user_id:
        return jsonify({"error": "Please log in to join events"}), 401

    conn = db()
    event = conn.execute("SELECT id, title FROM events WHERE UPPER(code)=?", (code,)).fetchone()
    if not event:
        conn.close()
        return jsonify({"error": f"Event code '{code}' not found. Check code and try again."}), 404

    existing = conn.execute(
        "SELECT 1 FROM registrations WHERE event_id=? AND user_id=?",
        (event["id"], user_id)
    ).fetchone()

    if existing:
        conn.close()
        return jsonify({"error": f"You are already registered for '{event['title']}'!"}), 409

    # Insert registration
    conn.execute("INSERT INTO registrations (event_id, user_id) VALUES (?,?)",
                 (event["id"], user_id))
    conn.commit()
    conn.close()
    # Return event_id for feedback submission
    return jsonify({"message": f"Successfully registered for '{event['title']}'!", "event_id": event["id"]})


# --------------------------------------------------------------------- 7. OFFLINE AI ENGINE
def generate_offline_summary(category, rows):
    """Generate structured executive summary offline from database feedback records."""
    if not rows:
        return f"No feedback records found for '{category}' category."

    positives = [r for r in rows if r["sentiment"] == "Positive"]
    negatives = [r for r in rows if r["sentiment"] == "Negative"]
    neutrals = [r for r in rows if r["sentiment"] == "Neutral"]

    criticals = [r for r in rows if r["severity"] == "Critical"]
    highs = [r for r in rows if r["severity"] == "High"]

    # Calculate Priority
    if len(criticals) > 0 or len(highs) >= 5:
        priority = "Critical"
    elif len(highs) > 0 or len(negatives) > len(positives):
        priority = "High"
    elif len(negatives) > 0:
        priority = "Medium"
    else:
        priority = "Low"

    # Select Key Strengths
    strengths_list = []
    if positives:
        sample_pos = random.sample(positives, min(3, len(positives)))
        for p in sample_pos:
            strengths_list.append(f"- {p['text']}")
    else:
        strengths_list.append("- Overall positive sentiments recorded in regular operations.")

    # Select Key Complaints
    complaints_list = []
    if negatives:
        priority_negs = [n for n in negatives if n["severity"] in ("Critical", "High")]
        if not priority_negs:
            priority_negs = negatives
        sample_neg = random.sample(priority_negs, min(3, len(priority_negs)))
        for n in sample_neg:
            complaints_list.append(f"- [{n['severity']} Severity] {n['text']}")
    else:
        complaints_list.append("- No major negative complaints recorded for this category.")

    # Actions matrix
    action_map = {
        "Lecturer": [
            "Conduct peer teaching reviews and upgrade lecture room multimedia equipment.",
            "Establish clearer office hours and digital communication channels for student queries."
        ],
        "Canteen": [
            "Implement stricter food hygiene standards and vendor quality audits.",
            "Review pricing and expand dietary options (vegetarian/halal) based on student preferences."
        ],
        "Hostel": [
            "Upgrade Wi-Fi access point bandwidth during peak evening study hours.",
            "Schedule immediate maintenance for water heaters and communal facility locks."
        ],
        "Cleanliness": [
            "Increase daily sanitation shifts in high-traffic restrooms and lecture halls.",
            "Deploy additional waste segregation bins across campus corridors."
        ],
        "Security": [
            "Repair broken perimeter lighting and enhance campus night security patrols.",
            "Upgrade turnstile access control systems at student hostel gates."
        ]
    }

    actions = action_map.get(category, [
        "Review operational guidelines and collect follow-up student feedback.",
        "Allocate maintenance budget to resolve highlighted operational issues."
    ])

    summary_text = (
        f"Executive Summary: {category} Campus Operations Analysis\n\n"
        f"1. Key Strengths ({len(positives)} Positive Submissions)\n"
        + "\n".join(strengths_list) + "\n\n"
        f"2. Key Complaints ({len(negatives)} Negative Submissions, {len(criticals)} Critical)\n"
        + "\n".join(complaints_list) + "\n\n"
        f"3. Major Themes\n"
        f"- Sentiment Ratio: {len(positives)} Positive, {len(neutrals)} Neutral, {len(negatives)} Negative.\n"
        f"- Operational Risk: {criticals[0]['text'] if criticals else ('Standard maintenance required' if negatives else 'High student satisfaction')}.\n\n"
        f"4. Recommended Actions\n"
        f"- {actions[0]}\n"
        f"- {actions[1]}\n\n"
        f"5. Priority Level\n"
        f"{priority}"
    )

    return summary_text


@app.post("/api/summary")
def summary():
    d = request.get_json() or {}
    user_id = d.get("user_id")
    if user_id:
        role = role_of(user_id)
        if role == "student":
            return jsonify({"error": "Access Denied: AI Summaries are restricted to Lecturers and Management."}), 403

    category = d.get("category")
    if not category or category not in CATEGORIES:
        return jsonify({"error": f"Select a valid category ({', '.join(CATEGORIES)})"}), 400

    conn = db()
    cached = conn.execute("SELECT summary FROM ai_cache WHERE category=?", (category,)).fetchone()
    if cached and not d.get("refresh"):
        conn.close()
        return jsonify({"summary": cached["summary"], "cached": True, "success": True})

    rows = conn.execute("SELECT text, sentiment, severity FROM feedback WHERE category=? ORDER BY id DESC LIMIT 50",
                        (category,)).fetchall()
    if not rows:
        conn.close()
        return jsonify({"error": f"No feedback recorded for '{category}' yet."}), 404

    text = generate_offline_summary(category, rows)

    conn.execute("INSERT OR REPLACE INTO ai_cache (category, summary) VALUES (?,?)", (category, text))
    conn.commit()
    conn.close()
    return jsonify({"summary": text, "cached": False, "success": True})


@app.post("/api/chat")
def chat():
    d = request.get_json() or {}
    user_id = d.get("user_id")
    if user_id:
        role = role_of(user_id)
        if role == "student":
            return jsonify({"error": "Access Denied: AI Assistant is restricted to Lecturers and Management."}), 403

    question = (d.get("question") or "").strip()
    if len(question) < 3:
        return jsonify({"error": "Please enter a valid question (at least 3 characters)"}), 400

    words = [w for w in question.lower().split() if len(w) > 2][:5]
    sql = "SELECT text, category, sentiment, severity FROM feedback"
    params = []
    if words:
        sql += " WHERE " + " OR ".join("LOWER(text) LIKE ?" for _ in words)
        params = [f"%{w}%" for w in words]

    conn = db()
    rows = conn.execute(sql + " ORDER BY id DESC LIMIT 15", params).fetchall()

    if not rows:
        rows = conn.execute("SELECT text, category, sentiment, severity FROM feedback ORDER BY id DESC LIMIT 5").fetchall()

    conn.close()

    matching_snippets = []
    for r in rows[:4]:
        matching_snippets.append(f"• [{r['category']} / {r['sentiment']}] \"{r['text']}\"")

    answer = (
        f"Based on student feedback records regarding '{question}':\n\n"
        + "\n".join(matching_snippets) + "\n\n"
        f"Summary Insight: Retrieved {len(rows)} relevant campus feedback records. "
        f"The primary issues revolve around {rows[0]['category']} operational management."
    )

    return jsonify({"answer": answer, "used_records": len(rows), "success": True})


# --------------------------------------------------------------- webpage
@app.get("/")
def home():
    return send_from_directory(HERE, "index.html")


# ---------------------------------------------------------------------------
# START-UP SEQUENCE
# ---------------------------------------------------------------------------
setup()
train_model()

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)), host="0.0.0.0")