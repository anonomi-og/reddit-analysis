import os
import re
import json
import openai
import praw
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from flask import Flask, request, render_template, redirect, url_for, flash, session
from functools import wraps
from dotenv import load_dotenv
from datetime import datetime, timezone
from urllib.parse import urlencode
from flask_session import Session  # Import this at the top of your app.py
# Import Auth0 client
from authlib.integrations.flask_client import OAuth
import secrets
import flask
import flask_session
from werkzeug.middleware.proxy_fix import ProxyFix
from google.cloud import firestore
db = firestore.Client()

import tiktoken   # <‑‑ new



MODEL = "gpt-3.5-turbo"   # keep a single source of truth
enc = tiktoken.encoding_for_model(MODEL)

def tokens_in(text: str) -> int:
    """Return how many tokens `text` will use with the chosen model."""
    return len(enc.encode(text))


# print("Flask version:", flask.__version__)
# print("Flask-Session version:", flask_session.__version__)

load_dotenv()

ADMIN_EMAILS = set(e.strip().lower()
                   for e in os.getenv("ADMIN_EMAILS", "").split(",")
                   if e.strip())


app = Flask(__name__)
# Add this line for compatibility with flask-session
app.session_cookie_name = app.config.get("SESSION_COOKIE_NAME", "session")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"  # Store sessions in a temporary file
app.config["PREFERRED_URL_SCHEME"] = "https"

Session(app)  # Initialize the session

# --- Auth0 Configuration ---
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")

# AUTH0_CALLBACK_URL = os.getenv("AUTH0_CALLBACK_URL", "http://localhost:8080/callback")   #local testing
AUTH0_CALLBACK_URL = os.getenv("AUTH0_CALLBACK_URL", "https://reddit-analyzer-793334408726.europe-west1.run.app/callback")  #deployed

AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE", f"https://{AUTH0_DOMAIN}/userinfo")

# Initialize OAuth and register Auth0
oauth = OAuth(app)
auth0 = oauth.register(
    'auth0',
    client_id=AUTH0_CLIENT_ID,
    client_secret=AUTH0_CLIENT_SECRET,
    api_base_url=f'https://{AUTH0_DOMAIN}',
    access_token_url=f'https://{AUTH0_DOMAIN}/oauth/token',
    authorize_url=f'https://{AUTH0_DOMAIN}/authorize',
    client_kwargs={
        'scope': 'openid profile email',
    },
    server_metadata_url=f'https://{AUTH0_DOMAIN}/.well-known/openid-configuration'
)


# --- Session-based Authentication Decorator ---
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# --- Auth0 Routes ---
@app.route("/login")
def login():
    # Generate a secure random state token and store it in the session.
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    # Pass the state to Auth0
    return auth0.authorize_redirect(redirect_uri=AUTH0_CALLBACK_URL, state=state)

@app.route("/callback")
def callback_handling():
    # Retrieve the state from the query parameters and from the session.
    state_from_response = request.args.get("state")
    state_from_session = session.get("oauth_state")
    
    if not state_from_session or state_from_response != state_from_session:
        print("[ERROR] State mismatch: session state:", state_from_session, "response state:", state_from_response)
        return "State mismatch error", 400

    # If the state matches, proceed to get the token.
    token = auth0.authorize_access_token()
    resp = auth0.get('userinfo')
    userinfo = resp.json()
    session['user'] = {
        'jwt_payload': userinfo,
        'name': userinfo.get('name'),
        'email': userinfo.get('email')
    }
    # Log the login event
    log_event("login", session['user']['email'])
    # Redirect to the dashboard instead of index
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    # Clear session and redirect to Auth0 logout endpoint
    session.clear()
    params = {'returnTo': url_for('index', _external=True), 'client_id': AUTH0_CLIENT_ID}
    return redirect(auth0.api_base_url + '/v2/logout?' + urlencode(params))

@app.route("/dashboard")
@requires_auth
def dashboard():
    return render_template("dashboard.html")



# --- Reddit API Credentials ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "RedditUserAnalysisScript/0.1")

# --- OpenAI API Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")

def collect_comments_within_budget(username,
                                   token_budget=1800,
                                   chunk_size=100,
                                   min_length=10):
    """
    Stream newest‑first comments until adding another would exceed `token_budget`.
    Returns a list of cleaned comment dicts.
    """
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    redditor = reddit.redditor(username)

    comments_data = []
    running_tkns = 0

    for comment in redditor.comments.new(limit=None):   # unlimited stream
        body_clean = clean_comment(comment.body or "")
        if len(body_clean) < min_length:
            continue

        comment_cost = tokens_in(body_clean) + 7   # small padding for JSON / newline
        if running_tkns + comment_cost > token_budget:
            break   # would blow the limit

        comments_data.append({
            "body": body_clean,
            "subreddit": comment.subreddit.display_name,
            "created_utc": comment.created_utc
        })
        running_tkns += comment_cost

        if len(comments_data) % chunk_size == 0:
            print(f"[DEBUG] {len(comments_data)} comments, {running_tkns}/{token_budget} tokens")

    print(f"[DEBUG] Final: {len(comments_data)} comments, {running_tkns}/{token_budget} tokens")
    return comments_data, running_tkns



def log_event(event_type, user_email, reddit_username=None):
    # event_type can be 'login' or 'usage'
    doc = {
        'user_email': user_email,
        'event_type': event_type,
        'timestamp': firestore.SERVER_TIMESTAMP,
    }
    if reddit_username:
        doc['reddit_username'] = reddit_username
    db.collection("events").add(doc)

def get_today_usage_count(user_email):
    # Get the start of the current UTC day
    now = datetime.now(timezone.utc)
    start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    
    events_ref = db.collection("events")
    query = (events_ref
             .where("user_email", "==", user_email)
             .where("event_type", "==", "usage")
             .where("timestamp", ">=", start_of_day))
    docs = query.stream()
    count = sum(1 for _ in docs)
    return count

def clean_comment(text):
    """
    Remove markdown links and URLs.
    """
    text = re.sub(r'\[([^\]]+)\]\((http[^\)]+)\)', r'\1', text)
    text = re.sub(r'http\S+', '', text)
    return text.strip()

def extract_username_from_input(user_input):
    """
    Extracts a Reddit username from a direct username string or a Reddit URL.
    """
    # Match up to the next slash OR end of string
    match = re.search(r'reddit\.com/(?:u|user)/([^/?#]+)', user_input)
    if match:
        return match.group(1)
    # If it's just a username without URL
    return user_input.split("/")[0].strip()


def fetch_comments(username, limit=150, min_length=10):
    """
    Fetch the last `limit` comments for a given Reddit username.
    """
    comments_data = []
    try:
        print(f"[DEBUG] Fetching comments for '{username}' with limit={limit}...")
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        redditor = reddit.redditor(username)
        for comment in redditor.comments.new(limit=limit):
            if not comment.body or len(comment.body.strip()) < min_length:
                continue
            cleaned = clean_comment(comment.body)
            if cleaned:
                comments_data.append({
                    "body": cleaned,
                    "subreddit": comment.subreddit.display_name,
                    "created_utc": comment.created_utc
                })
        print(f"[DEBUG] Fetched {len(comments_data)} comments for '{username}'.")
    except Exception as e:
        print(f"[ERROR] Error fetching comments for user '{username}': {e}")
    return comments_data

def fetch_user_info(username):
    """
    Retrieve additional user info such as karma, account age, etc.
    """
    try:
        print(f"[DEBUG] Fetching user info for '{username}'...")
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        redditor = reddit.redditor(username)

        created_utc = redditor.created_utc
        created_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc)
        created_date_str = created_dt.strftime('%Y-%m-%d')

        today = datetime.now(timezone.utc)
        account_age_days = (today - created_dt).days

        public_description = getattr(redditor.subreddit, 'public_description', 'No description available')

        user_info = {
            "username": username,
            "comment_karma": redditor.comment_karma,
            "link_karma": redditor.link_karma,
            "account_created": created_date_str,
            "account_age_days": account_age_days,
            "is_gold": getattr(redditor, "is_gold", False),
            "is_mod": getattr(redditor, "is_mod", False),
            "public_description": public_description
        }
        print(f"[DEBUG] User info for '{username}': {user_info}")
        return user_info
    except Exception as e:
        print(f"[ERROR] Error fetching user info for '{username}': {e}")
        return {}

def build_prompt(user_info, comments):
    """
    Build the prompt for the LLM that includes both user info and comments.
    """
    prompt = (
        "You are an analyst reviewing Reddit user behavior. Analyze the following Reddit user information and comments "
        "and provide your analysis in a strict JSON format with the following keys: "
        "\"personality\", \"interests\", \"writing_style\", \"political_spectrum\", \"location_guess\", \"comments_of_interest\", \"timing_analysis\". "
        "For the key \"political_spectrum\", choose one of the following options exactly: \"far left\", \"left\", \"centre\", \"right\", \"far right\", \"Unable to determine\". "
        "For the key \"location_guess\", return only the city or country name (for example, \"Leicester\" or \"United Kingdom\").\n\n try to guess the city if possible"
        "User Information:\n"
        f"{json.dumps(user_info, indent=2)}\n\n"
        "Reddit Comments:\n"
    )
    for idx, comment in enumerate(comments, start=1):
        prompt += f"{idx}. {comment['body']}\n"
    prompt += "\nPlease ensure the output is valid JSON and nothing else."
    print("[DEBUG] Prompt being sent to LLM (truncated to 500 chars):")
    print(prompt[:500], "...\n")
    return prompt

def call_llm_analysis(prompt):
    """
    Call OpenAI’s ChatCompletion endpoint using the latest OpenAI API format.
    """
    try:
        print("[DEBUG] Calling LLM with the prompt.")
        client = openai.Client()  # Initialize the OpenAI client

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a chatbot with a witty personality."},
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            max_tokens=2000,
            top_p=0.999
        )
        generated_content = response.choices[0].message.content
        print("[DEBUG] LLM analysis response (truncated to 300 chars):")
        print(generated_content[:300], "...\n")
    except Exception as e:
        print("[ERROR] Error calling LLM analysis:", e)
        generated_content = "Error calling LLM analysis."
    return generated_content

def generate_comment_heatmap(comments):
    """
    Create a heatmap showing comment activity by day of the week and hour of the day.
    Returns a base64-encoded PNG image.
    """
    try:
        print("[DEBUG] Generating heatmap for comment timings...")
        df = pd.DataFrame(comments)
        df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['hour'] = df['datetime'].dt.hour

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        pivot_table = pivot.pivot(index='day_of_week', columns='hour', values='count').reindex(days_order)
        pivot_table = pivot_table.fillna(0)

        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot_table, cmap="YlGnBu", linewidths=.5, annot=True, fmt=".0f")
        plt.title("Reddit Comment Activity Heatmap (Day vs Hour)")
        plt.ylabel("Day of Week")
        plt.xlabel("Hour of Day")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        heatmap_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        print("[DEBUG] Heatmap generated successfully.")
        return heatmap_base64
    except Exception as e:
        print("[ERROR] Error generating heatmap:", e)
        return None

# --- Main App Route ---
@app.route("/", methods=["GET", "POST"])
@requires_auth
def index():
    user_email = session['user']['email']
    if request.method == "POST":
        # Check usage count for today
        count = get_today_usage_count(user_email)

        # Admins get unlimited (or set a higher number here)
        if user_email.lower() not in ADMIN_EMAILS and count >= 5:
            flash("Daily usage limit reached. Please try again tomorrow.", "warning")
            return redirect(url_for("dashboard"))

            
        user_input = request.form.get("reddit_username", "").strip()
        print(f"[DEBUG] Received POST request with user_input='{user_input}'")

        reddit_username = extract_username_from_input(user_input)
        print(f"[DEBUG] Extracted reddit_username='{reddit_username}'")

        if not reddit_username:
            flash("Please enter a Reddit username or a valid profile URL.", "danger")
            return redirect(url_for("index"))
        
        user_info = fetch_user_info(reddit_username)
        # BEFORE
        # comments = fetch_comments(reddit_username, limit=150)

        # AFTER
        comments, prompt_tokens = collect_comments_within_budget(
            reddit_username,
            token_budget=1800   # adjust if you change model / output length
        )

        if not comments:
            flash("No valid comments found or an error occurred.", "warning")
            return redirect(url_for("index"))
        
        prompt = build_prompt(user_info, comments)
        analysis_summary = call_llm_analysis(prompt)

        location_guess = ""
        try:
            parsed_summary = json.loads(analysis_summary)
            location_guess = parsed_summary.get("location_guess", "")
            print(f"[DEBUG] location_guess extracted from LLM JSON: '{location_guess}'")
        except json.JSONDecodeError:
            print("[ERROR] Could not parse LLM JSON output")
        
        subreddit_counts = {}
        for comment in comments:
            sub = comment["subreddit"]
            subreddit_counts[sub] = subreddit_counts.get(sub, 0) + 1
        frequent_subreddits = sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"[DEBUG] frequent_subreddits = {frequent_subreddits}")

        heatmap_img = generate_comment_heatmap(comments)

        analysis = {
            "username": reddit_username,
            "total_comments_analyzed": len(comments),
            "prompt_tokens": prompt_tokens,                 
            "frequent_subreddits": frequent_subreddits,
            "user_info": user_info,
            "summary": analysis_summary,
            "heatmap": heatmap_img,
            "location_guess": location_guess
        }
        
        # Log the usage event with the searched Reddit username
        log_event("usage", user_email, reddit_username=reddit_username)

        print("[DEBUG] Final 'analysis' dict being sent to template:\n", json.dumps(analysis, indent=2)[:1000], "...\n")
        return render_template("result.html", analysis=analysis)
    
    print("[DEBUG] GET request on '/', rendering index.html.")
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=False)
