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

from flask import Flask, request, render_template, redirect, url_for, flash, Response
from functools import wraps
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")

# --- Basic Authentication Setup ---
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "password")

def check_auth(username, password):
    return username == APP_USERNAME and password == APP_PASSWORD

def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
         auth = request.authorization
         if not auth or not check_auth(auth.username, auth.password):
              return authenticate()
         return f(*args, **kwargs)
    return decorated

# --- Reddit API Credentials ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "RedditUserAnalysisScript/0.1")

# --- OpenAI API Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_comment(text):
    """
    Remove markdown links and URLs.
    """
    text = re.sub(r'\[([^\]]+)\]\((http[^\)]+)\)', r'\1', text)
    text = re.sub(r'http\S+', '', text)
    return text.strip()

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

        # Calculate readable account creation date
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
    print(prompt[:500], "...\n")  # prevent flooding console
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

@app.route("/", methods=["GET", "POST"])
@requires_auth
def index():
    if request.method == "POST":
        reddit_username = request.form.get("reddit_username", "").strip()
        print(f"[DEBUG] Received POST request with reddit_username='{reddit_username}'")

        if not reddit_username:
            flash("Please enter a Reddit username.", "danger")
            return redirect(url_for("index"))
        
        user_info = fetch_user_info(reddit_username)
        comments = fetch_comments(reddit_username, limit=150)
        if not comments:
            flash("No valid comments found or an error occurred.", "warning")
            return redirect(url_for("index"))
        
        prompt = build_prompt(user_info, comments)
        analysis_summary = call_llm_analysis(prompt)

        # Attempt to parse JSON from LLM result
        location_guess = ""
        try:
            parsed_summary = json.loads(analysis_summary)
            location_guess = parsed_summary.get("location_guess", "")
            print(f"[DEBUG] location_guess extracted from LLM JSON: '{location_guess}'")
        except json.JSONDecodeError:
            print("[ERROR] Could not parse LLM JSON output")
        
        # Build frequency data
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
            "frequent_subreddits": frequent_subreddits,
            "user_info": user_info,
            "summary": analysis_summary,
            "heatmap": heatmap_img,
            "location_guess": location_guess
        }

        print("[DEBUG] Final 'analysis' dict being sent to template:\n", json.dumps(analysis, indent=2)[:1000], "...\n")

        return render_template("result.html", analysis=analysis)
    
    # GET request => Show index
    print("[DEBUG] GET request on '/', rendering index.html.")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)
