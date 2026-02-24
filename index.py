import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import re
import requests
import time
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from groq import Groq
from supabase import create_client, Client
from streamlit_autorefresh import st_autorefresh

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

# --- SECRETS ---
def get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

NEWSAPI_KEY    = get_secret("NEWSAPI_KEY")
GROQ_API_KEY   = get_secret("GROQ_API_KEY")
EMAIL_SENDER   = get_secret("EMAIL_SENDER")
EMAIL_PASSWORD = get_secret("EMAIL_PASSWORD")
EMAIL_RECEIVER = get_secret("EMAIL_RECEIVER")
SUPABASE_URL   = get_secret("SUPABASE_URL")
SUPABASE_KEY   = get_secret("SUPABASE_KEY")
GROQ_MODEL     = "llama-3.1-8b-instant"

# --- RATE LIMITER ---
GROQ_RPM_LIMIT  = 28
GROQ_WINDOW_SEC = 60
RETRY_ATTEMPTS  = 3
RETRY_BASE_WAIT = 5

def init_rate_limiter():
    if "groq_request_times"   not in st.session_state:
        st.session_state.groq_request_times   = []
    if "groq_total_requests"  not in st.session_state:
        st.session_state.groq_total_requests  = 0
    if "groq_rate_waits"      not in st.session_state:
        st.session_state.groq_rate_waits      = 0

def rate_limit_groq():
    init_rate_limiter()
    now = time.time()
    st.session_state.groq_request_times = [
        t for t in st.session_state.groq_request_times
        if now - t < GROQ_WINDOW_SEC
    ]
    if len(st.session_state.groq_request_times) >= GROQ_RPM_LIMIT:
        oldest = st.session_state.groq_request_times[0]
        wait   = GROQ_WINDOW_SEC - (now - oldest) + 0.5
        if wait > 0:
            st.session_state.groq_rate_waits += 1
            time.sleep(wait)
    st.session_state.groq_request_times.append(time.time())
    st.session_state.groq_total_requests += 1

# --- SUPABASE ---
def get_supabase() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        st.warning(f"Supabase connection failed: {e}")
        return None

def save_sentiment_scores(ticker, rows):
    db = get_supabase()
    if not db:
        return
    try:
        now  = datetime.now().isoformat()
        data = [{
            "ticker": ticker, "date": row["Date"],
            "headline": row["Headline"], "source": row.get("Source", "Headline"),
            "score": float(row["Score"]), "created_at": now
        } for row in rows]
        db.table("sentiment_scores").insert(data).execute()
    except Exception as e:
        st.warning(f"Save sentiment error: {e}")

def save_correlation(ticker, correlation, signal, articles, period):
    db = get_supabase()
    if not db:
        return
    try:
        db.table("correlation_history").insert({
            "ticker": ticker, "correlation": float(correlation),
            "signal": signal, "articles": articles,
            "period": period, "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        st.warning(f"Save correlation error: {e}")

def save_run_log(tickers, interval):
    db = get_supabase()
    if not db:
        return
    try:
        db.table("run_log").insert({
            "tickers": ",".join(tickers),
            "interval": interval,
            "created_at": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        st.warning(f"Save run log error: {e}")

def clear_history():
    db = get_supabase()
    if not db:
        st.warning("Supabase not connected.")
        return
    try:
        db.table("sentiment_scores").delete().neq("id", 0).execute()
        db.table("correlation_history").delete().neq("id", 0).execute()
        db.table("run_log").delete().neq("id", 0).execute()
    except Exception as e:
        st.error(f"Clear error: {e}")

@st.cache_data(ttl=300)
def load_correlation_history(ticker=None):
    db = get_supabase()
    if not db:
        return pd.DataFrame()
    try:
        query = db.table("correlation_history").select("*").order("created_at", desc=True).limit(100)
        if ticker:
            query = query.eq("ticker", ticker)
        result = query.execute()
        return pd.DataFrame(result.data) if result.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Load correlation error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_sentiment_history(ticker=None):
    db = get_supabase()
    if not db:
        return pd.DataFrame()
    try:
        query = db.table("sentiment_scores").select("*").order("created_at", desc=True).limit(500)
        if ticker:
            query = query.eq("ticker", ticker)
        result = query.execute()
        return pd.DataFrame(result.data) if result.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Load sentiment error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_run_log():
    db = get_supabase()
    if not db:
        return pd.DataFrame()
    try:
        result = db.table("run_log").select("*").order("created_at", desc=True).limit(50).execute()
        return pd.DataFrame(result.data) if result.data else pd.DataFrame()
    except Exception as e:
        st.error(f"Load run log error: {e}")
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.title("⚙️ Configuration")
tickers_input = st.sidebar.text_input(
    "Stock Tickers (comma separated)",
    value="NVDA, AMD, TSLA",
    help="Enter up to 4 tickers to compare"
)
TICKERS = [t.strip().upper() for t in tickers_input.split(",") if t.strip()][:4]

days_back = st.sidebar.slider("News History (days)", min_value=7, max_value=30, value=30)
period    = st.sidebar.selectbox("Stock History Period", ["1mo", "3mo", "6mo"], index=1)
run_btn   = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**⏰ Auto-Run Schedule**")
auto_run = st.sidebar.toggle("Enable Auto-Run", value=False)
schedule_interval = st.sidebar.selectbox(
    "Run Every",
    ["15 minutes", "30 minutes", "1 hour", "6 hours", "24 hours"],
    index=2
)
interval_map = {
    "15 minutes": 15*60*1000, "30 minutes": 30*60*1000,
    "1 hour": 60*60*1000, "6 hours": 6*60*60*1000, "24 hours": 24*60*60*1000,
}
if auto_run:
    count = st_autorefresh(interval=interval_map[schedule_interval], key="auto_refresh")
    st.sidebar.success(f"✅ Auto-run active — every {schedule_interval}")
    st.sidebar.caption(f"Run count: {count}")
else:
    st.sidebar.info("⏸️ Auto-run disabled")

st.sidebar.markdown("---")
st.sidebar.markdown("**🔔 Alert Settings**")
alert_threshold = st.sidebar.slider("Alert Threshold (|correlation|)", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
email_alerts    = st.sidebar.toggle("Enable Email Alerts", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**🗄️ Database**")
show_history = st.sidebar.toggle("Show History Dashboard", value=False)
if not SUPABASE_URL:
    st.sidebar.warning("⚠️ Supabase not configured")
else:
    st.sidebar.success("✅ Supabase connected")
clear_btn = st.sidebar.button("🗑️ Clear All History", use_container_width=True)
if clear_btn:
    clear_history()
    st.sidebar.success("✅ History cleared!")

st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Rate Limiter**")
init_rate_limiter()
st.sidebar.markdown(f"Requests this session: `{st.session_state.groq_total_requests}`")
st.sidebar.markdown(f"Rate waits triggered: `{st.session_state.groq_rate_waits}`")
recent = len([t for t in st.session_state.groq_request_times if time.time() - t < 60])
st.sidebar.markdown(f"Requests in last 60s: `{recent}/{GROQ_RPM_LIMIT}`")
if recent >= GROQ_RPM_LIMIT * 0.8:
    st.sidebar.warning("⚠️ Approaching rate limit!")
else:
    st.sidebar.success("✅ Rate limit OK")

st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")
st.sidebar.markdown(f"Groq API: {'✅ Connected' if GROQ_API_KEY else '❌ Missing'}")
st.sidebar.markdown(f"NewsAPI:  {'✅ Connected' if NEWSAPI_KEY else '❌ Missing'}")
st.sidebar.markdown(f"Supabase: {'✅ Connected' if SUPABASE_URL else '❌ Missing'}")
st.sidebar.markdown(f"Email:    {'✅ Configured' if EMAIL_SENDER else '❌ Not Set'}")
st.sidebar.markdown(f"Model: `{GROQ_MODEL}`")
st.sidebar.markdown(f"Analyzing: `{', '.join(TICKERS)}`")

# --- MAIN TITLE ---
st.title("📈 AI Stock Sentiment Analyzer")
st.markdown("Correlates **AI-scored news sentiment** with **next-day stock price changes**.")

# --- CORE FUNCTIONS ---

@st.cache_data(ttl=3600, show_spinner=False)
def get_full_article_text(url):
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        text = re.sub(r'<[^>]+>', ' ', response.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:2000]
    except Exception:
        return ""

@st.cache_data(ttl=3600, show_spinner=False)
def get_sentiment(headline, ticker, article_text=""):
    if not GROQ_API_KEY:
        return 0.0
    for attempt in range(RETRY_ATTEMPTS):
        try:
            rate_limit_groq()
            client = Groq(api_key=GROQ_API_KEY)
            if article_text and len(article_text) > 100:
                content = f"Headline: {headline}\n\nArticle: {article_text[:1500]}"
                prompt  = f"Analyze sentiment for {ticker} stock based on this news article. Return ONLY a number between -1.0 (very negative) and 1.0 (very positive). No words, just the number.\n\n{content}"
            else:
                prompt = f"Analyze sentiment for {ticker} stock: '{headline}'. Return ONLY a number between -1.0 and 1.0. No words, just the number."
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10, temperature=0.1
            )
            score_text = response.choices[0].message.content.strip()
            match = re.search(r'-?\d+\.?\d*', score_text)
            if match:
                return max(-1.0, min(1.0, float(match.group())))
            return 0.0
        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err:
                wait = RETRY_BASE_WAIT * (2 ** attempt)
                st.warning(f"⏳ Rate limit hit — waiting {wait}s (attempt {attempt+1}/{RETRY_ATTEMPTS})")
                time.sleep(wait)
            else:
                st.warning(f"Groq error: {e}")
                return 0.0
    st.error("❌ Groq API failed after max retries.")
    return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def get_news_newsapi(ticker_name, days_back):
    if not NEWSAPI_KEY:
        return []
    try:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        response  = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": ticker_name, "language": "en",
                "sortBy": "publishedAt", "pageSize": 100,
                "from": from_date, "apiKey": NEWSAPI_KEY
            },
            timeout=15
        )
        articles = []
        for a in response.json().get("articles", []):
            if not a.get('title') or a['title'] == '[Removed]':
                continue
            dt = datetime.fromisoformat(a['publishedAt'].replace('Z', '+00:00')).replace(tzinfo=None)
            articles.append({
                "title": a['title'], "dt": dt,
                "url": a.get('url', ''), "description": a.get('description', '')
            })
        return articles
    except Exception as e:
        st.error(f"NewsAPI Error: {e}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_news_yfinance(ticker):
    articles = []
    stock = yf.Ticker(ticker)
    for article in stock.news:
        content  = article.get('content', article)
        title    = content.get('title', "")
        raw_date = content.get('pubDate', content.get('providerPublishTime'))
        if not title:
            continue
        if isinstance(raw_date, str):
            dt = datetime.fromisoformat(raw_date.replace('Z', '+00:00')).replace(tzinfo=None)
        else:
            dt = datetime.fromtimestamp(raw_date)
        articles.append({"title": title, "dt": dt, "url": "", "description": ""})
    return articles

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(ticker, period):
    stock = yf.Ticker(ticker)
    hist  = stock.history(period=period, interval="1d")
    if hist.empty:
        return None, None
    hist.index = hist.index.tz_localize(None)
    hist['Price_Change'] = hist['Close'].pct_change()
    hist_df = hist.reset_index()
    hist_df.rename(columns={hist_df.columns[0]: 'Timestamp'}, inplace=True)
    hist_df = hist_df.dropna(subset=['Price_Change'])
    hist_df['Price_Change_Next'] = hist_df['Price_Change'].shift(-1)
    hist_df = hist_df.dropna(subset=['Price_Change_Next'])
    hist_df['Timestamp'] = hist_df['Timestamp'].astype('datetime64[s]')
    hist_df = hist_df.sort_values('Timestamp')
    return hist_df, hist

def send_alert(ticker, correlation, signal):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        return False
    try:
        msg            = MIMEMultipart()
        msg['From']    = EMAIL_SENDER
        msg['To']      = EMAIL_RECEIVER
        msg['Subject'] = f"📈 Stock Alert: {ticker} — {signal} Signal Detected"
        body = f"""
        <html><body style="font-family:Arial;background:#0e1117;color:white;padding:20px;">
            <h2>📈 AI Stock Sentiment Alert</h2>
            <table style="border-collapse:collapse;width:100%;">
                <tr style="background:#1e1e2e;"><td style="padding:10px;border:1px solid #333;"><b>Ticker</b></td><td style="padding:10px;border:1px solid #333;">{ticker}</td></tr>
                <tr><td style="padding:10px;border:1px solid #333;"><b>Correlation</b></td><td style="padding:10px;border:1px solid #333;">{correlation:.4f}</td></tr>
                <tr style="background:#1e1e2e;"><td style="padding:10px;border:1px solid #333;"><b>Signal</b></td><td style="padding:10px;border:1px solid #333;">{signal}</td></tr>
                <tr><td style="padding:10px;border:1px solid #333;"><b>Time</b></td><td style="padding:10px;border:1px solid #333;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
            </table>
            <p style="color:#888;margin-top:20px;">Sent by AI Stock Sentiment Analyzer</p>
        </body></html>
        """
        msg.attach(MIMEText(body, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        return True
    except Exception as e:
        st.error(f"Email error: {e}")
        return False

# --- ANALYSIS PIPELINE ---

def analyze_ticker(ticker, period, days_back):
    hist_df, hist_raw = get_stock_data(ticker, period)
    if hist_df is None:
        st.error(f"❌ No market data for {ticker}")
        return None, None

    if NEWSAPI_KEY:
        raw_articles = get_news_newsapi(ticker_name=ticker, days_back=days_back)
        if not raw_articles:
            st.warning(f"⚠️ NewsAPI returned no articles for {ticker}, trying yfinance...")
            raw_articles = get_news_yfinance(ticker)
    else:
        st.warning("⚠️ No NEWSAPI_KEY — using yfinance fallback (~10 articles).")
        raw_articles = get_news_yfinance(ticker)

    if not raw_articles:
        st.warning(f"⚠️ No news found for {ticker}")
        return None, None

    total    = len(raw_articles)
    est_mins = round((total / GROQ_RPM_LIMIT) + 0.5)
    if total > GROQ_RPM_LIMIT:
        st.info(f"ℹ️ {total} articles — rate limiter active. Estimated: ~{est_mins} min.")

    processed_data = []
    rows           = []
    progress       = st.progress(0, text=f"Scoring {ticker} headlines...")
    live_table     = st.empty()
    rate_status    = st.empty()

    for i, article in enumerate(raw_articles):
        title        = article["title"]
        dt           = article["dt"]
        url          = article.get("url", "")
        description  = article.get("description", "")
        article_text = get_full_article_text(url) if url else ""
        if not article_text:
            article_text = description

        score  = get_sentiment(title, ticker, article_text)
        source = "📄 Full" if len(article_text) > 100 else "📰 Headline"

        processed_data.append({"Timestamp": dt, "Sentiment": score})
        rows.append({
            "Date": dt.strftime('%Y-%m-%d'), "Source": source,
            "Headline": title[:90] + "..." if len(title) > 90 else title,
            "Score": score
        })

        recent_count = len([t for t in st.session_state.groq_request_times if time.time() - t < 60])
        rate_status.caption(
            f"🔄 Rate limiter: {recent_count}/{GROQ_RPM_LIMIT} requests in last 60s | "
            f"Total: {st.session_state.groq_total_requests} | Waits: {st.session_state.groq_rate_waits}"
        )
        progress.progress((i + 1) / total, text=f"Scoring {ticker}: {i+1}/{total} ({source})")
        if i % 5 == 0:
            live_table.dataframe(
                pd.DataFrame(rows).style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-1, vmax=1),
                use_container_width=True
            )

    live_table.dataframe(
        pd.DataFrame(rows).style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-1, vmax=1),
        use_container_width=True
    )
    progress.progress(1.0, text=f"✅ {ticker} scoring complete!")
    rate_status.empty()

    save_sentiment_scores(ticker, rows)

    sent_df = pd.DataFrame(processed_data)
    sent_df['Timestamp'] = sent_df['Timestamp'].dt.round('1h')
    sent_df['Timestamp'] = sent_df['Timestamp'].astype('datetime64[s]')
    sent_df = sent_df.sort_values('Timestamp').reset_index(drop=True)
    sent_df['Sentiment_Rolling'] = sent_df['Sentiment'].rolling(window=7, min_periods=1).mean()

    combined = pd.merge_asof(
        sent_df, hist_df[['Timestamp', 'Price_Change_Next']],
        on='Timestamp', direction='nearest', tolerance=pd.Timedelta('7 days')
    )
    combined = combined.dropna()
    combined.rename(columns={'Price_Change_Next': 'Price_Change'}, inplace=True)

    if combined.empty:
        st.warning(f"⚠️ No overlapping data for {ticker}")
        return None, hist_raw
    if combined['Sentiment'].std() == 0:
        st.warning(f"⚠️ All sentiment scores identical for {ticker}")
        return None, hist_raw
    if combined['Price_Change'].std() == 0:
        st.warning(f"⚠️ All price changes identical for {ticker}")
        return None, hist_raw

    return combined, hist_raw

# --- COMPARISON DASHBOARD ---

def plot_comparison(all_results):
    """
    Side-by-side comparison of all tickers:
    - Correlation bar chart
    - Normalized price performance
    - Sentiment heatmap
    - Summary table
    """
    st.markdown("---")
    st.markdown("## 🔀 Multi-Ticker Comparison")

    valid = {t: d for t, d in all_results.items() if d["combined"] is not None}
    if len(valid) < 2:
        st.info("Need at least 2 tickers with valid data for comparison.")
        return

    tickers     = list(valid.keys())
    colors      = ['#00b4d8', '#f72585', '#00f5d4', '#ffd166']
    ticker_clr  = {t: colors[i] for i, t in enumerate(tickers)}

    # --- 1. Summary Metrics Table ---
    st.markdown("### 📊 Summary")
    summary_rows = []
    for t, d in valid.items():
        corr   = d["combined"]["Sentiment"].corr(d["combined"]["Price_Change"])
        signal = "📈 Bullish" if corr > 0.3 else "📉 Bearish" if corr < -0.3 else "➡️ Neutral"
        summary_rows.append({
            "Ticker":        t,
            "Correlation":   round(corr, 4),
            "Signal":        signal,
            "Articles Used": len(d["combined"]),
            "Avg Sentiment": round(d["combined"]["Sentiment"].mean(), 3),
            "Avg Price Chg": f"{round(d['combined']['Price_Change'].mean() * 100, 3)}%"
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
    st.dataframe(
        summary_df.style.background_gradient(subset=["Correlation"], cmap="RdYlGn", vmin=-1, vmax=1),
        use_container_width=True
    )

    # --- 2. Correlation Bar Chart ---
    st.markdown("### 🎯 Correlation Comparison")
    fig1, ax = plt.subplots(figsize=(10, 4))
    fig1.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    corrs  = [d["combined"]["Sentiment"].corr(d["combined"]["Price_Change"]) for d in valid.values()]
    bar_colors = ['#00f5d4' if c > 0 else '#ef233c' for c in corrs]
    bars = ax.bar(tickers, corrs, color=bar_colors, alpha=0.8, width=0.4)
    for bar, val in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', color='white', fontsize=11, fontweight='bold')
    ax.axhline(0,    color='white', linewidth=0.8, linestyle='--')
    ax.axhline(0.3,  color='#00f5d4', linewidth=0.5, linestyle=':', alpha=0.6, label='Bullish threshold')
    ax.axhline(-0.3, color='#ef233c', linewidth=0.5, linestyle=':', alpha=0.6, label='Bearish threshold')
    ax.set_ylabel('Sentiment-Price Correlation', color='white')
    ax.set_ylim(-1.1, 1.1)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e2e', labelcolor='white', fontsize=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('gray')
    plt.title("Sentiment vs Price Correlation by Ticker", color='white', fontsize=12)
    fig1.tight_layout()
    st.pyplot(fig1)

    # --- 3. Normalized Price Performance ---
    st.markdown("### 📈 Normalized Price Performance (base 100)")
    fig2, ax = plt.subplots(figsize=(14, 4))
    fig2.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    for t, d in valid.items():
        if d["hist_raw"] is not None:
            close  = d["hist_raw"]['Close']
            normed = (close / close.iloc[0]) * 100
            ax.plot(normed.index, normed.values,
                    color=ticker_clr[t], linewidth=2, label=t)
    ax.axhline(100, color='white', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_ylabel('Normalized Price (base 100)', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e2e', labelcolor='white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('gray')
    plt.title("Price Performance (Normalized)", color='white', fontsize=12)
    plt.xticks(rotation=45, color='white')
    fig2.tight_layout()
    st.pyplot(fig2)

    # --- 4. Sentiment Rolling Average Comparison ---
    st.markdown("### 🧠 Sentiment Trend Comparison")
    fig3, ax = plt.subplots(figsize=(14, 4))
    fig3.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    for t, d in valid.items():
        df = d["combined"].sort_values('Timestamp')
        ax.plot(
            df['Timestamp'].dt.strftime('%m-%d'),
            df['Sentiment_Rolling'],
            color=ticker_clr[t], linewidth=2, label=f"{t} 7d Avg"
        )
    ax.axhline(0, color='white', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Rolling Avg Sentiment', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e1e2e', labelcolor='white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('gray')
    plt.title("7-Day Rolling Sentiment Average by Ticker", color='white', fontsize=12)
    plt.xticks(rotation=45, color='white')
    fig3.tight_layout()
    st.pyplot(fig3)

    # --- 5. Sentiment Heatmap ---
    st.markdown("### 🌡️ Sentiment Heatmap")
    pivot_rows = []
    for t, d in valid.items():
        df = d["combined"].copy()
        df['date']   = df['Timestamp'].dt.strftime('%m-%d')
        df['ticker'] = t
        pivot_rows.append(df[['date', 'ticker', 'Sentiment']])

    if pivot_rows:
        pivot_df = pd.concat(pivot_rows)
        pivot    = pivot_df.groupby(['ticker', 'date'])['Sentiment'].mean().unstack(fill_value=0)
        fig4, ax = plt.subplots(figsize=(14, len(tickers) * 1.2 + 1))
        fig4.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, color='white')
        step = max(1, len(pivot.columns) // 20)
        ax.set_xticks(range(0, len(pivot.columns), step))
        ax.set_xticklabels(pivot.columns[::step], rotation=45, color='white')
        plt.colorbar(im, ax=ax, label='Sentiment')
        plt.title("Sentiment Heatmap by Ticker & Date", color='white', fontsize=12)
        fig4.tight_layout()
        st.pyplot(fig4)

    # --- 6. Win Rate: Does positive sentiment predict positive next-day price? ---
    st.markdown("### 🎯 Signal Accuracy")
    acc_cols = st.columns(len(valid))
    for col, (t, d) in zip(acc_cols, valid.items()):
        df       = d["combined"]
        correct  = ((df['Sentiment'] > 0) & (df['Price_Change'] > 0)) | \
                   ((df['Sentiment'] < 0) & (df['Price_Change'] < 0))
        accuracy = correct.mean() * 100
        col.metric(
            label=t,
            value=f"{accuracy:.1f}%",
            delta="Signal Accuracy",
            delta_color="normal"
        )

# --- INDIVIDUAL TICKER PLOT ---

def plot_ticker(ticker, combined, hist_raw):
    correlation = combined['Sentiment'].corr(combined['Price_Change'])
    signal      = "📈 Bullish" if correlation > 0.3 else "📉 Bearish" if correlation < -0.3 else "➡️ Neutral"

    save_correlation(ticker, correlation, signal, len(combined), period)

    if email_alerts and abs(correlation) >= alert_threshold:
        with st.spinner(f"Sending alert for {ticker}..."):
            sent = send_alert(ticker, correlation, signal)
        st.success("✅ Alert sent!") if sent else st.warning("⚠️ Alert failed.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Correlation",   f"{correlation:.4f}")
    col2.metric("Articles Used", len(combined))
    col3.metric("Date Range",    f"{combined['Timestamp'].min().strftime('%m-%d')} → {combined['Timestamp'].max().strftime('%m-%d')}")
    col4.metric("Signal",        signal)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')
    ax1.set_facecolor('#0e1117')
    ax1.bar(
        combined['Timestamp'].dt.strftime('%m-%d'),
        combined['Sentiment'],
        color=['#00b4d8' if s >= 0 else '#ef233c' for s in combined['Sentiment']],
        alpha=0.6, label='AI Sentiment'
    )
    ax1.plot(
        combined['Timestamp'].dt.strftime('%m-%d'),
        combined['Sentiment_Rolling'],
        color='yellow', linewidth=1.5, linestyle='--', label='7-day Rolling Avg'
    )
    ax1.set_ylabel('Sentiment (-1 to 1)', color='#00b4d8')
    ax1.tick_params(axis='y', labelcolor='#00b4d8')
    ax1.tick_params(axis='x', colors='white')
    ax1.axhline(0, color='white', linewidth=0.5, linestyle='--')
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_color('gray')

    ax2 = ax1.twinx()
    ax2.plot(
        combined['Timestamp'].dt.strftime('%m-%d'),
        combined['Price_Change'] * 100,
        color='#f72585', marker='o', linewidth=2, label='Next Day Price Change %'
    )
    ax2.set_ylabel('Next Day Price Change %', color='#f72585')
    ax2.tick_params(axis='y', labelcolor='#f72585')
    ax2.spines['right'].set_color('gray')
    ax2.spines['top'].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               facecolor='#1e1e2e', labelcolor='white', fontsize=8)
    plt.title(f"{ticker} — Sentiment vs Next-Day Price | Correlation: {correlation:.2f}", color='white', fontsize=12)
    plt.xticks(rotation=45, color='white')
    fig.tight_layout()
    st.pyplot(fig)

    fig2, ax = plt.subplots(figsize=(14, 3))
    fig2.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    close_prices = hist_raw['Close']
    ax.plot(close_prices.index, close_prices.values, color='#00f5d4', linewidth=1.5, label=f'{ticker} Close Price')
    ax.fill_between(close_prices.index, close_prices.values, alpha=0.1, color='#00f5d4')
    ax.set_ylabel('Price (USD)', color='#00f5d4')
    ax.tick_params(axis='y', labelcolor='#00f5d4')
    ax.tick_params(axis='x', colors='white')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('gray')
    ax.legend(facecolor='#1e1e2e', labelcolor='white')
    plt.title(f"{ticker} — Price History", color='white', fontsize=11)
    plt.xticks(rotation=45, color='white')
    fig2.tight_layout()
    st.pyplot(fig2)

    st.markdown("#### 🔍 Key Headlines")
    col_bull, col_bear = st.columns(2)
    with col_bull:
        st.markdown("**📈 Most Bullish**")
        st.dataframe(
            combined.nlargest(3, 'Sentiment')[['Timestamp', 'Sentiment']]
            .style.background_gradient(subset=['Sentiment'], cmap='Greens', vmin=0, vmax=1),
            use_container_width=True
        )
    with col_bear:
        st.markdown("**📉 Most Bearish**")
        st.dataframe(
            combined.nsmallest(3, 'Sentiment')[['Timestamp', 'Sentiment']]
            .style.background_gradient(subset=['Sentiment'], cmap='Reds_r', vmin=-1, vmax=0),
            use_container_width=True
        )
    with st.expander("📋 View Raw Combined Data"):
        st.dataframe(
            combined[['Timestamp', 'Sentiment', 'Sentiment_Rolling', 'Price_Change']]
            .style.background_gradient(subset=['Sentiment'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True
        )

# --- HISTORY DASHBOARD ---

def show_history_dashboard():
    st.markdown("---")
    st.markdown("## 🗄️ Historical Data")
    if not SUPABASE_URL:
        st.warning("⚠️ Supabase not configured.")
        return

    tab_corr, tab_sent, tab_runs = st.tabs([
        "📊 Correlation History", "🧠 Sentiment Scores", "🕐 Run Log"
    ])

    with tab_corr:
        hist_df = load_correlation_history()
        if hist_df.empty:
            st.info("No correlation history yet.")
        else:
            fig, ax = plt.subplots(figsize=(14, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            for t in hist_df['ticker'].unique():
                td = hist_df[hist_df['ticker'] == t].sort_values('created_at')
                ax.plot(td['created_at'], td['correlation'], marker='o', linewidth=2, label=t)
            ax.axhline(0, color='white', linewidth=0.5, linestyle='--')
            ax.axhline(0.3, color='green', linewidth=0.5, linestyle=':', alpha=0.5)
            ax.axhline(-0.3, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
            ax.set_ylabel('Correlation', color='white')
            ax.tick_params(colors='white')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('gray')
            ax.legend(facecolor='#1e1e2e', labelcolor='white')
            plt.title("Correlation Over Time", color='white')
            plt.xticks(rotation=45, color='white')
            fig.tight_layout()
            st.pyplot(fig)
            st.dataframe(
                hist_df.style.background_gradient(subset=['correlation'], cmap='RdYlGn', vmin=-1, vmax=1),
                use_container_width=True
            )

    with tab_sent:
        ticker_filter = st.selectbox("Filter by Ticker", ["All"] + list(TICKERS))
        sent_hist = load_sentiment_history(ticker=None if ticker_filter == "All" else ticker_filter)
        if sent_hist.empty:
            st.info("No sentiment history yet.")
        else:
            st.markdown(f"**{len(sent_hist)} scored headlines stored**")
            st.dataframe(
                sent_hist.style.background_gradient(subset=['score'], cmap='RdYlGn', vmin=-1, vmax=1),
                use_container_width=True
            )
            st.download_button(
                label="⬇️ Download as CSV",
                data=sent_hist.to_csv(index=False),
                file_name=f"sentiment_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with tab_runs:
        run_log = load_run_log()
        if run_log.empty:
            st.info("No runs logged yet.")
        else:
            st.markdown(f"**{len(run_log)} runs recorded**")
            st.dataframe(run_log, use_container_width=True)

# --- ENTRY POINT ---
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing.")
    st.code('''
# .streamlit/secrets.toml
GROQ_API_KEY   = "your_groq_key"
NEWSAPI_KEY    = "your_newsapi_key"
SUPABASE_URL   = "https://your-project.supabase.co"
SUPABASE_KEY   = "your-anon-key"
EMAIL_SENDER   = "your_gmail@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "receiver@gmail.com"
    ''', language="toml")
    st.stop()

auto_triggered = auto_run and 'auto_refresh' in st.session_state

if run_btn or auto_triggered:
    st.caption(f"🕐 Last run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mode: {'⏰ Auto' if auto_triggered else '🖱️ Manual'}")
    save_run_log(TICKERS, schedule_interval if auto_triggered else "manual")

    # Collect all results for comparison
    all_results = {}

    if len(TICKERS) == 1:
        ticker = TICKERS[0]
        st.markdown(f"## {ticker}")
        combined, hist_raw = analyze_ticker(ticker, period, days_back)
        if combined is not None:
            plot_ticker(ticker, combined, hist_raw)
        all_results[ticker] = {"combined": combined, "hist_raw": hist_raw}
    else:
        tabs = st.tabs([f"📊 {t}" for t in TICKERS] + ["🔀 Comparison"])
        for tab, ticker in zip(tabs[:-1], TICKERS):
            with tab:
                st.markdown(f"## {ticker}")
                combined, hist_raw = analyze_ticker(ticker, period, days_back)
                if combined is not None:
                    plot_ticker(ticker, combined, hist_raw)
                all_results[ticker] = {"combined": combined, "hist_raw": hist_raw}

        # Comparison tab — always last
        with tabs[-1]:
            plot_comparison(all_results)

else:
    st.info("👈 Enter ticker symbols in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    ### ✨ Features
    - 📊 **Multi-ticker comparison** — correlation bars, normalized price, sentiment heatmap, signal accuracy
    - 🧠 **AI sentiment scoring** — Groq Llama 3.1 on full articles
    - 📄 **Full article text** — fetches complete articles, not just headlines
    - 📰 **30 days of news** — via NewsAPI (100 articles)
    - 📈 **Next-day price correlation** — does sentiment predict price?
    - 🔄 **Rolling 7-day average** — smooth out noise
    - 🔔 **Email alerts** — notified when correlation crosses threshold
    - ⏰ **Auto-run scheduling** — runs every 15min / 1hr / 24hr
    - 🗄️ **Supabase history** — persists forever across restarts
    - ⚡ **Cached results** — won't re-score same headlines for 1 hour
    - 🚦 **Rate limiter** — sliding window keeps Groq under 28 RPM
    """)

if show_history:
    show_history_dashboard()