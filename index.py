import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import requests
import time
import os
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from groq import Groq

# --- DATABASE SETUP ---
DB_PATH = "sentiment_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT,
            date        TEXT,
            headline    TEXT,
            source      TEXT,
            score       REAL,
            created_at  TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS correlation_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT,
            correlation REAL,
            signal      TEXT,
            articles    INTEGER,
            period      TEXT,
            created_at  TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_sentiment_scores(ticker, rows):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for row in rows:
        c.execute('''
            INSERT INTO sentiment_scores (ticker, date, headline, source, score, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (ticker, row['Date'], row['Headline'], row.get('Source', 'Headline'), row['Score'], now))
    conn.commit()
    conn.close()

def save_correlation(ticker, correlation, signal, articles, period):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO correlation_history (ticker, correlation, signal, articles, period, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (ticker, correlation, signal, articles, period, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

@st.cache_data(ttl=300)
def load_correlation_history(ticker=None):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM correlation_history"
    if ticker:
        query += f" WHERE ticker = '{ticker}'"
    query += " ORDER BY created_at DESC LIMIT 100"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data(ttl=300)
def load_sentiment_history(ticker=None):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM sentiment_scores"
    if ticker:
        query += f" WHERE ticker = '{ticker}'"
    query += " ORDER BY created_at DESC LIMIT 500"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sentiment_scores")
    conn.execute("DELETE FROM correlation_history")
    conn.commit()
    conn.close()

# Initialize DB immediately
init_db()

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
GROQ_MODEL     = "llama-3.1-8b-instant"

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
st.sidebar.markdown("**🔔 Alert Settings**")
alert_threshold = st.sidebar.slider("Alert Threshold (|correlation|)", min_value=0.1, max_value=1.0, value=0.3, step=0.05)
email_alerts    = st.sidebar.toggle("Enable Email Alerts", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**🗄️ Database**")
show_history = st.sidebar.toggle("Show History Dashboard", value=False)
clear_btn    = st.sidebar.button("🗑️ Clear All History", use_container_width=True)
if clear_btn:
    clear_history()
    st.sidebar.success("✅ History cleared!")

st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")
st.sidebar.markdown(f"Groq API: {'✅ Connected' if GROQ_API_KEY else '❌ Missing'}")
st.sidebar.markdown(f"NewsAPI: {'✅ Connected' if NEWSAPI_KEY else '❌ Missing'}")
st.sidebar.markdown(f"Email Alerts: {'✅ Configured' if EMAIL_SENDER else '❌ Not Set'}")
st.sidebar.markdown(f"Model: `{GROQ_MODEL}`")
st.sidebar.markdown(f"Analyzing: `{', '.join(TICKERS)}`")

# --- MAIN TITLE ---
st.title("📈 AI Stock Sentiment Analyzer")
st.markdown("Correlates **AI-scored news sentiment** with **next-day stock price changes**.")

# --- CORE FUNCTIONS ---

@st.cache_data(ttl=3600, show_spinner=False)
def get_full_article_text(url):
    """Fetch and clean full article text from URL"""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        text = re.sub(r'<[^>]+>', ' ', response.text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:2000]
    except Exception:
        return ""

@st.cache_data(ttl=3600, show_spinner=False)
def get_sentiment(headline, ticker, article_text=""):
    """Score sentiment using Groq — full article if available, else headline"""
    if not GROQ_API_KEY:
        return 0.0
    try:
        client = Groq(api_key=GROQ_API_KEY)
        if article_text and len(article_text) > 100:
            content = f"Headline: {headline}\n\nArticle: {article_text[:1500]}"
            prompt  = f"Analyze sentiment for {ticker} stock based on this news article. Return ONLY a number between -1.0 (very negative) and 1.0 (very positive). No words, just the number.\n\n{content}"
        else:
            prompt = f"Analyze sentiment for {ticker} stock: '{headline}'. Return ONLY a number between -1.0 and 1.0. No words, just the number."

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.1
        )
        score_text = response.choices[0].message.content.strip()
        match = re.search(r'-?\d+\.?\d*', score_text)
        if match:
            return max(-1.0, min(1.0, float(match.group())))
        return 0.0
    except Exception:
        time.sleep(1)
        return 0.0

@st.cache_data(ttl=3600, show_spinner=False)
def get_news_newsapi(ticker_name, days_back):
    """Fetch articles from NewsAPI with URL and description"""
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
                "title":       a['title'],
                "dt":          dt,
                "url":         a.get('url', ''),
                "description": a.get('description', '')
            })
        return articles
    except Exception as e:
        st.error(f"NewsAPI Error: {e}")
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def get_news_yfinance(ticker):
    """Fallback: yfinance news"""
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
    """Fetch and process stock OHLCV data"""
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
    """Send email alert"""
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
                <tr style="background:#1e1e2e;">
                    <td style="padding:10px;border:1px solid #333;"><b>Ticker</b></td>
                    <td style="padding:10px;border:1px solid #333;">{ticker}</td>
                </tr>
                <tr>
                    <td style="padding:10px;border:1px solid #333;"><b>Correlation</b></td>
                    <td style="padding:10px;border:1px solid #333;">{correlation:.4f}</td>
                </tr>
                <tr style="background:#1e1e2e;">
                    <td style="padding:10px;border:1px solid #333;"><b>Signal</b></td>
                    <td style="padding:10px;border:1px solid #333;">{signal}</td>
                </tr>
                <tr>
                    <td style="padding:10px;border:1px solid #333;"><b>Time</b></td>
                    <td style="padding:10px;border:1px solid #333;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
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
    """Full pipeline: fetch → score → merge → return combined df"""

    # 1. Stock data
    hist_df, hist_raw = get_stock_data(ticker, period)
    if hist_df is None:
        st.error(f"❌ No market data for {ticker}")
        return None, None

    # 2. News
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

    # 3. Sentiment scoring
    processed_data = []
    rows           = []
    progress       = st.progress(0, text=f"Scoring {ticker} headlines...")
    live_table     = st.empty()

    for i, article in enumerate(raw_articles):
        title       = article["title"]
        dt          = article["dt"]
        url         = article.get("url", "")
        description = article.get("description", "")

        # Priority: full article → description → headline only
        article_text = get_full_article_text(url) if url else ""
        if not article_text:
            article_text = description

        score  = get_sentiment(title, ticker, article_text)
        source = "📄 Full" if len(article_text) > 100 else "📰 Headline"

        processed_data.append({"Timestamp": dt, "Sentiment": score})
        rows.append({
            "Date":     dt.strftime('%Y-%m-%d'),
            "Source":   source,
            "Headline": title[:90] + "..." if len(title) > 90 else title,
            "Score":    score
        })

        progress.progress((i + 1) / len(raw_articles), text=f"Scoring {ticker}: {i+1}/{len(raw_articles)} ({source})")
        if i % 5 == 0:
            live_table.dataframe(
                pd.DataFrame(rows).style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-1, vmax=1),
                use_container_width=True
            )

    # Final table update
    live_table.dataframe(
        pd.DataFrame(rows).style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-1, vmax=1),
        use_container_width=True
    )
    progress.progress(1.0, text=f"✅ {ticker} scoring complete!")

    # Save to DB
    save_sentiment_scores(ticker, rows)

    # 4. Build sentiment df
    sent_df = pd.DataFrame(processed_data)
    sent_df['Timestamp'] = sent_df['Timestamp'].dt.round('1h')
    sent_df['Timestamp'] = sent_df['Timestamp'].astype('datetime64[s]')  # cast AFTER round
    sent_df = sent_df.sort_values('Timestamp').reset_index(drop=True)
    sent_df['Sentiment_Rolling'] = sent_df['Sentiment'].rolling(window=7, min_periods=1).mean()

    # 5. Merge
    combined = pd.merge_asof(
        sent_df,
        hist_df[['Timestamp', 'Price_Change_Next']],
        on='Timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('7 days')
    )
    combined = combined.dropna()
    combined.rename(columns={'Price_Change_Next': 'Price_Change'}, inplace=True)

    # 6. Validate
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

# --- VISUALIZATION ---

def plot_ticker(ticker, combined, hist_raw):
    correlation = combined['Sentiment'].corr(combined['Price_Change'])
    signal      = "📈 Bullish" if correlation > 0.3 else "📉 Bearish" if correlation < -0.3 else "➡️ Neutral"

    # Always save correlation to DB
    save_correlation(ticker, correlation, signal, len(combined), period)

    # Email alert if enabled and threshold crossed
    if email_alerts and abs(correlation) >= alert_threshold:
        with st.spinner(f"Sending alert for {ticker}..."):
            sent = send_alert(ticker, correlation, signal)
        if sent:
            st.success(f"✅ Alert sent! {ticker} correlation: {correlation:.4f}")
        else:
            st.warning("⚠️ Alert failed — check email secrets.")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Correlation", f"{correlation:.4f}")
    col2.metric("Articles Used", len(combined))
    col3.metric("Date Range", f"{combined['Timestamp'].min().strftime('%m-%d')} → {combined['Timestamp'].max().strftime('%m-%d')}")
    col4.metric("Signal", signal)

    # Chart 1: Sentiment vs Price Change
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
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor='#1e1e2e', labelcolor='white', fontsize=8)
    plt.title(f"{ticker} — Sentiment vs Next-Day Price | Correlation: {correlation:.2f}", color='white', fontsize=12)
    plt.xticks(rotation=45, color='white')
    fig.tight_layout()
    st.pyplot(fig)

    # Chart 2: Stock Price History
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

    # Top Bullish/Bearish Headlines
    st.markdown("#### 🔍 Key Headlines")
    col_bull, col_bear = st.columns(2)
    top_bullish = combined.nlargest(3, 'Sentiment')[['Timestamp', 'Sentiment']]
    top_bearish = combined.nsmallest(3, 'Sentiment')[['Timestamp', 'Sentiment']]

    with col_bull:
        st.markdown("**📈 Most Bullish**")
        st.dataframe(top_bullish.style.background_gradient(subset=['Sentiment'], cmap='Greens', vmin=0, vmax=1), use_container_width=True)
    with col_bear:
        st.markdown("**📉 Most Bearish**")
        st.dataframe(top_bearish.style.background_gradient(subset=['Sentiment'], cmap='Reds_r', vmin=-1, vmax=0), use_container_width=True)

    with st.expander("📋 View Raw Combined Data"):
        st.dataframe(
            combined[['Timestamp', 'Sentiment', 'Sentiment_Rolling', 'Price_Change']].style.background_gradient(
                subset=['Sentiment'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True
        )

# --- HISTORY DASHBOARD ---

def show_history_dashboard():
    st.markdown("---")
    st.markdown("## 🗄️ Historical Data")
    tab_corr, tab_sent = st.tabs(["📊 Correlation History", "🧠 Sentiment Scores"])

    with tab_corr:
        hist_df = load_correlation_history()
        if hist_df.empty:
            st.info("No correlation history yet. Run an analysis first.")
        else:
            fig, ax = plt.subplots(figsize=(14, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            for t in hist_df['ticker'].unique():
                td = hist_df[hist_df['ticker'] == t].sort_values('created_at')
                ax.plot(td['created_at'], td['correlation'], marker='o', linewidth=2, label=t)
            ax.axhline(0,    color='white', linewidth=0.5, linestyle='--')
            ax.axhline(0.3,  color='green', linewidth=0.5, linestyle=':', alpha=0.5)
            ax.axhline(-0.3, color='red',   linewidth=0.5, linestyle=':', alpha=0.5)
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
            st.info("No sentiment history yet. Run an analysis first.")
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

# --- ENTRY POINT ---
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing.")
    st.code('''
# .streamlit/secrets.toml
GROQ_API_KEY   = "your_groq_key"
NEWSAPI_KEY    = "your_newsapi_key"
EMAIL_SENDER   = "your_gmail@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_RECEIVER = "receiver@gmail.com"
    ''', language="toml")
    st.stop()

if run_btn:
    if len(TICKERS) == 1:
        ticker = TICKERS[0]
        st.markdown(f"## {ticker}")
        combined, hist_raw = analyze_ticker(ticker, period, days_back)
        if combined is not None:
            plot_ticker(ticker, combined, hist_raw)
    else:
        tabs = st.tabs([f"📊 {t}" for t in TICKERS])
        for tab, ticker in zip(tabs, TICKERS):
            with tab:
                st.markdown(f"## {ticker}")
                combined, hist_raw = analyze_ticker(ticker, period, days_back)
                if combined is not None:
                    plot_ticker(ticker, combined, hist_raw)
else:
    st.info("👈 Enter ticker symbols in the sidebar and click **Run Analysis** to start.")
    st.markdown("""
    ### ✨ Features
    - 📊 **Multi-ticker comparison** — up to 4 stocks side by side in tabs
    - 🧠 **AI sentiment scoring** — Groq Llama 3.1 scores full articles
    - 📄 **Full article text** — fetches complete articles, not just headlines
    - 📰 **30 days of news** — via NewsAPI (100 articles)
    - 📈 **Next-day price correlation** — does sentiment predict price?
    - 🔄 **Rolling 7-day average** — smooth out noise
    - 🔔 **Email alerts** — get notified when correlation crosses threshold
    - 🗄️ **SQLite history** — all results stored and downloadable as CSV
    - ⚡ **Cached results** — won't re-score same headlines for 1 hour
    """)

if show_history:
    show_history_dashboard()