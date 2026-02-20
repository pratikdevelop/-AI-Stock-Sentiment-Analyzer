import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import requests
from groq import Groq

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

# --- SECRETS (Streamlit Cloud) ---
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", "")
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"

# --- SIDEBAR ---
st.sidebar.title("⚙️ Configuration")
TICKER = st.sidebar.text_input("Stock Ticker", value="NVDA").upper()
days_back = st.sidebar.slider("News History (days)", min_value=7, max_value=30, value=30)
period = st.sidebar.selectbox("Stock History Period", ["1mo", "3mo", "6mo"], index=1)
run_btn = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Status**")
st.sidebar.markdown(f"Groq API: {'✅ Connected' if GROQ_API_KEY else '❌ Missing'}")
st.sidebar.markdown(f"NewsAPI: {'✅ Connected' if NEWSAPI_KEY else '❌ Missing'}")
st.sidebar.markdown(f"Model: `{GROQ_MODEL}`")

# --- MAIN TITLE ---
st.title("📈 AI Stock Sentiment Analyzer")
st.markdown("Correlates **AI-scored news sentiment** with **next-day stock price changes** using Groq + Llama 3.1.")

# --- FUNCTIONS ---
def get_sentiment(headline, ticker):
    """Score headline sentiment using Groq Llama 3.1"""
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not set in Streamlit secrets.")
        return 0.0
    try:
        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"Analyze sentiment for {ticker} stock: '{headline}'. Return ONLY a number between -1.0 (very negative) and 1.0 (very positive). No words, just the number."
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
    except Exception as e:
        st.warning(f"Groq error: {e}")
        return 0.0

def get_news_newsapi(ticker_name, days_back):
    """Fetch articles from NewsAPI"""
    if not NEWSAPI_KEY:
        return []
    try:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": ticker_name,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 100,
                "from": from_date,
                "apiKey": NEWSAPI_KEY
            },
            timeout=15
        )
        articles = []
        for a in response.json().get("articles", []):
            if not a.get('title') or a['title'] == '[Removed]':
                continue
            dt = datetime.fromisoformat(a['publishedAt'].replace('Z', '+00:00')).replace(tzinfo=None)
            articles.append({"title": a['title'], "dt": dt})
        return articles
    except Exception as e:
        st.error(f"NewsAPI Error: {e}")
        return []

def get_news_yfinance(stock):
    """Fallback: yfinance news (~10 articles)"""
    articles = []
    for article in stock.news:
        content = article.get('content', article)
        title = content.get('title', "")
        raw_date = content.get('pubDate', content.get('providerPublishTime'))
        if not title:
            continue
        if isinstance(raw_date, str):
            dt = datetime.fromisoformat(raw_date.replace('Z', '+00:00')).replace(tzinfo=None)
        else:
            dt = datetime.fromtimestamp(raw_date)
        articles.append({"title": title, "dt": dt})
    return articles

def run_analysis(ticker, period, days_back):

    # 1. Stock Data
    with st.status("📊 Fetching stock data...", expanded=True) as status:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval="1d")

        if hist.empty:
            st.error("❌ No market data found. Check the ticker symbol.")
            return

        hist.index = hist.index.tz_localize(None)
        hist['Price_Change'] = hist['Close'].pct_change()
        hist_df = hist.reset_index()
        hist_df.rename(columns={hist_df.columns[0]: 'Timestamp'}, inplace=True)
        hist_df = hist_df.dropna(subset=['Price_Change'])

        # Lagged: news today → price change tomorrow
        hist_df['Price_Change_Next'] = hist_df['Price_Change'].shift(-1)
        hist_df = hist_df.dropna(subset=['Price_Change_Next'])

        # Fix datetime precision
        hist_df['Timestamp'] = hist_df['Timestamp'].astype('datetime64[s]')
        hist_df = hist_df.sort_values('Timestamp')

        status.update(label=f"✅ Loaded {len(hist_df)} trading days for {ticker}", state="complete")

    # 2. News
    with st.status("📰 Fetching news...", expanded=True) as status:
        if NEWSAPI_KEY:
            raw_articles = get_news_newsapi(ticker_name=ticker, days_back=days_back)
            source = "NewsAPI"
        else:
            st.warning("⚠️ No NEWSAPI_KEY — falling back to yfinance (~10 articles only).")
            raw_articles = get_news_yfinance(stock)
            source = "yfinance"

        if not raw_articles:
            st.error("❌ No news articles found.")
            return

        status.update(label=f"✅ {len(raw_articles)} articles from {source}", state="complete")

    # 3. Sentiment Scoring
    st.markdown("### 🧠 Scoring Headlines...")
    progress_bar = st.progress(0, text="Starting...")
    live_table = st.empty()

    processed_data = []
    rows = []

    for i, article in enumerate(raw_articles):
        title = article["title"]
        dt = article["dt"]
        score = get_sentiment(title, ticker)

        processed_data.append({"Timestamp": dt, "Sentiment": score})
        rows.append({
            "Date": dt.strftime('%Y-%m-%d'),
            "Headline": title[:90] + "..." if len(title) > 90 else title,
            "Score": score
        })

        pct = (i + 1) / len(raw_articles)
        progress_bar.progress(pct, text=f"Scored {i+1}/{len(raw_articles)} articles...")
        live_table.dataframe(
            pd.DataFrame(rows).style.background_gradient(subset=['Score'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True
        )

    progress_bar.progress(1.0, text="✅ Scoring complete!")

    # 4. Merge
    sent_df = pd.DataFrame(processed_data)
    sent_df['Timestamp'] = sent_df['Timestamp'].dt.round('1h')
    sent_df['Timestamp'] = sent_df['Timestamp'].astype('datetime64[s]')  # cast AFTER round
    sent_df = sent_df.sort_values('Timestamp').reset_index(drop=True)

    combined = pd.merge_asof(
        sent_df,
        hist_df[['Timestamp', 'Price_Change_Next']],
        on='Timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('7 days')
    )
    combined = combined.dropna()
    combined.rename(columns={'Price_Change_Next': 'Price_Change'}, inplace=True)

    # 5. Validate
    if combined.empty:
        st.error("⚠️ No overlapping data between news dates and stock history.")
        return
    if combined['Sentiment'].std() == 0:
        st.error("⚠️ All sentiment scores identical — can't compute correlation.")
        return
    if combined['Price_Change'].std() == 0:
        st.error("⚠️ All price changes identical — not enough price variation.")
        return

    # 6. Results
    correlation = combined['Sentiment'].corr(combined['Price_Change'])

    st.markdown("---")
    st.markdown("### 📊 Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📉 Correlation", f"{correlation:.4f}")
    col2.metric("📰 Articles Used", len(combined))
    col3.metric("📅 Date Range", f"{combined['Timestamp'].min().strftime('%m-%d')} → {combined['Timestamp'].max().strftime('%m-%d')}")
    col4.metric("🧭 Signal",
        "📈 Bullish" if correlation > 0.3 else
        "📉 Bearish" if correlation < -0.3 else
        "➡️ Neutral"
    )

    # 7. Chart
    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0e1117')
    ax1.set_facecolor('#0e1117')

    bars = ax1.bar(
        combined['Timestamp'].dt.strftime('%m-%d'),
        combined['Sentiment'],
        color=['#00b4d8' if s >= 0 else '#ef233c' for s in combined['Sentiment']],
        alpha=0.7,
        label='AI Sentiment'
    )
    ax1.set_ylabel('Sentiment (-1 to 1)', color='#00b4d8')
    ax1.tick_params(axis='y', labelcolor='#00b4d8')
    ax1.tick_params(axis='x', colors='white')
    ax1.axhline(0, color='white', linewidth=0.5, linestyle='--')
    ax1.spines['bottom'].set_color('gray')
    ax1.spines['left'].set_color('gray')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(
        combined['Timestamp'].dt.strftime('%m-%d'),
        combined['Price_Change'] * 100,
        color='#f72585',
        marker='o',
        linewidth=2,
        label='Next Day Price Change %'
    )
    ax2.set_ylabel('Next Day Price Change %', color='#f72585')
    ax2.tick_params(axis='y', labelcolor='#f72585')
    ax2.spines['right'].set_color('gray')
    ax2.spines['top'].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', facecolor='#1e1e2e', labelcolor='white')

    plt.title(
        f"{ticker} — Sentiment vs Next-Day Price Change | Correlation: {correlation:.2f}",
        color='white', fontsize=13
    )
    plt.xticks(rotation=45, color='white')
    fig.tight_layout()
    st.pyplot(fig)

    # 8. Raw Data
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            combined[['Timestamp', 'Sentiment', 'Price_Change']].style.background_gradient(
                subset=['Sentiment'], cmap='RdYlGn', vmin=-1, vmax=1
            ),
            use_container_width=True
        )

# --- ENTRY POINT ---
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing. Add it to Streamlit secrets to continue.")
elif run_btn:
    run_analysis(TICKER, period, days_back)
else:
    st.info("👈 Set your ticker and click **Run Analysis** to start.")
