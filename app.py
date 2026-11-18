"""Streamlit app: lexicon-based sentiment analysis with a gauge chart."""

import io
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sentiment_core as sc

LEXICON_FILE = os.path.join(os.path.dirname(__file__), "Economic_Lexicon.xlsx")
MAX_URLS = 10
MAX_PDF_MB = 20

st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="wide")
st.title("📊 Economic Sentiment Analysis")
st.caption("Lexicon-based sentiment scoring of text, web pages and PDF documents.")


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def get_lexicon(path: str) -> dict:
    return sc.load_lexicon(path)


def gauge(score: float, title: str, rng: float) -> go.Figure:
    band = 0.05
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"valueformat": ".3f"},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [-rng, rng]},
            "bar": {"color": "#333333", "thickness": 0.25},
            "steps": [
                {"range": [-rng, -band], "color": "#e57373"},
                {"range": [-band, band], "color": "#d9d9d9"},
                {"range": [band, rng], "color": "#81c784"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=70, b=10))
    return fig


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sentiment")
    return buf.getvalue()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    uploaded_lex = st.file_uploader("Custom lexicon (optional, .xlsx)", type=["xlsx"])
    negation = st.checkbox("Handle negation (e.g. 'not good')", value=True)
    gauge_range = st.slider("Gauge range (±)", 0.1, 1.0, 0.5, 0.05,
                            help="Average scores are usually small; a narrower range "
                                 "makes the needle easier to read.")
    st.markdown("**Score** = average `sentiment` of the words found in the lexicon.")

# ------------------------------------------------------------------ lexicon
try:
    if uploaded_lex is not None:
        lexicon = sc.load_lexicon(uploaded_lex)
    else:
        lexicon = get_lexicon(LEXICON_FILE)
    st.sidebar.success(f"Lexicon loaded: {len(lexicon):,} tokens")
except Exception as e:
    st.error(f"Could not load the lexicon: {e}")
    st.stop()

# ------------------------------------------------------------------- inputs
tab_text, tab_url, tab_pdf = st.tabs(["✍️ Text", "🔗 URLs", "📄 PDF"])
with tab_text:
    text_input = st.text_area("Paste text", height=200)
with tab_url:
    urls_input = st.text_area(f"One URL per line (max {MAX_URLS})", height=150,
                              placeholder="https://example.com/article")
with tab_pdf:
    pdf_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if st.button("Analyze", type="primary"):
    sources, errors = {}, []

    if text_input.strip():
        sources["Pasted text"] = text_input

    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    if len(urls) > MAX_URLS:
        st.warning(f"Only the first {MAX_URLS} URLs will be analyzed.")
        urls = urls[:MAX_URLS]
    if urls:
        progress = st.progress(0.0, text="Fetching URLs...")
        for i, url in enumerate(urls, 1):
            try:
                if not sc.is_url(url):
                    raise ValueError("not a valid http(s) URL")
                sources[url] = sc.text_from_url(url)
            except Exception as e:
                errors.append(f"{url}: {e}")
            progress.progress(i / len(urls), text=f"Fetched {i}/{len(urls)}")
        progress.empty()

    for f in pdf_files or []:
        if f.size > MAX_PDF_MB * 1024 * 1024:
            errors.append(f"{f.name}: larger than {MAX_PDF_MB} MB")
            continue
        try:
            sources[f.name] = sc.text_from_pdf(f)
        except Exception as e:
            errors.append(f"{f.name}: {e}")

    for err in errors:
        st.warning(f"Skipped — {err}")

    sources = {k: v for k, v in sources.items() if v and v.strip()}
    if not sources:
        st.error("No readable text found. Add some text, a URL or a PDF.")
        st.stop()

    st.session_state["df"] = sc.analyze_texts(sources, lexicon, handle_negation=negation)

# ------------------------------------------------------------------ results
df = st.session_state.get("df")
if df is not None and not df.empty:
    overall = df.iloc[-1]
    st.subheader("Overall result")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(gauge(overall["avg_sentiment_matched"], "Overall sentiment", gauge_range),
                        width="stretch")
    with c2:
        st.metric("Label", str(overall["label"]).capitalize())
        st.metric("Words analyzed", f"{int(overall['total_words']):,}")
        st.metric("Words found in lexicon", f"{int(overall['matched_words']):,}")
        st.metric("Positive / Negative words",
                  f"{int(overall['positive_words'])} / {int(overall['negative_words'])}")
        if overall["matched_words"] == 0:
            st.warning("No lexicon words were found in this text.")

    if len(df) > 2:  # several sources: one gauge each
        st.subheader("By source")
        parts = df.iloc[:-1]
        cols = st.columns(min(3, len(parts)))
        for i, (_, row) in enumerate(parts.iterrows()):
            with cols[i % len(cols)]:
                st.plotly_chart(gauge(row["avg_sentiment_matched"], str(row["source"])[:40], gauge_range),
                                width="stretch", key=f"g{i}")

    st.subheader("Detailed table")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("⬇️ Download results (Excel)", to_excel_bytes(df),
                       file_name="sentiment_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
