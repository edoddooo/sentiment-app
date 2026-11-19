"""Streamlit app: lexicon-based sentiment analysis by stakeholder and over time."""

import io
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sentiment_core as sc
from stakeholders import STAKEHOLDERS

LEXICON_FILE = os.path.join(os.path.dirname(__file__), "Economic_Lexicon.xlsx")
MAX_URLS = 10            # URLs typed in the "single analysis" tab
MAX_BATCH_URLS = 100     # URLs in an uploaded Excel list
MAX_PDF_MB = 20
PATTERNS = sc.compile_stakeholder_patterns(STAKEHOLDERS)

st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="wide")
st.title("📊 Economic Sentiment Analysis")
st.caption("Lexicon-based sentiment scoring by stakeholder type and over time.")


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def get_lexicon(path: str) -> dict:
    return sc.load_lexicon(path)


@st.cache_data(show_spinner=False, ttl=86400, max_entries=200)
def fetch_cached(url: str) -> str:
    return sc.text_from_url(url)[:300_000]


def gauge(score: float, title: str, rng: float) -> go.Figure:
    steps = []
    for name, lo, hi in sc.CLASS_BOUNDS:
        lo_c, hi_c = max(lo, -rng), min(hi, rng)
        if lo_c < hi_c:
            steps.append({"range": [lo_c, hi_c], "color": sc.CLASS_COLORS[name]})
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"valueformat": ".3f"},
        title={"text": title, "font": {"size": 16}},
        gauge={"axis": {"range": [-rng, rng]},
               "bar": {"color": "#333333", "thickness": 0.25},
               "steps": steps},
    ))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=70, b=10))
    return fig


def stakeholder_bar(df: pd.DataFrame, rng: float) -> go.Figure:
    d = df.dropna(subset=["avg_sentiment"]).sort_values("avg_sentiment")
    fig = px.bar(d, x="avg_sentiment", y="stakeholder", orientation="h",
                 color="polarity_class", color_discrete_map=sc.CLASS_COLORS,
                 category_orders={"polarity_class": sc.CLASS_ORDER},
                 hover_data=["mentions", "matched_words"])
    fig.update_layout(xaxis=dict(range=[-rng, rng], title="Average sentiment"),
                      yaxis_title=None, legend_title="Polarity",
                      height=max(300, 40 * len(d) + 120), margin=dict(l=10, r=10, t=20, b=10))
    return fig


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sentiment")
    return buf.getvalue()


XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    uploaded_lex = st.file_uploader("Custom lexicon (optional, .xlsx)", type=["xlsx"])
    negation = st.checkbox("Handle negation (e.g. 'not good')", value=True)
    context = st.slider("Stakeholder context (± sentences)", 0, 2, 0,
                        help="0 = only the sentence mentioning the stakeholder is scored. "
                             "1-2 also uses neighbouring sentences.")
    gauge_range = st.slider("Gauge / chart range (±)", 0.1, 1.0, 1.0, 0.05)
    dayfirst = st.checkbox("Text dates are day-first (DD/MM/YYYY)", value=True)
    st.markdown("**Score** = average `sentiment` of the words found in the lexicon.")
    with st.expander("Polarity classes"):
        st.markdown("- very negative: [-1, -0.5]\n- negative: (-0.5, -0.1]\n"
                    "- neutral: (-0.1, 0.1)\n- positive: [0.1, 0.6)\n- very positive: [0.6, 1.0]")

try:
    lexicon = sc.load_lexicon(uploaded_lex) if uploaded_lex is not None else get_lexicon(LEXICON_FILE)
    st.sidebar.success(f"Lexicon loaded: {len(lexicon):,} tokens")
except Exception as e:
    st.error(f"Could not load the lexicon: {e}")
    st.stop()


# =============================================================== TAB 1: single
def render_single_tab():
    t_text, t_url, t_pdf = st.tabs(["✍️ Text", "🔗 URLs", "📄 PDF"])
    with t_text:
        text_input = st.text_area("Paste text", height=180)
    with t_url:
        urls_input = st.text_area(f"One URL per line (max {MAX_URLS})", height=130,
                                  placeholder="https://example.com/article")
    with t_pdf:
        pdf_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    if st.button("Analyze", type="primary", key="analyze_single"):
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
                    sources[url] = fetch_cached(url)
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
            return

        st.session_state["df"] = sc.analyze_texts(sources, lexicon, negation)
        st.session_state["stake"] = sc.analyze_stakeholders(
            "\n".join(sources.values()), lexicon, PATTERNS, negation, context)

    df = st.session_state.get("df")
    if df is None or df.empty:
        return

    overall = df.iloc[-1]
    st.subheader("Overall result")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(gauge(overall["avg_sentiment_matched"], "Overall sentiment", gauge_range),
                        width="stretch")
    with c2:
        st.metric("Polarity", str(overall["polarity_class"]).capitalize())
        st.metric("Words analyzed", f"{int(overall['total_words']):,}")
        st.metric("Words found in lexicon", f"{int(overall['matched_words']):,}")
        st.metric("Positive / Negative words",
                  f"{int(overall['positive_words'])} / {int(overall['negative_words'])}")
        if overall["matched_words"] == 0:
            st.warning("No lexicon words were found in this text.")

    stake = st.session_state.get("stake")
    st.subheader("Sentiment by stakeholder type")
    if stake is None or stake.dropna(subset=["avg_sentiment"]).empty:
        st.info("No stakeholder with sentiment-bearing words was found in this text.")
    else:
        st.plotly_chart(stakeholder_bar(stake, gauge_range), width="stretch")
        st.dataframe(stake, width="stretch", hide_index=True)

    if len(df) > 2:
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
                       file_name="sentiment_results.xlsx", mime=XLSX)


# ========================================================= TAB 2: time series
def render_timeseries_tab():
    mode = st.radio("Data source", ["Analyze a URL list (Excel)", "Load previous results (.xlsx)"],
                    horizontal=True)

    if mode.startswith("Analyze"):
        st.write("Upload an Excel file with the columns **country**, **date**, **url** "
                 f"(max {MAX_BATCH_URLS} URLs per run on the free server; use the local "
                 "`batch_run.py` for bigger lists).")
        st.download_button("Download input template", sc.url_template_bytes(),
                           file_name="urls_template.xlsx", mime=XLSX)
        f = st.file_uploader("URL list", type=["xlsx"], key="urls_xlsx")
        if f is not None:
            try:
                table = sc.load_url_table(f, dayfirst)
            except Exception as e:
                st.error(str(e))
                return
            n_bad = int(table["date"].isna().sum())
            st.write(f"{len(table)} URLs, {table['country'].nunique()} countries.")
            if n_bad:
                st.warning(f"{n_bad} row(s) have an unreadable date: analyzed but excluded from time charts.")
            if len(table) > MAX_BATCH_URLS:
                st.warning(f"Only the first {MAX_BATCH_URLS} rows will be analyzed.")
                table = table.head(MAX_BATCH_URLS)
            if st.button("Run analysis", type="primary", key="run_batch"):
                bar = st.progress(0.0, text="Starting...")
                docs, stakes, errors = sc.run_batch(
                    table, lexicon, PATTERNS, negation, context, fetch=fetch_cached,
                    progress=lambda i, n: bar.progress(i / n, text=f"Processed {i}/{n} URLs"))
                bar.empty()
                st.session_state["batch"] = (docs, stakes, errors)
    else:
        f = st.file_uploader("Results workbook (from this app or batch_run.py)", type=["xlsx"], key="res_xlsx")
        if f is not None:
            try:
                st.session_state["batch"] = sc.read_results_workbook(f)
            except Exception as e:
                st.error(str(e))
                return

    batch = st.session_state.get("batch")
    if batch is None:
        return
    docs, stakes, errors = batch
    if docs.empty:
        st.error("No URL could be analyzed.")
        if len(errors):
            st.dataframe(errors, width="stretch", hide_index=True)
        return

    st.divider()
    st.subheader("Explore results")
    f1, f2, f3 = st.columns(3)
    countries = sorted(docs["country"].astype(str).unique())
    sel_countries = f1.multiselect("Countries", countries, default=countries)
    freq = f2.selectbox("Time aggregation", list(sc.FREQS), index=0)
    groups = sorted(stakes["stakeholder"].unique()) if len(stakes) else []
    sel_groups = f3.multiselect("Stakeholder types", groups, default=groups)

    d = docs[docs["country"].astype(str).isin(sel_countries)]
    s = stakes[stakes["country"].astype(str).isin(sel_countries)] if len(stakes) else stakes
    if len(s):
        s = s[s["stakeholder"].isin(sel_groups)]

    dated = d.dropna(subset=["date"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Documents analyzed", len(d))
    k2.metric("Countries", d["country"].nunique())
    k3.metric("Period", f"{dated['date'].min():%Y-%m} → {dated['date'].max():%Y-%m}" if len(dated) else "n/a")
    k4.metric("URLs failed", len(errors))
    if len(d) - len(dated):
        st.caption(f"{len(d) - len(dated)} document(s) without a valid date are not in the time charts.")
    if dated.empty:
        st.warning("No documents with a valid date for the current selection.")
        return

    # 1. overall trend
    st.markdown("#### Overall sentiment over time")
    split = st.checkbox("One line per country", value=False)
    tso = sc.ts_overall(d, freq, by_country=split)
    if tso.empty:
        st.info("No lexicon words found in the selected documents.")
    else:
        fig = px.line(tso, x="period", y="avg_sentiment", markers=True,
                      color="country" if split else None, hover_data=["n_docs"])
        for _, lo, hi in sc.CLASS_BOUNDS[2:3]:
            fig.add_hrect(y0=lo, y1=hi, fillcolor="#bdbdbd", opacity=0.2, line_width=0)
        fig.update_layout(yaxis=dict(range=[-gauge_range, gauge_range], title="Average sentiment"),
                          xaxis_title=None, height=380, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

    # 2. polarity class shares
    st.markdown("#### Share of documents by polarity class")
    share = sc.class_share(d, freq)
    if not share.empty:
        fig = px.bar(share, x="period", y="share", color="polarity_class",
                     color_discrete_map=sc.CLASS_COLORS,
                     category_orders={"polarity_class": sc.CLASS_ORDER})
        fig.update_layout(yaxis_title="% of documents", xaxis_title=None,
                          legend_title="Polarity", height=360, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

    # 3. stakeholders
    st.markdown("#### Sentiment by stakeholder type over time")
    if len(s) == 0 or s["avg_sentiment"].notna().sum() == 0:
        st.info("No stakeholder sentiment for the current selection.")
    else:
        tss = sc.ts_stakeholder(s, freq)
        fig = px.line(tss, x="period", y="avg_sentiment", color="stakeholder",
                      markers=True, hover_data=["n_docs", "mentions"])
        fig.update_layout(yaxis=dict(range=[-gauge_range, gauge_range], title="Average sentiment"),
                          xaxis_title=None, legend_title=None, height=420, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

        heat = tss.pivot(index="stakeholder", columns="period", values="avg_sentiment")
        heat.columns = [c.strftime("%Y-%m") for c in heat.columns]
        fig = px.imshow(heat, color_continuous_scale="RdYlGn", zmin=-gauge_range, zmax=gauge_range,
                        aspect="auto", labels=dict(color="Sentiment"))
        fig.update_layout(height=max(300, 35 * len(heat) + 120), margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Stakeholder ranking (whole period)")
        rank = (s.dropna(subset=["avg_sentiment"]).groupby("stakeholder")
                  .agg(avg_sentiment=("avg_sentiment", "mean"), mentions=("mentions", "sum"),
                       matched_words=("matched_words", "sum")).reset_index())
        rank["avg_sentiment"] = rank["avg_sentiment"].round(4)
        rank["polarity_class"] = rank["avg_sentiment"].map(sc.polarity_class)
        st.plotly_chart(stakeholder_bar(rank, gauge_range), width="stretch")

    with st.expander("Data tables"):
        st.write("Documents"); st.dataframe(d, width="stretch", hide_index=True)
        st.write("Stakeholders (per document)"); st.dataframe(s, width="stretch", hide_index=True)
    if len(errors):
        with st.expander(f"URLs that could not be analyzed ({len(errors)})"):
            st.dataframe(errors, width="stretch", hide_index=True)

    st.download_button("⬇️ Download full results (Excel)",
                       sc.results_to_excel_bytes(docs, stakes, errors, freq),
                       file_name="sentiment_timeseries_results.xlsx", mime=XLSX)


# --------------------------------------------------------------------- main
tab_single, tab_time = st.tabs(["🔎 Analyze text / URLs / PDF", "📈 Over time & stakeholders"])
with tab_single:
    render_single_tab()
with tab_time:
    render_timeseries_tab()
