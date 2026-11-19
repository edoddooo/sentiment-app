"""Core logic: lexicon, text extraction, scoring, stakeholders, time series."""

import io
import ipaddress
import re
import socket
import time
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

HEADERS = {"User-Agent": "Mozilla/5.0 (sentiment-analysis app)"}
REQUEST_DELAY = 1                 # seconds between URL requests
MAX_DOWNLOAD_BYTES = 10_000_000   # 10 MB per page / PDF
MAX_REDIRECTS = 5

# ------------------------------------------------------------ polarity classes
# very negative [-1, -0.5] | negative (-0.5, -0.1] | neutral (-0.1, 0.1)
# positive [0.1, 0.6)      | very positive [0.6, 1.0]
VERY_NEG_MAX = -0.5
NEG_MAX = -0.1
POS_MIN = 0.1
VERY_POS_MIN = 0.6

CLASS_ORDER = ["very negative", "negative", "neutral", "positive", "very positive"]
CLASS_COLORS = {
    "very negative": "#c62828",
    "negative": "#e57373",
    "neutral": "#bdbdbd",
    "positive": "#81c784",
    "very positive": "#2e7d32",
}
# (class, lower edge, upper edge) used to draw gauge bands
CLASS_BOUNDS = [
    ("very negative", -1.0, VERY_NEG_MAX),
    ("negative", VERY_NEG_MAX, NEG_MAX),
    ("neutral", NEG_MAX, POS_MIN),
    ("positive", POS_MIN, VERY_POS_MIN),
    ("very positive", VERY_POS_MIN, 1.0),
]


def polarity_class(score) -> str:
    if score is None or pd.isna(score):
        return "no data"
    if score <= VERY_NEG_MAX:
        return "very negative"
    if score <= NEG_MAX:
        return "negative"
    if score < POS_MIN:
        return "neutral"
    if score < VERY_POS_MIN:
        return "positive"
    return "very positive"


# ---------------------------------------------------------------- lexicon
def load_lexicon(source) -> dict:
    """`source` = path or file-like object of an .xlsx with token/sentiment/polarity."""
    df = pd.read_excel(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"token", "sentiment", "polarity"} - set(df.columns)
    if missing:
        raise ValueError(f"Lexicon is missing column(s): {missing}")
    df = df.dropna(subset=["token"])
    df["token"] = df["token"].astype(str).str.strip().str.lower()
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0.0)
    df["polarity"] = pd.to_numeric(df["polarity"], errors="coerce").fillna(0).astype(int)
    df = df.drop_duplicates(subset="token", keep="first")
    return {r.token: (r.sentiment, r.polarity) for r in df.itertuples()}


# ------------------------------------------------------------ text input
def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", str(s).strip(), re.IGNORECASE))


def _check_public_url(url: str) -> None:
    """Reject non-http(s) URLs and hosts resolving to private/local addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("Only http/https URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror:
        raise ValueError("Host could not be resolved")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError("Private or local addresses are not allowed")


def text_from_pdf(file_or_path) -> str:
    reader = PdfReader(file_or_path)
    return " ".join((page.extract_text() or "") for page in reader.pages)


def text_from_url(url: str) -> str:
    """Download a web page (or a PDF behind a URL) and return its text."""
    current = str(url).strip()
    for _ in range(MAX_REDIRECTS + 1):
        _check_public_url(current)
        resp = requests.get(current, headers=HEADERS, timeout=15,
                            allow_redirects=False, stream=True)
        if resp.status_code in (301, 302, 303, 307, 308):
            current = urljoin(current, resp.headers.get("Location", ""))
            resp.close()
            continue
        resp.raise_for_status()
        raw = b""
        for chunk in resp.iter_content(chunk_size=65536):
            raw += chunk
            if len(raw) >= MAX_DOWNLOAD_BYTES:
                break
        ctype = resp.headers.get("Content-Type", "").lower()
        resp.close()
        break
    else:
        raise ValueError("Too many redirects")

    time.sleep(REQUEST_DELAY)
    if "pdf" in ctype or raw[:5] == b"%PDF-":
        return text_from_pdf(io.BytesIO(raw))
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return soup.get_text(separator=" ")


# --------------------------------------------------------------- scoring
NEGATIONS = {"not", "no", "never", "without", "neither", "nor", "hardly",
             "barely", "cannot", "can't", "don't", "doesn't", "didn't",
             "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't"}


def tokenize(text: str) -> list:
    text = text.lower().replace("’", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def lookup(token: str, lexicon: dict):
    if token in lexicon:
        return lexicon[token]
    for suf in ("ing", "ed", "es", "s", "ly"):
        if token.endswith(suf) and len(token) > len(suf) + 2:
            stem = token[: -len(suf)]
            for cand in (stem, stem + "e"):
                if cand in lexicon:
                    return lexicon[cand]
    return None


def _score_raw(tokens, lexicon, handle_negation=True, neg_window=3):
    """Return (sum_sentiment, matched, positive, negative)."""
    s, matched, pos, neg = 0.0, 0, 0, 0
    for i, tok in enumerate(tokens):
        hit = lookup(tok, lexicon)
        if hit is None:
            continue
        sent, pol = hit
        if handle_negation and any(t in NEGATIONS for t in tokens[max(0, i - neg_window):i]):
            sent, pol = -sent, -pol
        s += sent
        matched += 1
        pos += pol > 0
        neg += pol < 0
    return s, matched, pos, neg


@dataclass
class Result:
    source: str
    total_words: int
    matched_words: int
    positive_words: int
    negative_words: int
    neutral_words: int
    sum_sentiment: float
    avg_sentiment_matched: float   # main score: mean sentiment of lexicon words
    avg_sentiment_all: float
    net_polarity: float
    polarity_class: str


def score_tokens(tokens, lexicon, handle_negation=True, source="") -> Result:
    total = len(tokens)
    s, matched, pos, neg = _score_raw(tokens, lexicon, handle_negation)
    avg_m = s / matched if matched else 0.0
    avg_a = s / total if total else 0.0
    net = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
    return Result(source, total, matched, pos, neg, matched - pos - neg,
                  round(s, 4), round(avg_m, 4), round(avg_a, 4), round(net, 4),
                  polarity_class(avg_m) if matched else "no data")


def analyze_texts(sources: dict, lexicon: dict, handle_negation=True) -> pd.DataFrame:
    """sources = {name: text}. Adds an OVERALL row when there are 2+ sources."""
    results, all_tokens = [], []
    for name, text in sources.items():
        tokens = tokenize(text)
        all_tokens.extend(tokens)
        results.append(score_tokens(tokens, lexicon, handle_negation, source=name))
    if len(results) > 1:
        results.append(score_tokens(all_tokens, lexicon, handle_negation,
                                    source="OVERALL (all sources)"))
    return pd.DataFrame([asdict(r) for r in results])


# ---------------------------------------------------------- stakeholders
def compile_stakeholder_patterns(stakeholders: dict) -> dict:
    patterns = {}
    for name, kws in stakeholders.items():
        parts = [re.escape(k.lower()) + r"(?:s|es)?"
                 for k in sorted(set(kws), key=len, reverse=True)]
        patterns[name] = re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)
    return patterns


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


STAKE_COLS = ["stakeholder", "mentions", "matched_words", "positive_words",
              "negative_words", "avg_sentiment", "polarity_class"]


def analyze_stakeholders(text, lexicon, patterns, handle_negation=True, context=0) -> pd.DataFrame:
    """
    Sentence-level stakeholder sentiment. Every sentence that mentions a
    stakeholder keyword is scored (optionally with `context` neighbouring
    sentences on each side) and its score is credited to that stakeholder.
    avg_sentiment is NaN when the stakeholder is mentioned but no lexicon word
    appears in its sentences (no signal, not "zero").
    """
    sentences = split_sentences(text)
    token_sents = [tokenize(s) for s in sentences]
    acc = {n: dict(mentions=0, s=0.0, matched=0, pos=0, neg=0) for n in patterns}

    for i, sent in enumerate(sentences):
        hits = [n for n, p in patterns.items() if p.search(sent)]
        if not hits:
            continue
        lo, hi = max(0, i - context), min(len(sentences), i + context + 1)
        toks = [t for ts in token_sents[lo:hi] for t in ts]
        s, matched, pos, neg = _score_raw(toks, lexicon, handle_negation)
        for n in hits:
            a = acc[n]
            a["mentions"] += 1
            a["s"] += s
            a["matched"] += matched
            a["pos"] += pos
            a["neg"] += neg

    rows = []
    for n, a in acc.items():
        if a["mentions"] == 0:
            continue
        avg = a["s"] / a["matched"] if a["matched"] else np.nan
        rows.append(dict(stakeholder=n, mentions=a["mentions"],
                         matched_words=a["matched"], positive_words=a["pos"],
                         negative_words=a["neg"],
                         avg_sentiment=round(avg, 4) if not np.isnan(avg) else np.nan,
                         polarity_class=polarity_class(avg)))
    return pd.DataFrame(rows, columns=STAKE_COLS)


# -------------------------------------------------------- batch (URL list)
def load_url_table(source, dayfirst=True) -> pd.DataFrame:
    """Excel with columns: country, date, url (country optional)."""
    df = pd.read_excel(source)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"url", "date"} - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing column(s): {missing}. "
                         "Expected: country, date, url")
    if "country" not in df.columns:
        df["country"] = "N/A"
    df = df.dropna(subset=["url"]).copy()
    df["url"] = df["url"].astype(str).str.strip()
    df["country"] = df["country"].fillna("N/A").astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=dayfirst)
    return df[["country", "date", "url"]].reset_index(drop=True)


def run_batch(table, lexicon, patterns, handle_negation=True, context=0,
              fetch=text_from_url, progress=None):
    """Fetch and score every URL. Returns (docs_df, stakeholders_df, errors_df)."""
    docs, stakes, errors = [], [], []
    n = len(table)
    for i, row in enumerate(table.itertuples(index=False), 1):
        try:
            text = fetch(row.url)
            if not text or not text.strip():
                raise ValueError("no text extracted")
            tokens = tokenize(text)
            r = score_tokens(tokens, lexicon, handle_negation)
            docs.append(dict(country=row.country, date=row.date, url=row.url,
                             total_words=r.total_words, matched_words=r.matched_words,
                             positive_words=r.positive_words, negative_words=r.negative_words,
                             avg_sentiment_matched=r.avg_sentiment_matched,
                             polarity_class=r.polarity_class))
            st_df = analyze_stakeholders(text, lexicon, patterns, handle_negation, context)
            for rec in st_df.to_dict("records"):
                stakes.append(dict(country=row.country, date=row.date, url=row.url, **rec))
        except Exception as e:
            errors.append(dict(country=row.country, date=row.date, url=row.url, error=str(e)))
        if progress:
            progress(i, n)
    return pd.DataFrame(docs), pd.DataFrame(stakes), pd.DataFrame(errors)


# --------------------------------------------------------------- time series
FREQS = {"Month": "M", "Quarter": "Q", "Year": "Y"}


def add_period(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    out = df.dropna(subset=["date"]).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["period"] = out["date"].dt.to_period(FREQS[freq]).dt.to_timestamp()
    return out


def ts_overall(docs, freq, by_country=False) -> pd.DataFrame:
    d = add_period(docs[docs["matched_words"] > 0], freq)
    keys = ["period"] + (["country"] if by_country else [])
    out = (d.groupby(keys)
             .agg(avg_sentiment=("avg_sentiment_matched", "mean"), n_docs=("url", "count"))
             .reset_index())
    out["avg_sentiment"] = out["avg_sentiment"].round(4)
    return out.sort_values(keys)


def ts_stakeholder(stakes, freq, by_country=False) -> pd.DataFrame:
    d = add_period(stakes.dropna(subset=["avg_sentiment"]), freq)
    keys = ["period", "stakeholder"] + (["country"] if by_country else [])
    out = (d.groupby(keys)
             .agg(avg_sentiment=("avg_sentiment", "mean"), n_docs=("url", "nunique"),
                  mentions=("mentions", "sum"))
             .reset_index())
    out["avg_sentiment"] = out["avg_sentiment"].round(4)
    return out.sort_values(keys)


def class_share(docs, freq) -> pd.DataFrame:
    d = add_period(docs[docs["matched_words"] > 0], freq)
    ct = pd.crosstab(d["period"], d["polarity_class"], normalize="index") * 100
    ct = ct.reindex(columns=CLASS_ORDER, fill_value=0)
    return ct.reset_index().melt(id_vars="period", var_name="polarity_class", value_name="share")


# -------------------------------------------------------- excel in / out
def results_to_excel_bytes(docs, stakes, errors, freq="Month") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        docs.to_excel(w, sheet_name="documents", index=False)
        stakes.to_excel(w, sheet_name="stakeholders", index=False)
        errors.to_excel(w, sheet_name="errors", index=False)
        if len(docs) and docs["date"].notna().any():
            ts_overall(docs, freq, by_country=True).to_excel(w, sheet_name="timeseries_overall", index=False)
        if len(stakes) and stakes["date"].notna().any():
            ts_stakeholder(stakes, freq, by_country=True).to_excel(w, sheet_name="timeseries_stakeholder", index=False)
    return buf.getvalue()


def read_results_workbook(source):
    sheets = pd.read_excel(source, sheet_name=None)
    if "documents" not in sheets or "stakeholders" not in sheets:
        raise ValueError("Not a results file: sheets 'documents' and 'stakeholders' are required")
    docs, stakes = sheets["documents"], sheets["stakeholders"]
    errors = sheets.get("errors", pd.DataFrame(columns=["country", "date", "url", "error"]))
    for df in (docs, stakes, errors):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return docs, stakes, errors


def url_template_bytes() -> bytes:
    df = pd.DataFrame({"country": ["Italy", "Kenya"],
                       "date": ["2024-03-15", "2024-04-02"],
                       "url": ["https://example.com/article-1", "https://example.com/article-2"]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()
