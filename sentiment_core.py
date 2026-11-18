"""Core logic: lexicon loading, text extraction (URL / PDF), scoring."""

import ipaddress
import re
import socket
import time
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

HEADERS = {"User-Agent": "Mozilla/5.0 (sentiment-analysis app)"}
REQUEST_DELAY = 1             # seconds between URL requests
MAX_DOWNLOAD_BYTES = 5_000_000  # 5 MB per page
MAX_REDIRECTS = 5


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
    return bool(re.match(r"^https?://", s.strip(), re.IGNORECASE))


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


def text_from_url(url: str) -> str:
    current = url.strip()
    for _ in range(MAX_REDIRECTS + 1):
        _check_public_url(current)
        resp = requests.get(current, headers=HEADERS, timeout=15,
                            allow_redirects=False, stream=True)
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            current = urljoin(current, resp.headers.get("Location", ""))
            continue
        resp.raise_for_status()
        raw = resp.raw.read(MAX_DOWNLOAD_BYTES, decode_content=True)
        break
    else:
        raise ValueError("Too many redirects")

    time.sleep(REQUEST_DELAY)
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def text_from_pdf(file_or_path) -> str:
    reader = PdfReader(file_or_path)
    return " ".join((page.extract_text() or "") for page in reader.pages)


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


@dataclass
class Result:
    source: str
    total_words: int
    matched_words: int
    positive_words: int
    negative_words: int
    neutral_words: int
    sum_sentiment: float
    avg_sentiment_matched: float
    avg_sentiment_all: float
    net_polarity: float
    label: str


def label_from_score(score: float, threshold: float = 0.05) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def score_tokens(tokens, lexicon, handle_negation=True, neg_window=3, source="") -> Result:
    total = len(tokens)
    scores, pols = [], []
    for i, tok in enumerate(tokens):
        hit = lookup(tok, lexicon)
        if hit is None:
            continue
        sent, pol = hit
        if handle_negation and any(t in NEGATIONS for t in tokens[max(0, i - neg_window):i]):
            sent, pol = -sent, -pol
        scores.append(sent)
        pols.append(pol)

    matched = len(scores)
    pos = sum(1 for p in pols if p > 0)
    neg = sum(1 for p in pols if p < 0)
    s = float(sum(scores))
    avg_m = s / matched if matched else 0.0
    avg_a = s / total if total else 0.0
    net = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
    return Result(source, total, matched, pos, neg, matched - pos - neg,
                  round(s, 4), round(avg_m, 4), round(avg_a, 4),
                  round(net, 4), label_from_score(avg_m))


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
