"""
Local batch run for big URL lists (no 100-URL limit).

    python batch_run.py -l Economic_Lexicon.xlsx -u urls.xlsx -o results.xlsx

urls.xlsx must have the columns: country, date, url.
The output workbook can be loaded in the app ("Load previous results").
"""
import argparse

import sentiment_core as sc
from stakeholders import STAKEHOLDERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--lexicon", required=True)
    ap.add_argument("-u", "--urls", required=True, help="Excel with country, date, url")
    ap.add_argument("-o", "--output", default="results.xlsx")
    ap.add_argument("--context", type=int, default=0, help="neighbouring sentences (0-2)")
    ap.add_argument("--no-negation", action="store_true")
    ap.add_argument("--freq", default="Month", choices=list(sc.FREQS))
    a = ap.parse_args()

    lexicon = sc.load_lexicon(a.lexicon)
    table = sc.load_url_table(a.urls)
    patterns = sc.compile_stakeholder_patterns(STAKEHOLDERS)
    print(f"{len(lexicon)} lexicon tokens, {len(table)} URLs")

    docs, stakes, errors = sc.run_batch(
        table, lexicon, patterns, not a.no_negation, a.context,
        progress=lambda i, n: print(f"  {i}/{n}", end="\r"))
    with open(a.output, "wb") as f:
        f.write(sc.results_to_excel_bytes(docs, stakes, errors, a.freq))
    print(f"\nDone: {len(docs)} analyzed, {len(errors)} failed -> {a.output}")


if __name__ == "__main__":
    main()
