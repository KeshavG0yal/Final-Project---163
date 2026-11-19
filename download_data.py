"""
Keshav Goyal
CSE 163 Final Project

Downloads the data for this project from FRED, Yahoo Finance, and the
SEC EDGAR Company Facts API, cleans it, and saves CSV files in data/:

  - macro.csv:        monthly federal funds rate and CPI (FRED)
  - prices_daily.csv: daily adjusted closing prices (Yahoo Finance)
  - sec_raw.csv:      every net interest income fact the SEC API returned
  - bank_nii.csv:     one cleaned net interest income value per
                      bank per quarter

Run this once with an internet connection. The analysis code only reads
the saved CSV files, so it does not need internet access.
"""
import io
import os
import time

import pandas as pd
import requests

DATA_DIR = "data"
START = "2000-01-01"
END = "2025-12-31"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
SEC_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"
HEADERS = {"User-Agent": "Keshav Goyal keshavg1@uw.edu"}
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
OTHER_TICKERS = ["SPY", "JPM", "BAC", "WFC", "C"]
BANKS = {"JPM": "0000019617", "BAC": "0000070858",
         "WFC": "0000072971", "C": "0000831001"}
# Tags are listed in order of preference. Some banks have used a
# different tag for net interest income in some years.
NII_TAGS = ["InterestIncomeExpenseNet", "InterestIncomeExpenseOperatingNet"]
RAW_COLUMNS = ["bank", "tag", "start", "end", "val", "fy", "fp", "form",
               "filed"]
KEEP_FORMS = ["10-Q", "10-K"]


def clean_fred_csv(text, series_id):
    """
    Takes the text of a FRED CSV download and the series name, and
    returns a DataFrame with a 'date' column and one numeric column named
    after the series in lowercase. FRED marks missing values with '.',
    which become NaN.
    """
    df = pd.read_csv(io.StringIO(text))
    df = df.iloc[:, :2]
    df.columns = ["date", series_id.lower()]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id.lower()] = pd.to_numeric(df[series_id.lower()],
                                          errors="coerce")
    return df


def fetch_fred_series(series_id):
    """
    Downloads one FRED series from its public CSV link and returns it
    cleaned (see clean_fred_csv).
    """
    response = requests.get(FRED_URL.format(series_id), headers=HEADERS,
                            timeout=60)
    response.raise_for_status()
    return clean_fred_csv(response.text, series_id)


def build_macro(fedfunds, cpi):
    """
    Takes the cleaned FEDFUNDS and CPIAUCSL DataFrames and returns one
    monthly DataFrame with columns date, fedfunds, cpiaucsl, rate_change
    (change in the federal funds rate from the previous month, in
    percentage points), and cpi_yoy (percent change in CPI from 12 months
    earlier). Months missing from FRED are added as rows of NaN so that
    the 1-month and 12-month comparisons always use the right months.
    """
    macro = fedfunds.merge(cpi, on="date", how="outer")
    macro = macro.sort_values("date").set_index("date")
    full_range = pd.date_range(macro.index.min(), macro.index.max(),
                               freq="MS")
    macro = macro.reindex(full_range)
    macro.index.name = "date"
    macro["rate_change"] = macro["fedfunds"] - macro["fedfunds"].shift(1)
    macro["cpi_yoy"] = (macro["cpiaucsl"] / macro["cpiaucsl"].shift(12)
                        - 1) * 100
    macro = macro.loc[START:END]
    return macro.reset_index()


def fetch_prices():
    """
    Downloads daily dividend- and split-adjusted closing prices for all
    tickers from Yahoo Finance. Starts in December 1999 so that a return
    can be computed for January 2000.
    """
    import yfinance as yf
    tickers = SECTORS + OTHER_TICKERS
    raw = yf.download(tickers, start="1999-12-01", end="2026-01-01",
                      auto_adjust=True, progress=False)
    prices = raw["Close"][tickers].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "date"
    return prices.reset_index()


def fetch_sec_facts(cik):
    """
    Downloads the Company Facts JSON for one company from the SEC API.
    """
    response = requests.get(SEC_URL.format(cik), headers=HEADERS,
                            timeout=60)
    response.raise_for_status()
    return response.json()


def extract_nii_rows(facts, bank):
    """
    Takes a Company Facts JSON (as a dict) and a bank ticker and returns a
    DataFrame with one row for every net interest income fact found under
    any tag in NII_TAGS, before any cleaning.
    """
    rows = []
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in NII_TAGS:
        units = us_gaap.get(tag, {}).get("units", {}).get("USD", [])
        for fact in units:
            row = {"bank": bank, "tag": tag}
            for key in RAW_COLUMNS[2:]:
                row[key] = fact.get(key)
            rows.append(row)
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def clean_nii(raw):
    """
    Takes the raw SEC rows for any number of banks and returns one net
    interest income value per bank per calendar quarter, with columns
    bank, quarter, quarter_end, nii, source, and tag.

    Steps:
      1. Keep only 10-Q and 10-K filings with a start and end date.
      2. The same period is often reported in several filings (again as
         a comparison number in later reports). Keep one row per period:
         the preferred tag, and the earliest filing.
      3. Keep rows covering about 3 months (quarterly values).
      4. 10-K filings only give a full-year total, so compute
         Q4 = full year - (Q1 + Q2 + Q3) when all three are present.
    """
    df = raw[raw["form"].isin(KEEP_FORMS)].dropna(subset=["start", "end"])
    df = df.copy()
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df["days"] = (df["end"] - df["start"]).dt.days
    df["tag_rank"] = df["tag"].map({t: i for i, t in enumerate(NII_TAGS)})
    df = df.sort_values(["bank", "tag_rank", "filed"])
    df = df.drop_duplicates(["bank", "start", "end"], keep="first")

    quarters = df[(df["days"] >= 80) & (df["days"] <= 100)]
    annual = df[(df["days"] >= 350) & (df["days"] <= 380)]

    rows = []
    for _, q in quarters.iterrows():
        rows.append({"bank": q["bank"], "quarter_end": q["end"],
                     "nii": q["val"], "source": "reported",
                     "tag": q["tag"]})
    for _, year in annual.iterrows():
        same_bank = quarters[quarters["bank"] == year["bank"]]
        in_year = same_bank[(same_bank["start"] >= year["start"])
                            & (same_bank["end"] < year["end"])]
        has_q4 = (same_bank["end"] == year["end"]).any()
        if len(in_year) == 3 and not has_q4:
            rows.append({"bank": year["bank"], "quarter_end": year["end"],
                         "nii": year["val"] - in_year["val"].sum(),
                         "source": "derived_q4", "tag": year["tag"]})

    result = pd.DataFrame(rows, columns=["bank", "quarter_end", "nii",
                                         "source", "tag"])
    result["quarter"] = result["quarter_end"].dt.to_period("Q").astype(str)
    result = result.sort_values(["bank", "quarter_end", "source"])
    result = result.drop_duplicates(["bank", "quarter"], keep="last")
    columns = ["bank", "quarter", "quarter_end", "nii", "source", "tag"]
    return result[columns].reset_index(drop=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading FRED data...")
    fedfunds = fetch_fred_series("FEDFUNDS")
    cpi = fetch_fred_series("CPIAUCSL")
    macro = build_macro(fedfunds, cpi)
    macro.to_csv(os.path.join(DATA_DIR, "macro.csv"), index=False)
    print("  macro.csv:", macro.shape)

    print("Downloading Yahoo Finance prices...")
    prices = fetch_prices()
    prices.to_csv(os.path.join(DATA_DIR, "prices_daily.csv"), index=False)
    print("  prices_daily.csv:", prices.shape)

    print("Downloading SEC data...")
    raw_frames = []
    for bank, cik in BANKS.items():
        facts = fetch_sec_facts(cik)
        rows = extract_nii_rows(facts, bank)
        print("  " + bank + ":", len(rows), "raw rows")
        raw_frames.append(rows)
        time.sleep(0.5)  # stay well under the SEC's rate limit
    sec_raw = pd.concat(raw_frames, ignore_index=True)
    sec_raw.to_csv(os.path.join(DATA_DIR, "sec_raw.csv"), index=False)
    bank_nii = clean_nii(sec_raw)
    bank_nii.to_csv(os.path.join(DATA_DIR, "bank_nii.csv"), index=False)
    print("  sec_raw.csv:", sec_raw.shape)
    print("  bank_nii.csv:", bank_nii.shape)
    print("Done.")


if __name__ == "__main__":
    main()
