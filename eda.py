"""
Keshav Goyal
CSE 163 Final Project

Exploratory data analysis. Reads the CSV files saved by download_data.py,
prints the size, missing data, and summaries of each dataset, checks the
assumptions of the statistical tests planned for RQ1, RQ3, and RQ4, and
saves the EDA plots in results/figures/.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

DATA_DIR = "data"
FIG_DIR = os.path.join("results", "figures")
SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
SECTOR_NAMES = {"XLB": "Materials", "XLE": "Energy", "XLF": "Financials",
                "XLI": "Industrials", "XLK": "Technology",
                "XLP": "Consumer Staples", "XLU": "Utilities",
                "XLV": "Health Care", "XLY": "Consumer Discretionary"}
BANKS = ["JPM", "BAC", "WFC", "C"]
HIKING_CYCLES = [("2015-12-01", "2018-12-31"), ("2022-03-01", "2023-07-31")]
HIGH_INFLATION = 4.0
FIRST_MONTH = "2000-01"
LAST_MONTH = "2025-12"


def load_data(data_dir=DATA_DIR):
    """
    Loads the four CSV files and returns them as a dict of DataFrames with
    the date columns parsed.
    """
    macro = pd.read_csv(os.path.join(data_dir, "macro.csv"),
                        parse_dates=["date"])
    prices = pd.read_csv(os.path.join(data_dir, "prices_daily.csv"),
                         parse_dates=["date"])
    sec_raw = pd.read_csv(os.path.join(data_dir, "sec_raw.csv"))
    bank_nii = pd.read_csv(os.path.join(data_dir, "bank_nii.csv"),
                           parse_dates=["quarter_end"])
    return {"macro": macro, "prices": prices, "sec_raw": sec_raw,
            "bank_nii": bank_nii}


def period_returns(prices, freq):
    """
    Takes daily prices (a 'date' column plus one column per ticker) and a
    pandas period frequency ('M' for monthly, 'Q' for quarterly). Uses the
    last price in each period and returns the percent return from one
    period to the next, indexed by period. The first period is dropped
    because it has no previous price.
    """
    df = prices.set_index("date").sort_index()
    last = df.groupby(df.index.to_period(freq)).last()
    returns = last.pct_change(fill_method=None) * 100
    return returns.iloc[1:]


def monthly_returns(prices):
    """
    Returns monthly percent returns from Jan 2000 through Dec 2025.
    """
    returns = period_returns(prices, "M")
    return returns.loc[FIRST_MONTH:LAST_MONTH]


def label_months(macro):
    """
    Adds a 'month' period column, an 'inflation' column ('high' if
    cpi_yoy is above 4%, otherwise 'low', missing if cpi_yoy is missing),
    and a 'rate_move' column ('hike', 'cut', or 'no change').
    """
    macro = macro.copy()
    macro["month"] = macro["date"].dt.to_period("M")
    macro["inflation"] = None
    macro.loc[macro["cpi_yoy"] > HIGH_INFLATION, "inflation"] = "high"
    macro.loc[macro["cpi_yoy"] <= HIGH_INFLATION, "inflation"] = "low"
    macro["rate_move"] = None
    macro.loc[macro["rate_change"] > 0, "rate_move"] = "hike"
    macro.loc[macro["rate_change"] < 0, "rate_move"] = "cut"
    macro.loc[macro["rate_change"] == 0, "rate_move"] = "no change"
    return macro


def join_returns_macro(returns, macro):
    """
    Joins monthly sector returns with the monthly FRED data on month
    (inner join), returning one row per month.
    """
    labeled = label_months(macro).set_index("month")
    columns = ["fedfunds", "rate_change", "cpi_yoy", "inflation",
               "rate_move"]
    return returns.join(labeled[columns], how="inner")


def join_nii_rates(bank_nii, macro, prices):
    """
    Joins each bank's quarterly net interest income with the average
    federal funds rate in that quarter and the bank's quarterly stock
    return minus SPY's return (excess_return). Adds nii_billions and
    nii_growth (percent change from the bank's previous quarter).
    """
    rates = macro.copy()
    rates["quarter"] = rates["date"].dt.to_period("Q").astype(str)
    avg_rate = rates.groupby("quarter")["fedfunds"].mean()
    avg_rate = avg_rate.rename("avg_fedfunds").reset_index()

    q_returns = period_returns(prices, "Q")
    excess = q_returns[BANKS].sub(q_returns["SPY"], axis=0)
    excess.index = excess.index.astype(str)
    excess = excess.reset_index(names="quarter").melt(
        id_vars="quarter", var_name="bank", value_name="excess_return")

    df = bank_nii.merge(avg_rate, on="quarter", how="inner")
    df = df.merge(excess, on=["quarter", "bank"], how="left")
    df = df.sort_values(["bank", "quarter"]).reset_index(drop=True)
    df["nii_billions"] = df["nii"] / 1e9
    df["nii_growth"] = df.groupby("bank")["nii"].pct_change() * 100
    return df


def overview(df):
    """
    Returns a dict with the number of rows and columns of df and the
    number of missing values in each column that has any.
    """
    missing = df.isna().sum()
    missing = {col: int(n) for col, n in missing.items() if n > 0}
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]),
            "missing": missing}


def seven_number(series):
    """
    Returns the seven-number summary of a numeric Series as a dict:
    mean, std, min, 25%, 50%, 75%, max.
    """
    described = series.describe()
    keys = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    return {key: float(described[key]) for key in keys}


def lag1_autocorr(series):
    """
    Returns the correlation between a series and itself shifted by one
    period. Values near 0 mean neighboring observations are close to
    independent.
    """
    return float(series.dropna().autocorr(lag=1))


def check_rq1(joined):
    """
    Checks the assumptions of Pearson's correlation test for RQ1 for each
    sector: approximate normality (Shapiro-Wilk), outliers (|z| > 3), and
    independence (lag-1 autocorrelation of returns).
    """
    result = {}
    rate = joined["rate_change"].dropna()
    result["rate_change"] = {
        "shapiro_p": float(stats.shapiro(rate).pvalue),
        "share_zero": float((rate == 0).mean()),
        "outliers": int((abs(stats.zscore(rate)) > 3).sum())}
    for sector in SECTORS:
        returns = joined[sector].dropna()
        result[sector] = {
            "n": int(len(returns)),
            "shapiro_p": float(stats.shapiro(returns).pvalue),
            "outliers": int((abs(stats.zscore(returns)) > 3).sum()),
            "autocorr": lag1_autocorr(returns)}
    return result


def check_rq3(joined):
    """
    Checks the assumptions of Welch's t-test for RQ3 for each sector:
    group sizes, approximate normality within each group (Shapiro-Wilk),
    and whether the two groups have equal variances (Levene's test).
    Welch's test does not need equal variances; Levene's test shows
    whether that matters for this data.
    """
    high = joined[joined["inflation"] == "high"]
    low = joined[joined["inflation"] == "low"]
    result = {"n_high": int(len(high)), "n_low": int(len(low))}
    for sector in SECTORS:
        h = high[sector].dropna()
        lo = low[sector].dropna()
        result[sector] = {
            "mean_high": float(h.mean()), "mean_low": float(lo.mean()),
            "std_high": float(h.std()), "std_low": float(lo.std()),
            "shapiro_p_high": float(stats.shapiro(h).pvalue),
            "shapiro_p_low": float(stats.shapiro(lo).pvalue),
            "levene_p": float(stats.levene(h, lo).pvalue)}
    return result


def check_rq4(nii):
    """
    Checks the assumptions of Pearson's correlation test for RQ4 for each
    bank: number of quarters, normality of net interest income and of its
    quarterly growth, and lag-1 autocorrelation (a value near 1 means the
    series trends over time instead of moving independently).
    """
    result = {}
    for bank in BANKS:
        df = nii[nii["bank"] == bank]
        growth = df["nii_growth"].dropna()
        result[bank] = {
            "n": int(len(df)),
            "first": str(df["quarter"].min()),
            "last": str(df["quarter"].max()),
            "shapiro_p_nii": float(stats.shapiro(df["nii"]).pvalue),
            "shapiro_p_growth": float(stats.shapiro(growth).pvalue),
            "autocorr_nii": lag1_autocorr(df["nii"]),
            "autocorr_growth": lag1_autocorr(growth),
            "r_nii_rate": float(df["nii"].corr(df["avg_fedfunds"]))}
    return result


def plot_rates_inflation(macro, path):
    """
    Line chart of the federal funds rate and CPI inflation over time,
    with the two rate-hiking periods shaded.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(macro["date"], macro["fedfunds"], label="Federal funds rate")
    ax.plot(macro["date"], macro["cpi_yoy"],
            label="CPI inflation (year over year)")
    ax.axhline(HIGH_INFLATION, color="gray", linestyle="--", linewidth=1,
               label="4% high-inflation cutoff")
    for i, (start, end) in enumerate(HIKING_CYCLES):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="orange",
                   alpha=0.2, label="Hiking period" if i == 0 else None)
    ax.set_title("Federal Funds Rate and CPI Inflation, 2000-2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_rate_moves(macro, path):
    """
    Bar chart of how many months the federal funds rate went up, went
    down, or did not change.
    """
    counts = label_months(macro)["rate_move"].value_counts()
    counts = counts.reindex(["cut", "no change", "hike"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4c72b0", "gray", "#dd8452"]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Monthly Change in the Federal Funds Rate, 2000-2025")
    ax.set_xlabel("Direction of change from previous month")
    ax.set_ylabel("Number of months")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sector_growth(returns, path):
    """
    Line chart of the growth of $1 invested in each sector ETF and SPY
    at the start of 2000, on a log scale.
    """
    growth = (1 + returns[SECTORS + ["SPY"]] / 100).cumprod()
    growth.index = growth.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(9, 5))
    for ticker in SECTORS:
        ax.plot(growth.index, growth[ticker], linewidth=1.2,
                label=ticker + " (" + SECTOR_NAMES[ticker] + ")")
    ax.plot(growth.index, growth["SPY"], color="black", linewidth=2.2,
            label="SPY (whole market)")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 Invested in Jan 2000, by Sector")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value of $1 (log scale)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_return_boxes(returns, path):
    """
    Box plot of monthly returns for each sector ETF.
    """
    long = returns[SECTORS].melt(var_name="sector", value_name="ret")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=long, x="sector", y="ret", ax=ax, color="#8fb3d9")
    ax.set_title("Distribution of Monthly Returns by Sector, 2000-2025")
    ax.set_xlabel("Sector ETF")
    ax.set_ylabel("Monthly return (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bank_nii(bank_nii, path):
    """
    Line chart of each bank's quarterly net interest income, with the
    Q4 values computed from 10-K totals marked separately.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for bank in BANKS:
        df = bank_nii[bank_nii["bank"] == bank]
        line = ax.plot(df["quarter_end"], df["nii"] / 1e9, label=bank)
        derived = df[df["source"] == "derived_q4"]
        ax.scatter(derived["quarter_end"], derived["nii"] / 1e9, s=12,
                   color=line[0].get_color())
    ax.scatter([], [], color="gray", s=12,
               label="Q4 computed from 10-K total")
    ax.set_title("Quarterly Net Interest Income of Four Large Banks")
    ax.set_xlabel("Quarter end")
    ax.set_ylabel("Net interest income ($ billions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sec_cleaning(sec_raw, bank_nii, path):
    """
    Grouped bar chart comparing the number of raw SEC rows for each bank
    with the number of cleaned quarterly values that were kept.
    """
    raw_counts = sec_raw["bank"].value_counts().reindex(BANKS).fillna(0)
    kept = bank_nii["bank"].value_counts().reindex(BANKS).fillna(0)
    df = pd.DataFrame({"Raw rows from SEC API": raw_counts,
                       "Cleaned quarterly values": kept})
    fig, ax = plt.subplots(figsize=(7, 4))
    df.plot.bar(ax=ax, rot=0)
    ax.set_title("SEC Net Interest Income Rows Before and After Cleaning")
    ax.set_xlabel("Bank")
    ax.set_ylabel("Number of rows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_inflation_boxes(joined, path):
    """
    Box plot of monthly sector returns, split into high-inflation and
    low-inflation months.
    """
    df = joined.dropna(subset=["inflation"])
    long = df[SECTORS + ["inflation"]].melt(
        id_vars="inflation", var_name="sector", value_name="ret")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.boxplot(data=long, x="sector", y="ret", hue="inflation",
                hue_order=["low", "high"], ax=ax)
    ax.set_title("Monthly Sector Returns in High- vs. Low-Inflation Months")
    ax.set_xlabel("Sector ETF")
    ax.set_ylabel("Monthly return (%)")
    ax.legend(title="Inflation (cutoff 4%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nii_vs_rate(nii, path):
    """
    Scatter plot of each bank's quarterly net interest income against the
    average federal funds rate in that quarter.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(data=nii, x="avg_fedfunds", y="nii_billions",
                    hue="bank", hue_order=BANKS, ax=ax)
    ax.set_title("Net Interest Income vs. Federal Funds Rate")
    ax.set_xlabel("Average federal funds rate in quarter (%)")
    ax.set_ylabel("Net interest income ($ billions)")
    ax.legend(title="Bank")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_eda(data_dir=DATA_DIR, fig_dir=FIG_DIR):
    """
    Runs the whole EDA: loads the data, computes summaries and
    assumption checks, saves the figures, and returns all the results in
    a dict.
    """
    os.makedirs(fig_dir, exist_ok=True)
    data = load_data(data_dir)
    macro = data["macro"]
    prices = data["prices"]
    sec_raw = data["sec_raw"]
    bank_nii = data["bank_nii"]

    returns = monthly_returns(prices)
    joined = join_returns_macro(returns, macro)
    nii = join_nii_rates(bank_nii, macro, prices)
    labeled = label_months(macro)

    r = {"overview": {"macro": overview(macro),
                      "prices": overview(prices),
                      "returns": overview(returns),
                      "sec_raw": overview(sec_raw),
                      "bank_nii": overview(bank_nii),
                      "joined": overview(joined),
                      "nii_joined": overview(nii)}}
    r["macro_missing_dates"] = [
        str(d.date()) for d in
        macro.loc[macro["cpiaucsl"].isna() | macro["fedfunds"].isna(),
                  "date"]]
    r["price_first_date"] = str(prices["date"].min().date())
    r["price_last_date"] = str(prices["date"].max().date())
    r["summary_macro"] = {col: seven_number(macro[col]) for col in
                          ["fedfunds", "rate_change", "cpi_yoy"]}
    r["summary_returns"] = {t: seven_number(returns[t]) for t in
                            SECTORS + ["SPY", "JPM", "BAC", "WFC", "C"]}
    r["summary_nii"] = {b: seven_number(
        bank_nii.loc[bank_nii["bank"] == b, "nii"] / 1e9) for b in BANKS}
    r["counts_inflation"] = {
        k: int(v) for k, v in labeled["inflation"].value_counts().items()}
    r["counts_rate_move"] = {
        k: int(v) for k, v in labeled["rate_move"].value_counts().items()}
    r["counts_source"] = {
        k: int(v) for k, v in bank_nii["source"].value_counts().items()}
    r["counts_form"] = {
        k: int(v) for k, v in sec_raw["form"].value_counts().items()}
    r["counts_tag"] = {
        k: int(v) for k, v in bank_nii["tag"].value_counts().items()}
    r["raw_rows_by_bank"] = {
        k: int(v) for k, v in sec_raw["bank"].value_counts().items()}
    r["kept_rows_by_bank"] = {
        k: int(v) for k, v in bank_nii["bank"].value_counts().items()}
    growth = (1 + returns[SECTORS + ["SPY"]] / 100).prod()
    r["growth_of_1"] = {k: float(v) for k, v in growth.items()}
    yearly = bank_nii.assign(year=bank_nii["quarter_end"].dt.year)
    yearly = yearly.groupby(["bank", "year"])["nii"].mean() / 1e9
    r["nii_2021_2023"] = {
        b: [float(yearly.get((b, 2021), float("nan"))),
            float(yearly.get((b, 2023), float("nan")))] for b in BANKS}
    r["rq1"] = check_rq1(joined)
    r["rq3"] = check_rq3(joined)
    r["rq4"] = check_rq4(nii)

    plot_rates_inflation(macro, os.path.join(fig_dir, "fig1.png"))
    plot_rate_moves(macro, os.path.join(fig_dir, "fig2.png"))
    plot_sector_growth(returns, os.path.join(fig_dir, "fig3.png"))
    plot_return_boxes(returns, os.path.join(fig_dir, "fig4.png"))
    plot_bank_nii(bank_nii, os.path.join(fig_dir, "fig5.png"))
    plot_sec_cleaning(sec_raw, bank_nii, os.path.join(fig_dir, "fig6.png"))
    plot_inflation_boxes(joined, os.path.join(fig_dir, "fig7.png"))
    plot_nii_vs_rate(nii, os.path.join(fig_dir, "fig8.png"))
    return r


def print_results(r):
    """
    Prints the EDA results in a readable form.
    """
    print("=== Size and missing data ===")
    for name, info in r["overview"].items():
        print(name, info)
    print("Months with missing FRED values:", r["macro_missing_dates"])
    for key in ["summary_macro", "summary_returns", "summary_nii"]:
        print("===", key, "===")
        print(pd.DataFrame(r[key]).T.round(3))
    for key in ["counts_inflation", "counts_rate_move", "counts_source",
                "counts_form", "counts_tag"]:
        print("===", key, "===", r[key])
    for key in ["rq1", "rq3", "rq4"]:
        print("=== Assumption checks:", key, "===")
        for name, value in r[key].items():
            print(" ", name, value)


def main():
    plt.switch_backend("Agg")
    results = run_eda()
    print_results(results)
    print("Figures saved in", FIG_DIR)


if __name__ == "__main__":
    main()
