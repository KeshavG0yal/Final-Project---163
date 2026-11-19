"""
Keshav Goyal
CSE 163 Final Project

Builds the EDA report (eda.pdf) from the results of eda.py, so every
number and figure in the report comes straight from the data.

Usage: python build_report.py <GitHub repository URL>
"""
import sys

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, ListFlowable,
                                ListItem, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

import eda

TITLE = ("How Interest Rates and Inflation Relate to Sector Stock "
         "Returns and Bank Earnings, 2000&ndash;2025")
SS = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=SS["Normal"], fontSize=11,
                      leading=15, spaceAfter=6)
H1 = ParagraphStyle("h1", parent=SS["Heading1"], fontSize=15,
                    spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=SS["Heading2"], fontSize=12.5,
                    spaceBefore=8, spaceAfter=4)
CAPTION = ParagraphStyle("cap", parent=BODY, fontSize=11, leading=14,
                         fontName="Helvetica-Oblique", spaceAfter=10)


def code(text):
    """
    Returns text formatted in a monospace font.
    """
    return "<font face='Courier'>" + text + "</font>"


def num(x, digits=2):
    """
    Formats a number with the given number of decimal places.
    """
    return ("{:,." + str(digits) + "f}").format(x)


def pval(p):
    """
    Formats a p-value, showing very small values as '< 0.001'.
    """
    return "&lt; 0.001" if p < 0.001 else num(p, 3)


class Report:
    """
    Collects the pieces of the report and writes them to a PDF.
    """

    def __init__(self):
        self._story = []
        self._fig = 0

    def p(self, text, style=BODY):
        self._story.append(Paragraph(text, style))

    def h1(self, text):
        self.p(text, H1)

    def h2(self, text):
        self.p(text, H2)

    def bullets(self, items, numbered=False):
        flow = [ListItem(Paragraph(i, BODY), leftIndent=16) for i in items]
        kind = "1" if numbered else "bullet"
        self._story.append(ListFlowable(flow, bulletType=kind,
                                        leftIndent=16))

    def table(self, header, rows, widths=None):
        data = [header] + rows
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
        self._story.append(t)
        self._story.append(Spacer(1, 8))

    def figure(self, path, caption, paragraph, width=4.4):
        """
        Adds a numbered figure, its one-sentence caption, and the
        paragraph explaining it.
        """
        self._fig += 1
        px_w, px_h = ImageReader(path).getSize()
        img_w = width * inch
        img = Image(path, width=img_w, height=img_w * px_h / px_w)
        label = "<b>Figure " + str(self._fig) + ".</b> " + caption
        self._story.append(KeepTogether([img, Paragraph(label, CAPTION)]))
        self.p(paragraph)

    def build(self, path):
        def number_page(canvas, doc):
            canvas.setFont("Helvetica", 10)
            canvas.drawCentredString(letter[0] / 2, 0.5 * inch,
                                     "Page " + str(doc.page))
        doc = SimpleDocTemplate(path, pagesize=letter,
                                leftMargin=0.8 * inch,
                                rightMargin=0.8 * inch,
                                topMargin=0.7 * inch,
                                bottomMargin=0.8 * inch)
        doc.build(self._story, onFirstPage=number_page,
                  onLaterPages=number_page)


def summary_rows(summary, names=None):
    """
    Turns a dict of seven-number summaries into table rows.
    """
    keys = ["mean", "std", "min", "25%", "50%", "75%", "max"]
    rows = []
    for name, s in summary.items():
        label = names.get(name, name) if names else name
        rows.append([label] + [num(s[k]) for k in keys])
    return rows


SUMMARY_HEADER = ["Variable", "Mean", "Std", "Min", "Q1", "Median", "Q3",
                  "Max"]


def missing_text(info, name):
    """
    Describes the missing values in one dataset overview.
    """
    if not info["missing"]:
        return ("Calling " + code("df.isna().sum()") + " on " + name +
                " returns 0 for every column, so it has no missing values.")
    parts = [code(c) + " (" + str(n) + " of " + str(info["rows"]) +
             " rows, " + num(100 * n / info["rows"], 1) + "%)"
             for c, n in info["missing"].items()]
    return ("Calling " + code("df.isna().sum()") + " on " + name +
            " shows missing values in: " + ", ".join(parts) + ".")


def write_intro(rep, r, url):
    rep.p(TITLE, SS["Title"])
    rep.p("Keshav Goyal", ParagraphStyle("a", parent=BODY, alignment=1))
    rep.p("GitHub repository: <link href='" + url + "' color='blue'>" + url +
          "</link>", ParagraphStyle("g", parent=BODY, alignment=1))

    first = min(v["first"] for v in r["rq4"].values())
    last = max(v["last"] for v in r["rq4"].values())
    rep.h1("Summary of Research Questions")
    rep.bullets([
        "<b>Which stock market sectors are most sensitive to interest rate "
        "changes?</b> For the nine S&amp;P 500 sector ETFs, I will find the "
        "correlation between each sector's monthly returns and monthly "
        "changes in the federal funds rate (Jan 2000&ndash;Dec 2025), then "
        "rank the sectors from most to least positively correlated.",
        "<b>Did sectors behave the same way in the 2015&ndash;2018 and "
        "2022&ndash;2023 rate-hiking periods?</b> I will compare each "
        "sector's total return to the overall market (SPY) in both periods "
        "and find which sectors switched from beating the market to "
        "trailing it.",
        "<b>Are sector returns different in high-inflation months?</b> For "
        "each sector, I will test whether the average monthly return is "
        "significantly different in months when yearly CPI inflation is "
        "above 4% compared to months when it is 4% or lower.",
        "<b>Do big U.S. banks earn more, and do their stocks do better, when "
        "interest rates rise?</b> For JPMorgan Chase, Bank of America, Wells "
        "Fargo, and Citigroup, I will measure the correlation between the "
        "net interest income they report to the SEC each quarter and the "
        "average federal funds rate, and check whether quarters where this "
        "income grew line up with the stock beating the market. "
        "(Updated: the SEC's machine-readable data only covers " + first +
        " to " + last + ", so RQ4 uses that period.)"], numbered=True)

    rep.h1("Motivation")
    rep.p("When the Federal Reserve changes interest rates, it makes "
          "headlines. Financial commentators often say that some sectors, "
          "like utilities, real estate, and technology, get hurt by higher "
          "rates, while others, like banks, benefit. People use claims like "
          "these to decide how to invest their retirement savings, but the "
          "claims are often made without much evidence or are based on only "
          "one recent example. The 2022&ndash;2023 rate hikes were the "
          "fastest in about forty years, which makes them a good chance to "
          "test whether the usual advice held up compared with earlier "
          "periods. The answers can help regular investors understand what "
          "a rate change does and does not mean for their money, and test "
          "whether the idea that \"banks profit from higher rates\" shows up "
          "in the banks' own official filings, not just in their stock "
          "prices.")

    rep.h1("Dataset")
    rep.p("All three sources are free and public. " + code("download_data.py")
          + " downloads and cleans them and saves CSV files in the " +
          code("data/") + " folder of the repository, so the analysis can "
          "be rerun without internet access. To download the data again, "
          "run " + code("python download_data.py") + ".")
    rep.bullets([
        "<b>FRED (Federal Reserve Bank of St. Louis).</b> Effective Federal "
        "Funds Rate, monthly (" + code("FEDFUNDS") + "): "
        "https://fred.stlouisfed.org/series/FEDFUNDS, and Consumer Price "
        "Index for All Urban Consumers (" + code("CPIAUCSL") + "): "
        "https://fred.stlouisfed.org/series/CPIAUCSL. Updated: the code "
        "reads FRED's public CSV links "
        "(https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS and "
        "?id=CPIAUCSL), which do not need an API key.",
        "<b>Daily ETF and stock prices from Yahoo Finance</b>, through the " +
        code("yfinance") + " library (https://pypi.org/project/yfinance/). "
        "Tickers: XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY (sector ETFs), "
        "SPY (the whole market), and JPM, BAC, WFC, C. Example page: "
        "https://finance.yahoo.com/quote/XLK/history/.",
        "<b>SEC EDGAR Company Facts API</b> "
        "(https://www.sec.gov/search-filings/edgar-application-programming-"
        "interfaces), for example "
        "https://data.sec.gov/api/xbrl/companyfacts/CIK0000019617.json for "
        "JPMorgan Chase. I use net interest income (" +
        code("InterestIncomeExpenseNet") + ") from 10-Q and 10-K filings. "
        "Company IDs: JPM 0000019617, BAC 0000070858, WFC 0000072971, "
        "C 0000831001."])


def write_method(rep):
    rep.h1("Method")
    rep.p("<b>Data pipeline (" + code("download_data.py") + ").</b> (1) "
          "Download " + code("FEDFUNDS") + " and " + code("CPIAUCSL") +
          " from FRED, turn \".\" into missing values, and add any month "
          "FRED skipped as a missing row so that comparisons always use the "
          "right months. Compute " + code("rate_change") + " (this month's "
          "rate minus last month's) and " + code("cpi_yoy") + " (percent "
          "change in CPI from 12 months earlier). (2) Download daily "
          "adjusted closing prices for all 14 tickers with " +
          code("yfinance") + ". (3) Download each bank's Company Facts JSON "
          "from the SEC, pull out every net interest income fact, keep 10-Q "
          "and 10-K rows, remove repeated reports of the same period "
          "(keeping the first one filed), keep values that cover about "
          "three months, and compute Q4 = full-year total &minus; (Q1 + Q2 "
          "+ Q3).")
    rep.p("<b>EDA (" + code("eda.py") + ").</b> For each dataset I counted "
          "rows, columns, and missing values with " + code("shape") +
          " and " + code("isna().sum()") + ", computed seven-number "
          "summaries with " + code("describe()") + ", counted the values of "
          "categorical columns with " + code("value_counts()") + ", and "
          "made two plots. I turned daily prices into monthly returns using "
          "the last price of each month, joined them with the FRED data by "
          "month, and joined each bank's quarterly income with the average "
          "federal funds rate in that quarter. On these joined tables I "
          "checked the assumptions of the tests planned for RQ1, RQ3, and "
          "RQ4.")
    rep.p("<b>Plan for the research questions (unchanged from the "
          "proposal).</b> RQ1: correlation of each sector's monthly return "
          "with " + code("rate_change") + ", with p-values and bootstrap "
          "confidence intervals. RQ2: each sector's total return minus SPY's "
          "in the two hiking periods. RQ3: Welch's t-test and a permutation "
          "test comparing high- and low-inflation months. RQ4: correlation "
          "of net interest income with the average federal funds rate, and "
          "of income growth with the stock's return minus SPY's. A "
          "Bonferroni correction (p &lt; 0.05/9) is used for the nine "
          "sectors.")


def write_fred(rep, r):
    o = r["overview"]["macro"]
    s = r["summary_macro"]
    inf = r["counts_inflation"]
    moves = r["counts_rate_move"]
    rep.h1("EDA Results")
    rep.h2("Dataset 1: FRED interest rate and inflation data "
           "(" + code("macro.csv") + ")")
    rep.p("<b>Size.</b> " + str(o["rows"]) + " rows and " + str(o["cols"]) +
          " columns. Each row is one month from January 2000 to December "
          "2025. The columns are " + code("date") + " (first day of the "
          "month), " + code("fedfunds") + " (average federal funds rate, "
          "%), " + code("cpiaucsl") + " (CPI index level), " +
          code("rate_change") + " (change in " + code("fedfunds") +
          " from the previous month, percentage points), and " +
          code("cpi_yoy") + " (inflation compared to 12 months earlier, %).")
    text = missing_text(o, code("macro.csv"))
    if r["macro_missing_dates"]:
        text += (" The missing months are " +
                 ", ".join(r["macro_missing_dates"]) + ". A missing CPI "
                 "value also makes " + code("cpi_yoy") + " missing for that "
                 "month. My plan is to leave these months out of any test "
                 "that needs the missing value instead of filling them in, "
                 "since it is a very small share of the 312 months and "
                 "guessing a CPI value could bias the inflation labels.")
    rep.p("<b>Missing data.</b> " + text)
    rep.p("<b>Variables of interest.</b> " + code("rate_change") + " is the "
          "x variable for RQ1 and is used to find the hiking periods in "
          "RQ2. " + code("cpi_yoy") + " decides which months count as high "
          "inflation in RQ3. " + code("fedfunds") + " is averaged by quarter "
          "for RQ4. I use the change in the rate for RQ1 because stock "
          "prices react to news, so a change should matter more than the "
          "level.")
    rep.table(SUMMARY_HEADER, summary_rows(s), [1.1 * inch] +
              [0.72 * inch] * 7)
    rep.p("<b>Categorical variables</b> (made from the columns above): "
          "inflation label &mdash; high (above 4%): " +
          str(inf.get("high", 0)) + " months, low: " +
          str(inf.get("low", 0)) + " months; direction of rate change "
          "&mdash; hike: " + str(moves.get("hike", 0)) + ", cut: " +
          str(moves.get("cut", 0)) + ", no change: " +
          str(moves.get("no change", 0)) + " months.")
    rep.figure(
        "results/figures/fig1.png",
        "Both shaded hiking periods show the federal funds rate climbing "
        "steadily, and inflation was above the 4% cutoff in only " +
        str(inf.get("high", 0)) + " of the months.",
        "This line chart shows " + code("fedfunds") + " and " +
        code("cpi_yoy") + " over time, with the 4% cutoff from RQ3 as a "
        "dashed line and the two hiking periods from RQ2 shaded. I chose a "
        "line chart because both are time series and I wanted to see how "
        "they move together. The takeaway is that the two hiking periods I "
        "picked match the data, and that high-inflation months are rare, so "
        "the high-inflation group in RQ3 will be small. The rate's peak was "
        + num(s["fedfunds"]["max"]) + "% and inflation's peak was " +
        num(s["cpi_yoy"]["max"]) + "%.")
    zero = 100 * moves.get("no change", 0) / max(sum(moves.values()), 1)
    rep.figure(
        "results/figures/fig2.png",
        "The federal funds rate did not change in " + num(zero, 0) +
        "% of months, so " + code("rate_change") + " is mostly zeros.",
        "This bar chart counts months by the direction of " +
        code("rate_change") + ". A bar chart fits because the direction is "
        "a category. The reader should notice that most months have no "
        "change, so " + code("rate_change") + " is far from normally "
        "distributed. This matters for RQ1, because a correlation driven "
        "by a few large moves may not be reliable, which is why I plan to "
        "use bootstrap confidence intervals (see Result Validity).",
        width=3.4)


def write_prices(rep, r):
    o = r["overview"]["prices"]
    ro = r["overview"]["returns"]
    rep.h2("Dataset 2: Yahoo Finance prices (" + code("prices_daily.csv") +
           ")")
    rep.p("<b>Size.</b> " + str(o["rows"]) + " rows and " + str(o["cols"]) +
          " columns. Each row is one trading day from " +
          r["price_first_date"] + " to " + r["price_last_date"] + ". The "
          "columns are " + code("date") + " and one column per ticker "
          "holding that day's closing price, adjusted for dividends and "
          "splits. After turning prices into monthly returns, the table has "
          + str(ro["rows"]) + " rows (months) and " + str(ro["cols"]) +
          " columns (tickers).")
    text = missing_text(o, code("prices_daily.csv"))
    if o["missing"]:
        text += (" These are days with no price for that ticker. Since I "
                 "only use the last available price of each month, a "
                 "missing day does not affect the monthly returns. ")
    text += " " + missing_text(ro, "the monthly returns table")
    rep.p("<b>Missing data.</b> " + text)
    rep.p("<b>Variables of interest.</b> The monthly return of each sector "
          "ETF (XLB&ndash;XLY) is the y variable for RQ1, RQ2, and RQ3. "
          "SPY's return is the market benchmark for RQ2 and RQ4, and the "
          "returns of JPM, BAC, WFC, and C are used in RQ4. Summary of "
          "monthly returns (%):")
    rep.table(["Ticker", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"],
              summary_rows(r["summary_returns"]),
              [0.8 * inch] + [0.75 * inch] * 7)
    growth = {k: v for k, v in r["growth_of_1"].items() if k != "SPY"}
    best = max(growth, key=growth.get)
    worst = min(growth, key=growth.get)
    rep.figure(
        "results/figures/fig3.png",
        "From 2000 to 2025, $1 in " + best + " (" + eda.SECTOR_NAMES[best] +
        ") grew to $" + num(growth[best]) + ", while $1 in " + worst +
        " (" + eda.SECTOR_NAMES[worst] + ") grew to $" + num(growth[worst]) +
        " (SPY: $" + num(r["growth_of_1"]["SPY"]) + ").",
        "This line chart shows the value of $1 invested in each sector "
        "ETF and in SPY at the start of 2000, built by compounding the "
        "monthly returns. I used a log scale so that equal percentage "
        "changes look the same size at any point in time. The takeaway is "
        "that sectors ended up in very different places, and that they "
        "split apart most during big events like 2008 and 2020. That is "
        "why RQ2 compares each sector to SPY instead of using raw returns.")
    std = {k: r["summary_returns"][k]["std"] for k in eda.SECTORS}
    most = max(std, key=std.get)
    least = min(std, key=std.get)
    rep.figure(
        "results/figures/fig4.png",
        most + " (" + eda.SECTOR_NAMES[most] + ") had the widest spread of "
        "monthly returns (standard deviation " + num(std[most]) + "%) and " +
        least + " (" + eda.SECTOR_NAMES[least] + ") the narrowest (" +
        num(std[least]) + "%).",
        "This box plot shows the distribution of monthly returns for each "
        "sector. I chose a box plot to compare the center, spread, and "
        "outliers of nine distributions side by side. The reader should "
        "see that every sector has outliers far from the box, so returns "
        "have heavier tails than a normal distribution. Sectors also "
        "differ a lot in spread, which is one reason I use Welch's t-test "
        "(it does not assume equal variances) in RQ3.")


def write_sec(rep, r):
    o_raw = r["overview"]["sec_raw"]
    o = r["overview"]["bank_nii"]
    raw_total = sum(r["raw_rows_by_bank"].values())
    kept_total = sum(r["kept_rows_by_bank"].values())
    src = r["counts_source"]
    rep.h2("Dataset 3: SEC net interest income (" + code("sec_raw.csv") +
           " and " + code("bank_nii.csv") + ")")
    rep.p("<b>Size.</b> The raw table has " + str(o_raw["rows"]) + " rows "
          "and " + str(o_raw["cols"]) + " columns. Each row is one net "
          "interest income number found in one filing. The columns are " +
          code("bank") + ", " + code("tag") + " (the label the bank used), "
          + code("start") + " and " + code("end") + " (the period covered), "
          + code("val") + " (dollars), " + code("fy") + " and " +
          code("fp") + " (fiscal year and period of the filing), " +
          code("form") + " (filing type), and " + code("filed") + " (filing "
          "date). After cleaning, " + code("bank_nii.csv") + " has " +
          str(o["rows"]) + " rows and " + str(o["cols"]) + " columns: one "
          "row per bank per quarter, with columns " + code("bank") + ", " +
          code("quarter") + ", " + code("quarter_end") + ", " +
          code("nii") + " (dollars), " + code("source") + " (reported or "
          "derived Q4), and " + code("tag") + ".")
    rep.p("<b>Missing data.</b> " + missing_text(o_raw, code("sec_raw.csv"))
          + " " + missing_text(o, code("bank_nii.csv")) + " The bigger "
          "issue is that Q4 is never reported on its own. I fill it in by "
          "computing it from the 10-K full-year total (" +
          str(src.get("derived_q4", 0)) + " of " + str(o["rows"]) +
          " quarters), not by guessing.")
    rep.p("<b>Variables of interest.</b> " + code("nii") + " is the main "
          "variable for RQ4. " + code("source") + " shows which values I "
          "computed. " + code("quarter") + " is the key for joining with the "
          "FRED and price data. Categorical counts: form in the raw data "
          "&mdash; " + ", ".join(k + ": " + str(v) for k, v in
                                 r["counts_form"].items()) +
          "; source in the cleaned data &mdash; " +
          ", ".join(k + ": " + str(v) for k, v in src.items()) +
          "; tag in the cleaned data &mdash; " +
          ", ".join(k + ": " + str(v) for k, v in r["counts_tag"].items()) +
          ". Summary of quarterly net interest income ($ billions):")
    rep.table(["Bank", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"],
              summary_rows(r["summary_nii"]),
              [0.8 * inch] + [0.75 * inch] * 7)
    rose = [b for b, (a, c) in r["nii_2021_2023"].items() if c > a]
    rep.figure(
        "results/figures/fig5.png",
        "Average quarterly net interest income was higher in 2023 than in "
        "2021 for " + str(len(rose)) + " of the 4 banks, the period when "
        "the Fed raised rates fastest.",
        "This line chart shows each bank's quarterly net interest income, "
        "with dots marking the Q4 values computed from the 10-K totals. I "
        "chose a line chart to see how income changes over time and to "
        "check that the computed Q4 values fit in smoothly with the "
        "reported quarters (a computed value far off the line would point "
        "to a cleaning bug). The takeaway is that income clearly moves with "
        "the rate cycle, which RQ4 will test formally.")
    rep.figure(
        "results/figures/fig6.png",
        "Cleaning reduced " + str(raw_total) + " raw SEC rows to " +
        str(kept_total) + " quarterly values, because most raw rows repeat "
        "a period that was already reported.",
        "This bar chart compares, for each bank, the number of raw rows "
        "from the SEC API with the number of cleaned quarterly values. I "
        "chose a grouped bar chart to compare two counts for four banks. "
        "The takeaway is how messy the raw data is: the same quarter shows "
        "up in several filings and many rows cover six months, nine months, "
        "or a full year. This is the main reason this project meets the "
        "Messy Data challenge goal.", width=3.8)


def write_joined(rep, r):
    o = r["overview"]["joined"]
    on = r["overview"]["nii_joined"]
    rq3 = r["rq3"]
    wider = [s for s in eda.SECTORS
             if rq3[s]["std_high"] > rq3[s]["std_low"]]
    rq4 = r["rq4"]
    pos = [b for b in eda.BANKS if rq4[b]["r_nii_rate"] > 0]
    rs = [rq4[b]["r_nii_rate"] for b in eda.BANKS]
    rep.h2("How the datasets fit together")
    rep.p("The datasets are linked by time. Monthly sector returns are "
          "joined with the FRED data on month (inner join, " +
          str(o["rows"]) + " rows, " + str(o["cols"]) + " columns), which "
          "is the table used for RQ1&ndash;RQ3. Each bank's quarterly income "
          "is joined with the average federal funds rate in that quarter "
          "and with the bank's quarterly return minus SPY's (" +
          str(on["rows"]) + " rows, " + str(on["cols"]) + " columns), which "
          "is the table used for RQ4. " + missing_text(o, "the monthly "
                                                       "joined table") +
          " " + missing_text(on, "the quarterly joined table") + " The "
          "missing " + code("nii_growth") + " values are each bank's first "
          "quarter, which has no previous quarter to compare to.")
    rep.figure(
        "results/figures/fig7.png",
        "In " + str(len(wider)) + " of 9 sectors, monthly returns were more "
        "spread out in the " + str(rq3["n_high"]) + " high-inflation months "
        "than in the " + str(rq3["n_low"]) + " low-inflation months.",
        "This box plot splits each sector's monthly returns into high- and "
        "low-inflation months. I chose it because RQ3 compares these two "
        "groups, and a box plot shows both the center and the spread of "
        "each group. The reader should take away that the high-inflation "
        "group is small and the spreads of the two groups differ, which "
        "affects how RQ3 should be tested (see Result Validity).")
    rep.figure(
        "results/figures/fig8.png",
        "Net interest income is positively correlated with the federal "
        "funds rate for " + str(len(pos)) + " of 4 banks (r from " +
        num(min(rs)) + " to " + num(max(rs)) + "), as a first look before "
        "the formal test.",
        "This scatter plot shows each bank's quarterly net interest income "
        "against the average federal funds rate in that quarter. A scatter "
        "plot is the natural choice for the relationship between two "
        "numeric variables. The banks sit at different levels because they "
        "are different sizes, so RQ4 will measure the correlation for each "
        "bank separately rather than pooling them.")


def write_validity(rep, r):
    rq1 = r["rq1"]
    rq3 = r["rq3"]
    rq4 = r["rq4"]
    rep.h1("Result Validity: Hypotheses and Assumption Checks")
    rate = rq1["rate_change"]
    nonnormal = [s for s in eda.SECTORS if rq1[s]["shapiro_p"] < 0.05]
    max_ac = max(abs(rq1[s]["autocorr"]) for s in eda.SECTORS)
    rep.p("<b>RQ1.</b> Null hypothesis: for each sector, the correlation "
          "between monthly return and " + code("rate_change") + " is 0. "
          "Test: Pearson's correlation test (" +
          code("scipy.stats.pearsonr") + ") plus a bootstrap 95% confidence "
          "interval. Assumptions: observations are independent, the "
          "relationship is roughly linear, there are no extreme outliers, "
          "and both variables are roughly normal (needed for the p-value). "
          "Checks: " + code("rate_change") + " is 0 in " +
          num(100 * rate["share_zero"], 0) + "% of months and fails the "
          "Shapiro-Wilk normality test (p " + pval(rate["shapiro_p"]) +
          "); " + str(len(nonnormal)) + " of 9 sectors' returns also fail "
          "it (p &lt; 0.05). The largest lag-1 autocorrelation of any "
          "sector's returns is " + num(max_ac) + ", which is "
          + ("small, so treating months as independent is reasonable."
             if max_ac < 0.2 else
             "not tiny, so months are not fully independent.") +
          " Because normality does not hold, I will rely on the bootstrap "
          "interval, which does not assume normality, and also report "
          "Spearman's rank correlation, which is less affected by outliers.")
    rep.table(["Sector", "n", "Shapiro p", "Outliers (|z|>3)",
               "Lag-1 autocorr"],
              [[s, str(rq1[s]["n"]), pval(rq1[s]["shapiro_p"]),
                str(rq1[s]["outliers"]), num(rq1[s]["autocorr"])]
               for s in eda.SECTORS],
              [0.9 * inch, 0.7 * inch, 1.0 * inch, 1.3 * inch, 1.2 * inch])

    unequal = [s for s in eda.SECTORS if rq3[s]["levene_p"] < 0.05]
    rep.p("<b>RQ3.</b> Null hypothesis: for each sector, the mean monthly "
          "return in high-inflation months equals the mean in low-inflation "
          "months. Test: Welch's t-test, checked with a permutation test. "
          "Assumptions: independent observations, each group roughly normal "
          "or large enough for the Central Limit Theorem, and (unlike the "
          "regular t-test) no need for equal variances. Checks: the groups "
          "have " + str(rq3["n_high"]) + " and " + str(rq3["n_low"]) +
          " months. Levene's test finds unequal variances in " +
          str(len(unequal)) + " of 9 sectors, which supports using Welch's "
          "test. " + ("The high-inflation group is small (under 30), so I "
                      "cannot rely on the Central Limit Theorem, and the "
                      "permutation test, which makes no normality "
                      "assumption, will be the main check."
                      if rq3["n_high"] < 30 else
                      "Both groups have at least 30 months, so the Central "
                      "Limit Theorem makes the t-test reasonable even "
                      "though returns are not perfectly normal.") +
          " High-inflation months also mostly come in one stretch of time, "
          "so they are not fully independent, which I will mention when "
          "interpreting the results.")
    rep.table(["Sector", "Std high", "Std low", "Shapiro p high",
               "Shapiro p low", "Levene p"],
              [[s, num(rq3[s]["std_high"]), num(rq3[s]["std_low"]),
                pval(rq3[s]["shapiro_p_high"]),
                pval(rq3[s]["shapiro_p_low"]), pval(rq3[s]["levene_p"])]
               for s in eda.SECTORS],
              [0.8 * inch, 0.9 * inch, 0.9 * inch, 1.15 * inch, 1.1 * inch,
               0.9 * inch])

    trending = [b for b in eda.BANKS if rq4[b]["autocorr_nii"] > 0.5]
    rep.p("<b>RQ4.</b> Null hypotheses: for each bank, (a) the correlation "
          "between quarterly net interest income and the average federal "
          "funds rate is 0, and (b) the correlation between quarterly income "
          "growth and the stock's return minus SPY's is 0. Test: Pearson's "
          "correlation test with bootstrap confidence intervals. "
          "Assumptions: same as RQ1. Checks: net interest income has a "
          "lag-1 autocorrelation above 0.5 for " + str(len(trending)) +
          " of 4 banks, meaning it trends slowly over time instead of "
          "moving independently from quarter to quarter. "
          "Two series that both trend can look correlated even when one does "
          "not cause the other, and the p-value will be too small. Based on "
          "this, I added a step: I will also correlate quarter-over-quarter "
          "<i>changes</i> in income with changes in the rate, since the "
          "growth series is much less autocorrelated (see table).")
    rep.table(["Bank", "Quarters", "Lag-1 autocorr (NII)",
               "Lag-1 autocorr (growth)", "Shapiro p (growth)"],
              [[b, str(rq4[b]["n"]), num(rq4[b]["autocorr_nii"]),
                num(rq4[b]["autocorr_growth"]),
                pval(rq4[b]["shapiro_p_growth"])] for b in eda.BANKS],
              [0.8 * inch, 0.9 * inch, 1.5 * inch, 1.6 * inch, 1.3 * inch])


def write_end(rep, r):
    raw_total = sum(r["raw_rows_by_bank"].values())
    kept_total = sum(r["kept_rows_by_bank"].values())
    rep.h1("Challenge Goals Update")
    rep.p("<b>Messy Data (kept).</b> The SEC data comes from an API as "
          "deeply nested JSON. Cleaning it means removing repeated reports "
          "of the same quarter, dropping six-month, nine-month, and "
          "full-year values, and computing Q4 from the 10-K totals: " +
          str(raw_total) + " raw rows become " + str(kept_total) +
          " quarterly values. One change: I now get the FRED data from its "
          "public CSV links instead of the FRED API, because the API needs "
          "a private key. I am not counting FRED toward this goal, so the "
          "goal now rests on the SEC data and on turning daily prices into "
          "monthly and quarterly returns.")
    rep.p("<b>Result Validity (kept and expanded).</b> The assumption "
          "checks above showed that returns are not normal and that bank "
          "income trends over time. Because of this I added Spearman's rank "
          "correlation for RQ1 and a test on quarterly changes for RQ4, on "
          "top of the bootstrap intervals, permutation test, and Bonferroni "
          "correction from the proposal.")
    rep.p("<b>Dropped:</b> I am no longer claiming Multiple Datasets or New "
          "Library as challenge goals. I still join the three sources and "
          "still use " + code("yfinance") + " and " + code("scipy") +
          ", but focusing on two goals keeps the project manageable.")

    rep.h1("Work Plan Evaluation")
    rep.p("Done so far: Task 1 (data pipeline), Task 2 (SEC quarterly "
          "cleanup), and part of Task 5 (unit tests). My estimate of 16 hours "
          "for Tasks 1 and 2 was too high, mainly because I used an AI "
          "assistant (Claude) to help write the pipeline code. The SEC "
          "cleanup was the hardest part, as I expected, because of the "
          "repeated filings and the missing Q4 values. Still to do: Task 3 "
          "(RQ1 and RQ2 analysis, about 8 hours), Task 4 (RQ3 and RQ4 "
          "analysis, about 8 hours, now including the extra tests above), "
          "the rest of Task 5 (bootstrap, permutation, and Bonferroni code "
          "and tests, about 4 hours), and Task 6 (report and presentation, "
          "about 8 hours).")

    rep.h1("Testing")
    rep.p(code("test_eda.py") + " tests the cleaning and EDA functions on "
          "small datasets where I worked out the right answer by hand, "
          "using " + code("assert") + " statements. It checks that FRED's "
          "\".\" becomes a missing value; that " + code("rate_change") +
          " and " + code("cpi_yoy") + " compare to the right months even "
          "when FRED skips a month; that SEC cleaning keeps the first-filed "
          "value of a repeated quarter, drops six-month and 8-K rows, and "
          "computes Q4 correctly (100 &minus; (10 + 20 + 30) = 40); that "
          "monthly returns use the last price of each month; that months "
          "are labeled high or low inflation correctly at exactly 4%; and "
          "that the missing-value counts and seven-number summaries match "
          "hand calculations. I tested this way because the cleaning steps "
          "are where silent mistakes are most likely, and small hand-made "
          "examples make each mistake visible. I also used the plots as "
          "checks: Figure 5 shows the computed Q4 values fitting in with the "
          "reported quarters, and Figure 1 shows the hiking periods lining "
          "up with the rate data. Finally, the cleaned SEC data has about "
          "four values per bank per year, which is what it should have.")

    rep.h1("Collaboration")
    rep.p("I used Claude (an AI assistant from Anthropic) to help write the "
          "data pipeline, EDA, and testing code and to help draft this "
          "report. I also used the documentation for FRED, the SEC EDGAR "
          "APIs, " + code("yfinance") + ", pandas, scipy, matplotlib, "
          "and seaborn.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python build_report.py <GitHub repository URL>")
        return
    plt.switch_backend("Agg")
    results = eda.run_eda()
    rep = Report()
    write_intro(rep, results, sys.argv[1])
    write_method(rep)
    write_fred(rep, results)
    write_prices(rep, results)
    write_sec(rep, results)
    write_joined(rep, results)
    write_validity(rep, results)
    write_end(rep, results)
    rep.build("eda.pdf")
    print("Saved eda.pdf")


if __name__ == "__main__":
    main()
