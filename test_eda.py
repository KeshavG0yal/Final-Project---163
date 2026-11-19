"""
Keshav Goyal
CSE 163 Final Project

Tests for the data cleaning and EDA functions, using small datasets
whose correct answers were worked out by hand.
"""
import math

import pandas as pd

import download_data
import eda


def close(a, b, tol=1e-9):
    """
    Returns True if a and b are within tol of each other.
    """
    return abs(a - b) < tol


def test_clean_fred_csv():
    """
    FRED's '.' marker should become NaN and numbers should be numeric.
    """
    text = "observation_date,FEDFUNDS\n2020-01-01,1.55\n2020-02-01,.\n"
    df = download_data.clean_fred_csv(text, "FEDFUNDS")
    assert list(df.columns) == ["date", "fedfunds"]
    assert close(df.loc[0, "fedfunds"], 1.55)
    assert math.isnan(df.loc[1, "fedfunds"])
    assert df.loc[1, "date"] == pd.Timestamp("2020-02-01")


def test_build_macro():
    """
    rate_change is this month minus last month, and cpi_yoy compares to
    12 months earlier. A month missing from FRED must not shift the
    comparison.
    """
    dates = pd.date_range("1999-01-01", "2000-02-01", freq="MS")
    fedfunds = pd.DataFrame({"date": dates,
                             "fedfunds": [1.0] * 13 + [1.5]})
    cpi = pd.DataFrame({"date": dates,
                        "cpiaucsl": [100.0] * 12 + [103.0, 110.0]})
    # Drop March 1999 CPI entirely to check missing months are handled.
    cpi = cpi[cpi["date"] != pd.Timestamp("1999-03-01")]
    macro = download_data.build_macro(fedfunds, cpi)
    # Only 2000 onward is kept.
    assert len(macro) == 2
    jan = macro[macro["date"] == pd.Timestamp("2000-01-01")].iloc[0]
    feb = macro[macro["date"] == pd.Timestamp("2000-02-01")].iloc[0]
    assert close(jan["cpi_yoy"], 3.0)      # 103 / 100 - 1
    assert close(feb["cpi_yoy"], 10.0)     # 110 / 100 - 1
    assert close(jan["rate_change"], 0.0)
    assert close(feb["rate_change"], 0.5)


def make_raw_sec():
    """
    Builds a small fake set of raw SEC rows for one bank in 2020:
    Q1-Q3 from 10-Qs, Q1 reported again in a later filing with a
    different value, a six-month year-to-date value, an 8-K row, and the
    full-year 10-K total.
    """
    rows = [
        ["X", "InterestIncomeExpenseNet", "2020-01-01", "2020-03-31", 10,
         2020, "Q1", "10-Q", "2020-05-01"],
        ["X", "InterestIncomeExpenseNet", "2020-01-01", "2020-03-31", 99,
         2021, "Q1", "10-Q", "2021-05-01"],
        ["X", "InterestIncomeExpenseNet", "2020-04-01", "2020-06-30", 20,
         2020, "Q2", "10-Q", "2020-08-01"],
        ["X", "InterestIncomeExpenseNet", "2020-01-01", "2020-06-30", 30,
         2020, "Q2", "10-Q", "2020-08-01"],
        ["X", "InterestIncomeExpenseNet", "2020-07-01", "2020-09-30", 30,
         2020, "Q3", "10-Q", "2020-11-01"],
        ["X", "InterestIncomeExpenseNet", "2020-07-01", "2020-09-30", 77,
         2020, "Q3", "8-K", "2020-10-15"],
        ["X", "InterestIncomeExpenseNet", "2020-01-01", "2020-12-31", 100,
         2020, "FY", "10-K", "2021-02-20"],
    ]
    return pd.DataFrame(rows, columns=download_data.RAW_COLUMNS)


def test_clean_nii():
    """
    Cleaning should keep the first-filed Q1 value (10, not 99), drop the
    six-month and 8-K rows, and compute Q4 = 100 - (10 + 20 + 30) = 40.
    """
    clean = download_data.clean_nii(make_raw_sec())
    assert list(clean["quarter"]) == ["2020Q1", "2020Q2", "2020Q3",
                                      "2020Q4"]
    assert list(clean["nii"]) == [10, 20, 30, 40]
    assert list(clean["source"]) == ["reported"] * 3 + ["derived_q4"]


def test_extract_nii_rows():
    """
    Only USD facts under the listed tags should be extracted.
    """
    facts = {"facts": {"us-gaap": {
        "InterestIncomeExpenseNet": {"units": {"USD": [
            {"start": "2020-01-01", "end": "2020-03-31", "val": 5,
             "fy": 2020, "fp": "Q1", "form": "10-Q",
             "filed": "2020-05-01"}]}},
        "Revenues": {"units": {"USD": [
            {"start": "2020-01-01", "end": "2020-03-31", "val": 9}]}}}}}
    rows = download_data.extract_nii_rows(facts, "X")
    assert len(rows) == 1
    assert rows.loc[0, "val"] == 5
    assert rows.loc[0, "bank"] == "X"


def test_period_returns():
    """
    Monthly returns use the last price of each month:
    Jan 100 -> Feb 110 is +10%, Feb 110 -> Mar 99 is -10%.
    """
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-03", "2000-01-31", "2000-02-15",
                                "2000-02-29", "2000-03-31"]),
        "A": [50.0, 100.0, 120.0, 110.0, 99.0]})
    returns = eda.period_returns(prices, "M")
    assert len(returns) == 2
    assert close(returns["A"].iloc[0], 10.0)
    assert close(returns["A"].iloc[1], -10.0)


def test_label_months():
    """
    Inflation above 4% is 'high', 4% or below is 'low'; rate changes are
    labeled hike, cut, or no change.
    """
    macro = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01", "2000-02-01",
                                "2000-03-01"]),
        "cpi_yoy": [4.5, 4.0, 1.0], "rate_change": [0.25, 0.0, -0.5]})
    labeled = eda.label_months(macro)
    assert list(labeled["inflation"]) == ["high", "low", "low"]
    assert list(labeled["rate_move"]) == ["hike", "no change", "cut"]


def test_overview_and_summary():
    """
    overview counts rows, columns, and missing values; seven_number
    matches values worked out by hand for 1, 2, 3, 4, 5.
    """
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5],
                       "b": [1, None, 3, None, 5]})
    info = eda.overview(df)
    assert info == {"rows": 5, "cols": 2, "missing": {"b": 2}}
    summary = eda.seven_number(df["a"])
    assert summary["mean"] == 3 and summary["50%"] == 3
    assert summary["min"] == 1 and summary["max"] == 5
    assert summary["25%"] == 2 and summary["75%"] == 4
    assert close(summary["std"], math.sqrt(2.5))


def main():
    test_clean_fred_csv()
    test_build_macro()
    test_extract_nii_rows()
    test_clean_nii()
    test_period_returns()
    test_label_months()
    test_overview_and_summary()
    print("All tests passed!")


if __name__ == "__main__":
    main()
