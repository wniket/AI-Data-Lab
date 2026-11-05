
import pandas as pd
import pytest

from src.analysis import DataAnalyzer


@pytest.fixture
def sample_dataframe():

    return pd.DataFrame({
        "rooms": [2.0, None, 4.0, 4.0],
        "income": [10, 20, 30, 30],
        "city": ["A", "B", "C", "C"]
    })


@pytest.fixture
def analyzer(sample_dataframe):

    return DataAnalyzer(sample_dataframe)


def test_get_overview(analyzer):

    overview = analyzer.get_overview()

    assert overview["rows"] == 4
    assert overview["columns"] == 3
    assert overview["missing"] == 1
    assert overview["duplicates"] == 1


def test_get_statistics(analyzer):

    statistics = analyzer.get_statistics()

    assert "rooms" in statistics.index
    assert "income" in statistics.index
    assert statistics.loc["rooms", "count"] == 3


def test_get_missing_values(analyzer):

    missing = analyzer.get_missing_values()

    assert missing["rooms"] == 1
    assert len(missing) == 1


def test_get_correlations(analyzer):

    correlations = analyzer.get_correlations()

    assert "rooms" in correlations.columns
    assert "income" in correlations.columns
    assert correlations.shape == (2, 2)



def test_get_column_types(analyzer):

    column_types = analyzer.get_column_types()

    assert column_types["rooms"] == "float64"
    assert column_types["income"] == "int64"

    # Compatible with Pandas 2.x and 3.x
    assert pd.api.types.is_string_dtype(
        analyzer.df["city"].dtype
    )

    assert column_types["city"] == str(
        analyzer.df["city"].dtype
    )

