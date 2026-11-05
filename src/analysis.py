
import pandas as pd


class DataAnalyzer:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def get_overview(self):
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "missing": int(self.df.isnull().sum().sum()),
            "duplicates": int(self.df.duplicated().sum())
        }

    def get_statistics(self):
        return self.df.describe().T

    def get_missing_values(self):
        missing = self.df.isnull().sum()

        return missing[missing > 0]

    def get_correlations(self):
        numeric_data = self.df.select_dtypes(
            include="number"
        )

        return numeric_data.corr()

    def get_column_types(self):
        return self.df.dtypes.astype(str)
