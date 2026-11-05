
import pandas as pd
from analysis import DataAnalyzer

df = pd.read_csv("data/housing.csv")

analyzer = DataAnalyzer(df)

print(analyzer.get_overview())
print(analyzer.get_statistics())
print(analyzer.get_missing_values())