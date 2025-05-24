import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport


df = pd.read_csv('league_leaders_2000_2024.csv')

profile = ProfileReport(df, title="Profiling Report league leaders")

profile.to_file("your_report.html")