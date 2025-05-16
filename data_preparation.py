from nba_api.stats.endpoints import LeagueLeaders
import pandas as pd
import time
import os
import requests
from bs4 import BeautifulSoup, Comment
from io import StringIO

output_file = "league_leaders_2000_2024.csv"

if os.path.exists("league_leaders_2000_2024.csv"):
    print(f"File '{output_file}' already exists.")
else:
    # The League Leaders Stats
    seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2000, 2024)]

    all_data = []

    for season in seasons:
        print(f"Downloading data from season: {season}")
        leaders = LeagueLeaders(season=season, season_type_all_star='Regular Season')
        df = leaders.get_data_frames()[0]
        df['SEASON'] = season
        all_data.append(df)
        time.sleep(1)  # Avoiding API throttling

    # Concatenating data into DataFrame
    league_leaders_df = pd.concat(all_data, ignore_index=True)
    league_leaders_df.to_csv("league_leaders_2000_2024.csv", index=False)

    print("The data is saved")

# web scraping from https://github.com/JK-Future-GitHub/NBA_Champion/blob/main/nba_html_crawler.ipynb
PARSER = 'lxml'

def filter_out_comment(soup: BeautifulSoup) -> BeautifulSoup:
    content = str(soup).replace('<!--', '')
    content = content.replace('-->', '')
    return BeautifulSoup(content, PARSER)


def request_data(url: str, sleep_time_sec: float = 1.0, with_comment: bool = True) -> BeautifulSoup:
    time.sleep(sleep_time_sec)

    if with_comment:
        return BeautifulSoup(requests.get(url).content, PARSER)
    return filter_out_comment(BeautifulSoup(requests.get(url).content, PARSER))

def season_to_int(cell_value: str):
    if cell_value[-2:] == "00":
        return (int(cell_value[:2]) + 1)*100
    else:
        return int(cell_value[:2] + cell_value[-2:])

ALL_NBA_URL = r"https://www.basketball-reference.com/awards/all_league.html"
NBA_Rookie = r"https://www.basketball-reference.com/awards/all_rookie.html"
PARSER = 'lxml'


content = request_data(url=ALL_NBA_URL, sleep_time_sec=4.0, with_comment=False)
table = content.find("table",id="awards_all_league")
df = pd.read_html(str(table))[0]
df = df.drop(columns=["Lg","Voting"])
df = df.dropna() # Drop empty spaces
for col in df.columns[2:]:
    df[col] = df[col].str.replace(r' [CFG]$', '', regex=True)
df = df[df['Season'].str[:4].astype(int) >= 2000] # Only seasons later than 2000
df.to_csv("nba_all_nba_teams.csv", index=False)

