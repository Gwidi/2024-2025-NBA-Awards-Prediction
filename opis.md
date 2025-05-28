# Przygotowanie danych do trenowania modeli

Dane do trenowania modeli zostały przygotowane w kilku krokach:

## 1. Pobranie danych statystycznych NBA

Dane o statystykach graczy dla każdego sezonu (2000–2024) pobrano z oficjalnego API NBA:

```python
from nba_api.stats.endpoints import LeagueLeaders

seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2000, 2024)]
all_data = []

for season in seasons:
    print(f"Pobieranie danych z sezonu: {season}")
    leaders = LeagueLeaders(season=season, season_type_all_star='Regular Season')
    df = leaders.get_data_frames()[0]
    df['SEASON'] = season
    all_data.append(df)
    time.sleep(1)  # Ograniczenie liczby zapytań do API

league_leaders_df = pd.concat(all_data, ignore_index=True)
league_leaders_df.to_csv("league_leaders_2000_2024.csv", index=False)
```

## 2. Dodanie informacji o debiutantach

Aby oznaczyć debiutantów, pobrano osobno statystyki rookies i zintegrowano je z główną tabelą:

```python
from nba_api.stats.endpoints import LeagueDashPlayerStats

league_leaders_df = pd.read_csv("../data/league_leaders_2024-25.csv")

rookie_stats = LeagueDashPlayerStats(
    season='2024-25',
    season_type_all_star='Regular Season',
    player_experience_nullable='Rookie',
    per_mode_detailed='Totals'
)
df_rookies = rookie_stats.get_data_frames()[0]
rookies_set = set(df_rookies['PLAYER_ID'])

league_leaders_df['IS_ROOKIE'] = league_leaders_df['PLAYER_ID'].isin(rookies_set)
league_leaders_df.to_csv("../data/final_dataset_for_prediction.csv", index=False)
```

## 3. Pozyskiwanie danych o nominacjach do All-NBA Teams (Web Scraping)

Dane o zawodnikach nominowanych do All-NBA Teams w poprzednich sezonach zostały **pozyskane metodą web scrapingu** z serwisu [basketball-reference.com](https://www.basketball-reference.com/awards/all_league.html). 

Tak przygotowane dane posłużyły jako **etykiety (labele) do trenowania modelu klasyfikacyjnego**, umożliwiając automatyczną predykcję wyróżnionych zawodników na podstawie ich statystyk.

### Przykładowy kod scrapowania danych

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

ALL_NBA_URL = "https://www.basketball-reference.com/awards/all_league.html"
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

# Pobierz stronę i znajdź tabelę z nominacjami All-NBA
content = request_data(url=ALL_NBA_URL, sleep_time_sec=4.0, with_comment=False)
table = content.find("table", id="awards_all_league")
df = pd.read_html(str(table))[0]

# Przetwarzanie i czyszczenie tabeli
df = df.drop(columns=["Lg", "Voting"])
df = df.dropna()  # Usuń puste wiersze
for col in df.columns[2:]:
    df[col] = df[col].str.replace(r' [CFG]$', '', regex=True)
df = df[df['Season'].str[:4].astype(int) >= 2000]  # Tylko sezony od 2000 roku
df.to_csv("nba_all_nba_teams.csv", index=False
```
### Zabezpieczenia na stronie basketball-reference – zakomentowane tabele

Podczas scrapowania danych z serwisu [basketball-reference.com](https://www.basketball-reference.com/awards/all_league.html) napotkano na dodatkowe zabezpieczenie: **kluczowe dane tabelaryczne były zakomentowane w kodzie HTML** za pomocą znaczników `<!-- ... -->`. To popularna technika stosowana przez twórców strony, która ma utrudnić automatyczne pobieranie danych przez boty i web scrapery.

#### Rozwiązanie problemu

Aby móc pobrać i przetworzyć te ukryte dane, konieczne było **odkomentowanie** zawartości HTML przed analizą kodu przez BeautifulSoup. W praktyce oznaczało to, że po pobraniu kodu strony należało najpierw usunąć znaczniki komentarzy, a dopiero potem przekazać taki "oczyszczony" HTML do parsera.

Poniżej przykład funkcji, która realizuje takie zadanie:

```python
from bs4 import BeautifulSoup

def filter_out_comment(soup: BeautifulSoup) -> BeautifulSoup:
    # Usuwa znaczniki komentarza z kodu HTML
    content = str(soup).replace('<!--', '')
    content = content.replace('-->', '')
```

## 4. Łączenie statystyk z informacjami o nominacjach

Statystyki zawodników z poprzednich sezonów, wraz z informacją o tym, czy w danym sezonie byli debiutantami, zostały **połączone z danymi o kategoriach nominacji do nagród**. 

W przypadku **All-NBA Teams** zawodnicy mogli być przypisani do jednej z kategorii:
- `1st` – pierwsza drużyna All-NBA
- `2nd` – druga drużyna All-NBA
- `3rd` – trzecia drużyna All-NBA

W przypadku **All-NBA Rookie Team** wyróżniano:
- `1st` – pierwsza drużyna debiutantów
- `2nd` – druga drużyna debiutantów

Wartość `0` w tych kolumnach oznaczała, że zawodnik **nie został nominowany** do żadnej drużyny w danym sezonie.

Aby umożliwić poprawne połączenie statystyk z informacjami o nominacjach, plik z nominacjami z poszczególnych lat został **przekształcony z formatu szerokiego na format długi** ("long format"). Dzięki temu każdy wiersz odpowiadał pojedynczemu zawodnikowi, konkretnemu sezonowi oraz przypisanej mu kategorii wyróżnienia. Takie przygotowanie danych pozwoliło na jednoznaczne powiązanie rekordów statystycznych z etykietami wykorzystywanymi podczas trenowania modeli klasyfikacyjnych.

---
