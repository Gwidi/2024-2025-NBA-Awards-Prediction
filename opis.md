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

# Dlaczego zastosowano RandomForestClassifier?

Do realizacji tego projektu wybrano algorytm **RandomForestClassifier** ze względu na specyfikę dostępnego zbioru danych oraz naturę zadania klasyfikacyjnego:

- **Wysoka liczba cech liczbowych:**  
  Statystyki graczy NBA obejmują wiele różnych zmiennych numerycznych (np. punkty, asysty, zbiórki, minuty, efektywność i inne), które są często ze sobą nieliniowo powiązane. Random Forest bardzo dobrze radzi sobie z wieloma cechami oraz ich interakcjami, a także automatycznie ocenia istotność poszczególnych zmiennych.

- **Złożone zależności i brak założeń o rozkładzie danych:**  
  Klasyczne metody liniowe mogą nie radzić sobie ze złożonością zależności w statystykach sportowych. RandomForestClassifier nie wymaga założenia liniowości ani normalności rozkładu cech, co czyni go uniwersalnym wyborem dla tego typu danych.

- **Odporność na przeuczenie:**  
  Dzięki zastosowaniu wielu drzew decyzyjnych oraz mechanizmowi losowego wyboru cech, Random Forest jest mniej podatny na przeuczenie, nawet jeśli poszczególne drzewa są głęboko dopasowane do fragmentów zbioru treningowego.

- **Radzenie sobie z niezbalansowanymi klasami:**  
  W projekcie liczba graczy wyróżnionych w danym sezonie (np. wybranych do All-NBA Team) jest zdecydowanie mniejsza niż liczba wszystkich zawodników. RandomForestClassifier, w połączeniu z technikami typu SMOTE do oversamplingu, dobrze radzi sobie w takich sytuacjach i umożliwia uzyskanie stabilnych wyników klasyfikacji.

- **Intuicyjna interpretacja wyników:**  
  Random Forest pozwala na ocenę ważności cech, dzięki czemu można zinterpretować, które statystyki mają największy wpływ na szanse zawodnika na nominację do All-NBA Teams lub All-NBA Rookie Teams.

Dzięki powyższym cechom, RandomForestClassifier jest narzędziem elastycznym, skutecznym oraz łatwym do wdrożenia i interpretacji przy analizie danych sportowych tego typu.


# Techniki wykorzystane do trenowania modeli

Podczas trenowania modeli zastosowano kilka nowoczesnych technik i narzędzi, które pozwoliły zoptymalizować proces budowy i oceny modeli klasyfikacyjnych.

## GridSearchCV

Do wyboru najlepszych hiperparametrów modeli zastosowano metodę **GridSearchCV**. GridSearch polega na automatycznym przeszukiwaniu zdefiniowanej siatki kombinacji parametrów modelu (np. liczba drzew, głębokość drzewa, minimalna liczba próbek w liściu itp.). Dla każdej kombinacji model jest trenowany i oceniany na zbiorze walidacyjnym przy użyciu techniki **cross-validation** (walidacji krzyżowej). 

Celem GridSearch jest znalezienie zestawu parametrów, który daje najlepsze wyniki (np. najwyższy wynik f1_score) na podstawie zdefiniowanej metryki.

**Przykład działania:**
- Tworzymy siatkę możliwych wartości hiperparametrów, np. liczba drzew: [100, 200, 500], głębokość drzewa: [10, 20, None].
- GridSearch testuje każdą możliwą kombinację tych wartości, trenując model za każdym razem na innych danych w ramach cross-validation.
- Wyniki są porównywane, a do końcowego modelu wybierana jest konfiguracja o najlepszym wyniku.

## MLOps i MLflow

Do zarządzania procesem eksperymentowania z modelami oraz ich wersjonowaniem wykorzystano narzędzie **MLflow**, które jest przykładem podejścia **MLOps** (Machine Learning Operations). Pozwala ono na:
- automatyczne zapisywanie parametrów, metryk, wykresów i artefaktów każdego eksperymentu,
- łatwe porównywanie różnych wersji modeli i konfiguracji,
- powtarzalność eksperymentów i przechowywanie całej historii trenowania.

Dzięki MLflow możliwe było systematyczne testowanie różnych ustawień modeli, rejestrowanie najlepszych wyników oraz szybki powrót do wybranych wersji w przyszłości.

## Podsumowanie

Zastosowanie GridSearchCV i MLOps (MLflow) pozwoliło nie tylko zoptymalizować proces wyboru najlepszego modelu, ale również zachować pełną transparentność, powtarzalność i kontrolę nad eksperymentami podczas całego cyklu życia projektu.

---

# Jak przebiega proces predykcji?

Proces predykcji został w pełni zautomatyzowany w pliku `predict.py` i obejmuje następujące etapy:

1. **Wczytanie danych zawodników**
   
   Do predykcji wykorzystywany jest plik `final_dataset_for_prediction.csv`, zawierający dane statystyczne graczy z ostatniego sezonu (m.in. punkty, minuty, zbiórki oraz informację, czy zawodnik jest debiutantem).

2. **Podział danych na grupy**
   
   Zawodnicy są dzieleni na dwie grupy:
   - zawodnicy z odpowiednią liczbą rozegranych meczów, którzy **nie są debiutantami** (do typowania All-NBA Teams),
   - zawodnicy z odpowiednią liczbą meczów, którzy **są debiutantami** (do typowania All-NBA Rookie Team).

3. **Wczytanie wytrenowanych modeli**
   
   Za pomocą MLflow ładowane są dwa modele:
   - model do typowania All-NBA Teams,
   - model do typowania All-NBA Rookie Teams.

4. **Predykcja przynależności do drużyn**
   
   - Dla każdej grupy zawodników wykonywana jest predykcja przynależności do odpowiedniej drużyny (`1st`, `2nd`, `3rd` lub brak nominacji dla All-NBA; `1st`, `2nd` lub brak nominacji dla All-Rookie).
   - Wyniki predykcji są przypisywane do DataFrame z danymi zawodników.

5. **Selekcja zawodników do drużyn**
   
   Po predykcji zawodnicy są sortowani według kategorii drużyny i ich liczby zdobytych punktów, a następnie wybierana jest piątka graczy do każdej z drużyn (pierwsza, druga, trzecia). Jeśli w którejś drużynie zabraknie graczy, kolejne osoby są dobierane zgodnie z rankingiem punktowym.

6. **Zapis wyników**
   
   Wyniki predykcji, czyli składy wszystkich drużyn (All-NBA i All-Rookie), zapisywane są do pliku JSON w czytelnym formacie.

## Fragment przykładowego kodu predykcji

```python
import mlflow.sklearn
import pandas as pd

# Wczytanie danych i modeli
df = pd.read_csv('../data/final_dataset_for_prediction.csv')
model = mlflow.sklearn.load_model('...ścieżka do modelu...')
results = model.predict(df[wybrane_kolumny])

# Przypisanie wyników do DataFrame i wybór drużyn
df['prediction'] = results
first_team = df[df['prediction'] == '1'].sort_values('PTS', ascending=False).head(5)['PLAYER'].tolist()
```
