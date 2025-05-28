# Klasyfikacja Zawodników NBA do All-NBA Teams i All-Rookie Teams

## Opis projektu

Celem projektu było stworzenie modeli uczenia maszynowego, które automatycznie typują zawodników do nagród **All-NBA Teams** oraz **All-NBA Rookie Teams** na podstawie ich statystyk z poprzedniego sezonu NBA. W tym celu powstały dwa modele klasyfikacyjne typu RandomForestClassifier, trenowane na danych z sezonów 2000–2024.

## Dane

- **league_leaders_2000_2024_with_rookies_and_all_rookie_team.csv**  
  Zawiera statystyki liderów NBA z uwzględnieniem debiutantów oraz członków All-Rookie Team.

## Sposób działania

1. **Przygotowanie danych:**  
   Dane podzielono na zawodników niebędących debiutantami (do All-NBA Teams) oraz debiutantów (do All-Rookie Teams). Uwzględniono wyłącznie graczy z ≥ 65 meczami.

2. **Trenowanie modeli:**
   - **All-NBA Teams:**  
     Model RandomForestClassifier przewiduje, czy zawodnik trafi do 1., 2. lub 3. drużyny All-NBA. Użyto GridSearchCV do optymalizacji hiperparametrów oraz SMOTE do zbalansowania klas.
   - **All-Rookie Teams:**  
     Oddzielny RandomForestClassifier przewiduje przynależność debiutantów do All-Rookie Teams.

3. **Logowanie eksperymentów:**  
   Modele i wyniki logowane są przez MLflow.

4. **Predykcja:**  
   Skrypt `predict.py` wczytuje wytrenowane modele i dokonuje predykcji na bazie statystyk z ostatniego sezonu. Wyniki zapisywane są do pliku JSON.

## Pliki projektu

- **train_all_nba_teams.py**  
  Trenuje model przewidujący przynależność do All-NBA Teams.
- **train_all_nba_teams_rookie.py**  
  Trenuje model dla All-Rookie Teams.
- **predict.py**  
  Wczytuje modele i generuje predykcje do pliku JSON.
- **league_leaders_2000_2024_with_rookies_and_all_rookie_team.csv**  
  Zbiór danych używany do treningu i testów.

## Uruchomienie

1. **Trenowanie modeli:**
   ```bash
   python train_all_nba_teams.py
   python train_all_nba_teams_rookie.py

2. **Predykcja na nowym sezonie**
   ```bash
   python predict.py
   ```

