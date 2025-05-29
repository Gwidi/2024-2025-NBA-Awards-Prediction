# Wyniki trenowania modelu All-NBA Teams Rookie

## Metodyka

Do przewidywania zawodników nominowanych do All-NBA Rookie Teams wykorzystano klasyfikator RandomForestClassifier, a optymalizacji hiperparametrów dokonano z użyciem GridSearchCV. Model był trenowany i testowany z użyciem walidacji krzyżowej (5-fold cross-validation) na wybranych kombinacjach parametrów.

## Wyniki klasyfikacji

Poniżej przedstawiono uzyskane metryki modelu dla poszczególnych klas:

| Klasa | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
|   0   |   0.70    |  0.59  |   0.64   |   32    |
|   1   |   0.61    |  0.85  |   0.71   |   20    |
|   2   |   0.44    |  0.37  |   0.40   |   19    |

**Łączna dokładność (accuracy): 0.61**  
**Średnia f1-score (weighted avg): 0.60**

- Klasa `0` oznacza zawodników, którzy nie zostali nominowani do żadnej drużyny.
- Klasa `1` odpowiada pierwszej drużynie All-Rookie.
- Klasa `2` odpowiada drugiej drużynie All-Rookie.

## Wybrane hiperparametry modelu (GridSearch)

Dzięki GridSearchCV uzyskano najlepsze ustawienia hiperparametrów:

- `max_depth`: None
- `min_samples_leaf`: 1
- `min_samples_split`: 2
- `n_estimators`: 100

## Najlepszy wynik f1_score (walidacja krzyżowa)

**f1_weighted = 0.9001**

## Podsumowanie

Model uzyskał najlepsze wyniki w przewidywaniu nominacji do pierwszej drużyny debiutantów (wysokie recall i f1-score dla klasy `1`). Skuteczność przewidywania zawodników, którzy nie zostali wyróżnieni (klasa `0`), jest również zadowalająca. Największym wyzwaniem okazało się dokładne wskazanie członków drugiej drużyny debiutantów (klasa `2`).

Uzyskane rezultaty potwierdzają skuteczność podejścia z wykorzystaniem RandomForestClassifier i optymalizacji hiperparametrów z użyciem GridSearchCV.

# Wyniki trenowania modelu – All-NBA Rookie Teams

## Opis

Model został wytrenowany do przewidywania nominacji do All-NBA Rookie Teams na podstawie dostępnych cech gracza. Klasyfikacja obejmowała następujące klasy:

* **0** – brak nominacji
* **1st** – All-Rookie First Team
* **2nd** – All-Rookie Second Team
* **3rd** – All-Rookie Third Team (jeśli istnieje taka kategoria w zbiorze)

## Parametry modelu (uzyskane dzięki GridSearch)

Do optymalizacji parametrów modelu wykorzystano GridSearch. Uzyskano następujące wartości dla najlepszego zestawu hiperparametrów Random Forest:

- **feature_names:** EFF, PTS, FTA, FGM, FTM, FGA, TOV, MIN, DREB
- **max_depth:** None
- **min_samples_leaf:** 2
- **min_samples_split:** 2
- **n_estimators:** 200

## Raport klasyfikacji

| Klasa | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.99      | 0.97   | 0.98     | 794     |
| 1st   | 0.44      | 0.87   | 0.59     | 23      |
| 2nd   | 0.20      | 0.26   | 0.23     | 19      |
| 3rd   | 0.00      | 0.00   | 0.00     | 19      |

### Podsumowanie metryk:

* **accuracy:** 0.93 (855 przypadków)
* **macro avg:** Precision 0.41, Recall 0.53, F1-score 0.45
* **weighted avg:** Precision 0.93, Recall 0.93, F1-score 0.93

## Wnioski

* **Model bardzo dobrze wykrywa przypadki braku nominacji** (klasa 0) – precision oraz recall na poziomie \~0.98.
* **Wykrywanie zawodników wyróżnionych** w All-Rookie First Team jest przyzwoite (recall 0.87, F1-score 0.59), ale precision jest umiarkowane (0.44), co oznacza sporą liczbę fałszywych alarmów.
* **All-Rookie Second Team** – model radzi sobie słabo (F1-score 0.23, recall 0.26).
* **All-Rookie Third Team** – model nie wykrywa tej klasy w ogóle (być może z powodu bardzo małej liczby przykładów lub złej jakości danych).
* **Makrośrednia F1-score jest niska (0.45)**, co wynika z niezbalansowanego zbioru i trudności z wykrywaniem mniejszych klas.
* **Wysoka dokładność (0.93)** wynika głównie z dużej liczby przykładów klasy 0.





