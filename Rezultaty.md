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

**f1_score = 0.9001**

## Podsumowanie

Model uzyskał najlepsze wyniki w przewidywaniu nominacji do pierwszej drużyny debiutantów (wysokie recall i f1-score dla klasy `1`). Skuteczność przewidywania zawodników, którzy nie zostali wyróżnieni (klasa `0`), jest również zadowalająca. Największym wyzwaniem okazało się dokładne wskazanie członków drugiej drużyny debiutantów (klasa `2`).

Uzyskane rezultaty potwierdzają skuteczność podejścia z wykorzystaniem RandomForestClassifier i optymalizacji hiperparametrów z użyciem GridSearchCV.

---



