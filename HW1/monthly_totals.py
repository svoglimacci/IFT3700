"""
Ce devoir est basé sur [le cours de science des données de Greg Baker à SFU

À la fin de ce devoir, vous devriez être convaincu qu'il vaut mieux utiliser les fonctionnalités natives de Pandas
que de faire tout vous même. Vous devez également vous sentir bien à l'aise avec les DataFrames
et savoir pivoter ces objets pour atteindre vos objectifs.

Toutes les zones qui nécessitent des travaux sont marquées d'une étiquette "TODO".
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from geopy import distance

from typing import Tuple


def get_precip_data(fp: str = "data/precipitation.csv") -> pd.DataFrame:
    return pd.read_csv(fp, parse_dates=[2])


def date_to_month(d: pd.Timestamp) -> str:
    """
    Vous devrez peut-être modifier cette fonction, en fonction de vos types de données (s'ils ne correspondent pas
    aux types d'entrée attendus)
    """
    return "%04i-%02i" % (d.year, d.month)


def pivot_months_pandas(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Créez des totaux mensuels de précipitations pour chaque station dans l'ensemble de données.

    Cela devrait utiliser les méthodes Pandas pour manipuler les données. Arrondir les précipitations (mm) au 1er
    décimale.
    """

    data["month"] = data["date"].apply(date_to_month)

    grouped_data = data.groupby(["name", "month"])

    monthly = grouped_data["precipitation"].sum().reset_index()
    monthly = (
        monthly.pivot(index="name", columns="month", values="precipitation")
        .fillna(0)
        .round(1)
    )

    counts = grouped_data["precipitation"].count().reset_index()
    counts = counts.pivot(index="name", columns="month", values="precipitation").fillna(
        0
    )

    return monthly, counts


def pivot_months_loops(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Créez des totaux mensuels de précipitations pour chaque station dans l'ensemble de données.
    La façon difficile: utiliser Pandas comme magasin de données stupide et itérer en Python.

    Ne faites jamais les choses de cette façon!!!
    """
    # Trouvez toutes les stations et tous les mois dans l'ensemble de données.
    stations = set()
    months = set()
    for i, r in data.iterrows():
        stations.add(r["name"])
        m = date_to_month(r["date"])
        months.add(m)

    # Agrégez dans des dictionnaires afin que nous puissions rechercher plus tard.
    stations = sorted(list(stations))
    row_to_station = dict(enumerate(stations))
    station_to_row = {s: i for i, s in row_to_station.items()}

    months = sorted(list(months))
    col_to_month = dict(enumerate(months))
    month_to_col = {m: i for i, m in col_to_month.items()}

    # Créez des tableaux pour les données et remplissez-les.
    precip_total = np.zeros((len(row_to_station), 12), dtype=np.float64)
    obs_count = np.zeros((len(row_to_station), 12), dtype=np.float64)

    for _, row in data.iterrows():
        m = date_to_month(row["date"])
        r = station_to_row[row["name"]]
        c = month_to_col[m]

        precip_total[r, c] += row["precipitation"]
        obs_count[r, c] += 1

    # Construisez les DataFrames dont nous avions besoin tout au long (en rangeant les noms d'index pendant que nous y sommes).
    totals = pd.DataFrame(
        data=np.round(precip_total, 1),
        index=stations,
        columns=months,
    )
    totals.index.name = "name"
    totals.columns.name = "month"

    counts = pd.DataFrame(
        data=obs_count.astype(int),
        index=stations,
        columns=months,
    )
    counts.index.name = "name"
    counts.columns.name = "month"

    return totals, counts


def compute_pairwise(df: pd.DataFrame, func: callable) -> pd.DataFrame:
    """
    Complétez cette fonction, qui prend un dataframe et une fonction d'une paire de colonnes du dataframe
    en entrée et retourne un dataframe contenant la fonction appliquée à
    **chaque paire de lignes du dataframe**.

    Pour cela nous utiliserons les fonctions `pdist` et `squareform` de la bibliothèque `scipy.spatial`.
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

    Astuce: Assurez-vous que le DataFrame d'entrée a le nom de la station comme index, pas un nombre! Vous pouvez le faire en faisant pivoter
    le DataFrame. Cela devrait ressembler à cet extrait :

    ```
                             column1     column2
    name
    BURNABY SIMON FRASER U   ...        ...
    CALGARY INTL A           ...        ...
    ```
    """

    pairwise_distances = pdist(df.values, metric=func)

    square_matrix = squareform(pairwise_distances)

    new_df = pd.DataFrame(square_matrix, index=df.index, columns=df.index)

    return new_df


def geodesic(latlon1, latlon2) -> int:
    """
    Définit une métrique entre deux points ; dans notre cas, nos deux points sont des coordonnées latitude/longitude.
    "Nous" devons faire de la géométrie si nous voulons obtenir la distance entre deux points sur un ellipsoïde (Terre),
    mais nous allons abstraire cette fonctionnalité à une autre géopie. Vous pouvez en savoir plus sur
    les mathématiques ici:
        - https://en.wikipedia.org/wiki/Geodesics_on_an_ellipsoid

    Une simplification de ceci consiste à considérer à la place une sphère:
        - https://en.wikipedia.org/wiki/Haversine_formula
    """
    return int(distance.distance(tuple(latlon1), tuple(latlon2)).km)


def compute_pairwise_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Étant donné les fonctions `compute_pairwise()` et `geodesic()` définies ci-dessus,
    calculez la distance entre chacune des stations. L'entrée doit être la trame de données brute
    d'origine chargée du CSV.
    """

    new_df = (
        df[["name", "latitude", "longitude"]]
        .drop_duplicates()
        .set_index("name")
        .sort_index()
    )

    return compute_pairwise(new_df, geodesic)


def correlation(u, v) -> float:
    """
    Calculez la corrélation entre deux ensembles de données
    - https://en.wikipedia.org/wiki/Correlation

    Plus précisément, l'équation du coefficient produit-moment de Pearson est :

        corr = E[(X - x_avg) * (Y - y_avg)] / (x_std * y_std)

    """
    corr = None

    # obtenez des indices appropriés (filtrez les NaN ; '~' est 'not' logique)
    idx_u = ~pd.isna(u)
    idx_v = ~pd.isna(v)
    idx = idx_u & idx_v

    #  calculez la moyenne (mean) et  l'écart-type (std) des entrées valides
    x_avg = u[idx].mean()
    y_avg = v[idx].mean()
    x_std = u[idx].std()
    y_std = v[idx].std()

    # Calculez la corrélation
    corr = ((u[idx] - x_avg) * (v[idx] - y_avg)).mean() / (x_std * y_std)

    return corr


def compute_pairwise_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """

    Étant donné les fonctions `compute_pairwise()` et `correlation()` complétés ci-dessus, calculez la
    corrélation par paires des précipitations quotidiennes entre les stations. Le but ici est de voir s'il y a une
    corrélation des précipitations entre les stations. Idéalement, nous nous attendrions à des stations plus proches
    les uns aux autres pour avoir une corrélation plus élevée. L'entrée doit être le dataframe brute d'origine chargé
    du CSV.

    Notez que vous aurez probablement une diagonale de zéros alors qu'il devrait s'agir de uns - c'est bien
    aux fins de cette mission. `pdist` s'attend à ce que la fonction métrique soit une métrique appropriée,
    c'est-à-dire que la distance entre un élément et lui-même est nulle.

    """
    new_df = df.pivot(index="name", columns="date", values="precipitation")

    return compute_pairwise(new_df, correlation)


def compute_pairwise_correlation_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Surprise! Pandas peut en fait faire le calcul de corrélation pour vous en un seul appel d'une fonction

    Vous allez faire pivoter la table légèrement différemment, puis faire un seul appel de fonction sur le dataframe:
    - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html

    Vous devriez obtenir le même résultat que ce que vous avez obtenu pour `compute_pairwise_correlation()`, avec
    à l'exception de uns (correctement) le long de la diagonale.
    """
    new_df = df.pivot(index="name", columns="date", values="precipitation")

    return new_df.T.corr()


def main():
    data = get_precip_data()
    totals, counts = pivot_months_loops(data)

    # Facultativement créez les données...
    # totals.to_csv("data/totals.csv")
    # counts.to_csv("data/counts.csv")
    # np.savez("data/monthdata.npz", totals=totals.values, counts=counts.values)

    # faites pivoter monthspandas
    totals_pd, counts_pd = pivot_months_pandas(data)
    assert all(abs(totals - totals_pd).max() < 1e-10), "totals != totals_pd"
    assert all(abs(counts - counts_pd).max() < 1e-10), "counts != counts_pd"

    # calculez pairwise
    test_df = pd.DataFrame(
        [[0, 0], [0, 1], [1, 0]], columns=["x", "y"], index=list("abc")
    )
    euclidean = lambda xy1, xy2: np.sqrt(
        (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2
    )
    expected_output = pd.DataFrame(
        {
            "a": {"a": 0, "b": 1, "c": 1},
            "b": {"a": 1, "b": 0, "c": np.sqrt(2)},
            "c": {"a": 1, "b": np.sqrt(2), "c": 0},
        }
    )
    output = compute_pairwise(test_df, euclidean)
    assert np.allclose(output, expected_output)

    # distances par paires
    print(compute_pairwise_distances(data))

    # corrélation par paires
    print(compute_pairwise_correlation(data))

    # corrélation par paires
    print(compute_pairwise_correlation_pandas(data))

    print("Fini!")


if __name__ == "__main__":
    main()
