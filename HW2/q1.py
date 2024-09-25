import json
import pandas as pd


def count_labels(labels: str) -> int:
    """
    Given a string of unparsed labels, return the number of distinct labels.

    For example:
    "/m/04rlf,/m/06_fw,/m/09x0r" -> 3
    """
    length = len(labels.split(","))
    return length


def convert_id(ID: str) -> str:
    """
    Create a function that takes in a label ID (e.g. "/m/09x0r") and returns the corresponding label name (e.g. "Speech")

    To do so, make use of the `json` library and the `data/ontology.json` file, a description of the file can be found
    at https://github.com/audioset/ontology

    While reading the file each time and looping through the elements to find a match works well enough for our
    purposes, think of ways this process could be sped up if say this function needed to be run 100000 times.
    """
    with open("data/ontology.json") as f:
        data = json.load(f)
        for i in data:
            if i["id"] == ID:
                return i["name"]
    return None


def convert_ids(labels: str) -> str:
    """
    Using convert_id() create a function that takes the label columns (i.e a string of comma-separated label IDs)
    and returns a string of label names, separated by pipes "|".

    For example:
    "/m/04rlf,/m/06_fw,/m/09x0r" -> "Music|Skateboard|Speech"
    """
    labels = labels.split(",")
    string = [convert_id(i) for i in labels]

    result = "|".join(string)

    return result


def contains_label(labels: pd.Series, label: str) -> pd.Series:
    """
    Create a function that takes a Series of strings where each string is formatted as above
    (i.e. "|" separated label names like "Music|Skateboard|Speech") and returns a Series with just
    the values that include `label`.

    For example, given the label "Music" and the following Series:
    "Music|Skateboard|Speech"
    "Voice|Speech"
    "Music|Piano"

    the function should just return
    "Music|Skateboard|Speech"
    "Music|Piano"
    """

    ## do not use contains()

    result = labels[labels.str.contains(label)]


    return result


def get_correlation(labels: pd.Series, label_1: str, label_2: str) -> float:
    """
    Create a function that, given a Series as described above, returns the proportion of rows
    with label_1 that also have label_2. Make use of the function you created above.

    For example, suppose the Series has 1000 values, of which 120 have label_1. If 30 of the 120
    have label_2, your function should return 0.25.
    """
    label_1_series = contains_label(labels, label_1)
    label_2_series = contains_label(label_1_series, label_2)

    return len(label_2_series) / len(label_1_series)


if __name__ == "__main__":
    print(count_labels("/m/04rlf,/m/06_fw,/m/09x0r"))
    print(convert_id("/m/04rlf"))
    print(convert_ids("/m/04rlf,/m/06_fw,/m/09x0r"))

    series = pd.Series([
        "Music|Skateboard|Speech",
        "Voice|Speech",
        "Music|Piano"
    ])
    print(contains_label(series, "Music"))
    print(get_correlation(series, "Music", "Piano"))
