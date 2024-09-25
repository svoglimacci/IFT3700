import re
import os
import pandas as pd
from tqdm import tqdm
from q2 import download_audio, cut_audio
from typing import List


def filter_df(csv_path: str, label: str) -> List[str]:
    """
    Write a function that takes the path to the processed csv from q1 (in the notebook) and returns a df of only the rows
    that contain the human readable label passed as argument

    For example:
    get_ids("audio_segments_clean.csv", "Speech")
    """
    df = pd.read_csv(csv_path)

    result = df[df['label_names'].str.contains(label)]

    return result


def data_pipeline(csv_path: str, label: str) -> None:
    """
    Using your previously created functions, write a function that takes a processed csv and for each video with the given label:
    (don't forget to create the audio/ folder and the associated label folder!).
    1. Downloads it to <label>_raw/<ID>.mp3
    2. Cuts it to the appropriate segment
    3. Stores it in <label>_cut/<ID>.mp3

    It is recommended to iterate over the rows of filter_df().
    Use tqdm to track the progress of the download process (https://tqdm.github.io/)

    Unfortunately, it is possible that some of the videos cannot be downloaded. In such cases, your pipeline should handle the failure by going to the next video with the label.
    """
    raw_path = label + "_raw"
    cut_path = label + "_cut"

    if not os.path.exists(raw_path):
        os.makedirs(raw_path)

    if not os.path.exists(cut_path):
        os.makedirs(cut_path)

    df = filter_df(csv_path, label)

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        label_id = row['# YTID']
        start = row[' start_seconds']
        end = row[' end_seconds']

        raw_file = raw_path + "/" + label_id + ".mp3"
        cut_file = cut_path + "/" + label_id + ".mp3"

        try:
            download_audio(label_id, raw_file)

        except Exception as e:
            print(e)
            continue


        cut_audio(raw_file, cut_file, start, end)





def rename_files(path_cut: str, csv_path: str) -> None:
    """
    Suppose we now want to rename the files we've downloaded in `path_cut` to include the start and end times as well as length of the segment. While
    this could have been done in the data_pipeline() function, suppose we forgot and don't want to download everything again.

    Write a function that, using regex (i.e. the `re` library), renames the existing files from "<ID>.mp3" -> "<ID>_<start_seconds_int>_<end_seconds_int>_<length_int>.mp3"
    in path_cut. csv_path is the path to the processed csv from q1. `path_cut` is a path to the folder with the cut audio.

    For example
    "--BfvyPmVMo.mp3" -> "--BfvyPmVMo_20_30_10.mp3"

    ## BE WARY: Assume that the YTID can contain special characters such as '.' or even '.mp3' ##
    """

    df = pd.read_csv(csv_path)
    files = os.listdir(path_cut)

    for i in range(len(files)):
        file = files[i]
        label_id = file.split(".")[0]
        row = df[df['# YTID'] == label_id]
        start = row[' start_seconds']
        end = row[' end_seconds']
        length = end - start

        new_name = label_id + "_" + str(int(start)) + "_" + str(int(end)) + "_" + str(int(length)) + ".mp3"

        os.rename(path_cut + "/" + file, path_cut + "/" + new_name)




if __name__ == "__main__":
    print(filter_df("data/audio_segments_clean.csv", "Laughter"))
    data_pipeline("data/audio_segments_clean.csv", "Laughter")
    rename_files("Laughter_cut", "data/audio_segments_clean.csv")
