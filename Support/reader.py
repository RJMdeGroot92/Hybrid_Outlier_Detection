import pandas as pd
from Support.Filters import df_id_filter


def read_parsed(path):
    """
    Reads log file using the path class as input
    :param path:
    :type path: object
    :return:
    :rtype: dataframe
    """
    df = pd.read_csv(path, compression='gzip', encoding="utf-8", quotechar="'", delimiter=',')
    df.drop("Cic", 1, inplace=True)
    return df


def read_CSV_file(path):
    df = pd.read_csv(path, compression='gzip', encoding="utf-8", quotechar="'", delimiter=',')
    df = df.drop("Cic", 1)
    return df


def detection_data_reader(path):
    """
    Reads a csv file without removing any variables
    :param path:
    :type path: str
    :return:
    :rtype: dataframe
    """
    print("reading")
    df = pd.read_csv(path, compression='gzip', encoding="utf-8", quotechar="'", delimiter=',')
    return df


def id_data_reader(path):
    """
    Reads and filters a csv file and only returns id_data
    :param path:
    :type path: str
    :return:
    :rtype: dataframe
    """
    df = pd.read_csv(path, compression='gzip', encoding="utf-8", quotechar="'", delimiter=',')
    df.drop("Cic", 1, inplace=True)
    df = df[(df['Path'].str.contains("client", case=False) | df['Resource'].str.contains("client", case=False))]
    dataframe = df
    # dataframe = df[df.Resource != "/multiple"]
    dataframe = dataframe[dataframe.Id.notnull()]
    # dataframe=dataframe[0:100]
    dataframe = df_id_filter(dataframe)
    return dataframe
