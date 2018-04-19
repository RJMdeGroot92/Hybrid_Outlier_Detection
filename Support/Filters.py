import pandas as pd
import numpy as np


def df_filter(df, test=False):
    """
    standard filter for dataframe
    :param df:
    :type df: dataframe
    :param test:
    :type test: bool
    :return:
    :rtype: dataframe
    """
    df = df.drop("Date", 1)
    df = df.drop("Status", 1)
    df = df.drop("Id", 1)

    if test:
        print("test is on!")
        df = df[0:1000]
        df = df.drop("Src", 1)
    return df


def df_id_filter(df):
    """
    removes the columns:Date, Status, Resource, Src, Method, Path
    this should leave only the ID column
    :param df:
    :type df:
    :return:
    :rtype:
    """
    df = df.drop("Date", 1)
    df = df.drop("Status", 1)
    df = df.drop("Resource", 1)
    df = df.drop("Src", 1)
    df = df.drop("Method", 1)
    df = df.drop("Path", 1)
    return df


def insignificant_user_remover(df, threshold=7):
    """
    Removes all users and their requests of the their total amount of requests is below the threshold
    :param df:
    :type df: dataframe
    :param threshold:
    :type threshold: int
    :return:
    :rtype: dataframe
    """
    df = df
    value_counts = df["Username"].value_counts()
    sum_value_counts = sum(value_counts.values)
    original_users = len(value_counts.values)
    print("unique_users: " + str(original_users)+" with a total of "+str(sum_value_counts)+" actions.")
    to_remove = value_counts[value_counts <= threshold].index
    cleaned_df = df[~df.Username.isin(to_remove)]
    cleaned_value_counts = cleaned_df["Username"].value_counts().values
    remain_users = len(cleaned_value_counts)
    sum_cleaned_value_counts = sum(cleaned_value_counts)
    print("with a threshold of "+str(threshold)+".\n"+str(((original_users-remain_users)/original_users)*100) +
          "% of users where removed.")
    print("Resulting in "+str(remain_users)+" unique users with a total of "+str(sum_cleaned_value_counts)+" actions.")
    return cleaned_df


def mpr_merger(dataframe):
    """
    merges the Method, Path and Resource columns to one column named MPR
    :param dataframe:
    :type dataframe:
    :return:
    :rtype: Dataframe
    """
    df = dataframe.fillna(value=" ", axis=1)
    df["MPR"] = df["Method"]+"_"+df["Path"]+"_"+df["Resource"]

    df.drop("Method", 1, inplace=True)
    df.drop("Path", 1, inplace=True)
    df.drop("Resource", 1, inplace=True)
    return df

def parsed_filter(df):
    """
    Filters the variables Data, Id, Status from the input.
    :param df:
    :type df: Dataframe
    :return:
    :rtype: Dataframe
    """
    df.drop("Date", 1, inplace=True)
    df.drop("Id", 1, inplace=True)
    df = df.drop("Status", 1)
    return df