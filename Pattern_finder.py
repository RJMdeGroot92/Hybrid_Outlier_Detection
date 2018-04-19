import Support.settings as sup
import Support.reader as reader
import pandas as pd
import numpy as np
from Support.Filters import mpr_merger
from Support.Filters import parsed_filter


def extracter(customer, period, method=3, cutoff=20):
    """
    Generates a baseline of percentages for each action for each group per user.

    :param customer:
    :type customer: int
    :param period:
    :type period: int
    :param method:
    :type method: int
    :param cutoff: if for every group the percentage of users that uses the action is lower then the cutoff, the action
                    is removed
    :type cutoff: int
    :return:
    :rtype:
    """
    path = sup.PatternPath(customer, period)
    route = path.expertise_file()
    group_df = pd.read_csv(route)
    group_df.columns = ["Username", "Group"]
    log_df = reader.parsed_filter(reader.read_parsed(path.log_file()))
    # log_df.drop("Src", 1, inplace=True)
    log_df_mpr = mpr_merger(log_df)
    df_merged = log_df_mpr.merge(group_df, on=["Username"])
    count = df_merged.groupby("Group")["Username"].nunique()
    result, method, cutoff = difference_maker(df_merged, method, cutoff)
    result.index = result.rename(index=lambda x: x + "-[" + str(count[x])+"]").index
    with open(path.destination("baseline_" + method), "w") as out_f:
        out_f.write(result.to_string(index=True))
        out_f.close()


def group_extractor(customer, period, method=3, cutoff=20, algorithm="ACL"):
    """
    generates a file that shows the percentage of users in a group that made an action
    :param customer:
    :type customer: int
    :param period:
    :type period: int
    :param method:
    :type method: int
    :param cutoff: if for every group the percentage of users that uses the action is lower then the cutoff, the action
                    is removed
    :type cutoff: int
    :param algorithm:
    :type algorithm: str
    :return:
    :rtype:
    """
    path = sup.GroupPatternPath(customer, period, algorithm)
    grouping_route = path.grouping_file()
    grouping_df = pd.read_csv(grouping_route)
    expertise_df = pd.read_csv(path.expertise_file())
    expertise_df.columns = ["Username", "Group"]
    log_df = parsed_filter(reader.read_parsed(path.log_file()))
    # log_df.drop("Src", 1, inplace=True)
    log_df_mpr = mpr_merger(log_df)
    df_merged = log_df_mpr.merge(grouping_df, on=["Username"])
    count = df_merged.groupby("Group")["Username"].nunique()
    result, method, cutoff = difference_maker(df_merged, method, cutoff)
    result.index = result.rename(index=lambda x: str(x) + "-[" + str(count[x]) + "]").index

    with open(path.destination(path.get_filename() + method), "w") as out_f:
        out_f.write(result.to_string(index=True))
        out_f.close()


def difference_maker(df_merged, method, cutoff=20):
    """
    Calculates the percentage of users that do an action via three approaches
    :param df_merged:
    :type df_merged: dataframe
    :param method:
    :type method: int
    :param cutoff:
    :type cutoff: int
    :return:
    :rtype:
    """
    methods = {1: "BinALL", 2: "SumAll", 3: "BinUser", 4: "IntFind"}
    method = methods[method]
    column_names = df_merged.columns.values.tolist()
    column_names.remove('Group')
    column_names.remove('Username')
    result = pd.DataFrame({})
    if method == methods[1]:
        df_merged.drop("Username", 1, inplace=True)
        for column in column_names:
            col = pd.get_dummies(df_merged[column], prefix=None) \
                .groupby(df_merged["Group"]) \
                .agg("max") \
                .apply(lambda x: x / x.sum() * 100, axis=1)
            col = col.round(2)
            col = col.loc[:, (col >= cutoff).any(axis=0)]
            result = pd.concat([result, col], axis=1)
    elif method == methods[2]:
        df_merged.drop("Username", 1, inplace=True)
        for column in column_names:
            col = pd.get_dummies(df_merged[column], prefix=None) \
                .groupby(df_merged["Group"]) \
                .sum() \
                .apply(lambda x: x / x.sum() * 100, axis=1)
            col = col.round(2)
            col = col.loc[:, (col >= cutoff).any(axis=0)]
            result = pd.concat([result, col], axis=1)
    elif method == methods[3]:
        df_grouped = df_merged[["Username", "Group"]].drop_duplicates("Username")
        df_grouped.set_index("Username", inplace=True)
        df_merged.drop("Group", 1, inplace=True)
        for column in column_names:
            user_col = pd.get_dummies(df_merged[column], prefix=None) \
                .groupby(df_merged["Username"], as_index=True)\
                .agg("max")
            group_col = pd.concat([user_col, df_grouped], axis=1, join="inner")
            col = group_col.groupby("Group").sum()\
                .apply(lambda x: x / x.sum() * 100, axis=1)
            col = col.round(2)
            col = col.loc[:, (col >= cutoff).any(axis=0)]
            result = pd.concat([result, col], axis=1)
    elif method == methods[4]:
        df_grouped = df_merged[["Username", "Group"]].drop_duplicates("Username")
        df_grouped.set_index("Username", inplace=True)
        df_merged.drop("Group", 1, inplace=True)
        for column in column_names:
            user_col = pd.get_dummies(df_merged[column], prefix=None) \
                .groupby(df_merged["Username"], as_index=True) \
                .agg("max")
            group_col = pd.concat([user_col, df_grouped], axis=1, join="inner")
            col = group_col.groupby("Group").sum() \
                .apply(lambda x: x / x.sum() * 100, axis=1)
            col = col.round(2)
            col = col.loc[:, ((col > 0) & (col <= cutoff)).any(axis=0)]
            result = pd.concat([result, col], axis=1)
    return result, method, cutoff


def interest_finder(customer, period, method=4, cutoff=0.5):
    """
    checks for actions that only a small percent of users make in a baseline
    :param customer:
    :type customer:int
    :param period:
    :type period: int
    :param method:
    :type method: int
    :param cutoff:
    :type cutoff: int
    :return:
    :rtype:
    """
    pd.set_option('display.width', 1000)
    path = sup.PatternPath(customer, period)
    route = path.expertise_file()
    group_df = pd.read_csv(route)
    group_df.columns = ["Username", "Group"]
    log_df = reader.read_parsed(path.log_file())
    log_df_mpr = mpr_merger(log_df)
    print("done with merging")
    log_df_mpr_filtered = parsed_filter(log_df_mpr.copy())
    df_merged = log_df_mpr_filtered.merge(group_df, on=["Username"])
    log_df_mpr_filtered = None
    count = df_merged.groupby("Group")["Username"].nunique()
    print("count: ", count)
    result, method, cutoff = difference_maker(df_merged, method, cutoff)
    df_merged = None
    with open(path.destination("baseline_" + method), "w") as out_f:
        out_f.write(result.to_string(index=True))
        out_f.close()
    with open(path.destination("baseline_" + method+"_finds"), "w") as out_f2:
        for col in result.columns.values:
            if log_df_mpr["Src"].isin([col]).any():
                c_name = "Src"
            elif log_df_mpr["MPR"].isin([col]).any():
                c_name = "MPR"
            interest = result[((result[col] > 0) & (result[col] <= 0.5))][col]
            for group in interest.index.values:
                selected_expertise = group_df.loc[group_df["Group"] == group]
                selected_merge = pd.merge(log_df_mpr, selected_expertise, how="inner", on=["Username"])
                selected_resource = selected_merge.loc[selected_merge[c_name] == col]
                out_f2.write("Group: "+group+"/Resource: "+col+"/Value: "+str(interest[group])+"\n")
                out_f2.write(selected_resource.to_string(index=False, header=False)+"\n\n")
    out_f2.close()


def grouping_interest_finder(customer, period, method=4, cutoff=0.5, algorithm="ACL"):
    """
    checks for actions that only a small percent of users make in a group
    :param customer:
    :type customer: int
    :param period:
    :type period: int
    :param method:
    :type method: int
    :param cutoff:
    :type cutoff: int
    :param algorithm:
    :type algorithm: str
    :return:
    :rtype:
    """
    pd.set_option('display.width', 1000)
    path = sup.GroupPatternPath(customer, period, algorithm)
    grouping_route = path.grouping_file()
    grouping_df = pd.read_csv(grouping_route)
    expertise_df = pd.read_csv(path.expertise_file())
    expertise_df.columns = ["Username", "Group"]
    log_df = reader.read_parsed(path.log_file())
    log_df_mpr = mpr_merger(log_df)
    print("done with merging")
    log_df_mpr_filtered = parsed_filter(log_df_mpr.copy())
    df_merged = log_df_mpr_filtered.merge(grouping_df, on=["Username"])
    log_df_mpr_filtered = None
    count = df_merged.groupby("Group")["Username"].nunique()
    print("count: ", count)
    result, method, cutoff = difference_maker(df_merged, method, cutoff)
    df_merged = None
    with open(path.destination("baseline_" + method), "w") as out_f:
        out_f.write(result.to_string(index=True))
        out_f.close()
    with open(path.destination("baseline_" + method+"_finds"), "w") as out_f2:
        for col in result.columns.values:
            if log_df_mpr["Src"].isin([col]).any():
                c_name = "Src"
            elif log_df_mpr["MPR"].isin([col]).any():
                c_name = "MPR"
            interest = result[((result[col] > 0) & (result[col] <= 0.5))][col]
            for group in interest.index.values:
                selected_expertise = grouping_df.loc[grouping_df["Group"] == group]
                selected_merge = pd.merge(log_df_mpr, selected_expertise, how="inner", on=["Username"])
                selected_resource = selected_merge.loc[selected_merge[c_name] == col]
                out_f2.write("Group: "+str(group)+"/Resource: "+col+"/Value: "+str(interest[group])+"\n")
                out_f2.write(selected_resource.to_string(index=False, header=False)+"\n\n")
    out_f2.close()


def interest_checker(customer, period, expertise, resource_type, resource):
    """
    Return a datafame consisting only of request of a certain variable for a specific expertise.
    :param customer:
    :type customer: int
    :param period:
    :type period: int
    :param expertise: for example "Caretaker"
    :type expertise: str
    :param resource_type: for example "SRC"
    :type resource_type: str
    :param resource: for example "Dossier"
    :type resource: str
    :return:
    :rtype: dataframe
    """
    pd.set_option('display.width', 1000)
    path = sup.InterestPatternPath(customer, period)
    print(path.log_file())
    group_df = pd.read_csv(path.expertise_file())
    group_df.columns = ["Username", "Group"]
    print(path.log_file())
    log_df = reader.read_parsed(path.log_file())
    log_df_mpr = mpr_merger(log_df)
    selected_expertise = group_df.loc[group_df["Group"] == expertise]
    merged_df = pd.merge(log_df_mpr, selected_expertise, how="inner", on=["Username"])
    selected_resource = merged_df.loc[merged_df[resource_type] == resource]
    return selected_resource


def figure_extracter(customer, period, method=3, cutoff=20):
    """
    Generates a barchart of requests that are made by users per group.
     x-axis is the expertise
     y-axis is the percentage of users that made the request
     each color represent a type of request
    :param customer:
    :type customer: int
    :param period:
    :type period: int
    :param method:
    :type method: int
    :param cutoff:
    :type cutoff: int
    :return:
    :rtype:
    """
    path = sup.PatternPath(customer, period)
    expertise_df = pd.read_csv(path.expertise_file())
    expertise_df.columns = ["Username", "Group"]
    log_df = parsed_filter(reader.read_parsed(path.log_file()))
    log_df = log_df[log_df.Src != "HubDeployment"]
    log_df_mpr = mpr_merger(log_df)
    df_merged = log_df_mpr.merge(expertise_df, on=["Username"])
    count = df_merged.groupby("Group")["Username"].nunique()
    print(count)
    result, method, cutoff = difference_maker(df_merged, method, cutoff)
    delete_count = count > 20
    result = result[delete_count]
    print(result)
    result.index = result.rename(index=lambda x: x + "-[" + str(count[x])+"]").index
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    fig.autofmt_xdate()
    result.plot(kind="bar", ax=ax)
    plt.show()

figure_extracter(0, 0, 3, 20)
