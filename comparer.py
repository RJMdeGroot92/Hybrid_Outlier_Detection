import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
import seaborn as sns
from Support.Exceptions import *
import Support.settings as sup
import numpy as np

comparer_dir = os.getcwd()+"/User_groupings"
shield_dir = comparer_dir+"/deskundigheden"
comparisons_dir = comparer_dir+"/Comparisons"
colors2d = ["g.", "r.", "b.", "c.", "m.", "y.", "k.", "b."]


def percentile_plot(path):
    """
    makes a visualisation of the user-grouping described in path

    :param path:
    :type path: class
    :return:
    :rtype:
    """
    try:
        list = baseline_list(path)
        dataframe_shield = pd.read_csv(path.expertise_file())
        dataframe_grouped = pd.read_csv(path.grouping_file())
        g_users = dataframe_grouped["Username"].value_counts()
        print(len(dataframe_grouped["Username"].value_counts().values))
        dataframe_shield["MD5(u.username)"].drop_duplicates(inplace=True)
        total_amount_users = dataframe_grouped.count()["Username"]
        dataframe_shield = dataframe_shield.groupby("MD5(u.username)").sum()
        dataframe_shield = dataframe_shield.sort_values("description")
        dataframe_grouped = dataframe_grouped.set_index("Username")
        dict = {}

        for group in dataframe_grouped["Group"].unique():
            dict[group] = []
        for username in g_users.index.values:
            group = dataframe_grouped.get_value(username, "Group")
            description = "no_data"
            if username in dataframe_shield.index:
                description = dataframe_shield.get_value(username, "description")
            dict[group].append(description)

        style.use("ggplot")
        fig, ax = plt.subplots(len(dict), sharex=True, num=101)
        count = percentage_calc(dict)

        total_count = count[0]
        group_count = count[1]
        colours = cm.rainbow(np.linspace(0, 1, len(list.tolist())))
        for group in group_count:
            percentage_array = [0]*len(list.tolist())
            array_index = list.tolist()
            Amount_array = [0]*len(list.tolist())
            for discription in group_count[group]:
                amount = group_count[group][discription]
                total = total_count[discription]
                percentage = float('%.2f' % round((amount / total) * 100, 2))
                Amount_array[array_index.index(discription)] = amount
                percentage_array[array_index.index(discription)] = percentage
            ax[group-1].bar(range(len(array_index)), percentage_array, align='center', color=colours)
            title = "Users:" + str(sum(Amount_array))+"_"+str('%.0f'%(sum(Amount_array)/total_amount_users*100))+"%"
            # rotation could help turning the titel for better visualization
            ax[group - 1].set_title(title, loc="right", rotation=0, va="bottom")
            ax[group - 1].set_ylim([0,100])
            ax[group-1].set_ylabel("Group: "+str(group))
            for rect, amount in zip(ax[group-1].patches, Amount_array):
                if amount > 0:
                    height = rect.get_height()
                    ax[group-1].text(rect.get_x() + rect.get_width() / 2, height + 5, str(amount), ha='center',
                                     va='bottom')
        plt.xticks(range(len(array_index)), array_index)
        fig.autofmt_xdate()
        fig.set_size_inches(34.65, 13.67)
        if True:
            print("saving figure as "+path.filename+".png")
            savefig(path.figure_destination("Png"), bbox_inches='tight',dpi=100)
        show()
    except FileNotFoundError as e:
        print(e)


def percentage_calc(dict):
    """
    Calculates for each group the total amount of occurrences and the amount of occurrences per group

    :param dict:
    :type dict:
    :return: list containing 2 dictionaries
    :rtype: list
    """
    total_count = {}
    group_count = {}
    for group in dict:
        if group not in group_count:
            group_count[group] = {}
        for Discription in dict[group]:
            if Discription in total_count:
                total_count[Discription] = total_count[Discription]+1
                if Discription in group_count[group]:
                    group_count[group][Discription] = group_count[group][Discription]+1
                else:
                    group_count[group][Discription] = 1
            else:
                total_count[Discription] = 1
                group_count[group][Discription] = 1
    return [total_count, group_count]


def percentile_matrix_writer(dict):
    """
    depreciated function that generates a file, with statistics about a given usergrouping
    :param dict:
    :type dict:
    :return:
    :rtype:
    """
    count = percentage_calc(dict)
    total_count = count[0]
    group_count = count[1]
    with open(comparer_dir+'/percentage matrix.txt', 'w') as f:
        for group in group_count:
            array = []
            array_index = []
            for discription in group_count[group]:
                amount = group_count[group][discription]
                total = total_count[discription]
                percentage = float('%.2f' % round((amount/total)*100, 2))
                array.append([percentage, amount, total])
                array_index.append(discription)
            df = pd.DataFrame(data=array, columns=["Percentage", "Amount", "Total"], index=array_index)
            df = df.sort_values(["Percentage"], ascending=False)
            f.write("Group: "+str(group)+"\n")
            f.write(df.to_string()+"\n\n")
    f.close()


def baseline(path, list):
    """
    generates a visualasation of the amount of request made by a certain role
    :param path:
    :type path: class
    :param list:
    :type list: list
    :return:
    :rtype:
    """
    fig = plt.figure(9, figsize=(25, 15))
    dataframe_shield = pd.read_csv(path.expertise_file())
    dataframe_shield = dataframe_shield.sort_values(["description"])
    dataframe_shield = dataframe_shield.drop_duplicates()
    dataframe_shield = dataframe_shield.groupby("MD5(u.username)").sum()
    dataframe_shield = dataframe_shield.reset_index(drop=True)
    ax = sns.countplot(x="description", data=dataframe_shield, order=list)
    fig.autofmt_xdate()
    ax.set_title("Baseline")
    return ax


def baseline_list(path):
    """
    provides a baseline for a certain customer
    :param path:
    :type path: class
    :return:
    :rtype: list
    """
    dataframe_shield = pd.read_csv(path.expertise_file())
    dataframe_shield = dataframe_shield.sort_values(["description"])
    dataframe_shield = dataframe_shield.drop_duplicates()
    dataframe_shield = dataframe_shield.groupby("MD5(u.username)").sum()
    list = dataframe_shield["description"].unique()
    list = np.append(list, "no_data")
    list.dtype
    return list


def manager(check, customer, period):
    """
    manager that helps in selecting right functions
    1. Generate visulisation of user_grouping for a certain customer
    2. generate visulisation of a baseline for a certain customer
    :param check:
    :type check: int
    :param customer:
    :type customer: int
    :param period:
    :type period: int
    :return:
    :rtype:
    """
    # other choice for algorithm is Kmeans
    algorithm = "ACL"
    if check == 1:
        path = sup.ComparePath(customer, period, algorithm)
        print(path.figure_destination())
        percentile_plot(path)
    elif check == 2:
        path = sup.ComparePath(customer, period, algorithm)
        baseline(path, baseline_list(path))
        show()


manager(1, 1, 0)

