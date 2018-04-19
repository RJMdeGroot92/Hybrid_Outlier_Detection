from gzipparser import parser
import gzip
import pandas as pd
import sys
import numpy as np
import os
import glob
import re
import Datamining as miner
from Support.Exceptions import *
from Support.Weka_format import *
from Support.reader import read_CSV_file
from Support.Filters import insignificant_user_remover
from Support.Filters import df_filter
from Support.Filters import df_id_filter
import Support.settings as sup

log_folder = "logs"
curDir = os.getcwd()
logDir = os.getcwd()+"/" + log_folder
newLogDir = os.getcwd()+"/logs/new"
parsedLogDir = os.getcwd()+"/logs/parsed"
comparer_dir = os.getcwd()+"/User_groupings"
# pattern for cupido_p_mathijs.log-date-machineName
date_pattern = re.compile("\w*_\w_\w*\.\w{3}-([0-9]*)-*.*")
date_pattern2 = re.compile("\w{2}[0-9]{2}\w[0-9]{4}-([0-9]{8})\.\w{3}\.\w{2}")
modus = ["Weka", "CSV", "QoS"]
# "MF1267" niet in shield.
customers = ["MF2023", "MF0874", "MF0122", "MF0058", "MF1955"]


def base_folder_maker():
    """
    Generates the basic folders needed if they do not exist
    :return:
    :rtype:
    """
    if not os.path.exists(logDir):
        os.mkdir(log_folder)
    if not os.path.exists(newLogDir):
        os.mkdir(newLogDir)
    if not os.path.exists(parsedLogDir):
        os.mkdir(parsedLogDir)


def folder_maker(path):
    """
    generates a folder at the specified path if does not exist
    :param path:
    :type path: str
    :return:
    :rtype:
    """
    if not os.path.exists(path):
        os.mkdir(path)


def new_log_reader():
    """
    Searches for new logs to parse. Data is extracted from the filename, and a new folder for the parsed log is made
    using this data
    :return:
    :rtype: list
    """
    try:
        files = []
        for file in glob.glob(newLogDir+"/*"):
            new_file = file.replace(newLogDir+"/", "")
            # Extract data from filename
            date = re.search(date_pattern2, file).group(1)
            # make new folder with date if none exists
            folder_maker(parsedLogDir+"/"+date)
            files.append(new_file)
        return files
    except AttributeError:
        print("AttributeError because of: ", file)


def date_finder(new_logs):
    """
    orders and sorts the logs on date
    :param new_logs:
    :type new_logs: list
    :return: a list of dates with a dict of logs belonging to each date
    :rtype: list
    """
    dates = []
    date_log_dict = {}
    for logs in new_logs:
        log_date = re.search(date_pattern2, logs).group(1)
        if log_date not in dates:
            dates.append(log_date)
            date_log_dict[log_date] = [logs]
        else:
            date_log_dict[log_date].append(logs)
    dates.sort()
    print(dates)
    print(date_log_dict)
    return [dates, date_log_dict]


def new_log_parser(modus):
    """
    Parse all new logs in the format of the given modus

    :param modus:
    :type modus: str
    :return:
    :rtype:
    """
    try:
        new_logs = new_log_reader()
        date_logs = date_finder(new_logs)
        dates = date_logs[0]
        ordered_logs = date_logs[1]
        for date in dates:
            customer_dict = {}
            for customer in customers:
                customer_dict.update({customer: gzip.open(parsedLogDir + "/" + date + "/" + customer + ".gz", 'wb')})
                if modus == "Weka":
                    customer_dict[customer].writelines(get_weka_complete())
                elif modus == "CSV":
                    customer_dict[customer].write("Date,Method,Path,Id,Resource,Status,Cic,Src,Username\n".
                                                  encode("utf-8"))
                elif modus == "QoS":
                    customer_dict[customer].write("Username,QoS\n".encode("utf-8"))
            # loop trough every log associated with that data
            for log in ordered_logs[date]:
                print(log)
                with gzip.open(newLogDir+"/"+log, 'rb') as in_f:
                    for line in in_f:
                        parsed_line = parser(line, modus)
                        if parsed_line is not None:
                            customer_dict[parsed_line[1]].write(parsed_line[0])
                in_f.close()
                print(log, " is done!")
            for customer in customers:
                customer_dict[customer].close()
                del customer_dict[customer]
            print(date, " has been processed!")
    except EmptyUsernameException as e:
        print("Empty username Exception:", e)
    except AttributeError as e:
        print("AttributeError", e)
        print(sys.exc_info())
    except:
        print("this is a base exception")
        print(sys.exc_info())


def period_merger(customer, dates=["20170901", "20170902", "20170903", "20170904", "20170905", "20170906", "20170907",
                            "20170908", "20170909", "20170910", "20170911", "20170912", "20170913", "20170914",
                            "20170915", "20170916", "20170917", "20170918", "20170919", "20170920", "20170921",
                            "20170922", "20170923", "20170924", "20170925", "20170926", "20170927", "20170928",
                            "20170929", "20170930"]):
    """
    Merges logs of the given date into one
    :param customer:
    :type customer: int
    :param dates:
    :type dates: list
    :return:
    :rtype:
    """
    dates.sort()
    begindate = dates[0]
    enddate = dates[len(dates)-1]
    customer = customers[customer]
    print("Merging on customer "+customer+" from "+begindate+" to "+enddate)
    outfile = begindate+"-"+enddate
    folder_maker(parsedLogDir+"/"+outfile)
    header_saved = False
    with gzip.open(parsedLogDir+"/"+outfile+"/"+customer+".gz", 'wb') as out_f:
        for date in dates:
            with gzip.open(parsedLogDir+"/"+date+"/"+customer+".gz", "rb") as in_f:
                header = next(in_f)
                if not header_saved:
                    out_f.write(header)
                    header_saved = True
                for line in in_f:
                    out_f.write(line)
            in_f.close()


def filemanager(check, customer=0, period=0, test=False):
    """
    Overarching manager of functions. depending on the value of check it starts different processes.
    process related to check:
    1: start processing new logs
    2: start making user-groupings using Agglomerative clustering
    3: start making user-groupings using k-means
    4: start the merging of different logs into one
    5: depreciated method used for generating groupings based on SubjectID without using multiple Ids
    :param check:
    :type check: int
    :param customer:
    :type customer: int
    :param period:
    :type period: int
    :param test:
    :type test: bool
    :return:
    :rtype:
    """
    if check == 1:
        print("Starting_to_parse:")
        check = input("are your sure?Y/N")
        if check == "Y":
            new_log_parser("CSV")
        else:
            print("Parsing canceled")
            check = 1
    if check == 2:
        threshold = 7
        clusters = 8
        # preprocessing has several options such as:
        # preprocessing = "Original"
        # preprocessing == "Minmax_scale"
        # preprocessing == "Standard_scalar"
        # preprocessing == "Wout"
        preprocessing = "Bin_Proc"
        path = sup.GroupingPath(customer, period, "ACL", preprocessing, threshold)
        path.set_clusters(clusters)
        print(path.log_file())

        dataframe = read_CSV_file(path.log_file())
        print(dataframe.Status.value_counts())
        dataframe = df_filter(dataframe, test)
        # remove users that have less action than a certain threshold.
        dataframe = insignificant_user_remover(dataframe, threshold=threshold)
        dataframe = miner.merge_path_method_resource(dataframe)

        # possibility of dropping certain variables such as:
        # dataframe = dataframe.drop("Src", 1)
        # dataframe = dataframe.drop("Method", 1)
        # dataframe = dataframe.drop("Path", 1)
        # dataframe = dataframe.drop("Resource", 1)

        arr_features = dataframe.columns.values
        path.set_features(",".join(np.delete(arr_features, np.argwhere(arr_features == "Username"))))
        dismatrix = miner.matrix_generator(dataframe, preprocessing, "rmse")
        groupings = miner.acl_username_grouper(dismatrix, clusters)

        get_mean(dataframe, path=path, user_groupings=groupings)
        groupings.to_csv(path.destination("data", "Csv"))

    if check == 3:
        threshold = 7
        clusters = 10
        # preprocessing has several options such as:
        # preprocessing = "Original"
        # preprocessing = "Wout"
        # preprocessing= "Minmax_scale"
        # preprocessing = "Standard_scalar"
        preprocessing = "Bin_Proc"
        path = sup.GroupingPath(customer, period, "Kmeans", preprocessing, threshold)
        path.set_clusters(clusters)
        dataframe = read_CSV_file(path.log_file())

        test = False
        dataframe = df_filter(dataframe, test)

        dataframe=insignificant_user_remover(dataframe, threshold=threshold) # remove users that have less action than a certain threshold.
        dataframe = miner.merge_path_method_resource(dataframe)

        # possibility of dropping certain variables such as:
        # dataframe = dataframe.drop("Src", 1)
        # dataframe = dataframe.drop("Method", 1)
        # dataframe = dataframe.drop("Path", 1)
        # dataframe = dataframe.drop("Resource", 1)

        arr_features = dataframe.columns.values
        path.set_features(",".join(np.delete(arr_features, np.argwhere(arr_features == "Username"))))
        dismatrix = miner.matrix_generator(dataframe, preprocessing, "rmse")
        user_groupings = miner.kmeans_username_grouping(dismatrix, clusters)

        get_mean(dataframe, path=path, user_groupings=user_groupings)
        user_groupings.to_csv(path.destination("data", "Csv"))

    if check == 4:
        # The period to merge depends on the variable dates defined here. Examples are:
        # dates = ["20170901", "20170902", "20170903", "20170904", "20170905", "20170906", "20170907"]
        # dates = ["20170908", "20170909", "20170910", "20170911", "20170912", "20170913", "20170914"]
        dates = ["20170901", "20170902", "20170903", "20170904", "20170905", "20170906", "20170907", "20170908",
                 "20170909", "20170910", "20170911", "20170912", "20170913", "20170914"]

        period_merger(customer, dates=dates)

    if check == 5:
        threshold = 7
        clusters = 9
        preprocessing = "Original"
        path = sup.GroupingPath(customer, period, "ACL", preprocessing, threshold)
        path.set_clusters(clusters)
        dataframe = read_CSV_file(path.log_file())
        dataframe = dataframe[dataframe.Resource != "/multiple"]
        dataframe = dataframe[dataframe.Id.notnull()]
        dataframe = df_id_filter(dataframe)
        dataframe = insignificant_user_remover(dataframe,
                                               threshold=threshold)  # remove users that have less action than a certain threshold.
        arr_features = dataframe.columns.values
        path.set_features(",".join(np.delete(arr_features, np.argwhere(arr_features == "Username"))))

        dismatrix = miner.matrix_generator(dataframe, preprocessing, "rmse")
        groupings = miner.acl_username_grouper(dismatrix, clusters)

        get_mean(dataframe, path=path, user_groupings=groupings)
        groupings.to_csv(path.destination("data", "Csv"))


def get_mean(dataframe, path, user_groupings):
    """

    :param dataframe:
    :type dataframe: dataframe
    :param path:
    :type path: class
    :param user_groupings:
    :type user_groupings: dict
    :return:
    :rtype:
    """
    print("getting mean")
    dummied_data = miner.data_processor(dataframe, path.preprocessing)
    with open(path.destination("stats", "txt"), 'w') as f:

        for group in user_groupings["Group"].unique():
            group_data = dummied_data.loc[user_groupings["Group"] == group]
            group_stats = pd.concat([group_data.mean().to_frame("Mean").round(3).replace(0.0, " "),
                                     group_data.std().to_frame("Std").round(3).replace(0.0, " "),
                                     group_data.max().to_frame("Max").round(3).replace(0.0, " "),
                                     group_data.min().to_frame("Min").round(3)], axis=1)
            group_stats = group_stats
            f.write("Group: " + str(group)+"\n")
            f.write(group_stats.to_string()+"\n\n")
    f.close()

filemanager(5, 2, 0)
