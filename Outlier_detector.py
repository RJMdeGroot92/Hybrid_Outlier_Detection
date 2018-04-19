import pandas as pd
import numpy as np
from pylab import *
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import Datamining as miner
from datetime import datetime
from Support.Exceptions import *
import Support.settings as sup
from Support.reader import detection_data_reader
from Support.reader import id_data_reader
from Support.reader import read_CSV_file

detectionDir = os.getcwd()+"/logs/detection"


def single_user_gen(df):
    """
    Generates a dataframe of a single user
    :param df:
    :type df: dataframe
    :return:
    :rtype: dataframe
    """
    users_count = df["Username"].value_counts()
    top_10 = users_count.head(10)
    user = top_10.index.values[2]
    user_df = df[df["Username"] == user]
    user_df = user_df.drop("Username", 1)
    return user_df


def single_user_df_gen(df, user):
    """
    Generates a dataframe of al request made by a single user
    :param df:
    :type df: dataframe
    :param user:
    :type user: str
    :return:
    :rtype:
    """
    user_df = df[df["Username"] == user]
    user_df = user_df.drop("Username", 1)
    return user_df


def groupdata_generator(path):
    """
    generates a dataframe to be used for analysing an expertise
    :param path:
    :type path: class
    :return:
    :rtype:
    """
    pd.set_option('display.width', 1000)
    description = path.expertise
    shield_data = pd.read_csv(path.expertise_file())
    users = shield_data[shield_data["description"] == description]["MD5(u.username)"].values
    dataframe = read_CSV_file(path.log_file())
    dataframe = dataframe[dataframe["Username"].isin(users)]
    file_name = path.customer+"_"+str(path.period)+"_"+path.expertise
    sup.folder_maker(detectionDir)
    dataframe.to_csv(detectionDir+"/"+file_name+".gz", compression='gzip', index=False)


def single_user_analytics(user_df):
    """
    analyses the data of a user using the IsolationForest algorithm and returns outliers.
    :param user_df: dataframe should only have the columns: Method, Path, Id, Resource, Status, Src
    :type user_df: dataframe
    :return:
    :rtype: Series
    """
    print('analyzing')
    converted_user_df = user_df.drop("Date", 1)
    converted_user_df = pd.get_dummies(converted_user_df, columns=["Method", "Path", "Id", "Resource", "Status", "Src"])
    clf = IsolationForest(contamination=0, max_features=len(converted_user_df.columns.values))
    clf.fit(converted_user_df)
    pred = clf.predict(converted_user_df)
    temp_df = user_df
    temp_df['e'] = pd.Series(pred, user_df.index)
    novelty = user_df.loc[temp_df["e"] == -1]
    return novelty.index.values


def single_user_analytics2(user_df):
    """
    analyses the data of a user using the IsolationForest algorithm and returns outliers.
    using alternative input
    :param user_df: dataframe should only have the columns: MPR, Id, Status, Src
    :type user_df: dataframe
    :return:
    :rtype:
    """
    print('analyzing')
    converted_user_df = user_df.drop("Date", 1)
    converted_user_df = pd.get_dummies(converted_user_df,columns=["MPR", "Id", "Status", "Src"])
    clf = IsolationForest(contamination=0, max_features=len(converted_user_df.columns.values))
    clf.fit(converted_user_df)
    pred = clf.predict(converted_user_df)
    unique, counts = np.unique(pred, return_counts=True)
    temp_df = user_df
    temp_df['e'] = pd.Series(pred, user_df.index)
    novelty = user_df.loc[temp_df["e"] == -1]
    return novelty.index.values


def multi_step_analytics(df):
    """
    Analyses multiple users to find deviant users
    :param df:
    :type df: dataframe
    :return:
    :rtype: dictionary
    """
    print("Starting multi_step")
    customer_data = miner.merge_path_method_resource(df)
    train_data = customer_data.drop("Date", 1)
    train_data = train_data.drop("Username", 1)
    train_data = train_data.drop("Id", 1)
    dummy_data = pd.get_dummies(train_data,columns=["MPR", "Status", "Src"])
    clf = IsolationForest(contamination=0, max_features=len(dummy_data.columns.values))
    clf.fit(dummy_data)
    user_count = df["Username"].value_counts()
    deviant_users = {}
    for user in user_count.index.values:
        sus_actions = single_user_analytics(single_user_df_gen(df, user))
        lines = customer_data.loc[sus_actions]
        pred = clf.predict(dummy_data.loc[sus_actions])
        if -1 in pred:
            pred_df = lines
            pred_df["pred"] = pd.Series(pred, lines.index)
            novelty = pred_df.loc[pred_df["pred"] == -1]
            actions_dict = action_dict_maker(novelty)
            deviant_users[user] = actions_dict
    print(deviant_users)
    return deviant_users


def action_dict_maker(df):
    """
    generates a dictionary of all requests in the given dataframe
    :param df:
    :type df: dataframe
    :return:
    :rtype: dictionary
    """
    actions_dict = {}
    action_count = df['MPR'].value_counts()
    for action in action_count.index.values:
        count = action_count[action]
        action_df = df.loc[df["MPR"] == action]
        different_users = len(action_df["Id"].index.values)
        status = action_df["Status"].value_counts().index.values
        status = ','.join(str(e) for e in status)
        action_data = {"count": count, "different_users": different_users, "status": status}
        actions_dict[action] = action_data
    return actions_dict


def multi_user_analytics(df):
    """
    Analyses multiple users for suspicious behaviour
    this function has a high risk of making the system run out of memory
    :param df:
    :type df: dataframe
    :return:
    :rtype:
    """
    print("analyzing")
    df = miner.merge_path_method_resource(df)
    df2 = df.drop("Date", 1)
    df2 = df2.drop("Username", 1)
    df2 = df2.drop("Id", 1)

    df3 = pd.get_dummies(df2, columns=["MPR", "Status", "Src"])
    clf = IsolationForest(contamination=0, max_features=len(df3.columns.values))
    clf.fit(df3)
    pred = clf.predict(df3)
    temp_df = df
    temp_df['e'] = pd.Series(pred, df.index)
    novelty = df.loc[temp_df["e"] == -1]
    vcu = novelty['Username'].value_counts()
    sus_users = {}
    for User in vcu.index:
        user_actions_data = {}
        user_df = novelty.loc[novelty["Username"] == User]
        uvc = user_df['MPR'].value_counts()
        user_actions = uvc.index.values
        for action in user_actions:
            count = uvc[action]
            print(count)
            action_df = user_df.loc[user_df["MPR"] == action]
            different_users = len(action_df["Id"].index.values)
            status = action_df["Status"].value_counts().index.values
            status = ','.join(str(e) for e in status)
            action_data = {"count": count, "different_users": different_users, "status": status}
            user_actions_data[action] = action_data
        sus_users[User] = user_actions_data

    return sus_users


def id_encoder(user_df):
    """
    encodes id data in a numpy array
    :param user_df:
    :type user_df: dataframe
    :return:
    :rtype: list[np.array, Series, Series]
    """
    unique_clients = user_df["Id"].unique()
    unique_users = user_df["Username"].unique()
    iter = 0
    client_dict = {}
    for client in unique_clients:
        client_dict[client]=iter
        iter += 1
    print("client dict is made!")
    iter = 0
    user_dict = {}
    for user in unique_users:
        user_dict[user] = iter
        iter += 1
    print("user_dict is made!")
    print("amount of unique clients: ", len(unique_clients))
    print("amount of unique users:", len(unique_users))
    array = np.zeros(shape=(len(unique_users), len(unique_clients)), dtype=np.int8)

    iter = 0
    for index, row in user_df.iterrows():
        iter += 1
        array[user_dict[row["Username"]], client_dict[row["Id"]]] = 1
        if iter == 300000:
            print(datetime.now())
            iter = 0
    return array, unique_clients, unique_users


def id_encoder2(user_df):
    """
    encodes the data but returns a dataframe
    :param user_df:
    :type user_df: dataframe
    :return:
    :rtype: dataframe
    """
    unique_clients = user_df["Id"].unique()
    unique_users = user_df["Username"].unique()
    print("amount of unique clients: ", len(unique_clients))
    print("amount of unique users:", len(unique_users))
    print("pre_dummy:", datetime.now())
    dummy_df = pd.get_dummies(user_df, columns=["Id"])
    print("pre_grouped_df:", datetime.now())
    grouped_df = dummy_df.groupby('Username', group_keys=False, sort=False).agg("max")
    print("pre_column renaming:", datetime.now())
    new_col = [item[3:] for item in grouped_df.columns.values]
    grouped_df.columns = new_col
    return grouped_df


def principal_component_analyser(values, components=2):
    """
    perform principal components analysis and reduces the data
    :param values:
    :type values:
    :param components: the amount of components that are left after reduction 
    :type components: int
    :return:
    :rtype:
    """
    pca = PCA(n_components=components)
    pca.fit(values)
    pca_matrix = pca.transform(values)
    return pca_matrix


def client_remover(path, threshold=1):
    """
    returns a list consisting of three np.array
    :param path:
    :type path:
    :param threshold:
    :type threshold:
    :return:
    :rtype: list
    """
    arraytrain = np.load(path.origin("data"))
    onecheck = np.where((arraytrain.sum(axis=0) <= threshold))
    print("original users: ", len(arraytrain))
    print("original clients: ", len(arraytrain[1]))
    cleanarray = np.delete(arraytrain, onecheck, axis=1)
    clients = pd.read_csv(path.origin("clients"), delimiter='\n', header=None, names=["clients"])
    users = pd.read_csv(path.origin("users"), delimiter='\n', header=None, names=["users"])
    cleaned_clients = clients.drop(onecheck[0]).reset_index(drop=True)
    users_check = np.where((cleanarray.sum(axis=1)) < 1)
    cleaned_users = users.drop(users_check[0]).reset_index(drop=True)
    cleanarray = np.delete(cleanarray, users_check[0], axis=0)
    lenusers = len(arraytrain) - len(cleanarray)
    lenclients = len(arraytrain[1]) - len(cleanarray[1])
    print("Users removed: ", lenusers)
    print("Clients removed: ", lenclients)
    print("len users: ", len(cleanarray))
    print("Len clients: ", len(cleanarray[1]))
    return cleanarray, cleaned_clients, cleaned_users


def reporter(report, filename, type=""):
    """
    depreciated function that creates a textfile with the given input was used to report results
    :param report: the input
    :type report:
    :param filename:
    :type filename: str
    :param type:
    :type type: str
    :return:
    :rtype:
    """
    print("reporting")
    report_dir = os.getcwd()+'/'+"reports"
    print(report_dir)
    sup.folder_maker(report_dir)
    with open(report_dir + "/" + filename[len(detectionDir+'/'):-3]+'_'+type, 'w') as f:
        for user in report:
            print(user)
            f.write("User: "+user+"\n")
            for action in report[user]:
                action_data = report[user][action]
                actionline = "        Occurences: %(count)s\n        Different_ids_requested: %(different_users)s\n  " \
                             "      returned_Statuses: %(status)s\n" % action_data
                f.write("   Action:"+action+'\n'+actionline+'\n')
    f.close()


def id_data_writer(path, info_df, sorted_big_groups):
    """
    generates a file that can be used to analyse the results of id_data analysing
    :param path:
    :type path: class
    :param info_df:
    :type info_df: dataframe
    :param sorted_big_groups:
    :type sorted_big_groups: dict
    :return:
    :rtype:
    """
    bg_name = "bf:" + str(info_df["Branching factor"].values[0]) + "_T:" + str(info_df["Threshold"].values[0]) \
              + "G-a:" + str(info_df["Group-amount"].values[0])
    if os.path.isfile(path.destination("csv", "data")):
        old_df = pd.read_csv(path.destination("csv", "data"),)
        info_df = old_df.append(info_df)
        info_df.to_csv(path.destination("csv", "data"), index=False)
    else:
        info_df.to_csv(path.destination("csv", "data"), index=False)
    with open(path.destination("txt", "text"), "w") as f:
        f.write(info_df.to_string(index=False))
    f.close()

    with open(path.destination("txt", bg_name), "w") as out_f:
        for bg_key in sorted(sorted_big_groups, reverse=True):
            out_f.write("Group-size: "+str(bg_key)+"\n")
            out_f.write(sorted_big_groups[bg_key].to_string(index=True))
            out_f.write("\n\n")
    out_f.close()


def similarity_finder(path, labels, users, info_df=None):
    """
    finds similarity between users.
    :param path:
    :type path: path
    :param labels:
    :type labels: list
    :param users:
    :type users: list
    :param info_df:
    :type info_df: dataframe
    :return:
    :rtype:
    """
    import Support.reader as reader
    bg_name = "test"
    if info_df is not None:
        bg_name = "bf:" + str(info_df["Branching factor"].values[0]) + "_T:" + str(
            info_df["Threshold"].values[0]) + "G-a:" +\
                  str(info_df["Group-amount"].values[0])
    uni_cl = np.unique(labels, return_counts=True)
    selected_nums = np.where(np.logical_and(uni_cl[1] >= 10, uni_cl[1] <= 50))[0]
    user_data = {}
    group_data = {}
    for i in range(len(np.unique(labels))):
        group_data[i] = []
    for i in range(len(labels)):
        user_data[users.get_value(i, "users")] = labels[i]
        group_data[labels[i]].append(users.get_value(i, "users"))

    df = reader.read_parsed(path.log_file())
    expertise_df = pd.read_csv(path.expertise_file())
    expertise_df.columns = ["Username", "Group"]
    for group in selected_nums:
        selected_num = group
        selected_users = group_data[selected_num]
        print("Amount of selected users:", selected_users)
        selected_expertise_df = expertise_df.loc[expertise_df["Username"].isin(selected_users)]
        count_selected_expertise_users = len(selected_expertise_df.index)
        selected_expertise_df = selected_expertise_df["Group"].value_counts(normalize=True)*100
        gdf = df.loc[df["Username"].isin(selected_users)]
        count = gdf[(gdf['Path'].str.contains("client", case=False) |
                     gdf['Resource'].str.contains("client", case=False))].nunique()
        gdf = gdf.drop("Id", 1)
        gdf = gdf.drop("Date", 1)
        gdf = miner.merge_path_method_resource(gdf)
        column_names = gdf.columns.values.tolist()
        column_names.remove('Username')
        result_df = pd.get_dummies(gdf, columns=column_names, prefix="", prefix_sep="") \
            .groupby(gdf["Username"])
        result_df_sum = result_df.sum()
        result_df_agg = result_df.agg("max").drop("Username", 1)
        if len(result_df_sum.index) != count_selected_expertise_users:
            print("users in group: ", len(result_df_sum.index))
            print("users in expertise_group: ", count_selected_expertise_users)
            print("Missing " + str(len(result_df_sum.index) - count_selected_expertise_users) + " users!")
        with open(path.similarity_destination("txt", bg_name+"_similarities", str(group)), "w") as out_f:
            out_f.write("Group_size="+str(len(result_df_sum.index))+"\n")
            out_f.write("Amount of ClientIds: " + str(count["Id"])+"\n")
            out_f.write("Expertise_division:\n")
            out_f.write(selected_expertise_df.to_string()+"\n")
            out_f.write("Group_summary:\n")
            out_f.write(result_df_agg.sum().apply(lambda x: (int(x)/len(result_df_sum.index))*100).
                        sort_values(ascending=False).to_string())
            out_f.write("\n\nUser_information\n")
            for index, row in result_df_sum.iterrows():
                out_f.write("user: "+index+"\n")
                out_f.write(row.iloc[row.nonzero()].sort_values(ascending=False).to_string()+"\n\n")


def manager(check, customer=1, periodid=0, optional=[50, 7]):
    """
    Overarching manager of functions
    :param check:
    :type check:
    :param customer:
    :type customer:
    :param periodid:
    :type periodid:
    :param optional:
    :type optional:
    :return:
    :rtype:
    """
    import Support.settings as sup
    pd.set_option('display.width', 1000)
    if check == 1:
        path = sup.DetectionPath(customer, periodid, expertise_id=str(0))
        detec = detection_data_reader(path=path.origin())
        su_detec = single_user_gen(detec)
        print(single_user_analytics(su_detec))
    elif check == 2:
        path = sup.DetectionPath(customer, periodid, expertise_id=str(0))
        detec = detection_data_reader(path=path.origin())
        report = multi_user_analytics(detec)
        reporter(report, path.filename(), type='MUA')
    elif check == 3:
        path = sup.DetectionPath(customer, periodid, expertise_id=str(0))
        detec = detection_data_reader(path=path.origin())
        su_detec = single_user_gen(detec)
        print(su_detec)
    elif check == 4:
        path = sup.DetectionPath(customer, periodid, analyzing=True)
        print(path.origin())
        clean_array, cleaned_clients, cleaned_users = client_remover(path, threshold=10)
        branching_factor = optional[0]
        threshold = optional[1]
        print(clean_array.dtype)
        clean_array = clean_array.astype(int)
        print(clean_array.dtype)
        labels, centroids, algo, transf = miner.birch_clustering(clean_array, branching_factor, threshold)
        print("len labels: ", len(labels))
        print("len unique labels:", len(np.unique(labels)))
        print("len centroids:", len(centroids))
        print("len array: ", len(clean_array))
        print("len array[1]: ", len(clean_array[1]))
        print("size max group", np.amax(np.unique(labels, return_counts=True)[1]))
        print("size group 3: ", np.unique(labels, return_counts=True)[1][3])
        print(len(np.argwhere(labels == 3)))
        dict = {}
        for i in np.argwhere(labels == 0):
            print(cleaned_users.get_value(i[0], "users"))
            print(clean_array[i[0]])
            dict[cleaned_users.get_value(i[0], "users")] = clean_array[i[0]]
        print(dict)
        print(cleaned_clients["clients"].values)
        print("kaas")
        dataf = pd.DataFrame.from_dict(dict, orient="index")
        dataf.columns = cleaned_clients["clients"].values
        dataf = dataf.loc[:, (dataf != 0).any(axis=0)]
        print(dataf)
        sumdf = dataf.sum()
        print(sumdf)

        print(sumdf[sumdf == 1].sum())
    elif check == 5:
        path = sup.DetectionPath(customer, periodid, expertise_id=str(0))
        detec = detection_data_reader(path.origin())
        multi_step = multi_step_analytics(detec)
        reporter(multi_step, path.filename(), type='MSA')
    elif check == 6:
        path = sup.IdPath(customer, period=periodid, analyzing=True)
        clean_array, cleaned_clients, cleaned_users = client_remover(path, threshold=1)
        arraytrain = clean_array.astype(int)
        branching_factor = optional[0]
        threshold = optional[1]
        labels, centroids, algo, transf = miner.birch_clustering(arraytrain, branching_factor,
                                                                 threshold)
        print("length-trainingset: ", len(arraytrain))
        print("length-labels: ", len(labels))
        print("unique_labels: ", len(np.unique(labels)))
        uni_cl = np.unique(labels, return_counts=True)
        report = [len(arraytrain), branching_factor, threshold, len(np.unique(labels)), np.amax(uni_cl[1])]
        user_data = {}
        client_data = {}
        data_dict = {}
        for i in np.unique(labels):
            data_dict[i] = {}
        clients = cleaned_clients
        users = cleaned_users
        for i in range(len(labels)):
            user_data[users.get_value(i, "users")] = labels[i]
            if labels[i] not in data_dict:
                data_dict[labels[i]] = {}
            for cid in argwhere(arraytrain[i] == 1):
                client = clients.get_value(cid[0], "clients")
                if client in client_data:
                    client_data[client] += 1
                else:
                    client_data[client] = 1
                if client not in data_dict[labels[i]]:
                    data_dict[labels[i]][client] = 1
                else:
                    data_dict[labels[i]][client] += 1
        # data_dict= a dictionary per group. in the dictioary of a group is a dict of clients and
        #  amount of occurences in tha group.
        data = data_dict
        df = pd.DataFrame.from_dict(data=data, orient='index')
        argsorted = np.argsort(uni_cl[1])
        biggest_numbers = [argsorted[len(argsorted)-1],argsorted[len(argsorted)-2], argsorted[len(argsorted)-3]]
        sorted_big_groups = {uni_cl[1][biggest_numbers[0]]: df.loc[biggest_numbers[0]].dropna().
            sort_values(ascending=False), uni_cl[1][biggest_numbers[1]]:df.loc[biggest_numbers[1]].dropna().
            sort_values(ascending=False), uni_cl[1][biggest_numbers[2]]: df.loc[biggest_numbers[2]].dropna().
            sort_values(ascending=False)}
        print("Unique_clients:", len(clients["clients"].unique()))
        iteronce = 0
        itertwice = 0
        itertest = 0
        for key, value in client_data.items():
            if value == 1:
                iteronce += 1
            if value == 2:
                itertwice += 1
            if value >= 3:
                itertest += 1
        # print("clients that occur only once:",iteronce)
        # print("clients that occur only twice", itertwice)
        # print("clients that are viewed by 3 or more users:", itertest)
        # print("user_data:",len(user_data))
        columns = ["Users", "Branching factor", "Threshold", "Group-amount", "Biggest-User-group", "Unique_clients",
                   "Once-occurring clients", "Twice-occurring clients"]
        report.append(len(clients["clients"].unique()))
        report.append(iteronce)
        report.append(itertwice)
        print(str(report))
        data_df = pd.DataFrame(data=[report], index=None, columns=columns)
        id_data_writer(path, data_df, sorted_big_groups)
        similarity_finder(path, labels, users, data_df)
    elif check == 7:
        path = sup.DetectionPath(customer, period=periodid)
        print(path.description())
        df = id_data_reader(path.origin())

        mode = "old"
        if mode == "old":
            data, clients, users = id_encoder(df)
            np.save(path.destination(data_type="data"), data)
            np.savetxt(path.destination(data_type="clients"), clients, fmt="%s", delimiter=',')
            np.savetxt(path.destination(data_type="users"), users, fmt="%s", delimiter=',')
        if mode == "new":
            df = id_encoder2(df)
            df.to_csv(path.destination("csv"))
    elif check == 8:
        path = sup.IdPath(customer, period=periodid, analyzing=True)
        print(path.origin())
        clean_array, cleaned_clients, cleaned_users = client_remover(path, threshold=1)
        arraytrain = clean_array.astype(int)
        print(arraytrain.dtype)
        branching_factor = optional[0]
        threshold = optional[1]
        labels, centroids, algo, transf = miner.birch_clustering(arraytrain, branching_factor,
                                                                 threshold)
        print("length-trainingset: ", len(arraytrain))
        print("length-labels: ", len(labels))
        print("unique_labels: ", len(np.unique(labels)))

        similarity_finder(path, labels, cleaned_users)
    elif check == 9:
        path = sup.DetectionPath(customer, periodid, expertise_id=str(0))
        groupdata_generator(path)

manager(5, 1)
