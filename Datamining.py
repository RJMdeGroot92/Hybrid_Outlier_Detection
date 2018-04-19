from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as ACL
import pandas as pd
from pylab import *
from matplotlib import style
from sklearn import preprocessing
from sklearn.cluster import Birch

markers = ['D', 'v', '^', '<', "o", "*", "8", "s", 'p']
colors3d = ["g", "r", "b", "c", "m", "y", "k", "b"]


def dissimilarity_matrix(processed_data):
    """
    Transforms the data into a dissimilarity matrix

    :param processed_data:
    :type processed_data:
    :return:
    :rtype: dataframe
    """
    print("unique usernames: k", len(processed_data.index))
    ds_array = []
    for index1, row in processed_data.iterrows():
        rmse1 = rmse(np.array(processed_data.values), np.array(row))
        ds_array.append(rmse1)
    dissimilarity_df = pd.DataFrame(ds_array, columns=processed_data.index, index=processed_data.index)
    return dissimilarity_df


def vector_normalize(dataframe):
    """
    Normalization method based on the unit vector

    :param dataframe:
    :type dataframe:
    :return: normalized version of input
    :rtype: dataframe
    """
    df = dataframe
    processed_data = pd.DataFrame({})
    for column in df:
        if column != "Username":
            temp_df = pd.get_dummies(df, columns=[column])
            grouped_df = temp_df.groupby("Username").sum()
            normal = preprocessing.normalize(grouped_df, axis=1)
            temp_df = pd.DataFrame(normal, columns=grouped_df.columns, index=grouped_df.index)
            processed_data = pd.concat([processed_data, temp_df], axis=1)
    return processed_data


def wout_normalize(dataframe):
    """
    Normalization method done by squaring every variable by 1/n
    :param dataframe:
    :type dataframe:
    :return:
    :rtype: dataframe
    """
    pd.set_option('display.width', 1000)
    df = dataframe
    processed_data = pd.DataFrame({})
    for column in df:
        if column != "Username":
            temp_df = pd.get_dummies(df, columns=[column])
            grouped_df = temp_df.groupby("Username").sum()
            normal = grouped_df.apply(power, axis=1)
            temp_df = pd.DataFrame(normal, columns=grouped_df.columns, index=grouped_df.index)
            processed_data = pd.concat([processed_data, temp_df], axis=1)
    return processed_data


def power(x):
    """
    returns x^(1/50)
    :param x:
    :type x: int
    :return:
    :rtype: int
    """
    return np.power(x, (1/50))


def binary_processing(df):
    """
    Normalizes input by making it binary

    :param df:
    :type df: dataframe
    :return:
    :rtype: dataframe
    """
    column_names = df.columns.values.tolist()
    column_names.remove("Username")
    result = pd.DataFrame({})
    for column in column_names:
        col = pd.get_dummies(df[column], prefix=None) \
            .groupby(df["Username"]) \
            .agg("max")
        result = pd.concat([result, col], axis=1)
    return result


def minmax_scale(dataframe):
    """
    Normalizes input using a minmax scaler
    :param dataframe:
    :type dataframe:
    :return:
    :rtype: dataframe
    """
    df = dataframe
    processed_data = pd.DataFrame({})
    for column in df:
        if column != "Username":
            temp_df = pd.get_dummies(df, columns=[column])
            grouped_df = temp_df.groupby("Username").sum()
            normal = preprocessing.minmax_scale(grouped_df, axis=1)
            temp_df = pd.DataFrame(normal, columns=grouped_df.columns, index=grouped_df.index)
            processed_data = pd.concat([processed_data, temp_df], axis=1)
    return processed_data


def standard_scalar(dataframe):
    """
    normalizes the input by removing the mean and scaling it to unit variance
    :param dataframe:
    :type dataframe:
    :return:
    :rtype: dataframe
    """
    df = dataframe
    processed_data = pd.DataFrame({})
    for column in df:
        if column != "Username":
            temp_df = pd.get_dummies(df, columns=[column])
            grouped_df = temp_df.groupby("Username").sum()
            normal = preprocessing.scale(grouped_df, axis=1)
            temp_df = pd.DataFrame(normal, columns=grouped_df.columns, index=grouped_df.index)
            processed_data = pd.concat([processed_data, temp_df], axis=1)
    return processed_data


def data_processor(dataframe, preprocessing="Original"):
    """
    Determines how the input should be preprocessed
    :param dataframe:
    :type dataframe: dataframe
    :param preprocessing: determines the type of preprocessing to be done
    :type preprocessing: str
    :return:
    :rtype: dataframe
    """
    processed_data = "-"
    if preprocessing == "Original":
        processed_data = vector_normalize(dataframe)
    elif preprocessing == "Minmax_scale":
        print("percentage_normalize")
        processed_data = minmax_scale(dataframe)
    elif preprocessing == "Standard_scalar":
        processed_data = standard_scalar(dataframe)
    elif preprocessing == "Wout":
        processed_data = wout_normalize(dataframe)
    elif preprocessing == "Bin_Proc":
        processed_data = binary_processing(dataframe)
    return processed_data


def matrix_generator(dataframe, preprocessing="Original", type="rmse"):
    """
    transform the input into a dissimilarity matrix after preprocessing
    :param dataframe:
    :type dataframe: dataframe
    :param preprocessing: determines the type of preprocessing to be done
    :type preprocessing: str
    :param type:
    :type type: str
    :return: a dissimilarity matrix
    :rtype: dataframe
    """
    processed_data = data_processor(dataframe, preprocessing)
    if type == "rmse":
        return dissimilarity_matrix(processed_data)
    elif type == "new":
        print("new")


def merge_path_method_resource(dataframe):
    """
    Merges the variables path, method and resource into mpr, removes the old columns in the dataframe
    :param dataframe: a dataframe with the columns Method, Path, Resource
    :type dataframe:
    :return: Dataframe with column MPR and without Method, Path, Resource
    :rtype: dataframe
    """
    dataframe = dataframe.fillna(value=" ", axis=1)
    print(dataframe.columns)
    dataframe["MPR"] = dataframe["Method"]+"_"+dataframe["Path"]+"_"+dataframe["Resource"]
    dataframe = dataframe.drop("Method", 1)
    dataframe = dataframe.drop("Resource", 1)
    dataframe = dataframe.drop("Path", 1)
    return dataframe


def kmeans_clustering(values, clusters=8):
    """
    Clusters input using Kmeans
    :param values:
    :type values:
    :param clusters: the amount of clusters to be made
    :type clusters: int
    :return: returns the Kmeans class
    :rtype: class
    """
    X = values
    clusters = clusters
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)
    return kmeans


def acl_clustering(values, clusters=8):
    """
    Clusters input using ACL
    :param values:
    :type values: dataframe
    :param clusters: the amount of clusters to be made
    :type clusters: int
    :return: returns the ACL class
    :rtype: class
    """
    Acl = ACL(n_clusters=clusters+1, linkage='complete').fit(values)
    print("ACL", Acl)
    print("label", Acl.labels_)
    print("leaves", Acl.n_leaves_)
    print("components", Acl.n_components_)
    print("children", Acl.children_)
    return Acl


def birch_clustering(values, branching_factor=50, threshold=0.5):
    """
    Clusters input using the birch algorithm
    :param values:
    :type values:
    :param branching_factor:
    :type branching_factor: int
    :param threshold: treshold, default=0.5 this is very high
    :type threshold: int
    :return: return list[labels, centroids, class, fitted class]
    :rtype: list
    """
    birchc = Birch(branching_factor=branching_factor, n_clusters=None, threshold=threshold, compute_labels=True)
    x_new = birchc.fit_transform(values)
    labels = birchc.labels_
    subc_centroids = birchc.subcluster_centers_
    return [labels, subc_centroids, birchc, x_new]


def kmeans_plotting2d(kmeans, values):
    """
    Depreciated 2d plotting function that makes a visualisation after clustering the input using kmeans
    :param kmeans:
    :type kmeans:
    :param values:
    :type values:
    :return:
    :rtype:
    """
    print("PLOTTING")
    style.use("ggplot")
    plt.figure()
    X = values
    colors2d = ["g.", "r.", "b.", "c.", "m.", "y.", "k.", "b.","g.", "r.", "b.", "c.", "m.", "y.", "k.", "b.","g.", "r.", "b.", "c.", "m.", "y.", "k.", "b.","g.", "r.", "b.", "c.", "m.", "y.", "k.", "b.","g.", "r.", "b.", "c.", "m.", "y.", "k.", "b.","g.", "r.", "b.", "c.", "m.", "y.", "k.", "b.","g.", "r.", "b.", "c.", "m.", "y.", "k.", "b.","g.", "r.", "b.", "c.", "m.", "y.", "k.", "b.","g.", "r.", "b.", "c.", "m.", "y.", "k.", "b.","g.", "r.", "b.", "c.", "m.", "y.", "k.", "b."]
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    for i in range(len(X)):
        print("coordinate:", X[i], "label:", labels[i], "colour:",colors2d[labels[i]])
        plt.plot(X[i][0], X[i][1], colors2d[labels[i]], markersize=10)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=150, linewidths=5, zorder=10)
    plt.gray()
    plt.show()


def kmeans_plotting3d(kmeans, values, grouped_df):
    """
    Depreciated 2d plotting function that makes a visualisation after clustering the input using kmeans
    :param kmeans:
    :type kmeans:
    :param values:
    :type values:
    :param grouped_df:
    :type grouped_df:
    :return:
    :rtype:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = values
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    for i in range(len(X)):
        print("coordinate:", X[i], "label:", labels[i])
        ax.scatter(X[i][0], X[i][1], X[i][2], c=colors3d[labels[i]], marker=markers[labels[i]])

    for i in range(len(centroids)):
        ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2], marker="x", s=150, c=colors3d[i])
    ax.set_xlabel(grouped_df.columns[0])
    ax.set_ylabel(grouped_df.columns[1])
    ax.set_zlabel(grouped_df.columns[2])
    plt.show()


def rmse(pred_array, target):
    """
    Calculates the rmse of the input:
    Documentation of behaviour of np.arrays
        pred=np.array([[0,0,1,0],[0,0,0,1],[0,0,0,1]])
        target=np.array([1,0,0,1])
        (pred-target)=[[-1  0  1 -1],[-1  0  0  0],[-1  0  0  0]]
    :param pred_array:
    :type pred_array: mp.array
    :param target:
    :type target: np.array
    :return:
    :rtype: np.array
    """
    array = []
    for i in range(len(pred_array)):
        array.append(np.sqrt(((pred_array[i] - target) ** 2).mean()))
    return array


def acl_username_grouper(data, clusters=8):
    """
    Groups users on similarity by using the ACL clustering algorithm
    :param data:
    :type data:
    :param clusters:
    :type clusters: int
    :return: list of dataframe, int
    :rtype:list
    """
    import scipy.cluster.hierarchy as sch
    y = sch.linkage(data.values, method='complete')
    t = sch.fcluster(y, clusters, criterion='maxclust')
    df = pd.DataFrame(t, index=data.index, columns=['Group'])
    sdf = df.sort_values('Group')
    return sdf


def kmeans_username_grouping(data, clusters=8):
    """
    Groups users on similarity by using the Kmeans clustering algorithm
    :param data:
    :type data: dataframe
    :param clusters:
    :type clusters: int
    :return: returns sorted datframe, that consists of rows for every user and a column Group that contains the group
                number for every user
    :rtype: dataframe
    """
    cluster = kmeans_clustering(data, clusters)
    labels = cluster.labels_
    print(data.index)
    df = pd.DataFrame(labels, index=data.index, columns=["Group"])
    sorted_df = df.sort_values('Group')
    return sorted_df
