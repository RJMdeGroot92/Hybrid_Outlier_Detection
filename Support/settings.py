import os
import glob
import errno
# These customers are place holders
customers = ["Cust1", "Cust2", "Cust3", "Cust4", "Cust5"]
expertise_users = {"Cust1": "expertise_cust1.csv", "Cust2": "expertise_cust2.csv",
                   "Cust3": "expertise_cust3.csv"}
expertises = {"Cust1": {}, "Cust2": {"0": "Verzorgende D"}, "Cust3": {"0": "Dietist"}}
periods = ["20170901", "20170902", "20170903", "20170901-20170907", "20170901-20170914", "20170901-20170925",
           "20170901-20170930", "20170908-20170914"]
# directories
# log dirs
parsedLogDir = os.getcwd()+"/logs/parsed"
# comparing dirs
comparer_dir = os.getcwd()+"/User_groupings"
comparisons_dir = comparer_dir+"/Comparisons"
expertiseDir = comparer_dir+"/deskundigheden"
# detection dirs
id_dataDir = os.getcwd()+"/id_data"
detectionDir = os.getcwd()+"/logs/detection"
reportDir = os.getcwd()+"/reports"


class Path(object):
    """
    Class that is used to store information about logs and other files.
    """

    def __init__(self, customer, period):
        self.customer = customers[customer]
        self.period = periods[period]

    def filename(self):
        pass

    def description(self):
        pass

    def log_file(self):
        return parsedLogDir+"/"+self.period+"/"+self.customer+".gz"

    def expertise_file(self):
        return "{}/{}".format(expertiseDir, expertise_users[self.customer])

    def destination(self, name):
        pass

    def extension_finder(self, extension_type=""):
        extensions = {"Png": ".png", "Csv": ".csv", "txt": ".txt", "gzip": ".gz", "gz": ".gz"}
        extension = ""
        if extension_type in extensions:
            extension = extensions[extension_type]
        return extension


class PatternPath(Path):
    """
    Extension of Path class used when generating patterns.
    """

    def __init__(self, customer, period):
        Path.__init__(self, customer, period)

    def filename(self):
        pass

    def destination(self, name):
        destination = reportDir+"/"+self.customer
        folder_maker(destination)
        destination = destination+"/"+self.period
        folder_maker(destination)
        destination = destination+"/"+name
        print(destination)
        return destination


class GroupPatternPath(PatternPath):
    """
    Class that is used to store information when generating patterns on groups
    """
    def __init__(self, customer, period, algorithm):
        PatternPath.__init__(self, customer, period)
        self.algorithm = algorithm
        self.filename = "*data.csv"

    def get_filename(self):
        if self.filename != "*data.csv":
            return self.filename[:-9]
        else:
            print("filename error!")

    def grouping_file(self):
        folder = "{}/{}/{}/{}".format(comparer_dir, self.customer, self.period, self.algorithm)
        file_place = "{}/{}".format(folder, self.filename)
        files = glob.glob(file_place)
        if len(files) > 1:
            file_iter = 0
            for file in files:
                print("{}:{}\n".format(file_iter, file))
                file_iter += 1
            check = input("which file? ")
            self.filename = os.path.basename(files[int(check)])
        elif len(files) == 1:
            self.filename = os.path.basename(files[0])
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_place)
        file_place = "{}/{}".format(folder, self.filename)
        return file_place


class InterestPatternPath(PatternPath):
    """
    Class that is used to store information when finding requests that have a low occurrence rate

    """
    def __init__(self, customer, period):
        PatternPath.__init__(self, customer, period)

    def destination(self, name):
        destination = reportDir + "/" + self.customer
        folder_maker(destination)
        destination = destination + "/" + self.period
        folder_maker(destination)
        destination = destination + "/" + name
        print(destination)
        return destination


class GroupingPath(Path):
    """
    Class that is used to store information when generating user-groupings.
    """

    def __init__(self, customer, period, algorithm, preprocessing, threshold):
        Path.__init__(self, customer, period)
        self.algorithm = algorithm
        self.preprocessing = preprocessing
        self.threshold = threshold
        self.features = None
        self.clusters = None

    def set_features(self, features):
        self.features = features

    def set_clusters(self, clusters):
        self.clusters = clusters

    def destination(self, name="", extension_type=""):
        extension = self.extension_finder(extension_type)
        destination = comparer_dir+"/"+self.customer
        folder_maker(destination)
        destination = destination+"/"+self.period
        folder_maker(destination)
        destination = destination+"/"+self.algorithm
        folder_maker(destination)
        destination = "{}/{}_{}_{}_T:{}_{}{}".format(destination, self.clusters, self.features, self.preprocessing,
                                                     self.threshold, name, extension)
        print(destination)
        return destination


class ComparePath(Path):
    """
    Class that is used to store information when generating visualisations of user-groupings.
    """

    def __init__(self, customer, period, algorithm):
        Path.__init__(self, customer, period)
        self.algorithm = algorithm
        self.filename = "*data.csv"

    def grouping_file(self):
        folder = "{}/{}/{}/{}".format(comparer_dir, self.customer, self.period, self.algorithm)
        file_place = "{}/{}".format(folder, self.filename)
        files = glob.glob(file_place)
        if len(files) > 1:
            file_iter = 0
            for file in files:
                print("{}:{}\n".format(file_iter, file))
                file_iter += 1
            check = input("which file? ")
            self.filename = os.path.basename(files[int(check)])
        elif len(files) == 1:
            self.filename = os.path.basename(files[0])
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_place)
        file_place = "{}/{}".format(folder, self.filename)
        return file_place

    def figure_destination(self, extension_type="Png"):
        extension = self.extension_finder(extension_type)
        destination = comparisons_dir+"/"+self.customer
        folder_maker(destination)
        destination = destination+"/"+self.period
        folder_maker(destination)
        destination = destination+"/"+self.algorithm
        folder_maker(destination)
        return "{}/{}{}".format(destination, self.filename[:-9], extension)


class IdPath(Path):
    """
    Class that is used to store information when evaluating groupings purely on ClientID
    """
    def __init__(self, customer, period, analyzing=False):
        Path.__init__(self, customer, period)
        self.analyzing = analyzing
        self.mode = "id"

    def filename(self):
        return self.customer

    def description(self):
        return self.period+"/"+self.customer

    def origin(self, data_type=""):
        if self.analyzing:
            if data_type == "data":
                data_type = "data.npy"
            return detectionDir+"/"+self.mode+"/"+self.period+"/"+self.customer+"/" + data_type
        else:
            return parsedLogDir+"/"+self.description()+".gz"

    def destination(self, extension_type="", data_type=""):
        extension = self.extension_finder(extension_type)
        if self.analyzing:
            destination = id_dataDir + "/" + self.period
            folder_maker(destination)
            destination = destination + "/" + self.customer
            folder_maker(destination)
            return destination + "/" + data_type + extension
        else:
            destination = detectionDir + "/" + self.mode
            folder_maker(destination)
            destination = destination + "/" + self.period
            folder_maker(destination)
            destination = destination + "/" + self.customer
            folder_maker(destination)
            return destination + "/" + data_type + extension

    def similarity_destination(self, extension_type="", data_type="", group=""):
        extension = self.extension_finder(extension_type)
        destination = id_dataDir + "/" + self.period
        folder_maker(destination)
        destination = destination + "/" + self.customer
        folder_maker(destination)
        destination = destination + "/" + data_type
        folder_maker(destination)
        return destination + "/" + group + extension


class ExpertisePath(Path):
    """
    Depreciated class used when analysing expertises of customers.
    """
    def __init__(self, customer, period, expertise_id=""):
        Path.__init__(self, customer, period)
        self.mode = "expertise"
        self.expertise = str(expertises[expertise_id])

    def filename(self):
        return self.expertise

    def description(self):
        return self.customer + "/" + self.period + "/" + self.expertise

    def origin(self):
        return detectionDir + "/" + self.customer + "_" + self.period + "_" + self.expertise + '.gz'

    def destination(self, extension_type="", data_type=""):
        extension = self.extension_finder(extension_type)
        destination = detectionDir + self.mode
        folder_maker(destination)
        destination = destination + "/" + self.period
        folder_maker(destination)
        destination = destination + "/" + self.customer
        folder_maker(destination)
        return destination + "/" + self.expertise + extension


class DetectionPath(Path):
    """
    Class that is used to store information when detecting outliers
    """

    def __init__(self, customer, period, expertise_id="", analyzing=False):
        Path.__init__(self, customer, period)
        self.analyzing = analyzing
        if expertise_id == "":
            self.mode = "id"
        else:
            self.mode = "expertise"
            self.expertise = str(expertises[self.customer][expertise_id])

    def filename(self):
        if self.mode == "id":
            return self.customer
        elif self.mode == "expertise":
            return self.expertise

    def description(self):
        if self.mode == "id":
            return self.period+"/"+self.customer
        else:
            return self.customer+"/"+self.period+"/"+self.expertise

    def origin(self, data_type=""):
        if self.mode == "id":
            if self.analyzing:
                if data_type == "data":
                    data_type = "data.npy"
                return detectionDir+"/"+self.mode+"/"+self.period+"/"+self.customer+"/" + data_type
            else:
                return parsedLogDir+"/"+self.description()+".gz"
        elif self.mode == "expertise":
            return detectionDir+"/"+self.customer+"_"+self.period+"_"+self.expertise+'.gz'

    def destination(self, extension_type="", data_type=""):
        # print("Generating destination...")
        if extension_type == "gzip":
            extension = ".gz"
        elif extension_type == "csv":
            extension = '.csv'
        else:
            extension = ""
        if self.mode == "id":
            if self.analyzing:
                destination = id_dataDir+"/"+self.period
                folder_maker(destination)
                destination = destination+"/"+self.customer
                folder_maker(destination)
                return destination+"/" + data_type + extension
            else:
                destination = detectionDir+"/"+self.mode
                folder_maker(destination)
                destination = destination+"/"+self.period
                folder_maker(destination)
                destination = destination+"/"+self.customer
                folder_maker(destination)
                return destination+"/" + data_type + extension
        if self.mode == "expertise":
            destination = detectionDir+self.mode
            folder_maker(destination)
            destination = destination+"/"+self.period
            folder_maker(destination)
            destination = destination+"/"+self.customer
            folder_maker(destination)
            return destination+"/"+self.expertise+extension


def folder_maker(path):
    """
       Creates a folder for the given path if it does not already exist
       :param path:
       :type path: str
       :return:
       :rtype:
       """
    if not os.path.exists(path):
        os.mkdir(path)
