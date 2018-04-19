import gzip
import hashlib
import re
import sys
from datetime import datetime

from Support.Exceptions import *
from Support.Weka_format import *


INFILE1 = "trueInput.gz"
OUTFILE1 = "OutputCSV.gz"
OUTFILE2 = "trueOutputOneClientCSV.gz"
OUTFILE3 = "test.gz"
customers1 = ["MF1267"]
customers = ["MF2023", "MF0874", "MF0122", "MF0058", "MF1955"]

# pattern that checks for 0 or more instances of \|
splitPat = re.compile(r'(.*?)\|(\w{3,6})\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*)')
# pattern that splits on /*
urlSplitPat = re.compile(r'/*')
# pattern that checks for multiple ids
multiples_split_pat = re.compile(r'id\[\]=(\d*)')
# url split construct /word/word/number/word
url_normal = re.compile(r'/(\w*/\w*)(/[0-9]*)(/\w*|$)')
# url split construct /word/word/word*/number
url_alt1 = re.compile(r'/(\w{2}/\w*)(/\w*|[\w*/\w*]*)/([0-9]+)')
# url split construct /word/word/word
url_alt2 = re.compile(r'/(\w{2}/\w*)([/\w*]*)$')
# url_split construct /word/word
url_alt3 = re.compile(r'/(\w{2})(/\w*)$')
# url split with Omaha care plan informatie. \w*/\w*/omaha/\w*/care_plan:[0-9]*:\w*
url_alt4 = re.compile(r'/(\w{2}/\w*)(/omaha/[\w*/]*):(\d*)[:\w.*?]*')
# url alternative with employee_id after the ?
url_alt5 = re.compile(r'/(\w{2})/(\w*)(/searchresult)?.*')
# url alternative with private
url_enc1 = re.compile(r'/(\w{2}/\w*)/(by_user_name)/(.*?)[\?]*')
# url alternative with private after ?
url_enc2 = re.compile(r'/(\w{2}/\w*)(/[\w*/]*?search_by_client)\?.*?id=([0-9]*).*')
# url alternative with private uuid. Think this is is actually hashed
url_enc3 = re.compile(r'/(\w{2}/\w*)(/by_uuid)/(.*)')
# \w{2}\w*?.*?client_id=[0-9]*.*
url_enc4 = re.compile(r'/(\w{2}/\w*)?.*?(client)_id=([0-9]*).*')
# url_alternative with multiples. only id[]=
url_multiple1 = re.compile(r'/(\w{2}/\w*)(/multiple)\?(id\[\]=[0-9]*.*)')
# dictionary for mapping usernames to hashmappings.
__UsernameDict = {"-": "-", "": ""}


def parseline(line):
    """
    split lines on pattern. adds appends \n at last attributes

    :param line:
    :type line: str
    :return: the line is split into the variables: date, method, uri, status, cic, src, username, QoS, Audit
    :rtype: dict
    """
    grouped_line = re.match(splitPat, line)
    parts = {"date": grouped_line.group(1), "method": grouped_line.group(2), "uri": grouped_line.group(3),
             "status": grouped_line.group(4), "cic": grouped_line.group(5), "src": grouped_line.group(6),
             "username": grouped_line.group(7), "QoS": grouped_line.group(8), "audit": grouped_line.group(9)}
    return parts


def parse_url(parts):
    """
    Parts is further updated by splitting Uri

    :param parts:
    :type parts: dict
    :return: Dictionary is update with: path, id, resource
    :rtype: dict
    """
    try:

        split_url = re.match(url_normal, parts["uri"])
        if split_url:
            parts.update({"path": split_url.group(1), "id": split_url.group(2), "resource": split_url.group(3)})
            parts = blind_id(parts)
        else:
            split_url = re.match(url_alt1, parts["uri"])
            if split_url:
                parts.update({"path": split_url.group(1), "id": split_url.group(3), "resource": split_url.group(2)})
                parts = blind_id(parts)
            else:
                split_url = re.match(url_alt2, parts["uri"])
                if split_url:
                    parts.update({"path": split_url.group(1), "id": "", "resource": split_url.group(2)})
                else:
                    split_url = re.match(url_alt3, parts["uri"])
                    if split_url:
                        parts.update({"path": split_url.group(1), "id": "", "resource": split_url.group(2)})
                    else:
                        split_url = re.match(url_enc1, parts["uri"])
                        if split_url:
                            parts.update({"path": split_url.group(1), "id": split_url.group(3),
                                          "resource": split_url.group(2)})
                            parts = blind_id(parts)
                        else:
                            split_url = re.match(url_multiple1, parts["uri"])
                            if split_url:
                                multiple_id = re.findall(multiples_split_pat, split_url.group(3))
                                blind_multiple = blind_multiple_id(multiple_id)
                                parts.update({"path": split_url.group(1), "id": blind_multiple,
                                              "resource": split_url.group(2)})

                            else:
                                split_url = re.match(url_alt4, parts["uri"])
                                if split_url:
                                    parts.update({"path": split_url.group(1), "id": split_url.group(3),
                                                  "resource": split_url.group(2)})
                                    parts = blind_id(parts)
                                else:
                                    split_url = re.match(url_enc2, parts["uri"])
                                    if split_url:
                                        parts.update({"path": split_url.group(1), "id": split_url.group(3),
                                                      "resource": split_url.group(2)})
                                        parts = blind_id(parts)
                                    else:
                                        split_url = re.match(url_enc3, parts["uri"])
                                        if split_url:
                                            parts.update({"path": split_url.group(1), "id": split_url.group(3),
                                                          "resource": split_url.group(2)})
                                        else:
                                            split_url = re.match(url_alt5, parts["uri"])
                                            if split_url:
                                                parts.update({"path": split_url.group(1), "id": "",
                                                              "resource": split_url.group(2)})
                                            else:
                                                split_url = re.match(url_enc4, parts["uri"])
                                                if split_url:
                                                    parts.update({"path": split_url.group(1), "id": split_url.group(2),
                                                                  "resource": split_url.group(3)})
                                                else:
                                                    print("URL not parsed")
                                                    raise UnparsedUrlException("url not parsed")
        return parts
    except UnparsedUrlException as e:
        print(parts)
        sys.exit("yeah, unparsed exception")
    except BaseException as e:
        print("error")
        print(e)
        print(parts)
        sys.exit("Base exception")


def filter(parts):
    """
    Checks parts for certain values
    :param parts:
    :type parts: dict
    :return: returns true for lines that should be kept, returns false for lines that should be excluded
    :rtype: bool
    """
    path1 = ["uc", "cc"]
    # excluded hub, admin, expertise groups, users
    path2 = ["clients", "dossier", "employees", "evs"]
    # return true for lines which you want to keep
    urlsplit = parts["uri"].split("/")
    urlsplit2 = []
    if len(urlsplit) >= 3:
        urlsplit2 = urlsplit[2].split("?")
    if urlsplit[1] in path1 and (urlsplit[2] in path2 or urlsplit2[0] == "presence_logs"):
        if len(urlsplit) != 5 or (len(urlsplit) >= 4 and urlsplit[4] != "devices"):

            if parts["cic"] in customers:
                    return True
    # When used this function prints on system values that should be excluded but are not
    if False:
        print("exclude checker active")
        exclude = ["admin", "hub", "deployements", "keystores", "monitor", "authorization", "users",
                   "/two_factor_check", "caren", "weeksheet_profiles", "utils", "deployments",
                   "expertise_profiles", "locations", "tasque", "connected_deployments", "expertise_groups", "survey",
                   "moves", "payroll", "springboard", "teams", "milo", "care_providers", "hermes", "client_collab",
                   "hour_types", "dbc", "nexus", "finance", "expense", "agenda_occurrences", "agenda_series", "valv",
                   "client_absence_reasons", "zorgmail", "care_orders", "location_assignments", "agbcodes",
                   "client_absences", "kermit", "hour_type_categories", "employee_addresses", "team_assignments",
                   "external_care_providers", "feedbep"]
        exclude1 = ["router", "broker"]
        exclude3 = ["documents"]
        exclude4 = ["devices"]
        if urlsplit[2] not in exclude and urlsplit[1] not in exclude1:
            if len(urlsplit) != 5 or urlsplit[4] not in exclude4:
                if len(urlsplit) >= 3:
                    urlsplit2 = urlsplit[2].split("?")
                    if urlsplit2[0] not in exclude3:
                        print(parts)
                        print(urlsplit)
                        sys.exit("not in exclude")
    return False


def blind_username(parts):
    """
    Hide/hash privacy sensitive data related to username
    :param parts:
    :type parts: dict
    :return:
    :rtype: dict
    """
    if parts['username'] in __UsernameDict:
        parts['username'] = __UsernameDict[parts['username']]
    else:
        __UsernameDict[parts["username"]] = hashlib.md5(parts["username"].encode()).hexdigest()
        parts['username'] = __UsernameDict[parts['username']]
    return parts


def blind_id(parts):
    """
    hide/hash privacy sensitive data related to id
    :param parts:
    :type parts: dict
    :return:
    :rtype: dict
    """
    if len(parts['id']) > 0:
        if parts['id'][0]=="/":
            parts['id']=parts['id'][1:]
    if parts['id'] in __UsernameDict:
        parts['id'] = __UsernameDict[parts["id"]]
    else:
        __UsernameDict[parts["id"]] = hashlib.md5(parts["id"].encode()).hexdigest()
        parts['id'] = __UsernameDict[parts['id']]
    return parts


def blind_multiple_id(id_array):
    """
    Hide/hash privacy sensitive data related to id when multiple are send in the same request
    :param id_array:
    :type id_array: list
    :return:
    :rtype: list
    """
    blind_array = []
    if len(id_array) > 0:
        for id in id_array:
            if id[0] == "/":
                id = id[1:]
            if id in __UsernameDict:
                blind_array.append(__UsernameDict[id])
            else:
                __UsernameDict[id] = hashlib.md5(id.encode()).hexdigest()
                blind_array.append(__UsernameDict[id])
    return blind_array


def rewrite(parts, type):
    """
    Rewrites parts to a line in a certain format
    :param parts:
    :type parts: dict
    :param type: options are Weka/CSV
    :type type: str
    :return: formatted line
    :rtype: string
    """
    new_format_line = '%(date)s,%(method)s,%(uri)s,%(status)s,%(cic)s,%(src)s,%(username)s,%(QoS)s,%(audit)s\n' % parts
    if type == "Weka":
        # weka time format function
        temptime = datetime.strptime(parts["date"], '%d/%b/%Y:%H:%M:%S %z')
        parts["date"] = temptime.strftime("%Y-%m-%dT%H:%M:%S")
        new_format_line = "%(date)s,%(method)s,'%(path)s','%(id)s','%(resource)s',%(status)s,'%(cic)s','%(src)s','%(username)s'\n" % parts
    elif type == "CSV":
        if parts["resource"] == "/multiple":
            new_format_line = ""
            id_array = parts["id"]
            for id in id_array:
                parts["id"] = id
                new_id_line = "%(date)s,%(method)s,'%(path)s','%(id)s','%(resource)s',%(status)s,'%(cic)s','%(src)s','%(username)s'\n" % parts
                new_format_line = new_format_line+new_id_line
        else:
            new_format_line = "%(date)s,%(method)s,'%(path)s','%(id)s','%(resource)s',%(status)s,'%(cic)s','%(src)s','%(username)s'\n" % parts
    elif type == "QoS":
        new_format_line = '%(username)s,%(QoS)s\n' % parts
    return new_format_line


def reader(modus, INFILE, OUTFILE):
    """
    depreciated reader function
    :param modus:
    :type modus:
    :param INFILE:
    :type INFILE:
    :param OUTFILE:
    :type OUTFILE:
    :return:
    :rtype:
    """
    with gzip.open(OUTFILE, 'wb') as out_f:
        if modus == "Weka":
            print(get_weka_complete())
            out_f.writelines(get_weka_complete())
        elif modus == "CSV":
            out_f.write("Date,Method,Path,Id,Resource,Status,Cic,Src,Username\n".encode("utf-8"))
        elif modus == "QoS":
            out_f.write("Username,QoS\n".encode("utf-8"))
        with gzip.open(INFILE, 'rb') as in_f:
            for line in in_f:
                parts = parseline(line.decode("utf-8"))
                if parts["QoS"] != "-" and parts["QoS"] != "" and parts["QoS"] !="mobile":
                    print(line)
                    sys.exit("interesting")
                if filter(parts):
                    if parts["username"] == "-" and parts["status"] != "403":
                        print(line)
                        print(parts["uri"][18:28])
                        print(parts["status"])
                        sys.exit("lege username")
                    parts = blind_username(parts)
                    parse_url(parts)

                    out_f.write(rewrite(parts, modus).encode("utf-8"))

            in_f.close()
        out_f.close()


def parser(line, modus):
    """
    Parses the line to a new format
    :param line:
    :type line: str
    :param modus: Type of formatting that should be done
    :type modus:
    :return:
    :rtype: str
    """
    try:
        parts = parseline(line.decode("utf-8"))
        if parts["QoS"] != "-" and parts["QoS"] != "" and parts["QoS"] != "mobile":
            print("SpecialQoSException")
            raise SpecialQoSException
        if filter(parts):
            if (parts["username"] == "-" or parts["username"] == "") and parts["status"] != "403":
                print("Empty username")
                raise EmptyUsernameException
            parts = blind_username(parts)
            parse_url(parts)
            return [rewrite(parts, modus).encode("utf-8"), parts["cic"]]
        else:
            return None
    except SpecialQoSException:
        print("specialQoSException:", line)
    except EmptyUsernameException:
        print("EmptyUsernameException:")
        print(line)
        sys.exit("Empty Username found, this is not possible")
