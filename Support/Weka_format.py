__Weka_dict_attributes = {"Date": "Time_local date", "Method": "Request_method {GET,POST,PUT,DELETE,PATCH}",
                          "Path": "Request_path string", "Id": "Request_ID string",
                          "Resource": "Request_resource string",
                          "Status":"Request_status {412,200,201,202,204,304,400,401,403,404,"
                                   "409,423,429,500,502,503,504}", "Cic": "Cic string",
                          "Src": "Src string", "Username": "Username string", "Qos": "QoS string",
                          "audit": "Audit string"}
__Weka_dict_relations = {"relation": "Access_logs", "data": "@data"}


def get_weka_attributes(attribute):
    return __Weka_dict_attributes[attribute]


def get_weka_relations(relation):
    return __Weka_dict_relations[relation]


def get_weka_attributelist():
    __weka_attribute_list=["@attribute " + get_weka_attributes("Date"), "@attribute "+ get_weka_attributes("Method"),
                           "@attribute "+ get_weka_attributes("Path"), "@attribute "+ get_weka_attributes("Id"),
                           "@attribute "+get_weka_attributes("Resource"), "@attribute "+get_weka_attributes("Status"),
                           "@attribute "+get_weka_attributes("Cic"), "@attribute "+get_weka_attributes("Src"),
                           "@attribute "+get_weka_attributes("Username")]
    return __weka_attribute_list


def get_weka_complete():
    __Weka_format_list = ["@relation " + get_weka_relations("relation")]
    __Weka_format_list = __Weka_format_list+get_weka_attributelist()
    __Weka_format_list = __Weka_format_list + [""] + [get_weka_relations("data")]
    final_list = [x+"\n" for x in __Weka_format_list]
    return [x.encode("utf_8") for x in final_list]

