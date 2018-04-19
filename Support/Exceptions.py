class UnparsedUrlException(Exception):
    def __init_(self, msg):
        if msg is None:
            msg = "Url is unparsable!"


class SpecialQoSException(Exception):
    def __init_(self, msg):
        if msg is None:
            msg = "QoS is Unique!"


class EmptyUsernameException(Exception):
    def __init_(self, msg):
        if msg is None:
            msg = "Username is empty!"


class WeirdStatusCodeException(Exception):
    def __init_(self, msg):
        if msg is None:
            msg = "Weird status code encountered"


class MultipleFilesFoundException(Exception):
    def __init_(self, msg):
        if msg is None:
            msg = "Multiple files have been found."


class MissingFilesException(Exception):
    def __init__(self, arg):
        self.args = arg
