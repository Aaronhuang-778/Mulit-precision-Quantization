

import os
import time
import datetime

class XXError():
    def __str__(self):
        return "Error"


class XXWarning():
    def __str__(self):
        return "Warning"

class XXInfo():
    def __str__(self):
        return "Info"


def xxlog(msg: str, type=None):
    today = datetime.datetime.now()
    if(not os.path.exists("logs")):
        os.system("mkdir logs")
    if(type is None):
        type = "Info"
    with open("logs/model_extract_log.log", "a") as f:
        string = "[%s-%s-%s %s:%s:%s] %s: "%(
            today.year, today.month, today.day, today.hour, today.minute, today.second,
            type
        )
        f.write(string + msg + "\n")


def clear_log():
    os.system("rm logs/*")