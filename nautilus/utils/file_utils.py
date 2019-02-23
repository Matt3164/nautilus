from os.path import exists, join, dirname, expandvars
from os import makedirs

def mkdir(folder: str):
    if not exists(folder):
        makedirs(folder)

def mk_parent(filename: str):
    if not exists(dirname(filename)):
        makedirs(dirname(filename))