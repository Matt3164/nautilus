
import os

from nautilus.utils import file_utils

NAUTILUS_ROOT=file_utils.expandvars("$HOME/.nautilus")

def root_dir(experiment_tag: str):
    return file_utils.join(NAUTILUS_ROOT, experiment_tag)

def init(experiment_tag: str):
    file_utils.mkdir(root_dir(experiment_tag))