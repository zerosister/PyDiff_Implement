# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import pydiff.archs
import pydiff.data
import pydiff.models

if __name__ == '__main__':
  # os.path is used to get the root path of the script.
  # osp.pardir is used to get the parent directory of the current directory.(i.e. "..")
  # osp.join get __file__ and then go up two levels to get the root path.
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
