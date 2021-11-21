#!/usr/bin/env python3
import argparse

from pyzm.helpers.new_yaml import process_config as proc_conf
from pyzm.helpers.pyzm_utils import LogBuffer
from pyzm.helpers.new_yaml import GlobalConfig
import pyzm.ml.face_train_dlib as train

g = GlobalConfig()
ap = argparse.ArgumentParser()
ap.add_argument('-c',
                '--config',
                default='./mlapiconfig.yml',
                help='config file with path (default ./mlapiconfig.yml)')

ap.add_argument('-s',
                '--size',
                type=int,
                help='resize amount (if you run out of memory)')

ap.add_argument('-d', '--debug', help='enables debug on console', action='store_true')
ap.add_argument('-bd', '--baredebug', help='enables debug on console', action='store_true')


args, u = ap.parse_known_args()
args = vars(args)

g.logger = LogBuffer()
mlc, g = proc_conf(args, conf_globals=g, type_='mlapi')
g.config = mlc.config

train.FaceTrain(globs=g).train(size=args['size'])
