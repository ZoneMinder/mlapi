#!/usr/bin/python3

import modules.utils as utils
import modules.common_params as g
import argparse
import pyzm.ml.face_train as train


ap = argparse.ArgumentParser()
ap.add_argument('-c',
                '--config',
                default='./mlapiconfig.ini',
                help='config file with path')

ap.add_argument('-s',
                '--size',
                type=int,
                help='resize amount (if you run out of memory)')

ap.add_argument('-d', '--debug', help='enables debug on console', action='store_true')


args, u = ap.parse_known_args()
args = vars(args)
utils.process_config(args)
train.FaceTrain(options=g.config).train(size=args['size'])

