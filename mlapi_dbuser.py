#!/usr/bin/env python3

import pyzm.helpers.globals as g
from pyzm.helpers.mlapi_db import Database
from argparse import ArgumentParser
from pyzm.helpers.new_yaml import process_config as proc_conf
from pyzm.helpers.pyzm_utils import LogBuffer

ap = ArgumentParser()
ap.add_argument('-u', '--user', help='username to create')
ap.add_argument('-p', '--password', help='password of user')
ap.add_argument('-d', '--dbpath', default='./db', help='path to DB')
ap.add_argument('-f', '--force', help='force overwrite user', action='store_true')
ap.add_argument('-l', '--list', help='list all users', action='store_true')
ap.add_argument('-r', '--remove', help='remove user')
ap.add_argument('-c', '--config', default='./mlapiconfig.yml')
args, u = ap.parse_known_args()
args = vars(args)
g.logger = LogBuffer()
mlc, g = proc_conf(args, conf_globals=g, type_='mlapi')
db = Database(prompt_to_create=False, db_globals=g)

if args.get('list'):
    print('----- Configured users ---------------')
    for i in db.get_all_users():
        print(f'User: {i.get("name")}')
    exit(0)

if args.get('remove'):
    u = args.get('remove')
    if not db.get_user(u):
        print(f'User: {u} not found')
    else:
        db.delete_user(args.get('remove'))
        print('OK - User Removed')
    exit(0)

if not args.get('user') or not args.get('password'):
    create_success = db.create_prompt()
else:
    if db.get_user(args.get('user')) and not args.get('force'):
        print(f"User: user '{args.get('user')}' already exists! you must --force or remove the user and re create\n")
        exit(1)
    db.add_user(args.get('user'), args.get('password'))