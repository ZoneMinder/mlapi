from tinydb import TinyDB, Query, where
from passlib.hash import sha256_crypt
import modules.common_params as g
import getpass

import modules.db as Database
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-u', '--user',  help='username to create')
ap.add_argument('-p', '--password', help='password of user')
ap.add_argument('-d', '--dbpath', default='./db', help='path to DB')
ap.add_argument('-f', '--force', help='force overwrite user', action='store_true')
ap.add_argument('-l', '--list', help='list all users', action='store_true')
ap.add_argument('-r', '--remove', help='remove user' )

args, u = ap.parse_known_args()
args = vars(args)


g.config['db_path']= args.get('dbpath')

db = Database.Database(prompt_to_create=False)

if args.get('list'):
    print ('----- Configured users ---------------')
    for i in db.get_all_users():
        print ('User: {}'.format(i.get('name')))
    exit(0)

if args.get('remove'):
    u = args.get('remove')
    if not db.get_user(u):
        print ('User: {} not found'.format(u))
    else:
        db.delete_user(args.get('remove'))
        print ('OK')
    exit(0)


if not args.get('user') or not args.get('password'):
    print ('--------------- User Creation ------------')
    while True:
        name = input ('\nuser name (Ctrl+C to exit):')
        if not name:
            print ('Error: username needed')
            continue
        p1 = getpass.getpass('Please enter password:')
        if not p1:
            print ('Error: password cannot be empty')
            continue
        p2 = getpass.getpass('Please re-enter password:')
        if  p1 != p2:
            print ('Passwords do not match, please re-try')
            continue
else:
    name = args.get('user')
    p1 = args.get('password')

if not db.get_user(name) or args.get('force'):
    db.add_user(name,p1)
    print ('User: {} created'.format(name))
else:
    print ('User: {} already exists. Use --force to override or --remove to remove old user first'.format(name))
