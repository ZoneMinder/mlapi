from tinydb import TinyDB, Query, where
from passlib.hash import sha256_crypt
import modules.common_params as g
import getpass

import modules.db as Database

g.config['db_path']='./db'
db = Database.Database()

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

    db.add_user(name,p1)
    print ('User: {} created'.format(name))

