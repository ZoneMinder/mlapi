import modules.log as g_log

DB_NAME='db/db.json'
SECRET_KEY = 'dont-a-worry-be-happy' # IMPORTANT - change this to your own secret key
UPLOAD_FOLDER = 'images/'
MAX_FILE_SIZE_MB = 5
ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.jpeg'])
ACCESS_TOKEN_EXPIRES = 60 * 60  # 1 hr

log = g_log.Log()
