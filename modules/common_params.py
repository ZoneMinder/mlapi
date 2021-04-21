import modules.log as g_log



MAX_FILE_SIZE_MB = 5
ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.jpeg'])
ACCESS_TOKEN_EXPIRES = 60 * 60  # 1 hr

log = g_log.ConsoleLog()
logger = log

config = {}
monitor_config = {}
monitor_polygons = {}
monitor_zone_patterns={}
config_vals = {
         'secrets':{
            'section': 'general',
            'default': None,
            'type': 'string',
        },
        'auth_enabled':{
            'section': 'general',
            'default': 'yes',
            'type': 'string',
        },
         'portal':{
            'section': 'general',
            'default': '',
            'type': 'string',
        },
        'api_portal':{
            'section': 'general',
            'default': '',
            'type': 'string',
        },
        'user':{
            'section': 'general',
            'default': None,
            'type': 'string'
        },
        'password':{
            'section': 'general',
            'default': None,
            'type': 'string'
        },
        'basic_auth_user':{
            'section': 'general',
            'default': None,
            'type': 'string'
        },
        'basic_auth_password':{
            'section': 'general',
            'default': None,
            'type': 'string'
        },
        'import_zm_zones':{
            'section': 'general',
            'default': 'no',
            'type': 'string',
        },
        'only_triggered_zm_zones':{
            'section': 'general',
            'default': 'no',
            'type': 'string',
        },
         'cpu_max_processes':{
            'section': 'general',
            'default': '1',
            'type': 'int',
        },
        'gpu_max_processes':{
            'section': 'general',
            'default': '1',
            'type': 'int',
        },
        'tpu_max_processes':{
            'section': 'general',
            'default': '1',
            'type': 'int',
        },

        'cpu_max_lock_wait':{
            'section': 'general',
            'default': '120',
            'type': 'int',
        },

        'gpu_max_lock_wait':{
            'section': 'general',
            'default': '120',
            'type': 'int',
        },
        'tpu_max_lock_wait':{
            'section': 'general',
            'default': '120',
            'type': 'int',
        },

        'processes':{
            'section': 'general',
            'default': '1',
            'type': 'int',
        },
        'port':{
            'section': 'general',
            'default': '5000',
            'type': 'int',
        },

        'wsgi_server':{
            'section': 'general',
            'default': 'flask',
            'type': 'string',
        },
      
        'images_path':{
            'section': 'general',
            'default': './images',
            'type': 'string',
        },
        'db_path':{
            'section': 'general',
            'default': './db',
            'type': 'string',
        },
        'wait': {
            'section': 'general',
            'default':'0',
            'type': 'int'
        },
        'mlapi_secret_key':{
            'section': 'general',
            'default': None,
            'type': 'string',
        },

         'max_detection_size':{
            'section': 'general',
            'default': '100%',
            'type': 'string',
        },
        'detection_sequence':{
            'section': 'general',
            'default': 'object',
            'type': 'str_split'
        },
        'detection_mode': {
            'section':'general',
            'default':'all',
            'type':'string'
        },
        'use_zm_logs': {
            'section':'general',
            'default':'no',
            'type':'string'
        },

         'pyzm_overrides': {
            'section': 'general',
            'default': "{}",
            'type': 'dict',

        },

        'allow_self_signed':{
            'section': 'general',
            'default': 'yes',
            'type': 'string'
        },

        'resize':{
            'section': 'general',
            'default': 'no',
            'type': 'string'
        },
       
       'object_framework':{
            'section': 'object',
            'default': 'opencv',
            'type': 'string'
        },

        'disable_locks': {
            'section': 'ml',
            'default': 'no',
            'type': 'string'
        },

        'use_sequence': {
            'section': 'ml',
            'default': 'no',
            'type': 'string'
        },
        'ml_sequence': {
            'section': 'ml',
            'default': None,
            'type': 'string'
        },
        'stream_sequence': {
            'section': 'ml',
            'default': None,
            'type': 'string'
        },
     
      

        'object_processor':{
            'section': 'object',
            'default': 'cpu',
            'type': 'string'
        },
        'object_config':{
            'section': 'object',
            'default': 'models/yolov3/yolov3.cfg',
            'type': 'string'
        },
        'object_weights':{
            'section': 'object',
            'default': 'models/yolov3/yolov3.weights',
            'type': 'string'
        },
        'object_labels':{
            'section': 'object',
            'default': 'models/yolov3/coco.names',
            'type': 'string'
        },
       

        'object_min_confidence': {
            'section': 'object',
            'default': '0.4',
            'type': 'float'
        },
        'object_detection_pattern': {
            'section': 'object',
            'default': '.*',
            'type': 'string'
        },
        # Face
        'face_detection_framework':{
            'section': 'face',
            'default': 'dlib',
            'type': 'string'
        },
        'face_recognition_framework':{
            'section': 'face',
            'default': 'dlib',
            'type': 'string'
        },
        'face_num_jitters':{
            'section': 'face',
            'default': '0',
            'type': 'int',
        },
        'face_upsample_times':{
            'section': 'face',
            'default': '1',
            'type': 'int',
        },
        'face_model':{
            'section': 'face',
            'default': 'hog',
            'type': 'string',
        },
        'face_train_model':{
            'section': 'face',
            'default': 'hog',
            'type': 'string',
        },
         'face_recog_dist_threshold': {
            'section': 'face',
            'default': '0.6',
            'type': 'float'
        },
        'face_recog_knn_algo': {
            'section': 'face',
            'default': 'ball_tree',
            'type': 'string'
        },
        'known_images_path':{
            'section': 'face',
            'default': './known_faces',
            'type': 'string',
        },
        'unknown_images_path':{
            'section': 'face',
            'default': './unknown_faces',
            'type': 'string',
        },
        'unknown_face_name':{
            'section': 'face',
            'default': 'unknown face',
            'type': 'string',
        },
         'save_unknown_faces':{
            'section': 'face',
            'default': 'yes',
            'type': 'string',
        },

        'save_unknown_faces_leeway_pixels':{
            'section': 'face',
            'default': '50',
            'type': 'int',
        },
        'face_detection_pattern': {
            'section': 'face',
            'default': '.*',
            'type': 'string'
        },
        #  ALPR

        'alpr_service': {
            'section': 'alpr',
            'default': 'plate_recognizer',
            'type': 'string',
        },
        'alpr_url': {
            'section': 'alpr',
            'default': None,
            'type': 'string',
        },
        'alpr_key': {
            'section': 'alpr',
            'default': '',
            'type': 'string',
        },
        'alpr_use_after_detection_only': {
            'section': 'alpr',
            'type': 'string',
            'default': 'yes',
        },

         'alpr_detection_pattern':{
            'section': 'general',
            'default': '.*',
            'type': 'string'
        },
        'alpr_api_type':{
            'section': 'alpr',
            'default': 'cloud',
            'type': 'string'
        },

        # Plate recognition specific
        'platerec_stats':{
            'section': 'alpr',
            'default': 'no',
            'type': 'string'
        },

       
        'platerec_regions':{
            'section': 'alpr',
            'default': None,
            'type': 'eval'
        },
        'platerec_payload':{
            'section': 'alpr',
            'default': None,
            'type': 'eval'
        },
        'platerec_config':{
            'section': 'alpr',
            'default': None,
            'type': 'eval'
        },
        'platerec_min_dscore':{
            'section': 'alpr',
            'default': '0.3',
            'type': 'float'
        },
       
        'platerec_min_score':{
            'section': 'alpr',
            'default': '0.5',
            'type': 'float'
        },

        # OpenALPR specific
        'openalpr_recognize_vehicle':{
            'section': 'alpr',
            'default': '0',
            'type': 'int'
        },
        'openalpr_country':{
            'section': 'alpr',
            'default': 'us',
            'type': 'string'
        },
        'openalpr_state':{
            'section': 'alpr',
            'default': None,
            'type': 'string'
        },

        'openalpr_min_confidence': {
            'section': 'alpr',
            'default': '0.3',
            'type': 'float'
        },

        # OpenALPR command line specfic

         'openalpr_cmdline_binary':{
            'section': 'alpr',
            'default': 'alpr',
            'type': 'string'
        },
        
         'openalpr_cmdline_params':{
            'section': 'alpr',
            'default': '-j',
            'type': 'string'
        },
        'openalpr_cmdline_min_confidence': {
            'section': 'alpr',
            'default': '0.3',
            'type': 'float'
        },
       


}

