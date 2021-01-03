from flask import Flask, send_file, request, jsonify, render_template
import requests as py_requests
from flask_restful import Resource, Api, reqparse, abort, inputs
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token, get_jwt_identity)
from werkzeug.security import safe_str_cmp
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException, default_exceptions
from werkzeug.datastructures import FileStorage
from functools import wraps
from mimetypes import guess_extension
#from collections import deque 

import os

import cv2
import uuid
import numpy as np
import argparse

import modules.common_params as g
import modules.db as Database
import modules.utils as utils
from modules.__init__ import __version__
from pyzm import __version__ as pyzm_version


from pyzm.ml.detect_sequence import DetectSequence
import pyzm.helpers.utils as pyzmutils
import ast
import pyzm.api as zmapi


def file_ext(str):
    f,e = os.path.splitext(str)
    return e.lower()

# Checks if filename is allowed
def allowed_ext(ext):
    return ext.lower() in g.ALLOWED_EXTENSIONS

# Assigns a unique name to the image and saves it locally for analysis

def parse_args():

    parser = reqparse.RequestParser()
    parser.add_argument('type', location='args',  default=None)
    parser.add_argument('response_format', location='args',  default='legacy')
    parser.add_argument('delete', location='args',
                        type=inputs.boolean, default=False)
    parser.add_argument('download', location='args',
                        type=inputs.boolean, default=False)
    parser.add_argument('url', default=False)
    parser.add_argument('file', type=FileStorage, location='files')
    return parser.parse_args()

def get_file(args):
  
    unique_filename = str(uuid.uuid4())
    file_with_path_no_ext = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    ext = None

   
    # uploaded as multipart data
    if args['file']:
        file = args['file']
        ext = file_ext(file.filename)
        if file.filename and allowed_ext(ext):
            file.save(file_with_path_no_ext+ext)
        else:
            abort (500, msg='Bad file type {}'.format(file.filename))

    # passed as a payload url
    elif args['url']:
        url = args['url']
        g.log.Debug (1,'Got url:{}'.format(url))
        ext = file_ext(url)
        r = py_requests.get(url, allow_redirects=True)
        
        cd = r.headers.get('content-disposition')
        ct = r.headers.get('content-type')
        if cd:
            ext = file_ext(cd)
            g.log.Debug (1,'extension {} derived from {}'.format(ext,cd))
        elif ct:
            ext = guess_extension(ct.partition(';')[0].strip())
            if ext == '.jpe': 
                ext = '.jpg'
            g.log.Debug (1,'extension {} derived from {}'.format(ext,ct))
            if not allowed_ext(ext):
                abort(400, msg='filetype {} not allowed'.format(ext))        
        else:
            ext = '.jpg'
        open(file_with_path_no_ext+ext, 'wb').write(r.content)
    else:
        abort(400, msg='could not determine file type')

    g.log.Debug (1,'get_file returned: {}{}'.format(file_with_path_no_ext,ext))
    return file_with_path_no_ext, ext
# general argument processing





class Detect(Resource):
    @jwt_required
    def post(self):
        args = parse_args()
        req = request.get_json()
        #print (req)
        fi = None
        stream_options={}
        stream = None 

        if req:
            stream = req.get('stream')
            stream_options = req.get('stream_options',{})
            if stream_options:
                stream_options['api'] = zmapi
            ml_overrides = req.get('ml_overrides',{})
        #g.log.Info ('I GOT: {} and {}'.format(stream, stream_options))        
        if args['type'] == 'face_names':
            g.log.Debug (1,'List of face names requested')
            print (face_obj.get_classes())
            face_list = {
                'names': face_obj.get_classes().tolist()
            }
            return face_list

        if args['type'] == 'face':
            g.log.Debug (1,'Face Recognition requested')
        
        elif args['type'] == 'alpr':
            g.log.Debug (1,'ALPR requested')

        elif args['type'] in [None, 'object']:
            g.log.Debug (1,'Object Recognition requested')
            #m = ObjectDetect.Object()
        else:
            abort(400, msg='Invalid Model:{}'.format(args['type']))

        if not stream:
            g.log.Debug (1, 'Stream info not found, looking at args...')
            fip,ext = get_file(args)
            fi = fip+ext
            stream = fi
            #image = cv2.imread(fi)
        #bbox,label,conf = m.detect(image)

        g.log.Debug (1, f'Calling detect streams with {stream} and {stream_options} and ml_overrides={ml_overrides} ml_options={ml_options}')
        #print (f'************************ {args}')

        matched_data,all_matches = m.detect_stream(stream=stream, options=stream_options, ml_overrides=ml_overrides)
        matched_data['image'] = None
       
        if args.get('response_format') == 'zm_detect':
            resp_obj= {
                'matched_data': matched_data,
                'all_matches': all_matches
            }
            g.log.Debug (1, 'Returning {}'.format(resp_obj))
            return resp_obj

        # legacy format
        bbox = matched_data['boxes']
        label = matched_data['labels']
        conf = matched_data['confidences']


        detections=[]
        for l, c, b in zip(label, conf, bbox):
            c = "{:.2f}%".format(c * 100)
            obj = {
                'type': 'object',
                'label': l,
                'confidence': c,
                'box': b
            }
            detections.append(obj)

        if args['delete'] and fi:
            #pass
            os.remove(fi)
        return detections


# generates a JWT token to use for auth
class Login(Resource):
    def post(self):
        if not request.is_json:
            abort(400, msg='Missing JSON in request')

        username = request.json.get('username', None)
        password = request.json.get('password', None)
        if not username:
            abort(400, message='Missing username')

        if not password:
            abort(400, message='Missing password')

        if not db.check_credentials(username,password):
            abort(401, message='incorrect credentials')
        # Identity can be any data that is json serializable
        access_token = create_access_token(identity=username)
        response = jsonify(access_token=access_token, expires=g.ACCESS_TOKEN_EXPIRES)
        response.status_code = 200
        return response

# implement a basic health check.
class Health(Resource):
    def get(self):
        response = jsonify("ok")
        response.status_code = 200
        return response

#-----------------------------------------------
# main init
#-----------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config',  help='config file with path')
ap.add_argument('-vv', '--verboseversion', action='store_true', help='print version and exit')
ap.add_argument('-v', '--version', action='store_true', help='print mlapi version and exit')

ap.add_argument('-d', '--debug', help='enables debug on console', action='store_true')

args, u = ap.parse_known_args()
args = vars(args)

if args.get('version'):
    print('{}'.format(__version__))
    exit(0)

if not args.get('config'):
    print ('--config required')
    exit(1)



utils.process_config(args)

app = Flask(__name__)

def get_http_exception_handler(app):
    """Overrides the default http exception handler to return JSON."""
    handle_http_exception = app.handle_http_exception
    @wraps(handle_http_exception)
    def ret_val(exception):
        exc = handle_http_exception(exception)
        return jsonify({'code': exc.code, 'msg': exc.description}), exc.code
    return ret_val


# Override the HTTP exception handler.
app.handle_http_exception = get_http_exception_handler(app)

api = Api(app, prefix='/api/v1')
app.config['UPLOAD_FOLDER'] = g.config['images_path']
app.config['MAX_CONTENT_LENGTH'] = g.MAX_FILE_SIZE_MB * 1024 * 1024
app.config['JWT_SECRET_KEY'] = g.config['mlapi_secret_key']
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = g.ACCESS_TOKEN_EXPIRES
app.config['PROPAGATE_EXCEPTIONS'] = True
app.debug = False
jwt = JWTManager(app)

db = Database.Database()

api.add_resource(Login, '/login')
api.add_resource(Detect, '/detect/object')
api.add_resource(Health, '/health')

secrets_conf = pyzmutils.read_config(g.config['secrets'])

api_options  = {
    'apiurl': pyzmutils.get(key='ZM_API_PORTAL', section='secrets', conf=secrets_conf),
    'portalurl':pyzmutils.get(key='ZM_PORTAL', section='secrets', conf=secrets_conf),
    'user': pyzmutils.get(key='ZM_USER', section='secrets', conf=secrets_conf),
    'password': pyzmutils.get(key='ZM_PASSWORD', section='secrets', conf=secrets_conf),
    'disable_ssl_cert_check':False if g.config['allow_self_signed']=='no' else True
}

g.log.set_level(5)

if not api_options.get('apiurl') or not api_options.get('portalurl'):
    g.log.Fatal('Missing API and/or Portal URLs. Your secrets file probably doesn\'t have these values')

zmapi = zmapi.ZMApi(options=api_options, logger=g.log)

     

ml_options = {}
stream_options = {}

if g.config['ml_sequence'] and g.config['use_sequence'] == 'yes':
        g.log.Debug(2,'using ml_sequence')
        ml_options = g.config['ml_sequence']
        secrets = pyzmutils.read_config(g.config['secrets'])
        ml_options = pyzmutils.template_fill(input_str=ml_options, config=None, secrets=secrets._sections.get('secrets'))
        #print (ml_options)
        ml_options = ast.literal_eval(ml_options)
        #print (ml_options)
else:
    g.logger.Debug(2,'mapping legacy ml data from config')
    ml_options = utils.convert_config_to_ml_sequence()

# stream options will come from zm_detect

#print(ml_options)

m = DetectSequence(options=ml_options, logger=g.log)


if __name__ == '__main__':
    g.log.Info ('--------| mlapi version:{}, pyzm version:{} |--------'.format(__version__, pyzm_version))
    
    #app.run(host='0.0.0.0', port=5000, threaded=True)
    #app.run(host='0.0.0.0', port=g.config['port'], threaded=False, processes=g.config['processes'])
    if g.config['wsgi_server'] == 'bjoern':
        g.log.Info ('Using bjoern as WSGI server')
        import bjoern
        bjoern.run(app,host='0.0.0.0',port=g.config['port'])
    else:
        g.log.Info ('Using flask as WSGI server')
        g.log.Info ('Starting server with max:{} processes'.format(g.config['processes']))
        app.run(host='0.0.0.0', port=g.config['port'], threaded=False, processes=g.config['processes'])
        