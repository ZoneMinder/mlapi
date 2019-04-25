from flask import Flask, send_file, request, jsonify, render_template
from flask_restful import Resource, Api, reqparse, abort, inputs
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token, get_jwt_identity)
from werkzeug.security import safe_str_cmp
from werkzeug.utils import secure_filename
from werkzeug.exceptions import HTTPException, default_exceptions
from functools import wraps

import os
import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import uuid
import numpy as np

import modules.face as FaceDetect
import modules.object as ObjectDetect
import modules.globals as g
import modules.db as Database

# Checks if filename is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in g.ALLOWED_EXTENSIONS

# Assigns a unique name to the image and saves it locally for analysis


def upload_file():
    file = request.files['file']
    if file.filename and allowed_file(file.filename):
        #filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4())
        file_with_path_no_ext = os.path.join(
            app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_with_path_no_ext+'.jpg')
        return file_with_path_no_ext

# general argument processing


def parse_args():
    parser = reqparse.RequestParser()
    parser.add_argument('type', location='args',  default=None)
    parser.add_argument('gender', location='args',
                        type=inputs.boolean, default=False)
    parser.add_argument('delete', location='args',
                        type=inputs.boolean, default=True)
    parser.add_argument('download', location='args',
                        type=inputs.boolean, default=False)
    return parser.parse_args()


class Detect(Resource):
    @jwt_required
    def post(self):
        args = parse_args()
        if args['type'] == 'face':
            m = FaceDetect.Face()
        elif args['type'] in [None, 'object']:
            m = ObjectDetect.Object()
        else:
            abort(400, msg='Invalid Model:{}'.format(args['type']))
        fip = upload_file()
        detections = m.detect(fip, args)
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
        response = jsonify(access_token=access_token)
        response.status_code = 200
        return response


# main init
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
app.config['UPLOAD_FOLDER'] = g.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = g.MAX_FILE_SIZE_MB * 1024 * 1024
app.config['JWT_SECRET_KEY'] = g.SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = g.ACCESS_TOKEN_EXPIRES
app.config['PROPAGATE_EXCEPTIONS'] = True
app.debug = False
jwt = JWTManager(app)

db = Database.Database()

api.add_resource(Login, '/login')
api.add_resource(Detect, '/detect/object')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
