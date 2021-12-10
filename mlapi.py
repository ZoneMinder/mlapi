#!/usr/bin/env python3
import json
import signal
import sys
import time
from argparse import ArgumentParser
from functools import wraps
from json import loads
from mimetypes import guess_extension
from collections import deque
from pathlib import Path
from threading import Thread
from traceback import format_exc
from typing import Optional, Union

import cryptography.exceptions
import cv2
# Pycharm hack for intellisense
# from cv2 import cv2
import numpy as np
from cryptography.fernet import Fernet
from flask import Flask, request, jsonify, Response
from flask_jwt_extended import (
    JWTManager,
    jwt_required,
    create_access_token,
    get_jwt_identity,

)
from flask_restful import Resource, Api, reqparse, abort, inputs
from pydantic import BaseModel, validator
from requests import get as req_get
from werkzeug.datastructures import FileStorage
from yaml import safe_load

import pyzm.helpers.mlapi_db as mlapi_user_db
from pyzm import __version__ as pyzm_version
from pyzm.api import ZMApi
from pyzm.helpers.new_yaml import ConfigParse, process_config as proc_conf
from pyzm.helpers.pyzm_utils import str2bool, import_zm_zones
from pyzm.interface import GlobalConfig
from pyzm.ml.detect_sequence import DetectSequence

JWT: JWTManager
MAX_FILE_SIZE_MB: int = 5
ALLOWED_EXTENSIONS: set = {'.png', '.jpg', '.gif', '.mp4'}
ACCESS_TOKEN_EXPIRES: int = 60 * 60  # 1 hr
DEFAULT_CONFIG: dict = safe_load("""
host: 0.0.0.0
processes: 1
port: 5000
wsgi_server: flask

model_sequence: 'object'
same_model_high_conf: no
sanitize_logs: no
sanitize_str: <sanitized>
log_user:
log_group:
log_name: 'zm_mlapi'
log_path: './logs'
base_data_path: '.'
match_past_detections: no
past_det_max_diff_area: '5%'


zmes_keys: {}
frame_set: snapshot,alarm,snapshot
force_mpd: no
secrets: ./mlapisecrets.yml
auth_enabled: yes
import_zm_zones: no
only_triggered_zm_zones: no
cpu_max_processes: 2
cpu_max_lock_wait: 120
gpu_max_processes: 2
gpu_max_lock_wait: 120
tpu_max_processes: 2
tpu_max_lock_wait: 120

image_path: './images'
db_path: './db'
wait: 0.0
mlapi_secret_key: 'ChangeMe this is for creating the auth JWT for users to connect'
max_detection_size: 90%
contained_area: '1px'
detection_sequence: object
pyzm_overrides: {}

smart_fs_thresh: 5
disable_locks: no
object_framework: opencv
object_detection_pattern: (person|dog|car|truck)
object_min_confidence: 0.6
fp16_target: no
show_models: no

face_detection_pattern: .*
face_detection_framework: dlib
face_recognition_framework: dlib
face_num_jitters: 0
face_upsample_times: 0
face_model: cnn
face_train_model: cnn
face_recog_dist_threshold: 0.6
face_recog_knn_algo: ball_tree
face_dlib_processor: gpu
unknown_face_name: Unknown_Face
save_unknown_faces: no
save_unknown_faces_leeway_pixels: 100
alpr_detection_pattern: .*
alpr_api_type: cloud
alpr_service: 
alpr_url: 
alpr_key: 
platerec_stats: no
platerec_regions: [ ]
platerec_min_dscore: 0.1
platerec_min_score: 0.2

openalpr_recognize_vehicle: 1
openalpr_country: us
openalpr_state: ca
openalpr_min_confidence: 0.3

openalpr_cmdline_binary: alpr
openalpr_cmdline_params: -j -d
openalpr_cmdline_min_confidence: 0.3

stream_sequence:
  frame_strategy: 'most'
  frame_set: snapshot,alarm,snapshot
  contig_frames_before_error: 2
  max_attempts: 4
  sleep_between_attempts: 2
  sleep_between_frames: 0.3
  sleep_between_snapshots: 2
  smart_fs_thresh: 5
  resize: no
  model_height: 416 
  model_width: 416
  tpu_model_height: 320
  tpu_model_width: 320

ml_sequence:
  general:
    model_sequence: object
    disable_locks: no

  object:
      general:
        object_detection_pattern: '(person|dog|cat|car|truck)'
        same_model_sequence_strategy: most
        contained_area: 1px

      sequence:
        - name: 'BUILT IN YOLO V4'
          enabled: 'yes'
          object_config: '{{yolo4_object_config}}'
          object_weights: '{{yolo4_object_weights}}'
          object_labels: '{{yolo4_object_labels}}'
          object_min_confidence: '{{object_min_confidence}}'
          object_framework: '{{yolo4_object_framework}}'
          object_processor: 'gpu'
          gpu_max_processes: '{{gpu_max_processes}}'
          gpu_max_lock_wait: '{{gpu_max_lock_wait}}'
          cpu_max_processes: '{{cpu_max_processes}}'
          cpu_max_lock_wait: '{{cpu_max_lock_wait}}'
          fp16_target: 'no'  # only applies to GPU, default is FP32
          show_models: 'no'  # at current moment this is a global setting turned on by just setting it to : yes
  alpr:
    general:
      same_model_sequence_strategy: 'first'
      alpr_detection_pattern: '.*'

    sequence: []

  face:
    general:
      face_detection_pattern: '.*'
        # combine results below
      same_model_sequence_strategy: 'union'

      sequence: []
""")

__version__ = "0.0.1"
lp: str = 'mlapi:'


class GatewayConfig(BaseModel):
    """
    GatewayConfig class for mlapi.py
    """
    processes: int = 1
    port: int = 5000
    host: str

    wsgi: str

    @validator('wsgi')
    def wsgi_validator(cls, v: str):
        accepted = {'flask', 'bjoern', 'uvicorn', 'starlette'}
        v = v.lower()
        if v not in accepted:
            raise ValueError("Config Error: 'wsgi_server' must be either flask or bjoern")
        return v


def _parse_args() -> dict:
    ap = ArgumentParser()
    ap.add_argument(
        "-c",
        "--config",
        help="config file with path"
    )
    ap.add_argument(
        "--from-docker",
        action="store_true",
        default=False
    )
    ap.add_argument(
        "--docker",
        action="store_true",
        default=False
    )
    ap.add_argument(
        "-vv",
        "--verboseversion",
        action="store_true",
        help="print version and exit"
    )
    ap.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="print mlapi version and exit"
    )
    ap.add_argument(
        "-d",
        "--debug",
        help="enables debug and outputs to console",
        action="store_true"
    )
    ap.add_argument(
        "-bd",
        "--baredebug",
        help="enables debug without console output",
        action="store_true",
    )
    ap.add_argument(
        "-fs",
        "--from-service",
        help="starting mlapi from a service wrapper that handles restarting mlapi, so mlapi "
             "doesnt handle its own restarts",
        action="store_true",
    )  # this may not matter at all
    ap.add_argument(
        "-l",
        "--logname",
        help="set the log file name (set to a filename not a path -> mlapi-test, the .log is implied)",
    )
    args, u = ap.parse_known_args()
    args = vars(args)
    if args.get("version"):
        print(f"{__version__}")
        exit(0)
    if not args.get("config"):
        if Path('./mlapiconfig.yml').is_file():
            args['config'] = './mlapiconfig.yml'
            print(f"{lp} there was no configuration file passed to '{Path(__file__).name}'. The built in default of "
                  f"'./mlapiconfig.yml' is being used as there is a file there.")
        else:
            print("--config/-c required (Default: ./mlapiconfig.yml) does not exist or is not a file")
            exit(1)
    return args


def main(globs):
    # How else to pass the GlobalConfig to the Detect class?
    app: Flask
    from pyzm.helpers.new_yaml import start_logs
    from pyzm.helpers.pyzm_utils import LogBuffer, set_g
    g = globs
    bg_logs: Thread
    args = _parse_args()
    g.DEFAULT_CONFIG = DEFAULT_CONFIG
    g.logger = LogBuffer()
    start = time.perf_counter()
    mlc, g = proc_conf(args, conf_globals=g, type_='mlapi')
    g.config = mlc.config
    # UGLY! but it works
    set_g(g)
    # start mlapi logs
    try:
        from pyzm.ZMLog import sig_intr, sig_log_rot
        g.logger.info(f"{lp}signal handlers: Setting up for log 'rotation' and log 'interrupt'")
        signal.signal(signal.SIGHUP, sig_log_rot)
        signal.signal(signal.SIGINT, sig_intr)
    except Exception as e:
        g.logger.error(f'{lp} Error setting up log rotate and interrupt signal handlers -> \n{e}\n')
        raise e
    bg_logs = Thread(
        target=start_logs,
        kwargs={
            'config': g.config,
            'args': args,
            'type_': 'mlapi',
            'no_signal': True
        }
    )
    db: mlapi_user_db = mlapi_user_db.Database(db_globals=g, prompt_to_create=True if args.get('debug') else False)
    if not db.get_all_users():
        print(f"{lp} No users found in DB, please create at least 1 user -> python3 mlapi_dbuser.py")
        g.logger.log_close(exit=1)
        exit(1)
    bg_logs.start()
    wsgi_config = GatewayConfig(
        host=g.config["host"],
        port=g.config["port"],
        processes=g.config["processes"],
        wsgi=g.config['wsgi_server']
    )
    g.logger.debug(
        f"perf:{lp}init: total time to build initial config -> {time.perf_counter() - start}"
    )
    if args.get('from_docker') or args.get('docker'):
        g.logger.info(f"{lp} {'--from-docker' if args.get('from_docker') else '--docker'} was passed, MLAPI is running inside of a docker environment")
    app = Flask(__name__)
    # Override the HTTP exception handler.
    app.handle_http_exception = get_http_exception_handler(app)
    flask_api = Api(app, prefix="/api/v1")
    app.config["UPLOAD_FOLDER"] = g.config["image_path"]
    app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024
    app.config["JWT_SECRET_KEY"] = g.config["mlapi_secret_key"]
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = ACCESS_TOKEN_EXPIRES
    app.config["PROPAGATE_EXCEPTIONS"] = True
    # reload on resource files change and better debug messages FOR FLASK ONLY, not bjoern
    app.debug = False
    # Construct the detector/filter pipeline.
    m = DetectSequence(options=g.config['ml_sequence'], globs=g)
    # JWT = JWTManager(app)
    configure_jwt(app)
    flask_api.add_resource(Login, "/login", resource_class_kwargs={'globs': g, 'db': db})
    flask_api.add_resource(Health, "/health", resource_class_kwargs={'globs': g})
    flask_api.add_resource(
        Detect,
        "/detect/object",
        resource_class_kwargs={
            'globs': g,
            'app': app,
            'args': args,
            'mlc': mlc,
            'm': m
        }
    )

    g.logger.info(
        f"|*** FORKED NEO - Machine Learning API (mlapi) version: {__version__} - pyzm version: {pyzm_version} - "
        f"OpenCV version: {cv2.__version__} ***|"
    )
    if wsgi_config.wsgi == "bjoern":
        try:
            import bjoern
        except ImportError:
            g.logger.error(f"{lp} you have specified bjoern as the 'wsgi_server' but it is not installed! Using "
                           f"Flask as WSGI")
            wsgi_config.wsgi = 'flask'
            bjoern = None
        except Exception as exc:
            g.logger.error(exc)
            g.logger.error(f"{lp} error trying to use bjoern as 'wsgi_server'! Using Flask as WSGI")
            wsgi_config.wsgi = 'flask'
            bjoern = None
        else:
            g.logger.info(
                f"mlapi: using 'bjoern' as WSGI server @ {wsgi_config.host}:{wsgi_config.port}"
            )
            try:
                bjoern.run(app, host=wsgi_config.host, port=wsgi_config.port)
            except Exception as exc:
                g.logger.error(exc)
                g.logger.error(f"{lp} error trying to use bjoern as 'wsgi_server'! Using Flask as WSGI")
                wsgi_config.wsgi = 'flask'
                bjoern = None

    if wsgi_config.wsgi == "flask" or not wsgi_config.wsgi:
        g.logger.info(
            f"mlapi: using 'Flask' with a maximum of '{wsgi_config.processes}' processes as WSGI server "
            f"@ {wsgi_config.host}:{wsgi_config.port}"
        )
        try:
            app.run(
                host=wsgi_config.host,
                port=wsgi_config.port,
                threaded=False,
                processes=wsgi_config.processes,
            )
        except Exception as exc:
            g.logger.error(exc)
            g.logger.error(f"{lp} error trying to use Flask as 'wsgi_server'")
            print(format_exc())
    if not wsgi_config.wsgi == 'flask' or not wsgi_config.wsgi == 'bjoern':
        g.logger.error(f"{lp} ERROR-> Only 'bjoern' and 'flask' for WSGI server! Exiting")
        g.logger.log_close(exit=1)
        exit(1)


# split filename and return extension
def file_ext(string):
    extension = Path(string).suffix
    return extension.lower()


# Checks if filename is allowed
def allowed_ext(ext):
    return ext.lower() in ALLOWED_EXTENSIONS


def parse_args():
    parser = reqparse.RequestParser()
    parser.add_argument("type", location="args", default=None)
    parser.add_argument("response_format", location="args", default="zm_detect")
    parser.add_argument("delete", location="args", type=inputs.boolean, default=False)
    parser.add_argument("download", location="args", type=inputs.boolean, default=False)
    parser.add_argument("url", default=False)
    parser.add_argument("file", type=FileStorage, location="files")
    parser.add_argument("json", location="args", default=None)

    return parser.parse_args()


# download file locally and store with unique name
def get_file(arguments, app: Flask):
    from uuid import uuid4
    lp = "mlapi:get_file:"
    unique_filename = str(uuid4())
    file_with_path_no_ext = Path(app.config["UPLOAD_FOLDER"] / unique_filename)
    ext = None
    # uploaded as multipart data
    if arguments["file"]:
        g.logger.debug(
            f"{lp}multipart data: {len(arguments['file'])} file{'s' if len(arguments['file']) > 1 else ''} "
            f"uploaded")
        ul_files = arguments["file"]  # is a FileStorage werkzeug object
        ext = file_ext(ul_files.filename)
        if ul_files.filename and allowed_ext(ext):
            ul_files.save(f"{file_with_path_no_ext}{ext}")
        else:
            abort(500, msg=f"Bad file type {ul_files.filename}")

    # passed as a payload url
    elif arguments["url"]:
        url = arguments["url"]
        g.logger.debug(f"{lp} uploaded file by payload URL: {url}")
        ext = file_ext(url)
        r = req_get(url, allow_redirects=True)

        cd = r.headers.get("content-disposition")
        ct = r.headers.get("content-type")
        if cd:
            ext = file_ext(cd)
            g.logger.debug(f"{lp}content-disposition: extension {ext} derived from {cd}")
        elif ct:
            ext = guess_extension(ct.partition(";")[0].strip())
            if ext == ".jpe":
                ext = ".jpg"
            g.logger.debug(f"{lp}content-type: extension {ext} derived from {ct}")
            if not allowed_ext(ext):
                abort(400, msg=f"filetype {ext} not allowed")
        else:
            ext = ".jpg"
        with open(f"{file_with_path_no_ext}{ext}", "wb") as o_file:
            o_file.write(r.content)
    else:
        abort(400, msg="could not determine file type")
    g.logger.debug(f"{lp} saving received object detection file as -> '{file_with_path_no_ext}{ext}'")
    return file_with_path_no_ext, ext


class Detect(Resource):
    def __init__(self, **kwargs):
        self.globs: GlobalConfig = kwargs["globs"]
        self.app: Flask = kwargs["app"]
        self.args: dict = kwargs["args"]
        self.mlc: ConfigParse = kwargs["mlc"]
        self.m: DetectSequence = kwargs["m"]

    @jwt_required
    def post(self):
        m: DetectSequence = self.m
        app: Flask = self.app
        args: dict = self.args
        g: GlobalConfig = self.globs
        mlc: Optional[ConfigParse] = self.mlc

        # noinspection PyCallingNonCallable
        def _crypt(crypt: Fernet, data: dict):
            # crypt will be either f.encode or f.decode objects. crypt() calls the method
            processed_data: dict = {}
            for enc_key, enc_val in data.items():
                enc_key: str
                enc_val: Optional[str]
                if enc_val is not None:
                    try:
                        dec_key: bytes = crypt(enc_key.encode('utf-8'))
                        dec_data: bytes = crypt(enc_val.encode('utf-8'))
                    except cryptography.exceptions.InvalidSignature:
                        g.logger.error(
                            f"{lp} The encryption key for '{route_name}' may not match! please check "
                            f"both ZMES and MLAPI configurations! (Invalid Signature)"
                        )
                        abort(400, msg=f"Please check that the encryption keys match!")
                    except cryptography.fernet.InvalidToken:
                        g.logger.error(
                            f"{lp} The encryption key for '{route_name}' may not match! please check "
                            f"both ZMES and MLAPI configurations! (Invalid Token)"
                        )
                        abort(400, msg=f"Please check that the encryption keys match!")

                    except Exception as exc:
                        g.logger.error(
                            f"{lp} the encrypted data is malformed! Please check that the encryption keys "
                            f"match!"
                        )
                        g.logger.error(f"{exc}")
                        abort(400, msg=f"Please check that the encryption keys match!")
                    else:

                        processed_data[dec_key.decode('utf-8')] = dec_data.decode('utf-8')
            return processed_data

        lp: str = 'mlapi:detect:'
        fi: Optional[str] = None
        stream: Optional[Union[str, int]] = None
        mid: Optional[int] = None
        stream_options: dict = {}
        ml_overrides: dict = {}
        ml_options: dict = {}
        file_uploaded: bool = False
        req: Optional[dict] = None

        remote_ip_address: Optional[str] = request.remote_addr or "N/A"
        if request.headers.get("X-Forwarded-For"):
            g.logger.debug(f"{lp} X-Forwarded-For header found - {request.headers.get('X-Forwarded-For')}")
            remote_ip_address = request.headers.get("X-Forwarded-For", 'N/A')
        elif request.headers.get("X-Real-IP"):
            g.logger.debug(f"{lp} X-Real-IP header found - {request.headers.get('X-Real-IP')}")
            remote_ip_address = request.headers.get("X-Real-IP", 'N/A')

        req_args: dict = parse_args()

        # Work around for when sending a file over from ZMES
        if request.files.get('json'):  # JSON
            req = loads(request.files['json'].read())
        else:
            req = request.get_json()

        if not req:
            g.logger.debug(f"{lp} the request is EMPTY")
            abort(400, msg="Request EMPTY")

        if request.files.get('image'):
            req_args['file'] = request.files.get('image')
            file_uploaded = True
        # todo: access.log
        encrypted_data: dict = req.get("encrypted data")
        route_name: str = ''
        route_data_str: str = ""
        if encrypted_data:
            route_name = encrypted_data.pop('name')
            route_data_str = f" coming in on route '{route_name}'"
        g.logger.debug(f"{lp} The detection request is for MLAPI DB user '{get_jwt_identity()}'"
                       f" using IP address -> {remote_ip_address}{route_data_str}")
        zmes_stream_options: Optional[dict] = req.get("stream_options")
        reason: Optional[str] = req.get("reason")
        ml_overrides: Optional[Union[str, dict]] = req.get("ml_overrides", {})
        g.eid = stream = req.get("stream")

        g.mid = mid = int(req.get("mid", 0))
        zm_keys: Optional[Union[str, dict]] = g.config.get('zmes_keys')
        # g.logger.debug(f"\n{req_args = }\n{req = }\n{request=}\n")

        # STREAM REQUEST
        if req_args["type"].startswith('stream-'):
            type_: str = req_args["type"].split('stream-')[1]

            g.logger.debug(f"{lp} STREAM requesting object detection for type: '{type_}'")
            g.config = mlc.config

            fip, ext = get_file(req_args, app)
            stream = fi = f"{fip}{ext}"
            if not stream:
                g.logger.error(f"{lp} there is something wrong with storing the downloaded file!")
                abort(400, msg="Error trying to store provided file (stream object detection)")

            ml_options = g.config['ml_sequence']
            # type_ is formatted same as config -> object,face,alpr or face,object
            # it follows the order so if you want object and then alpr it would be object,alpr
            ml_options['general']['model_sequence'] = type_
            # todo: configure stream.py with ml_overrides for patterns?
            ml_options['object']['general']['object_detection_pattern'] = ".*"
            ml_options['face']['general']['face_detection_pattern'] = ".*"
            ml_options['alpr']['general']['alpr_detection_pattern'] = ".*"

            m.set_ml_options(ml_options)  # set ml_options for detect_stream
            matched_data, all_matches, all_frames = m.detect_stream(
                stream=stream,
                options=stream_options,
                ml_overrides=ml_overrides,
                in_file=file_uploaded,
            )

        else:
            re_configure: bool = False
            sec_hash: bool = False
            perf_config_hash: Optional[time.perf_counter] = None
            if mlc is None:
                g.logger.error(f"{lp} SOMETHING IS WRONG! there is no config object? BUILDING NOW!")
                mlc, g = proc_conf(args, conf_globals=g, type_='mlapi')
            else:
                perf_config_hash = time.perf_counter()
                re_configure = mlc.hash_compare('config')

            if re_configure:
                g.logger.debug(f"{lp} the config file has not changed since it was last read!")
                sec_hash = mlc.hash_compare('secret')
                if sec_hash:
                    g.logger.debug(f"{lp} the secrets file has not changed since it was last read!")
                else:
                    g.logger.debug(f"{lp} the secrets file has changed since it was last read, rebuilding config!")
                    mlc = None
                    # reload the models
                    m.set_ml_options({}, force_reload=True)
                    mlc, g = proc_conf(args, conf_globals=g, type_='mlapi')
            else:
                g.logger.debug(f"{lp} the config file has changed since it was last read, rebuilding config!")
                mlc = None
                # reload the models
                m.set_ml_options({}, force_reload=True)
                mlc, g = proc_conf(args, conf_globals=g, type_='mlapi')
            if perf_config_hash:
                g.logger.debug(
                    f"perf:{lp} total time to hash config/secrets -> "
                    f"{time.perf_counter() - perf_config_hash}"
                )
            if mid in mlc.monitor_overrides:
                g.logger.debug(f"{lp} monitor {mid} has an overrode configuration built, switching to it...")
                g.config = mlc.monitor_overrides[mid]
            else:
                g.logger.debug(f"{lp} monitor {mid} has no overrode configuration built, using 'base' config...")
                g.config = mlc.config

            # End of hash and reconfigure
            # Cache the credentials?
            decrypted_data: dict = {}
            if encrypted_data:
                # zm_keys is from the config file
                if zm_keys:
                    # pop it so it doesnt mess up decrypting (it is not an encrypted string, will throw an exception)
                    g.logger.debug(2, f"{lp} encrypted credentials received, checking keystore "
                                      f"for '{route_name}'"
                                   )
                    if route_name not in zm_keys:
                        g.logger.error(f"{lp} There is not a matching key for "
                                       f"'{route_name}', check the config files for spelling "
                                       f"mistakes or key mismatch!"
                                       )
                        raise ValueError(f"No encryption key in zmes_keys for {route_name}!")
                    key: bytes = f'{zm_keys.get(route_name)}'.encode('utf-8')
                    f: Fernet = Fernet(key)
                    # noinspection PyTypeChecker
                    decrypted_data = _crypt(f.decrypt, encrypted_data)
                    g.config['allow_self_signed'] = str2bool(decrypted_data.get('allow_self_signed'))
                    # print(f"{decrypted_kickstart = }")
                    # print(f"{decrypted_data = }")
                    if decrypted_data:
                        # url, user, pass decrypted! name and self signed are plain text
                        g.logger.debug(2, f"{lp} credentials have been decrypted, attempting to login to the "
                                          f"ZoneMinder API for '{route_name}'"
                                       )
                    # Figure out if logging in or assuming a token
                else:
                    g.logger.error(f"{lp} ZMES sent encrypted data but there is no keystore configured in "
                                   f"'{Path(args['config']).name}' - Create the 'zmes_keys' option in the config file "
                                   f"and load with route: key!"
                                   )
                    abort(400, msg=f"No keystore configured in {Path(args['config']).name}")
            else:
                g.logger.error(f"{lp} ZMES did not send any encrypted credentials, unable to reply!")
                abort(400, msg=f"You must send encrypted credentials to reply with!")

            if decrypted_data:
                # we have decrypted data
                api_options: dict[str, Union[str, bool]] = {
                    # sent from ZMES
                    "apiurl": decrypted_data.get('api_url'),
                    "portalurl": decrypted_data.get("portal_url"),
                    "user": decrypted_data.get('user'),
                    "password": decrypted_data.get('password'),
                    # This was popped out of the encrypted_data dict before it went through the decrypter
                    "disable_ssl_cert_check": str2bool(g.config["allow_self_signed"]),
                    # from mlapi config file
                    "sanitize_portal": str2bool(g.config.get("sanitize_logs")),
                }

                if not api_options.get("apiurl") and not api_options.get("portalurl"):
                    g.logger.error(
                        f"{lp} missing ZoneMinder API and/or Portal URLs. ZMES sends these in the encrypted request."
                        f"FATAL ERROR")
                    g.logger.log_close(exit=1)
                else:
                    g.api = ZMApi(options=api_options, api_globals=g, kickstart=decrypted_data)
            g.Event, g.Monitor, g.Frame = g.api.get_all_event_data()

            stream_options = g.config.get("stream_sequence", {})
            if stream_options:  # if stream sequence in config use it
                g.logger.debug(2, f"{lp} found 'stream_sequence' in '{args.get('config')}'")
            else:
                g.logger.debug(2, f"{lp} 'stream_sequence' not configured, relying on ZMES stream_sequence")
                stream_options = zmes_stream_options
            if not stream_options:
                g.logger.error(f"{lp} NO STREAM_SEQUENCE ?!")
                abort(400, msg="No stream options after processing local and sent arguments")

            # Past event logic, pass along
            g.config["PAST_EVENT"] = stream_options["PAST_EVENT"] = zmes_stream_options.get("PAST_EVENT")
            # resize HAS to be sent from ZMES, if the 2 get out of sync on this, bounding boxes wont be correct
            rs = zmes_stream_options.get("resize")
            if rs:
                # resize should only be a whole number
                try:
                    if isinstance(rs, str) and rs != 'no':
                        rs = round(float(rs))
                except Exception:
                    g.logger.error(f"{lp} 'resize' can only be a number (xx / xx.yy) or 'no'! setting to 'no' ")
                    rs = 'no'
                finally:
                    g.config["resize"] = stream_options["resize"] = rs
                    g.logger.debug(f"{lp} ZMES has resize={rs} configured, propagating...")

            global_import = g.config.get('import_zm_zones')
            mid_import = mlc.monitor_overrides.get(mid, {}).get('import_zm_zones')
            if str2bool(global_import) or str2bool(mid_import):
                g.logger.info(f"{lp} importing ZM zones for monitor {mid}")
                mlc.polygons[mid] = import_zm_zones(reason, g, mlc.polygons.get(mid, {}))
            polygons = mlc.polygons.get(mid)

            # Delay in stream options only applies to the very first frame
            if not stream_options.get("delay") and g.config.get("wait"):
                stream_options["delay"] = g.config.get("wait")
            if not stream:
                g.logger.debug(
                    f"{lp} stream info not found (no event or local file to process) looking in request from "
                    f"{remote_ip_address} for an attached image/video file..."
                )
                fip, ext = get_file(req_args, app)
                fi = f"{fip}{ext}"
                stream = fi
                if stream is None:
                    g.logger.error(f"{lp} NO event ID or input file to process as a stream")
                    abort(400, msg="No stream data (image or event for API)")

            ml_options = g.config['ml_sequence']
            # ml_overrides, sequence and patterns right before we send detection off? does it matter on this end?
            if str2bool(ml_overrides.get('enable')):
                g.logger.debug(
                    f"{lp} using ML overrides received in request -> {ml_overrides}")
                ml_options['general']['model_sequence'] = ml_overrides['model_sequence']
                if ml_options.get('object', {}).get('general', {}).get('object_detection_pattern'):
                    ml_options['object']['general']['object_detection_pattern'] = ml_overrides['object'][
                        'object_detection_pattern']
                if ml_options.get('face', {}).get('general', {}).get('face_detection_pattern'):
                    ml_options['face']['general']['face_detection_pattern'] = ml_overrides['face'][
                        'face_detection_pattern']
                if ml_options.get('alpr', {}).get('general', {}).get('alpr_detection_pattern'):
                    ml_options['alpr']['general']['alpr_detection_pattern'] = ml_overrides['alpr'][
                        'alpr_detection_pattern']
            # set ml_options for detect_stream
            # todo ml_sequence before hashing total config so we know if anything in ml sequence changed,
            #  if it did we need to force_reload models
            m.set_ml_options(ml_options)
            # finish configuring stream options
            stream_options["polygons"] = polygons
            # run detections
            matched_data, all_matches, all_frames = m.detect_stream(
                stream=stream,
                options=stream_options,
                ml_overrides=ml_overrides,
                in_file=file_uploaded,
            )
        # Merge stream-<model sequences> and regular detections logic for constructing reply
        # img: cv2.imdecode = matched_data['image']
        # create new instance with copy of the image in bytes

        success: bool = False
        img: Optional[Union[bytes, np.ndarray]] = None
        from requests_toolbelt import MultipartEncoder
        # print(f'{matched_data = }')
        if matched_data.get("frame_id") and matched_data.get("image") is not None:
            success = True
            img = matched_data['image']
            img = cv2.imencode('.jpg', img)[1]
            # Save the matched frame to disk to verify
            # save_img = img.copy()
            # cv2.imwrite(
            #     f"/tmp/mlapi-match-{g.eid}-{matched_data['frame_id']}.jpg",
            #     cv2.imdecode(save_img, cv2.IMREAD_UNCHANGED)
            # )
            # print(f"saving matching image from to /tmp")
            img = img.tobytes()
            # Remove the numpy.ndarray formatted image from matched_data because it is not JSON serializable
            matched_data['image'] = None
            resp_json: dict[str, Optional[Union[bool, dict]]] = {
                'success': success,
                'matched_data': matched_data,
                'all_matches': None,
            }
            multipart_encoded_data: MultipartEncoder = MultipartEncoder(
                fields={
                    'json': (None, json.dumps(resp_json), 'application/json'),
                    'image': (f"event-{g.eid}-frame-{matched_data['frame_id']}.jpg", img, 'application/octet')
                }
            )
            g.logger.info(
                f"{lp} returning matched image and detection data -> {matched_data}"
            )
            # g.logger.debug(f"{lp} returning all detection data -> {all_matches}")
            # response = Response(multipart_encoded_data.to_string(), mimetype=multipart_encoded_data.content_type)
        else:
            resp_json: dict[str, Optional[Union[bool, dict]]] = {
                'success': success,
                'matched_data': matched_data,
                'all_matches': all_matches,
            }
            multipart_encoded_data = MultipartEncoder(
                fields={
                    'json': (None, json.dumps(resp_json), 'application/json'),
                }
            )
            g.logger.info(f"{lp} no detection data to return")

        return Response(multipart_encoded_data.to_string(), mimetype=multipart_encoded_data.content_type)


# generates a JWT token to use for auth
class Login(Resource):
    def __init__(self, **kwargs):
        self.globs: GlobalConfig = kwargs["globs"]
        self.db: mlapi_user_db = kwargs["db"]

    def post(self):
        # todo add rate limiter and access.log for Fail2Ban
        g: GlobalConfig = self.globs
        db: mlapi_user_db = self.db
        if not request.is_json:
            abort(400, msg="Missing JSON in request")
        remote_ip_address: str = request.remote_addr or "N/A"
        headers: dict = request.headers
        if headers.get('X-Forwarded-For'):
            g.logger.debug(f"{lp}login: X-Forwarded-For headers from {remote_ip_address} "
                           f"- {headers.get('X-Forwarded-For')}")
            remote_ip_address = headers.get('X-Forwarded-For')
        elif headers.get('X-Real-IP'):
            g.logger.debug(f"{lp}login: X-Real-IP headers changing remote address from {remote_ip_address} "
                           f"- {headers.get('X-Real-IP')}")
            remote_ip_address = headers.get('X-Real-IP')

        username: Optional[str] = request.json.get("username")
        password: Optional[str] = request.json.get("password")
        if not username:
            abort(400, msg="Missing username in request")
        elif not password:
            abort(400, message="Missing password")
        if not db.check_credentials(username, password, ip=remote_ip_address):
            abort(401, message="incorrect credentials")

        # Identity can be any data that is json serializable
        access_token: str = create_access_token(identity=username)
        response: jsonify = jsonify(access_token=access_token, expires=ACCESS_TOKEN_EXPIRES)
        response.status_code = 200
        return response


# implement a basic health check.
class Health(Resource):
    def __init__(self, **kwargs):
        self.globs: GlobalConfig = kwargs["globs"]

    def get(self):
        g: GlobalConfig = self.globs
        # ip_addr = request.remote_addr or "N/A"
        # headers = request.headers
        # if headers.get('X-Forwarded-For'):
        #     g.logger.debug(f"{lp}login: X-Forwarded-For headers from {ip_addr} - {headers.get('X-Forwarded-For')}")
        #     ip_addr = headers.get('X-Forwarded-For')
        # elif headers.get('X-Real-IP'):
        #     g.logger.debug(
        #         f"{lp}login: X-Real-IP headers changing remote address from {ip_addr} - {headers.get('X-Real-IP')}")
        #     ip_addr = headers.get('X-Real-IP')
        # todo: add the /health endpoint to access.log (may be helpful in identifying a pattern
        #  or something if you have MLAPI exposed)
        response: jsonify = jsonify("ok")
        response.status_code = 200
        return response


def get_http_exception_handler(app: Flask):
    """Overrides the default http exception handler to return JSON."""
    handle_http_exception: app.handle_http_exception = app.handle_http_exception

    @wraps(handle_http_exception)
    def ret_val(exception):
        exc: exception = handle_http_exception(exception)
        return jsonify({"code": exc.code, "msg": exc.description}), exc.code
    return ret_val


def configure_jwt(app):
    global JWT
    JWT = JWTManager(app)

    @JWT.unauthorized_loader
    def unauthorized_jwt(error):
        ip_addr = request.remote_addr or "N/A"
        g.logger.info(
            f"mlapi:JWT: FAILED IP: {ip_addr} -> [UNAUTHORIZED JWT]: {error}")
        return Response(
            response=json.dumps({
                "message": "Unauthorized token"
            }),
            status=401,
            mimetype='application/json'
        )

    @JWT.expired_token_loader
    def my_expired_token_callback(expired_token):
        ip_addr = request.remote_addr or "N/A"
        g.logger.info(
            f"mlapi:JWT: FAILED IP: {ip_addr} -> [EXPIRED JWT]: {expired_token}")
        return Response(
            response=json.dumps({
                "message": "Expired token"
            }),
            status=401,
            mimetype='application/json'
        )

    @JWT.invalid_token_loader
    def my_invalid_token_callback(invalid_token):
        ip_addr = request.remote_addr or "N/A"
        g.logger.info(
            f"mlapi:JWT: FAILED IP: {ip_addr} -> [INVALID JWT]: {invalid_token}")
        return Response(
            response=json.dumps({
                "message": "Invalid token"
            }),
            status=422,
            mimetype='application/json'
        )


if __name__ == "__main__":
    try:
        g: GlobalConfig = GlobalConfig()
        main(g)
    except Exception as ex:
        print(f"mlapi: MAIN LOGIC ERROR -> {ex}")
        print(format_exc())
    finally:
        # always set stdout and stderr back to original
        stderr = sys.__stderr__
        stdout = sys.__stdout__
