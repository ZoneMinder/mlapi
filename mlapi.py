#!/usr/bin/python3
# modified from source, original author @pliablepixels see: https://github.com/pliablepixels
import datetime
import json
import signal
import sys
from argparse import ArgumentParser
from functools import wraps
from json import loads
from mimetypes import guess_extension
from pathlib import Path
from threading import Thread
from traceback import format_exc
from typing import Optional

import cryptography.exceptions
import cv2
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

import pyzm.helpers.mlapi_db as mlapi_user_db
from pyzm import __version__ as pyzm_version
from pyzm.api import ZMApi
from pyzm.helpers.new_yaml import ConfigParse, process_config as proc_conf
from pyzm.helpers.pyzm_utils import str2bool, import_zm_zones
from pyzm.ml.detect_sequence import DetectSequence

__version__ = "3.0.3"
m: DetectSequence
app: Flask
db: mlapi_user_db
ml_options: dict = {}
stream_options: dict = {}
mlc: Optional[ConfigParse] = None
JWT: Optional[JWTManager] = None
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


def main():
    global app, args, secrets_conf, m, ml_options, stream_options, db, mlc, g
    bg_logs: Optional[Thread] = None
    lp = 'mlapi:'
    # -----------------------------------------------
    # main init
    # -----------------------------------------------
    ap = ArgumentParser()
    ap.add_argument(
        "--from-docker",
        action="store_true",
        default=False
    )
    ap.add_argument(
        "-c",
        "--config",
        help="config file with path"
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
        help="starting mlapi from a service wrapper that handles restarting mlapi, so mlapi doesnt handle its own restarts",
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

    import pyzm.helpers.globals as mlapi_g
    from pyzm.helpers.new_yaml import start_logs
    from pyzm.helpers.pyzm_utils import LogBuffer, set_g
    from pyzm.ZMLog import sig_intr, sig_log_rot
    mlapi_g.logger = LogBuffer()
    start = datetime.datetime.now()
    mlc, g = proc_conf(args, conf_globals=mlapi_g, type_='mlapi')
    g.config = mlc.config
    # UGLY! but it works
    set_g(g)
    # start mlapi logs
    g.logger.info(f"{lp}signal handlers: Setting up for log 'rotation' and log 'interrupt'")
    signal.signal(signal.SIGHUP, sig_log_rot)
    signal.signal(signal.SIGINT, sig_intr)
    bg_logs = Thread(
        target=start_logs,
        kwargs={
            'config': g.config,
            'args': args,
            '_type': 'mlapi',
            'no_signal': True
        }
    )
    bg_logs.start()
    wsgi_config = GatewayConfig(
        host=g.config["host"],
        port=g.config["port"],
        processes=g.config["processes"],
        wsgi=g.config['wsgi_server']
    )
    g.logger.debug(
        f"perf:{lp}init: total time to build initial config -> {(datetime.datetime.now() - start).total_seconds()}"
    )
    app = Flask(__name__)
    # Override the HTTP exception handler.
    app.handle_http_exception = get_http_exception_handler(app)
    flask_api = Api(app, prefix="/api/v1")
    app.config["UPLOAD_FOLDER"] = g.config["image_path"]
    app.config["MAX_CONTENT_LENGTH"] = g.MAX_FILE_SIZE_MB * 1024 * 1024
    app.config["JWT_SECRET_KEY"] = g.config["mlapi_secret_key"]
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = g.ACCESS_TOKEN_EXPIRES
    app.config["PROPAGATE_EXCEPTIONS"] = True
    # reload on resource files change and better debug messages FOR FLASK ONLY, not bjoern
    app.debug = False
    # JWT = JWTManager(app)
    configure_jwt(app)
    flask_api.add_resource(Login, "/login")
    flask_api.add_resource(Detect, "/detect/object")
    flask_api.add_resource(Health, "/health")
    # user DB for mlapi logins, mlapi does not allow for no auth, so there will always be a user DB of PW hashes
    # prompt to create wont work unless we are printing to console
    db = mlapi_user_db.Database(db_globals=g, prompt_to_create=True if args.get('debug') else False)
    if not db.get_all_users():
        print(f"{lp} No users found in DB, please create at least 1 user -> python3 mlapi_dbuser.py")
        g.logger.log_close(exit=1)
    # Construct the detector/filter pipeline.
    m = DetectSequence(options=g.config['ml_sequence'], globs=g)

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


# split filename and return extension
def file_ext(string):
    extension = Path(string).suffix
    return extension.lower()


# Checks if filename is allowed
def allowed_ext(ext):
    return ext.lower() in g.ALLOWED_EXTENSIONS


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
def get_file(arguments):
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
    @jwt_required
    def post(self):
        def cryptor(crypt: Fernet, data: dict):
            # crypt will be either f.encode or f.decode objects. crypt() calls the method
            processed_data = {}
            for enc_key, enc_val in data.items():
                if enc_val is not None:
                    try:
                        dec_key = crypt(enc_key.encode('utf-8'))
                        dec_data = crypt(enc_val.encode('utf-8'))
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

        lp = 'mlapi:detect:'
        global stream_options, ml_options, mlc, g
        fi = None
        stream = None
        mid = None
        stream_options = {}
        ml_overrides = {}
        ml_options = {}
        file_uploaded = False
        req = None
        ip_addr = request.remote_addr or "N/A"
        if request.headers.get("X-Forwarded-For"):
            g.logger.debug(f"{lp} X-Forwarded-For header found - {request.headers.get('X-Forwarded-For')}")
            ip_addr = request.headers.get("X-Forwarded-For", 'N/A')
        req_args = parse_args()

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

        g.logger.debug(f"{lp} The detection request is for MLAPI DB user '{get_jwt_identity()}'"
                       f" using IP address -> {ip_addr}")
        zmes_stream_options = req.get("stream_options")
        reason = req.get("reason")
        encrypted_data = req.get("encrypted data")
        ml_overrides = req.get("ml_overrides", {})
        g.eid = stream = req.get("stream")
        sub_options = None

        g.mid = mid = int(req.get("mid", 0))
        zm_keys = g.config.get('zmes_keys')
        # g.logger.debug(f"\n{req_args = }\n{req = }\n{request=}\n")

        # STREAM REQUEST
        if req_args["type"].startswith('stream-'):
            type_ = req_args["type"].split('stream-')[1]

            g.logger.debug(f"{lp} STREAM requesting object detection for type: '{type_}'")
            g.config = mlc.config

            fip, ext = get_file(req_args)
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
                sub_options=None,
                in_file=file_uploaded,
            )

        else:
            re_configure = None
            sec_hash = None
            perf_config_hash = None
            if mlc is None:
                g.logger.error(f"{lp} SOMETHING IS VERY WRONG! there is no config object? BUILDING!")
                mlc, g = proc_conf(args, conf_globals=g, type_='mlapi')
            else:
                perf_config_hash = datetime.datetime.now()
                re_configure = mlc.hash_compare('config')

            if re_configure:
                g.logger.debug(f"{lp} the config file has not changed since it was last read!")
                sec_hash = mlc.hash_compare('secret')
                if sec_hash:
                    g.logger.debug(f"{lp} the secrets file has not changed since it was last read!")
                else:
                    g.logger.debug(f"{lp} the secrets file has changed since it was last read, rebuilding config!")
                    mlc = None
                    mlc, g = proc_conf(args, conf_globals=g, type_='mlapi')
            else:
                g.logger.debug(f"{lp} the config file has changed since it was last read, rebuilding config!")
                mlc = None
                mlc, g = proc_conf(args, conf_globals=g, type_='mlapi')
            if perf_config_hash:
                g.logger.debug(
                    f"perf:{lp} total time to hash config/secrets -> "
                    f"{(datetime.datetime.now() - perf_config_hash).total_seconds()}"
                )
            if mid in mlc.monitor_overrides:
                g.logger.debug(f"{lp} monitor {mid} has an overrode configuration built, switching to it...")
                g.config = mlc.monitor_overrides[mid]
            else:
                g.logger.debug(f"{lp} monitor {mid} has no overrode configuration built, using 'base' config...")
                g.config = mlc.config

            # End of hash and reconfigure
            # Cache the credentials?
            route_name = ''
            decrypted_data = {}
            if encrypted_data:
                # zm_keys is from the config file
                if zm_keys:
                    route_name = encrypted_data.pop('name')
                    g.logger.debug(2, f"{lp} encrypted credentials received, checking keystore "
                                      f"for '{route_name}'"
                                   )
                    if route_name not in zm_keys:
                        g.logger.error(f"{lp} There is not a matching key for "
                                       f"'{route_name}', check the config files for spelling "
                                       f"mistakes or key mismatch!"
                                       )
                        raise ValueError(f"No encryption key in zmes_keys for {route_name}!")
                    key = f'{zm_keys.get(route_name)}'.encode('utf-8')
                    f = Fernet(key)
                    decrypted_data = cryptor(f.decrypt, encrypted_data)
                    g.config['allow_self_signed'] = str2bool(decrypted_data.get('allow_self_signed'))
                    # print(f"{decrypted_kickstart = }")
                    # print(f"{decrypted_data = }")
                    if decrypted_data:
                        # url, user, pass decrypted! name and self signed are plain text
                        g.logger.debug(2, f"{lp} credentials have been decrypted, attempting to login to the "
                                          f"ZoneMinder API for '{route_name}'"
                                       )
                    # Figure out if logging in or assuming a token
                    g.config['zm_creds'] = decrypted_data
                else:
                    g.logger.error(f"{lp} ZMES sent encrypted data but there is no keystore configured in "
                                   f"'{Path(args['config']).name}' - Create the 'zmes_keys' option in the config file "
                                   f"and load with route: key!"
                                   )
                    abort(400, msg=f"No keystore configured in {Path(args['config']).name}")
            else:
                g.logger.error(f"{lp} ZMES did not send any encrypted credentials, unable to reply!")
                abort(400, msg=f"You must send encrypted credentials to reply with!")

            if zm_keys:
                # we have decrypted data
                api_options = {
                    # sent from ZMES
                    "apiurl": g.config["zm_creds"].get('api_url'),
                    "portalurl": g.config["zm_creds"].get("portal_url"),
                    "user": g.config["zm_creds"].get('user'),
                    "password": g.config["zm_creds"].get('password'),
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

            stream_options = g.config.get("stream_sequence", {})
            if stream_options:  # if stream sequence in config use it
                g.logger.debug(2, f"{lp} found 'stream_sequence' in '{args.get('config')}'")
            else:
                g.logger.debug(2, f"{lp} 'stream_sequence' not configured, relying on ZMES stream_sequence")
                stream_options = zmes_stream_options
            # Past event logic, pass along
            g.config["PAST_EVENT"] = stream_options["PAST_EVENT"] = zmes_stream_options.get("PAST_EVENT")
            # resize HAS to be sent from ZMES, if the 2 get out of sync on this, bounding boxes wont be correct
            rs = zmes_stream_options.get("resize")
            if rs:
                # resize should only be a whole number
                try:
                    if isinstance(rs, str) and rs != 'no':
                        rs = round(float(rs))
                except Exception as exc:
                    g.logger.error(f"{lp} 'resize' can only be a number (xx / xx.yy) or 'no'! setting to 'no' ")
                    rs = 'no'
                finally:
                    g.config["resize"] = stream_options["resize"] = rs
                    g.logger.debug(f"{lp} ZMES has resize={rs} configured, propagating...")

            if not stream_options:
                g.logger.error(f"{lp} NO STREAM_SEQUENCE ?!")
                abort(400, msg="No stream options after processing local and sent arguments")

            # Create new object so modifying does not pollute source
            # There will be no reason if it's a Past event
            # DO zm zones and only triggered
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
                    f"{lp} stream info not found (no event or local file to process) looking in request from {ip_addr}"
                    f" for an attached image/video file..."
                )
                fip, ext = get_file(req_args)
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
            # todo ml_seqeuence before hashing total config so we know if anything in ml sequence changed,
            #  if it did we need to force_reload models
            m.set_ml_options(ml_options, force_reload=False)
            # finish configuring stream options
            stream_options["polygons"] = polygons
            # run detections
            matched_data, all_matches, all_frames = m.detect_stream(
                stream=stream,
                options=stream_options,
                ml_overrides=ml_overrides,
                sub_options=None,
                in_file=file_uploaded,
            )
        # Merge stream-<model sequences> and regular detections logic for constructing reply
        # img: cv2.imdecode = matched_data['image']
        # create new instance with copy of the image in bytes

        success = False
        from requests_toolbelt import MultipartEncoder
        if matched_data.get("frame_id") and matched_data.get("image") is not None:
            success = True
            img = matched_data['image']
            img = cv2.imencode('.jpg', img)[1]
            # save_img = img.copy()
            # cv2.imwrite(f"/tmp/mlapi-match-{g.eid}-{matched_data['frame_id']}.jpg", cv2.imdecode(save_img, cv2.IMREAD_UNCHANGED))
            # print(f"saveing matching image from to /tmp")
            img = img.tobytes()
            # Remove the numpy.ndarray formatted image from matched_data because it is not JSON serializable
            matched_data['image'] = None
            resp_json = {
                'success': success,
                'matched_data': matched_data,
                'all_matches': None,
            }
            multipart_encoded_data = MultipartEncoder(
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
            resp_json = {
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
    @staticmethod
    def post():
        # todo add rate limiter and access.log for Fail2Ban
        if not request.is_json:
            abort(400, msg="Missing JSON in request")
        ip_addr = request.remote_addr or "N/A"
        headers = request.headers
        if headers.get('X-Forwarded-For'):
            g.logger.debug(f"{lp}login: X-Forwarded-For headers from {ip_addr} - {headers.get('X-Forwarded-For')}")
            ip_addr = headers.get('X-Forwarded-For')
        elif headers.get('X-Real-IP'):
            g.logger.debug(f"{lp}login: X-Real-IP headers from {ip_addr} - {headers.get('X-Real-IP')}")
            ip_addr = headers.get('X-Real-IP')

        username = request.json.get("username", None)
        password = request.json.get("password", None)
        if not username:
            abort(400, msg="Missing username in request")
        elif not password:
            abort(400, message="Missing password")
        if not db.check_credentials(username, password, ip=ip_addr):
            abort(401, message="incorrect credentials")

        # Identity can be any data that is json serializable
        access_token = create_access_token(identity=username)
        response = jsonify(access_token=access_token, expires=g.ACCESS_TOKEN_EXPIRES)
        response.status_code = 200
        return response


# implement a basic health check.
class Health(Resource):
    @staticmethod
    def get():
        response = jsonify("ok")
        response.status_code = 200
        return response


def get_http_exception_handler(app):
    """Overrides the default http exception handler to return JSON."""
    handle_http_exception = app.handle_http_exception

    @wraps(handle_http_exception)
    def ret_val(exception):
        exc = handle_http_exception(exception)
        return jsonify({"code": exc.code, "msg": exc.description}), exc.code

    return ret_val


def configure_jwt(app):
    global JWT
    JWT = JWTManager(app)

    @JWT.unauthorized_loader
    def unauth_jwt(error):
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
        main()
    except Exception as ex:
        print(f"mlapi: MAIN LOGIC ERROR -> {ex}")
        print(format_exc())
    finally:
        # always set stdout and stderr back to original
        stderr = sys.__stderr__
        stdout = sys.__stdout__
