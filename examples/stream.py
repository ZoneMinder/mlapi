# Example of how you can use the MLAPI gateway
# with live streams

# Usage:
# python3 ./stream.py <local video file>
# if you leave out local video file, it will open your webcam

import cv2
import requests
import json
import imutils
import sys

# --------- Change to your needs---------------
BASE_API_URL = 'http://localhost:5000/api/v1'
# MLAPI user
USER = 'zm_user'
PASSWORD = 'zm_password'
FRAME_SKIP = 5

# All 3 work now, customizing/fine tuning is coming soon
# if you want object
PARAMS = {'delete': 'true', 'type': 'stream-object'}
# if you want face
# PARAMS = {'delete': 'true', 'type': 'stream-face'}
# if you want alpr
# PARAMS = {'delete': 'true', 'type': 'stream-alpr'}
# if you want object and then face
# PARAMS = {'delete': 'true', 'type': 'stream-object,face'}
# if you want face then object and finally alpr
# PARAMS = {'delete': 'true', 'type': 'stream-face,object,alpr'}

# adding ml_overrides to control the detection pattern soon

# If  you want to use webcam
CAPTURE_SRC = 0
# you can also point it to any media URL, like an RTSP one or a file
# CAPTURE_SRC = 'rtsp://whatever'

# If you want to use ZM
# note your URL may need /cgi-bin/zm/nph-zms - make sure you specify it correctly
# CAPTURE_SRC = 'https://demo.zoneminder.com/cgi-bin-zm/nph-zms?mode=jpeg&maxfps=5&buffer=1000&monitor=18&user=zmuser&pass=zmpass'
# --------- end ----------------------------

if sys.argv[1]:
    CAPTURE_SRC = sys.argv[1]

login_url = BASE_API_URL + '/login'
object_url = BASE_API_URL + '/detect/object'
auth_header = None

# Get API access token
r = requests.post(url=BASE_API_URL + '/login',
                  data=json.dumps({'username': USER, 'password': PASSWORD}),
                  headers={'content-type': 'application/json'})
data = r.json()
access_token = data.get('access_token')
if not access_token:
    print(data)
    print('Error retrieving access token from mlapi')
    exit()

# subsequent requests needs this JWT token
# Note it will expire in 2 hrs by default
auth_header = {'Authorization': 'Bearer ' + access_token}


# Draws bounding box around detections
def draw_boxes(frame, boxes, labels):
    color = (0, 255, 0)  # bgr
    for idx, bbox in enumerate(boxes):
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, labels[idx], (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


video_source = cv2.VideoCapture(CAPTURE_SRC)
frame_cnt = 0
if not video_source.isOpened():
    print("Could not open video_source")
    exit()

# read the video source, frame by frame and process it
while video_source.isOpened():
    status, frame = video_source.read()
    if not status:
        print('Error reading frame')
        exit()

    # resize width down to 800px before analysis
    # don't need more
    frame = imutils.resize(frame, width=800)
    frame_cnt += 1
    if frame_cnt % FRAME_SKIP:
        continue

    frame_cnt = 0
    # The API expects non-raw images, so lets convert to jpg
    ret, jpeg = cv2.imencode('.jpg', frame)
    # filename is important because the API checks filename type
    files = {'file': ('image.jpg', jpeg.tobytes(), 'application/octet')}

    r = requests.post(url=object_url, headers=auth_header, params=PARAMS, files=files)

    data = r.json()
    draw_boxes(frame, data['matched_data']['boxes'], data['matched_data']['labels'])

    cv2.imshow('Object detection via MLAPI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
video_source.release()
cv2.destroyAllWindows()
