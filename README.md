Note
=====
Release 2.1.0 onwards of mlapi requires ES 6.1.0 

What
=====
An API gateway that you can install in your own server to do object and face recognition.
Easy to extend to many/any other model. You can pass images as:
- a local file
- remote url

This can also be used as a remote face/recognition and object recognition server if you are using my [ZoneMinder Event Server](https://github.com/pliablepixels/zmeventnotification)!

This is an example of invoking `python ./stream.py video.mp4` ([video courtesy of pexels](https://www.pexels.com/video/people-walking-by-on-a-sidewalk-854100/))

<img src="https://media.giphy.com/media/YQ4f1xXHMaDLF7AZMe/giphy.gif"/>


Why
=====
Wanted to learn how to write an API gateway easily. Object detection was a good use-case since I use it extensively for other things (like my event server). This is the first time I've used flask/jwt/tinydb etc. so its very likely there are improvements that can be made. Feel free to PR.

Tip of the Hat
===============
A tip of the hat to [Adrian Rosebrock](https://www.pyimagesearch.com/about/) to get me started. His articles are great.

Containerized Fork
==================
themoosman maintains a containerized fork of this [repo](https://github.com/themoosman/mlapi).  This fork runs as a container and has been refactored to a WSGI (NGINX + Gunicorn + Flask) application. Please **do not** post questions about his containerized fork here. Please post issues in his fork.

Install
=======
- It's best to create a virtual environment with python3, but not mandatory 
- You need python3 for this to run
- face recognition requires cmake/gcc/standard linux dev libraries installed (if you have gcc, you likely have everything else. You may need to install cmake on top of it if you don't already have it)
- If you plan on using Tiny/Yolo V4, You need Open CV > 4.3
- If you plan on using the Google Coral TPU, please make sure you have all the libs
  installed as per https://coral.ai/docs/accelerator/get-started/ 
  

Note that this package also needs OpenCV which is not installed by the above step by default. This is because you may have a GPU and may want to use GPU support. If not, pip is fine. See [this page](https://zmeventnotification.readthedocs.io/en/latest/guides/hooks.html#opencv-install) on how to install OpenCV

Then:
```
 git clone https://github.com/pliablepixels/mlapi
 cd mlapi
 sudo -H pip3 install -r requirements.txt
 ```

Note: By default, `mlapiconfig.ini` uses the bjoern WSGI server. On debian, the following
dependencies are needed for bjoern:
```
sudo apt install libev-dev libevdev2
```
Alternately, you can just comment out `wsgi_server` and it will fall back to using flask.

Finally, you also need to get the inferencing models. Note this step is ONLY needed if you
don't already have the models downloaded. If you are running mlapi on the same server ZMES is 
running, you likely already have the models in `/var/lib/zmeventnotification/models/`.

To download all models, except coral edgetpu models:
```
./get_models.sh
```

To download all models, including coral edge tpu models:
(Coral needs the coral device, so it is not downloaded by default):
```
INSTALL_CORAL_EDGETPU=yes ./get_models.sh
```

**Please make sure you edit `mlapiconfig.ini` to meet your needs**


Running
========

Before you run, you need to create at least one user. Use `python3 mlapi_adduser.py` for that

Server: Manually
------------------
To run the server:
```
python3 ./mlapi.py -c mlapiconfig.ini
```

Server: Automatically
-----------------------
Take a look at `mlapi.service` and customize it for your needs


Client Side: From zm_detect
-----------------------------
One of the key uses of mlapi is to act as an API gateway for zm_detect, the ML 
python process for zmeventnotification. When run in this mode, zm_detect.py does not do local
inferencing. Instead if invokes an API call to mlapi. The big advantage is mlapi only loads the model(s) once 
and keeps them in memory, greatly reducing total time for detection.  If you downloaded mlapi to do this,
read ``objectconfig.ini`` in ``/etc/zm/`` to set it up. It is as simple as configuring the ``[remote]``
section of ``objectconfig.ini``. 

Client Side: From CLI
------------------------

(Note: The format of response that is returned for a CLI client is different from what is returned to zm_detect.
zm_detect uses a different format suited for its own needs)

To invoke detection from CLI, you need to:

Client Side:

(General note: I use [httpie](https://httpie.org) for command line http requests. Curl, while powerful has too many quirks/oddities. That being said, given curl is everywhere, examples are in curl. See later for a programmatic way)

- Get an access token
```
curl -H "Content-Type:application/json" -XPOST -d '{"username":"<user>", "password":"<password>"}' "http://localhost:5000/api/v1/login"
```
This will return a JSON object like:
```
{"access_token":"eyJ0eX<many more characters>","expires":3600}
```

Now use that token like so:

```
export ACCESS_TOKEN=<that access token>
```

Object detection for a remote image (via url):

```
curl -H "Content-Type:application/json" -H "Authorization: Bearer ${ACCESS_TOKEN}" -XPOST -d "{\"url\":\"https://upload.wikimedia.org/wikipedia/commons/c/c4/Anna%27s_hummingbird.jpg\"}" http://localhost:5000/api/v1/detect/object
```

**NOTE**: The payload shown below is when you invoke this from command line. When it is invoked by 
`zm_detect` a different format is returned that is compatible with the ES needs.

returns:

```
[{"type": "bird", "confidence": "99.98%", "box": [433, 466, 2441, 1660]}]
```

Object detection for a local image:
```
curl  -H "Authorization: Bearer ${ACCESS_TOKEN}" -XPOST -F"file=@IMG_1833.JPG" http://localhost:5000/api/v1/detect/object -v
```

returns:
```
[{"type": "person", "confidence": "99.77%", "box": [2034, 1076, 3030, 2344]}, {"type": "person", "confidence": "97.70%", "box": [463, 519, 1657, 1351]}, {"type": "cup", "confidence": "97.42%", "box": [1642, 978, 1780, 1198]}, {"type": "dining table", "confidence": "95.78%", "box": [636, 1088, 2370, 2262]}, {"type": "person", "confidence": "94.44%", "box": [22, 718, 640, 2292]}, {"type": "person", "confidence": "93.08%", "box": [408, 1002, 1254, 2016]}, {"type": "cup", "confidence": "92.57%", "box":[1918, 1272, 2110, 1518]}, {"type": "cup", "confidence": "90.04%", "box": [1384, 1768, 1564, 2044]}, {"type": "bowl", "confidence": "83.41%", "box": [882, 1760, 1134, 2024]}, {"type": "person", "confidence": "82.64%", "box": [2112, 984, 2508, 1946]}, {"type": "cup", "confidence": "50.14%", "box": [1854, 1520, 2072, 1752]}]
```

Face detection for the same image above:

```
curl  -H "Authorization: Bearer ${ACCESS_TOKEN}" -XPOST -F"file=@IMG_1833.JPG" "http://localhost:5000/api/v1/detect/object?type=face"
```

returns:

```
[{"type": "face", "confidence": "52.87%", "box": [904, 1037, 1199, 1337]}]
```

Object detection on a live Zoneminder feed:
(Note that ampersands have to be escaped as `%26` when passed as a data parameter)

```
curl -XPOST  "http://localhost:5000/api/v1/detect/object?delete=false" -d "url=https://demo.zoneminder.com/cgi-bin-zm/nph-zms?mode=single%26maxfps=5%26buffer=1000%26monitor=18%26user=zmuser%26pass=zmpass"
-H "Authorization: Bearer ${ACCESS_TOKEN}"
```

returns

```
[{"type": "bear", "confidence": "99.40%", "box": [6, 184, 352, 436]}, {"type": "bear
", "confidence": "72.66%", "box": [615, 219, 659, 279]}]
```

 Note that the server stores the images and the objects detected inside its `images/` folder. If you want the server to delete them after analysis add `&delete=true` to the query parameters.


Live Streams or Recorded Video files
======================================
This is an image based object detection API. If you want to pass a video file or live stream,
take a look at the full example below.


Full Example
=============
Take a look at [stream.py](https://github.com/pliablepixels/mlapi/blob/master/examples/stream.py). This program reads any media source and/or webcam and invokes detection via the API gateway


