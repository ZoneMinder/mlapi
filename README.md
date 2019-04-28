What
=====
An API gateway that you can install in your own server to do object, face and gender recognition.
Easy to extend to many/any other model. You can pass images as:
- a local file
- remote url

This is an example of invoking `python ./stream.py video.mp4` ([video courtesy of pexels](https://www.pexels.com/video/people-walking-by-on-a-sidewalk-854100/))
<img src="https://media.giphy.com/media/YQ4f1xXHMaDLF7AZMe/giphy.gif"/>

Technologies
=============
- I use  [cvlib](https://github.com/arunponnusamy/cvlib) by @arunponnusamy - easy to use wrapper for object detection
- Flask/Flask_restful for the API gateway
- TinyDB with bcrypt for password encryption
- flask_jwt_extended for JWT based access tokens
- I looked at Django too - too much to code. I found flask to be much easier and cleaner for this specific purpose

Why
=====
Wanted to learn how to write an API gateway easily. Object detection was a good use-case since I use it extensively for other things (like my event server). This is the first time I've used flask/jwt/tinydb etc. so its very likely there are improvements that can be made. Feel free to PR.

Install
=======
- It's best to create a virtual environment with python3 
- You need python3 for this to run

Then:
```
 git clone https://github.com/pliablepixels/mlapi
 cd mlapi
 pip install -r requirements.txt
 ```
Note: You may need other typical ml support libs. Forgot which. Feel free to PR and extend requirements.txt

Running
========
To run the server:
```
python ./api.py
```

To invoke detection, you need to:

Server Side:
- Make sure the username and password are created. Use `python adduser.py` for that

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

Face and gender detection for the same image above:

```
curl  -H "Authorization: Bearer ${ACCESS_TOKEN}" -XPOST -F"file=@IMG_1833.JPG" "http://localhost:5000/api/v1/detect/object?type=face&gender=true"
```

returns:

```
[{"type": "face", "confidence": "52.87%", "box": [904, 1037, 1199, 1337], "gender":
"man", "gender_confidence": "99.98%"}]
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


Live Streams or Recorded Video files
======================================
This is an image based object detection API. If you want to pass a video file or live stream,
take a look at the full example below.


Full Example
=============
Take a look at [stream.py](https://github.com/pliablepixels/mlapi/blob/master/examples/stream.py). This program reads any media source and/or webcam and invokes detection via the API gateway


Other Notes
============

- The first time you invoke a query, the ML engine inside will download weights/models and will take time. That will only happen once and from then on, it will be much faster

- Note that the server stores the images and the objects detected inside its `images/` folder. If you want the server to delete them after analysis add `&delete=true` to the query parameters.