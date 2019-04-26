What
=====
An API gateway that you can install in your own server to do object, face and gender recognition.
Easy to extend to many/any other model

Technologies
=============
- I use  [cvlib](https://github.com/arunponnusamy/cvlib) by @arunponnusamy - easy to use wrapper for object detection
- Flask/Flask_restful for the API gateway
- TinyDB with bcrypt for password encryption
- flask_jwt_extended for JWT based access tokens

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
- Get an access token
```
curl -H "Content-Type:application/json" -XPOST -d '{"username":"<user>", "password":"<password>"}' "http://localhost:5000/api/v1/login"
```
This will return a JSON object like:
```
{"access_token":"eyJ0eXAiOiJ<many more characters>"}
```

Now use that token like so:

```
export ACCESS_TOKEN=<that access token>

# Object detection on an image (1.jpg)
curl -F "file=@1.jpg" -H "Authorization:Bearer ${ACCESS_TOKEN}" -XPOST "http://localhost:5000/api/v1/detect/object"

# Face + Gender recognition on an image (1.jpg)
curl -F "file=@1.jpg" -H "Authorization:Bearer ${ACCESS_TOKEN}" -XPOST "http://localhost:5000/api/v1/detect/object?type=face&gender=true"

```


Sample responses for both of the commands above:

```
# for object recognition
[{"type": "person", "confidence": "99.69%", "box": [463, 126, 1051, 698]}, {"type": "person", "confidence": "99.54%
", "box": [277, 228, 511, 618]}]

# for face/gender
[{"type": "face", "confidence": "97.73%", "box": [403, 270, 470, 366], "gender": "man", "gender_confidence": "95.46%"}, {"type": "face", "confidence": "96.93%", "box": [594, 108, 708, 237], "gender": "man", "gender_confidence": "1
00.00%"}]

```

Full Example
=============
Take a look at [stream.py](https://github.com/pliablepixels/mlapi/blob/master/examples/stream.py). This program reads any media source and/or webcam and invokes detection via the API gateway


Other Notes
============

- The first time you invoke a query, the ML engine inside will download weights/models and will take time. That will only happen once and from then on, it will be much faster

- Note that the server stores the images and the objects detected inside its `images/` folder. If you want the server to delete them after analysis add `&delete=true` to the query parameters.