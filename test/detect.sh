#!/bin/bash

IMAGE=${1:-image.jpg}
MODE=${2:-object}

TOKEN=`curl -s -H "Content-Type:application/json" -XPOST -d '{"username":
"dbell", "password":"xT67&pXCRND$%6"}' "http://pliablepixels.duckdns.org:9988/api/v1/login" | jq -r '.access_token'
 2>/dev/null`

curl -F "file=@${IMAGE}" -H "Authorization:Bearer ${TOKEN}" -XPOST "http://pliablepixels.duckdns.org:9988/api/v1/detect/${MODE}?delete=true&gender=true"

