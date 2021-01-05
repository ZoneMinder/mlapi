#!/bin/bash

# To enable log rotation being processed in mlapi
# I edited /etc/logrotate.d/zoneminder to add an extra line 

#  /var/lib/zmeventnotification/mlapi/mlapi_logrot.sh  inside postrotate 
# at the end

#-----------------------------------------------------
# Handles HUP for logrot
#-----------------------------------------------------


if [ -f "/var/run/mlapi.pid" ]
then
    kill -HUP `cat /var/run/mlapi.pid` 
fi
