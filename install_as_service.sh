#!/bin/bash

#-----------------------------------------------------
# Install script to make mlapi run as a service
# Only for ubuntu
#-----------------------------------------------------


TARGET_MLAPI_DIR=/var/lib/zmeventnotification/mlapi

echo "This is really just my internal script to run mlapi as a service"
echo "You probably need to modify mlapi.service and this script"
echo "Which means, you likely don't want to run this..."
echo
read -p "Meh. I know what I'm doing. INSTALL! (Press some key...)"

if [[ $EUID -ne 0 ]]
then
    echo 
    echo "********************************************************************************"
    echo "           This script needs to run as root"
    echo "********************************************************************************"
    echo
    exit
fi

if [ ! -d "${TARGET_MLAPI_DIR}" ]
then
    echo "Creating ${TARGET_MLAPI_DIR}"
    mkdir -p "${TARGET_MLAPI_DIR}"
fi

echo "Copying files to ${TARGET_MLAPI_DIR}"
cp -R * "${TARGET_MLAPI_DIR}/"
install -m 755 -o "www-data" -g "www-data" mlapi.py "${TARGET_MLAPI_DIR}" 
install -m 755 -o "www-data" -g "www-data" mlapi_logrot.sh "${TARGET_MLAPI_DIR}" 

echo "Copying service file"
cp mlapi.service /etc/systemd/system
chmod 644 /etc/systemd/system/mlapi.service
systemctl enable mlapi.service 

echo "Starting mlapi service"
systemctl daemon-reload
service mlapi restart

