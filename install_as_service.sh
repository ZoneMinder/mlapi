#!/bin/bash

#-----------------------------------------------------
# Install script to make mlapi run as a service
# Only for ubuntu
#-----------------------------------------------------


TARGET_MLAPI_DIR="/var/lib/zmeventnotification/mlapi"
RSYNC="rsync -av --progress"

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

# Same dir check
touch .temp-dir-check.txt 2>/dev/null
if [ -f ${TARGET_MLAPI_DIR}/.temp-dir-check.txt ]
then
    echo "*** Error, the target and source directory seem to be the same! **"
    rm -f .temp-dir-check.txt 2>/dev/null
    exit 1
else
    rm -f .temp-dir-check.txt 2>/dev/null
fi

if [ ! -d "${TARGET_MLAPI_DIR}" ]
then
    echo "Creating ${TARGET_MLAPI_DIR}"
    mkdir -p "${TARGET_MLAPI_DIR}"
fi

echo "Syncing files to ${TARGET_MLAPI_DIR}"
EXCLUDE_PATTERN="--exclude .git"

if [ -d "${TARGET_MLAPI_DIR}/db" ]
then
    echo "Skipping db directory as it already exists in: ${TARGET_MLAPI_DIR}"
    EXCLUDE_PATTERN="${EXCLUDE_PATTERN} --exclude db"
fi

if [ -d "${TARGET_MLAPI_DIR}/known_faces" ]
then
    echo "Skipping known_faces directory as it already exists in: ${TARGET_MLAPI_DIR}"
    EXCLUDE_PATTERN="${EXCLUDE_PATTERN} --exclude known_faces"
fi

if [ -d "${TARGET_MLAPI_DIR}/unknown_faces" ]
then
    echo "Skipping unknown_faces directory as it already exists in: ${TARGET_MLAPI_DIR}"
    EXCLUDE_PATTERN="${EXCLUDE_PATTERN} --exclude unknown_faces"
fi

if [ -f "${TARGET_MLAPI_DIR}/mlapiconfig.ini" ]
then
    echo "Skipping mlapiconfig.ini file as it already exists in: ${TARGET_MLAPI_DIR}"
    EXCLUDE_PATTERN="${EXCLUDE_PATTERN} --exclude mlapiconfig.ini"
fi

if [ -f "${TARGET_MLAPI_DIR}/secrets.ini" ]
then
    echo "Skipping secrets.ini file as it already exists in: ${TARGET_MLAPI_DIR}"
    EXCLUDE_PATTERN="${EXCLUDE_PATTERN} --exclude secrets.ini"
fi


echo ${RSYNC} . ${TARGET_MLAPI_DIR} ${EXCLUDE_PATTERN}
${RSYNC} . ${TARGET_MLAPI_DIR} ${EXCLUDE_PATTERN}

#cp -R * "${TARGET_MLAPI_DIR}/"
install -m 755 -o "www-data" -g "www-data" mlapi.py "${TARGET_MLAPI_DIR}" 
install -m 755 -o "www-data" -g "www-data" mlapi_logrot.sh "${TARGET_MLAPI_DIR}" 
install -m 755 -o "www-data" -g "www-data" mlapi_face_train.py "${TARGET_MLAPI_DIR}" 


chown -R www-data:www-data ${TARGET_MLAPI_DIR}

echo "Copying service file"
cp mlapi.service /etc/systemd/system
chmod 644 /etc/systemd/system/mlapi.service
systemctl enable mlapi.service 


echo "Starting mlapi service"
systemctl daemon-reload
service mlapi restart

