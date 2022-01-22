#!/bin/bash

#-----------------------------------------------------
# Install script to make mlapi run as a service
# Only for ubuntu
#-----------------------------------------------------

# You can feed it USER_ GROUP_ and TARGET_MLAPI_DIR from the cli if desired.

TARGET_MLAPI_DIR=${TARGET_MLAPI_DIR:-"/var/lib/zmeventnotification/mlapi"}
RSYNC="rsync -av --progress"
# user and group
USER_=${USER_:-'www-data'}
GROUP_=${GROUP_:-'www-data'}


echo "This is really just my internal script to run mlapi as a service"
echo "You probably need to modify mlapi.service and this script"
echo "Which means, you likely don't want to run this..."
echo "*** IF YOU INSTALL MLAPI ON DIFF HOST THAN ZMES, make sure to change the user ($USER_) and group ($GROUP_) in these scripts!"
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

if [ -f "${TARGET_MLAPI_DIR}/mlapiconfig.yml" ]
then
    echo "Skipping mlapiconfig.yml file as it already exists in: ${TARGET_MLAPI_DIR}"
    EXCLUDE_PATTERN="${EXCLUDE_PATTERN} --exclude mlapiconfig.yml"
fi

if [ -f "${TARGET_MLAPI_DIR}/mlapisecrets.yml" ]
then
    echo "Skipping mlapisecrets.yml file as it already exists in: ${TARGET_MLAPI_DIR}"
    EXCLUDE_PATTERN="${EXCLUDE_PATTERN} --exclude mlapisecrets.yml"
fi


echo ${RSYNC} . ${TARGET_MLAPI_DIR} ${EXCLUDE_PATTERN}
${RSYNC} . ${TARGET_MLAPI_DIR} ${EXCLUDE_PATTERN}

#cp -R * "${TARGET_MLAPI_DIR}/"
install -m 755 -o "$USER_" -g "$GROUP_" mlapi.py "${TARGET_MLAPI_DIR}"
install -m 755 -o "$USER_" -g "$GROUP_" mlapi_logrot.sh "${TARGET_MLAPI_DIR}"
install -m 755 -o "$USER_" -g "$GROUP_" mlapi_face_train.py "${TARGET_MLAPI_DIR}"


chown -R "$USER_":"$GROUP_" ${TARGET_MLAPI_DIR}

echo "Copying service file"
cp mlapi.service /etc/systemd/system
chmod 644 /etc/systemd/system/mlapi.service
systemctl enable mlapi.service 


echo "Starting mlapi service"
systemctl daemon-reload
service mlapi restart

