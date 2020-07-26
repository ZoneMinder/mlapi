#!/bin/bash

#-----------------------------------------------------
# Install script for mlapi and the 

# /install.sh 
#
#-----------------------------------------------------

# --- Change these if you want --

PYTHON=python3
PIP=pip3

INSTALL_YOLOV3=${INSTALL_YOLOV3:-yes}
INSTALL_TINYYOLOV3=${INSTALL_TINYYOLOV3:-yes}
INSTALL_YOLOV4=${INSTALL_YOLOV4:-yes}
INSTALL_TINYYOLOV4=${INSTALL_TINYYOLOV4:-yes}
INSTALL_CORAL_EDGETPU=${INSTALL_CORAL_EDGETPU:-no}

TARGET_DIR='./models'
WGET=$(which wget)

# utility functions for color coded pretty printing
print_error() {
    COLOR="\033[1;31m"
    NOCOLOR="\033[0m"
    echo -e "${COLOR}ERROR:${NOCOLOR}$1"
}

print_important() {
    COLOR="\033[0;34m"
    NOCOLOR="\033[0m"
    echo -e "${COLOR}IMPORTANT:${NOCOLOR}$1"
}

print_warning() {
    COLOR="\033[0;33m"
    NOCOLOR="\033[0m"
    echo -e "${COLOR}WARNING:${NOCOLOR}$1"
}

print_success() {
    COLOR="\033[1;32m"
    NOCOLOR="\033[0m"
    echo -e "${COLOR}Success:${NOCOLOR}$1"
}


mkdir -p "${TARGET_DIR}/yolov3" 2>/dev/null
mkdir -p "${TARGET_DIR}/yolov4" 2>/dev/null
mkdir -p "${TARGET_DIR}/tinyyolov3" 2>/dev/null
mkdir -p "${TARGET_DIR}/tinyyolov4" 2>/dev/null
mkdir -p "${TARGET_DIR}/yolov4" 2>/dev/null
mkdir -p "${TARGET_DIR}/coral_edgetpu" 2>/dev/null

if [ "${INSTALL_CORAL_EDGETPU}" == "yes" ]
then
    # Coral files
    echo
   
    echo 'Checking for Google Coral Edge TPU data files...'
    targets=( 'coco_indexed.names' 'ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite')
    sources=('https://dl.google.com/coral/canned_models/coco_labels.txt'
                'https://github.com/google-coral/edgetpu/raw/master/test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
            )

    for ((i=0;i<${#targets[@]};++i))
    do
        if [ ! -f "${TARGET_DIR}/coral_edgetpu/${targets[i]}" ]
        then
            ${WGET} "${sources[i]}"  -O"${TARGET_DIR}/coral_edgetpu/${targets[i]}"
        else
            echo "${targets[i]} exists, no need to download"

        fi
    done
fi

if [ "${INSTALL_YOLOV3}" == "yes" ]
then
    # If you don't already have data files, get them
    # First YOLOV3
    echo 'Checking for YoloV3 data files....'
    targets=('yolov3.cfg' 'coco.names' 'yolov3.weights')
    sources=('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
            'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
            'https://pjreddie.com/media/files/yolov3.weights')

    [ -f "${TARGET_DIR}/yolov3/yolov3_classes.txt" ] && rm "${TARGET_DIR}/yolov3/yolov3_classes.txt"


    for ((i=0;i<${#targets[@]};++i))
    do
        if [ ! -f "${TARGET_DIR}/yolov3/${targets[i]}" ]
        then
            ${WGET} "${sources[i]}"  -O"${TARGET_DIR}/yolov3/${targets[i]}"
        else
            echo "${targets[i]} exists, no need to download"

        fi
    done
fi

if [ "${INSTALL_TINYYOLOV3}" == "yes" ]
then
    # Next up, TinyYOLOV3
    echo
    echo 'Checking for TinyYOLOV3 data files...'
    targets=('yolov3-tiny.cfg' 'coco.names' 'yolov3-tiny.weights')
    sources=('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg'
            'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
            'https://pjreddie.com/media/files/yolov3-tiny.weights')

    [ -f "${TARGET_DIR}/tinyyolov3/yolov3-tiny.txt" ] && rm "${TARGET_DIR}/yolov3/yolov3-tiny.txt"

    for ((i=0;i<${#targets[@]};++i))
    do
        if [ ! -f "${TARGET_DIR}/tinyyolov3/${targets[i]}" ]
        then
            ${WGET} "${sources[i]}"  -O"${TARGET_DIR}/tinyyolov3/${targets[i]}"
        else
            echo "${targets[i]} exists, no need to download"

        fi
    done
fi

if [ "${INSTALL_TINYYOLOV4}" == "yes" ]
then
    # Next up, TinyYOLOV4
    echo
    echo 'Checking for TinyYOLOV4 data files...'
    targets=('yolov4-tiny.cfg' 'coco.names' 'yolov4-tiny.weights')
    sources=('https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
            'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
            'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights')

    for ((i=0;i<${#targets[@]};++i))
    do
        if [ ! -f "${TARGET_DIR}/tinyyolov4/${targets[i]}" ]
        then
            ${WGET} "${sources[i]}"  -O"${TARGET_DIR}/tinyyolov4/${targets[i]}"
        else
            echo "${targets[i]} exists, no need to download"

        fi
    done
fi

if [ "${INSTALL_YOLOV4}" == "yes" ]
then
    
    echo
    echo 'Checking for YOLOV4 data files...'
    print_warning 'Note, you need OpenCV > 4.3 for Yolov4 to work'
    targets=('yolov4.cfg' 'coco.names' 'yolov4.weights')
    sources=('https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg'
            'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
            'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'
            )

    for ((i=0;i<${#targets[@]};++i))
    do
        if [ ! -f "${TARGET_DIR}/yolov4/${targets[i]}" ]
        then
            ${WGET} "${sources[i]}"  -O"${TARGET_DIR}/yolov4/${targets[i]}"
        else
            echo "${targets[i]} exists, no need to download"

        fi
    done
fi
