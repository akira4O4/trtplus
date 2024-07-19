#!/bin/bash
TRTEXEC_PATH=trtexec
#-----------------------------------------------------------------------
MODEL_NAME=koutu #onnx model name without suffix
#-----------------------------------------------------------------------
INPUT_PATH=/home/seeking/llf/code/tensorRT-lab #model path
OUTPUT_PATH=${INPUT_PATH}#engine output path
#OUTPUT_PATH=/home/seeking/llf/code/tensorRT-lab/temp/models/yolov8
#-----------------------------------------------------------------------
WORKSPACE_SIZE=1024
#-----------------------------------------------------------------------
${TRTEXEC_PATH} \
  --onnx=${INPUT_PATH}/${MODEL_NAME}.onnx \
  --workspace=${WORKSPACE_SIZE} \
  --saveEngine=${OUTPUT_PATH}/${MODEL_NAME}.engine
#-----------------------------------------------------------------------

