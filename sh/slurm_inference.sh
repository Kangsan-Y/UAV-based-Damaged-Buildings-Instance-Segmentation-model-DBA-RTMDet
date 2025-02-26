#!/usr/bin/env bash
source /share/home/yukangsan/anaconda3/etc/profile.d/conda.sh
conda activate openmmlab
cd /share/home/yukangsan/paper2/mmdetection
# export PYTHONPATH=$PYTHONPATH:`/share/home/yukangsan/research/transfiner`
# set -x

PARTITION=gpu
JOB_NAME=maskrcnn
#CONFIG=/share/home/yukangsan/research/mmdetection-main/configs/mask_rcnn/mask-rcnn_r50_fpn_2x_coco.py
# WORK_DIR=/share/home/yukangsan/research/mmdetection-main/result
# GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
# CPUS_PER_TASK=${CPUS_PER_TASK:-5}
# SRUN_ARGS=${SRUN_ARGS:-""}
# PY_ARGS=${@:5}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    # --ntasks=2 \
    # --ntasks-per-node=16 \
    # --cpus-per-task=${CPUS_PER_TASK} \
    # --kill-on-bad-exit=1 \
    # ${SRUN_ARGS} \

    # --launcher="slurm" ${PY_ARGS}
    # cd /share/home/yukangsan/research/transfiner
    python3 /share/home/yukangsan/paper2/mmdetection/demo/inference_damaged.py /share/home/yukangsan/paper2/mmdetection/results/inference1/gpj /share/home/yukangsan/paper2/mmdetection/configs/rtmdet/rtmdet-ins_s_8xb32-300e_coco_damaged.py --out-dir /share/home/yukangsan/paper2/mmdetection/results/inference/inference20241020_rtmdet_yangbi_base_epoch_280 --weights /share/home/yukangsan/paper2/mmdetection/yangbi/rtmdet_yangbi_base/epoch_280.pth --palette 'coco'