#!/bin/bash
TRAIN_STAGE=$1
COMPONENT=$2
DB=$3
CUDA=$4
GEREPING=$5


if [ "$COMPONENT" == "gereping" ] && [ -z "$GEREPING" ]
then
  echo "Gereping dataset needs id like 0250, 0440, 0455"
  exit 1
fi

if [ "$COMPONENT" == "gereping" ]
then
  DATAROOT=$COMPONENT/$GEREPING
  WANDB_PROJ=$COMPONENT$GEREPING
else
  DATAROOT=$COMPONENT
  WANDB_PROJ=$COMPONENT
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="$CUDA"
source activate ldm
if [ "$TRAIN_STAGE" == "1" ]
then
  python train_wgangp.py \
         --train_root /data3/lq/data/fucai/$DATAROOT \
         --save_dir /data4/lq/output/f_anogan \
         --save_interval 50 \
         --lr 0.0002 \
         --img_size 128 \
         --n_epochs 3000 \
         --channels 1 \
         --batchsize 128 \
         --n_critic 5\
         --seed 3452\
         --redis_db $DB \
         --sample_interval 200

elif [ "$TRAIN_STAGE" == "2" ]
then
  python train_encoder_izif.py \
         --train_root /data3/lq/data/fucai/$DATAROOT \
         --save_dir /data4/lq/output/f_anogan \
         --save_interval 50 \
         --lr 0.0002 \
         --img_size 128 \
         --n_epochs 3000 \
         --channels 1 \
         --batchsize 128 \
         --n_critic 5\
         --seed 3452\
         --redis_db $DB \
         --sample_interval 200

else
  echo "Unsupport Stage of train process"
  exit 1
fi