#!/bin/bash
EXPRIMENT_NAME=tumor_grade_x5_256_mobilenetv2_wtihout_es

LOG_FOLDER=/home/poahmadvand/ml/slurm/classification/subtypes/$EXPRIMENT_NAME
[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER
NODES=(dlhost02 dlhost02 dlhost04)
PARTITIONS=(dgxV100 dgxV100 rtx5000)

for (( SPLIT_NUMBER = 1; SPLIT_NUMBER < 4; SPLIT_NUMBER++ )); do

    sbatch -J $EXPRIMENT_NAME-$SPLIT_NUMBER -w ${NODES[$SPLIT_NUMBER-1]} -p ${PARTITIONS[$SPLIT_NUMBER-1]} -o $LOG_FOLDER/04_train_$SPLIT_NUMBER\.out \
 -e $LOG_FOLDER/04_train_$SPLIT_NUMBER\.err \
 --export=EXPRIMENT_NAME=$EXPRIMENT_NAME,SPLIT_NUMBER=$SPLIT_NUMBER tumor_grade.sh
    echo $LOG_FOLDER/04_train_$SPLIT_NUMBER\.err
#    sleep 60
done


