#!/bin/bash
#SBATCH --job-name=u_net_primer_intento
#SBATCH --output=Sr_train_%j.out
#SBATCH --error=Sr_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Parámetros
RUN_NAME="prueba_super_y_var"
LEARNING_RATE="0.01"
BATCH_SIZE="8"
EPOCHS="500"
RESUME="False"
RESUME_CHECKPOINT_PATH="/home/ocolom/outputs/run_prueba_super_y_var_lr0.0005_bs8_epochs3000_resumeTrue_criterionL1Loss_optimizerAdam/checkpoint.pth"
CRITERION="L1Loss"
OPTIMIZER="Adam"

# Crear una carpeta única para esta ejecución
output_dir="/home/ocolom/outputs/run_${RUN_NAME}_lr${LEARNING_RATE}_bs${BATCH_SIZE}_epochs${EPOCHS}_resume${RESUME}_criterion${CRITERION}_optimizer${OPTIMIZER}"
mkdir -p ${output_dir}

# Cargar el entorno de conda
module load conda
conda activate entorno_unet

# Ejecutar el script de Python con los parámetros
python /home/ocolom/u_net_super_DSA_and_var.py \
    --data_dir /data/upftfg04/ocolom/DSA-Self-real-L1A-dataset/ \
    --output_dir ${output_dir} \
    --run_name ${RUN_NAME} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --resume ${RESUME} \
    --resume_checkpoint_path ${RESUME_CHECKPOINT_PATH} \
    --criterion ${CRITERION} \
    --optimizer ${OPTIMIZER}