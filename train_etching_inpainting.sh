export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-inpainting"
export OUTPUT_DIR="diffusers_checkpoints"

export DATASET_1="gzguevara/mr_potato_head_masked"
export DATASET_2="gzguevara/cat_toy_masked"

# accelerate launch train_diffuser.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir $DATASET_1 $DATASET_2 \
#   --output_dir=$OUTPUT_DIR \
#   --resolution=512 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=2 \
#   --learning_rate=3e-6 \
#   --max_train_steps=500 \
#   --report_to_wandb

python train_diffuser.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir $DATASET_1 $DATASET_2 \
  --output_dir=$OUTPUT_DIR \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --learning_rate=3e-6 \
  --max_train_steps=500 \
  --report_to_wandb