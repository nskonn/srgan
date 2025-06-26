#!/bin/bash
# Запуск обучения
#python enhance_image/apply_checkpoint.py \
#  --input "inputs/0051x4.png" \
#  --output "outputs/result.jpg" \
#  --checkpoint "checkpoints/checkpoint_14.pth"

cd src
python main.py
