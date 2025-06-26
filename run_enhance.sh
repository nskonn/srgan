#!/bin/bash

python enhance_image/apply_checkpoint.py \
  --input "inputs/0069x4.png" \
  --output "outputs/result_151_69.jpg" \
  --checkpoint "checkpoints/checkpoint_151.pth"
