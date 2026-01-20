CUDA_VISIBLE_DEVICES=0 python3 main.py --device "0" \
  --cfg method_config/CUSTOM/Train/SOFS_dusan_hole.yaml --prior_layer_pointer 5 6 7 8 9 10 \
  --opts NUM_GPUS 1 DEVICE 0 RNG_SEED 54


