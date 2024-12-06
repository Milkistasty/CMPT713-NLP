#python scripts/merge_lora_weights.py \
#    --model-path "./downloads/checkpoints/llava_lora_weights" \
#    --model-base "liuhaotian/llava-v1.5-7b" \
#    --save-model-path "./downloads/checkpoint_merged/llava_merged/"


python scripts/merge_lora_weights.py \
    --model-path "./downloads/checkpoints/llava_fn_weights3" \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --save-model-path "./downloads/checkpoint_merged/llava_fn_weights3/"