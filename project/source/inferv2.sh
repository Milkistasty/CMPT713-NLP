#python ./llava/eval/run_llava.py \
#    --model-path "./downloads/checkpoint_merged/llava_merged2/" \
#    --image-file "/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"

#python ./llava/eval/run_llava.py \
#    --model-path "liuhaotian/llava-v1.5-7b" \
#    --image-file "/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"\

python ./llava/eval/run_llavav2.py \
    --model-path "./downloads/checkpoints/llava_lora_fn_weights7" \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --image-file "/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
    --query "why was this photo taken?"