#python ./llava/eval/run_llava.py \
#    --model-path "./downloads/checkpoint_merged/llava_merged2/" \
#    --image-file "/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"

#python ./llava/eval/run_llava.py \
#    --model-path "liuhaotian/llava-v1.5-7b" \
#    --image-file "/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"\

python ./llava/eval/run_llava.py \
    --model-path "./downloads/checkpoints/llava_lora_fn_cls1" \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --image-file "/dataset/cd_images/7fada345-8e1e-4956-a1bb-79735da8928f_no.jpg" \
    --query "Are given two images are similar ?"