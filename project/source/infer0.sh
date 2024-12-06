#python ./llava/eval/run_llava.py \
#    --model-path "./downloads/checkpoint_merged/llava_merged2/" \
#    --image-file "/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"

#python ./llava/eval/run_llava.py \
#    --model-path "liuhaotian/llava-v1.5-7b" \
#    --image-file "/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg" \
#    --query "why was this photo taken?"\

python ./llava/eval/run_llava0.py \
    --model-path "./downloads/checkpoints/llava_lora_fn_cls2_4cls" \
    --model-base "liuhaotian/llava-v1.5-7b" \
    --image-file "./dataset/cd_images/0a5d7e34-8369-4d71-b8cd-802440473c94.jpg" \
    --query "Are given two images are similar ?"

#copy ./downloads/colab_train to /content/colab_train
# python ./llava/eval/run_llava0.py --model-path "C:/Users/rajsh/Desktop/NLP Final Project/LLaVA./downloads/checkpoints/llava_lora_fn_cls2_4cls" --model-base "liuhaotian/llava-v1.5-7b" --image-file "C:/Users/rajsh/Desktop/NLP Final Project/LLaVA/dataset/images/0a3adc4a-cb66-4f28-8119-a2b3f7c0fae4.jpg" \ --query "Are given two images are similar ?"

# python ./llava/eval/run_llava0.py --model-path "./downloads/colab_train/checkpoints/llava_lora_fn_cls2_4cls/checkpoint-600" --model-base "liuhaotian/llava-v1.5-7b" --image-file "/dataset/dataset/cd_images/ffc34e2f-d6ae-4c15-8da3-a9022dc46486.jpg" \ --query "Are given two images are similar ?"