import argparse
import copy

import torch
import os,sys
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
print("current_dir : ",current_dir)
print("parent_dir : ",parent_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(parent_dir)
print("parent_dir : ",parent_dir)
import random


from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


# def eval_model(args):
#     # Model
#     disable_torch_init()
#
#     model_name = get_model_name_from_path(args.model_path)
#     print("model_name : ",model_name)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         args.model_path, args.model_base, model_name
#     )
#
#     qs = args.query
#     image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
#     if IMAGE_PLACEHOLDER in qs:
#         if model.config.mm_use_im_start_end:
#             qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
#         else:
#             qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
#     else:
#         if model.config.mm_use_im_start_end:
#             qs = image_token_se + "\n" + qs
#         else:
#             qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
#
#     if "llama-2" in model_name.lower():
#         conv_mode = "llava_llama_2"
#     elif "mistral" in model_name.lower():
#         conv_mode = "mistral_instruct"
#     elif "v1.6-34b" in model_name.lower():
#         conv_mode = "chatml_direct"
#     elif "v1" in model_name.lower():
#         conv_mode = "llava_v1"
#     elif "mpt" in model_name.lower():
#         conv_mode = "mpt"
#     else:
#         conv_mode = "llava_v0"
#
#     if args.conv_mode is not None and conv_mode != args.conv_mode:
#         print(
#             "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
#                 conv_mode, args.conv_mode, args.conv_mode
#             )
#         )
#     else:
#         args.conv_mode = conv_mode
#
#     conv = conv_templates[args.conv_mode].copy()
#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()
#
#     image_files = image_parser(args)
#     images = load_images(image_files)
#     image_sizes = [x.size for x in images]
#     images_tensor = process_images(
#         images,
#         image_processor,
#         model.config
#     ).to(model.device, dtype=torch.float16)
#
#     input_ids = (
#         tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#         .unsqueeze(0)
#         .cuda()
#     )
#
#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=images_tensor,
#             image_sizes=image_sizes,
#             do_sample=True if args.temperature > 0 else False,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             num_beams=args.num_beams,
#             max_new_tokens=args.max_new_tokens,
#             use_cache=True,
#         )
#
#     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#     print(outputs)

import torch
import re


def get_max_repeated_string(string_list):
    if not string_list:
        return None
    return max(set(string_list), key=string_list.count)

def eval_model(args):
    # Model Initialization
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print("Model Name: ", model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Determine Conversation Mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] The auto-inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    val_json = json.load(open("/dataset/cd_validation/dataset.json"))

    pds = []
    gts = []
    ctr = 0
    N = len(val_json)
    for val_idx,_ in enumerate(val_json):
        item = random.choice([0, 1])
        val = copy.deepcopy(val_json[val_idx])

        if item == 1:
            no_i = random.randint(0, N - 1)
            while no_i == val_idx:
                no_i = random.randint(0, N - 1)
            val2 = copy.deepcopy(val_json[no_i])

            for ech in val["conversations"]:
                if ech["from"] == "gpt":
                    ech["value"] = "No"
        else:
            val2 = copy.deepcopy(val)

        cur_out = []
        print(ctr,end='\r')
        ctr+=1
        for _ in range(1):
            print(val["image"])

            im_nam = ["./dataset/cd_images/"+val["image"]]
            im_nam2 = ["./dataset/cd_images/" + val2["image"]]
            for ech in val["conversations"]:
                if ech["from"] == "human":
                    qs = ech["value"]
                elif ech["from"] == "gpt":
                    gt = ech["value"]
            # print(item,val["image"],val2["image"],gt)
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            # Conversation Template
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Load and Process Image
            image_files = im_nam  # Wrap in list for compatibility
            images = load_images(image_files)
            image_sizes = [x.size for x in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            # Load and Process Image
            image_files2 = im_nam2  # Wrap in list for compatibility
            images2 = load_images(image_files2)
            image_sizes2 = [x.size for x in images2]
            images_tensor2 = process_images(
                images2,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            # Tokenize and Generate Output
            input_ids = (
                tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
            )

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[images_tensor,images_tensor2],
                    image_sizes=[image_sizes,image_sizes2],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print("\nResponse: ", outputs)
            cur_out.append(outputs)
        print(cur_out)
        pd = get_max_repeated_string(cur_out)
        print(pd,gt)
        pds.append(pd)
        gts.append(gt)

    # Calculating metrics
    classes = list(set(gts))
    accuracy = accuracy_score(gts, pds)
    f1 = f1_score(gts, pds, average='weighted')  # 'weighted' accounts for label imbalance
    precision = precision_score(gts, pds, average='weighted')
    recall = recall_score(gts, pds, average='weighted')
    conf_matrix = confusion_matrix(gts, pds, labels=classes)

    # Output results
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", conf_matrix)


    class_wise_accuracy = {}
    for idx, cls in enumerate(classes):
        true_positive = conf_matrix[idx, idx]  # Diagonal element for the class
        total_samples = conf_matrix[idx, :].sum()  # Total samples for the class
        accuracy = true_positive / total_samples if total_samples > 0 else 0
        class_wise_accuracy[cls] = accuracy
    print("Class-wise Accuracy:")
    for cls, acc in class_wise_accuracy.items():
        print(f"  {cls}: {acc:.2f}")



    # while True:
    #     # Get user input for query and image file
    #     user_query = input("Enter your query (or type 'quit' to exit): ").strip()
    #     if user_query.lower() == 'quit':
    #         print("Exiting...")
    #         break
    #
    #     user_image_file = input("Enter the path to the image file: ").strip()
    #
    #     # Prepare Query
    #     qs = user_query
    #     image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    #     if IMAGE_PLACEHOLDER in qs:
    #         if model.config.mm_use_im_start_end:
    #             qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    #         else:
    #             qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    #     else:
    #         if model.config.mm_use_im_start_end:
    #             qs = image_token_se + "\n" + qs
    #         else:
    #             qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    #
    #     # Conversation Template
    #     conv = conv_templates[args.conv_mode].copy()
    #     conv.append_message(conv.roles[0], qs)
    #     conv.append_message(conv.roles[1], None)
    #     prompt = conv.get_prompt()
    #
    #     # Load and Process Image
    #     image_files = [user_image_file]  # Wrap in list for compatibility
    #     images = load_images(image_files)
    #     image_sizes = [x.size for x in images]
    #     images_tensor = process_images(
    #         images,
    #         image_processor,
    #         model.config
    #     ).to(model.device, dtype=torch.float16)
    #
    #     # Tokenize and Generate Output
    #     input_ids = (
    #         tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    #         .unsqueeze(0)
    #         .cuda()
    #     )
    #
    #     with torch.inference_mode():
    #         output_ids = model.generate(
    #             input_ids,
    #             images=images_tensor,
    #             image_sizes=image_sizes,
    #             do_sample=True if args.temperature > 0 else False,
    #             temperature=args.temperature,
    #             top_p=args.top_p,
    #             num_beams=args.num_beams,
    #             max_new_tokens=args.max_new_tokens,
    #             use_cache=True,
    #         )
    #
    #     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    #     print("\nResponse: ", outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
