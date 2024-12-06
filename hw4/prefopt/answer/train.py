from pathlib import Path
import os, sys
import torch
import json
from tqdm import tqdm
import logging
from datasets import Dataset
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from trl import ORPOTrainer, ORPOConfig, setup_chat_format
from peft import prepare_model_for_kbit_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate answers from an LLM and train using ORPO")

    parser.add_argument("-m", "--model", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="model path or model id")
    parser.add_argument("-i", "--inputfile",
                        default=os.path.join('data', 'input', 'dev.txt'),
                        help="produce output for this input file")
    parser.add_argument("-d", "--device",
                        default='cpu',
                        help="cuda device if available")
    parser.add_argument("-l", "--logfile", dest="logfile", default=None,
                        help="log file for debugging")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if the results directory is not empty")
    args = parser.parse_args()

    if args.logfile is not None:
        logging.basicConfig(filename=args.logfile, filemode='w', level=logging.DEBUG)

    device = args.device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"using device {args.device}", file=sys.stderr)

    # Model
    base_model = args.model
    new_model = "qwen2.5-0.5B-Instruct-orpo"



    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
    )

    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)  # Only call setup_chat_format if no template exists

    model = prepare_model_for_kbit_training(model)

    # Load ORPO dataset from ./data/orpo.json
    json_file_path = './data/orpo.json'

    with open(json_file_path, 'r', encoding='utf-8') as f:
        orpo_dataset_dict = json.load(f)

    print(orpo_dataset_dict.keys())  # Output: dict_keys(['chosen', 'rejected'])

    dataset = Dataset.from_dict(orpo_dataset_dict)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.05)

    orpo_args = ORPOConfig(
        per_device_train_batch_size=4,
        max_steps=1000,
        learning_rate=8e-5,
        gradient_accumulation_steps=1,
        logging_steps=10,
        eval_steps=500,
        output_dir="./results/",
        optim="rmsprop",
        warmup_steps=150,
        bf16=True,
        logging_first_step=True,
        remove_unused_columns=False,
        report_to = 'none'
    )
    

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    for param in model.parameters():
        param.requires_grad = True

    trainer.train()
    trainer.save_model(new_model)

