import os
import gzip
import shutil
import json
import re
from pathlib import Path
from tqdm import tqdm


def is_gzipped(file_path):
    return file_path.endswith('.gz')


def open_file(file_path, mode='rt'):
    if is_gzipped(file_path):
        return gzip.open(file_path, mode, encoding='utf-8')
    else:
        return open(file_path, mode, encoding='utf-8')


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def merge_train_files(train_txt_path, train_out_path, output_path):
    """Merge `train.txt` and `train.out` into a single JSON file."""
    with open_file(train_txt_path, 'rt') as txt_file, \
         open_file(train_out_path, 'rt') as out_file, \
         open(output_path, 'w', encoding='utf-8') as merged_file:

        for txt_line, out_line in zip(txt_file, out_file):
            try:
                txt_json = json.loads(txt_line.strip())
                out_json = json.loads(out_line.strip())

                merged_json = {
                    "prompt": txt_json.get("prompt", ""),
                    "constraints": txt_json.get("constraints", ""),
                    "output": out_json.get("output", "")
                }

                merged_file.write(json.dumps(merged_json) + '\n')

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")

    print(f"Merging completed. Merged file saved to {output_path}")


def create_orpo_dataset(merged_file_path, default_file_path, output_path):
    """Create ORPO dataset from merged and default datasets."""
    orpo_dataset_dict = {"chosen": [], "rejected": []}

    # similarity_threshold = 50


    # Open both datasets
    with open(merged_file_path, 'rt', encoding='utf-8') as merged_f, \
        open(default_file_path, 'rt', encoding='utf-8') as default_f:

        merged_lines = merged_f.readlines()
        default_lines = default_f.readlines()

        # Iterate over the datasets
        for line_num, (merged_line, default_line) in enumerate(tqdm(zip(merged_lines, default_lines), desc="Processing samples", total=len(merged_lines)), 1):
            try:
                # Read data from merged dataset
                merged_data = json.loads(merged_line.strip())
                prompt = merged_data.get('prompt', '')
                constraints = merged_data.get('constraints', '')
                output_merged = merged_data.get('output', '').strip()

                default_data = json.loads(default_line.strip())
                output_default = default_data.get('output', '').strip()

                chosen = output_merged
                rejected = output_default

                prompt_text = prompt + '\n' + constraints
                chosen_messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides useful answers without too much extra output.",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt_text}"
                    },
                    {
                        "role": "assistant",
                        "content": f"{chosen}"
                    },
                ]

                rejected_messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides useful answers without too much extra output.",
                    },
                    {
                        "role": "user",
                        "content": f"{prompt_text}"
                    },
                    {
                        "role": "assistant",
                        "content": f"{rejected}"
                    },
                ]

                # orpo_dataset_dict['prompt'].append(messages)
                orpo_dataset_dict['chosen'].append(chosen_messages)
                orpo_dataset_dict['rejected'].append(rejected_messages)

            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue



def extract_gz_files(data_dir, files):
    """Extract multiple .gz files."""
    for gz_file in files:
        input_path = Path(data_dir) / gz_file
        output_path = input_path.with_suffix('') 
        with gzip.open(input_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted: {input_path} -> {output_path}")


def main():
    
    data_dir = './data'

    gz_files = ['train.out.gz', 'train.txt.gz', 'train_default.out.gz']
    extract_gz_files(data_dir, gz_files)


    train_txt = os.path.join(data_dir, 'train.txt')
    train_out = os.path.join(data_dir, 'train.out')
    merged_output = os.path.join(data_dir, 'train_merged.json')
    orpo_output = os.path.join(data_dir, 'orpo.json')

    # Merge train files
    print("Merging train files...")
    merge_train_files(train_txt, train_out, merged_output)

    # Create ORPO dataset
    print("Creating ORPO dataset...")
    create_orpo_dataset(merged_output, os.path.join(data_dir, 'train_default.out'), orpo_output)


if __name__ == '__main__':
    main()
