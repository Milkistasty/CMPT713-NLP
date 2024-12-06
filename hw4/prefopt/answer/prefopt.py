import os
import sys
import json
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def decode_all(model, device, inputfile, tokenizer):
    pipe = pipeline(
        "text-generation",
        device=device,
        model=model,
        tokenizer=tokenizer,  # Explicitly provide the tokenizer
        model_kwargs={
            "torch_dtype": torch.bfloat16,
        }
    )

    text = Path(inputfile).read_text().strip().split('\n')
    for line in (text):
        line = line.strip()
        data = json.loads(line)
        prompt_text = data['prompt'] + '\n' + data['constraints']
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides useful answers without too much extra output.",
            },
            {
                "role": "user",
                "content": f"{prompt_text}"
            },
        ]
        outputs = pipe(
            messages,
            do_sample=True,
            temperature=0.3,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
        )
        if outputs:
            print(json.dumps({'output': outputs[0]["generated_text"][-1]["content"]}))
        else:
            print(json.dumps({'output': "Sorry!"}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Decode using a trained LLM")
    parser.add_argument("-i", "--inputfile",
                        default=os.path.join('data', 'input', 'dev.txt'),
                        help="Input file containing prompts")
    parser.add_argument("-d", "--device",
                        default='cpu',
                        help="Device to use (e.g., 'cpu', 'cuda')")

    parser.add_argument("-m", "--model", type=str,
                        default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="model path or model id")

    parser.add_argument("-l", "--logfile", dest="logfile", default=None,
                        help="log file for debugging")


    args = parser.parse_args()

    # Set device
    device = args.device
    if torch.cuda.is_available() and device == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}", file=sys.stderr)

    # Load the trained model and tokenizer from the results directory
    results_dir = "./results/checkpoint-1000"
    tokenizer = AutoTokenizer.from_pretrained(results_dir)
    model = AutoModelForCausalLM.from_pretrained(results_dir)

    # Decode outputs
    decode_all(model, device, args.inputfile, tokenizer)
