from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# Path to your fine-tuned model
fine_tuned_model_path = "./downloads/checkpoint_merged/llava_merged/"

# Load the fine-tuned model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=fine_tuned_model_path,
    model_base=None,  # Adjust if necessary based on your training configuration
    model_name=get_model_name_from_path(fine_tuned_model_path)
)

# Evaluation setup
prompt = "why was this photo taken?"
image_file = "/dataset/images/be763e16-8714-4bf7-8bd5-3d4cc05bd160.jpg"
# Set up evaluation arguments
args = type('Args', (), {
    "model_path": fine_tuned_model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(fine_tuned_model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
# Perform evaluation with the fine-tuned model
eval_model(args)