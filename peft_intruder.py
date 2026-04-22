from peft import PeftModel
from peft.tuners.lora.intruders import reduce_intruder_dimension
from transformers import LlavaForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
import torch
import os
import json
import time

last_checkpoint = get_last_checkpoint("llava-emotion-lora")

logs_path = os.path.join(last_checkpoint, "trainer_state.json")
with open(logs_path, "r") as f:
    check_data = json.load(f)
log_history = check_data["log_history"]

base_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-1.5-7b-hf",
    torch_dtype=torch.float16,  # float32
    low_cpu_mem_usage=True,
)
tuned_model = PeftModel.from_pretrained(base_model, check_data["best_model_checkpoint"])

start_time = time.time()

reduce_intruder_dimension(
    tuned_model,
    #threshold_epsilon=0.7,
)

# about 12 min
print("Reduction Intruder Dimension took {:.2f} Sec".format(time.time() - start_time))