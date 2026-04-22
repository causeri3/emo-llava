from peft import PeftModel
from transformers import LlavaForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from peft.tuners.lora.intruders import reduce_intruder_dimension
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
    torch_dtype=torch.float16,  # consider float32 on MPS if needed
    low_cpu_mem_usage=True,
)
tuned_model = PeftModel.from_pretrained(base_model, check_data["best_model_checkpoint"])

# run on a subset of layers to keep it fast on CPU
# monkey-patch to limit iterations
# original_named_modules = tuned_model.named_modules
#
# counter = 0
# max_layers = 500  # start with just 10 layers
#
# def limited_named_modules():
#     global counter
#     for name, module in original_named_modules():
#         yield name, module
#         counter += 1
#         if counter >= max_layers:
#             return
#
# tuned_model.named_modules = limited_named_modules

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU], # CUDA, CPU
    record_shapes=True,
    with_stack=False
) as prof:
    reduce_intruder_dimension(tuned_model, threshold_epsilon=0.7)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

# GPU

from peft import PeftModel
from transformers import LlavaForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from peft.tuners.lora.intruders import reduce_intruder_dimension
import torch
import os
import json

last_checkpoint = get_last_checkpoint("llava-emotion-lora")
logs_path = os.path.join(last_checkpoint, "trainer_state.json")
with open(logs_path, "r") as f:
    check_data = json.load(f)
log_history = check_data["log_history"]

device = torch.device("cuda")

base_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

tuned_model = PeftModel.from_pretrained(base_model, check_data["best_model_checkpoint"])

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=False
) as prof:
    reduce_intruder_dimension(tuned_model, threshold_epsilon=0.7)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))



# GPU


from peft import PeftModel
from transformers import LlavaForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from peft.tuners.lora.intruders import reduce_intruder_dimension
import torch
import os
import json

last_checkpoint = get_last_checkpoint("llava-emotion-lora")
logs_path = os.path.join(last_checkpoint, "trainer_state.json")
with open(logs_path, "r") as f:
    check_data = json.load(f)
log_history = check_data["log_history"]

device = torch.device("cuda")

base_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

tuned_model = PeftModel.from_pretrained(base_model, check_data["best_model_checkpoint"])
# no need to .to(device) again — PeftModel inherits the base model's device

# run on a subset of layers to keep it fast
original_named_modules = tuned_model.named_modules
counter = 0
max_layers = 500

def limited_named_modules():
    global counter
    for name, module in original_named_modules():
        yield name, module
        counter += 1
        if counter >= max_layers:
            return

tuned_model.named_modules = limited_named_modules

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=False
) as prof:
    reduce_intruder_dimension(tuned_model, threshold_epsilon=0.7)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))  # <-- sort by GPU time now



# MPS


from peft import PeftModel
from transformers import LlavaForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from peft.tuners.lora.intruders import reduce_intruder_dimension
import torch
import os
import json

last_checkpoint = get_last_checkpoint("llava-emotion-lora")
logs_path = os.path.join(last_checkpoint, "trainer_state.json")
with open(logs_path, "r") as f:
    check_data = json.load(f)
log_history = check_data["log_history"]

device = torch.device("cuda")

base_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)  # <-- move model to GPU

tuned_model = PeftModel.from_pretrained(base_model, check_data["best_model_checkpoint"])
tuned_model = tuned_model.to("mps")
# warm up
reduce_intruder_dimension(tuned_model, threshold_epsilon=0.7)
# time it on MPS
torch.mps.synchronize()  # important - waits for MPS ops to complete
start = time.perf_counter()
reduce_intruder_dimension(tuned_model, threshold_epsilon=0.7)
torch.mps.synchronize()  # wait again before stopping timer
mps_time = time.perf_counter() - start
# time it on CPU for comparison
tuned_model = tuned_model.to("cpu")
start = time.perf_counter()
reduce_intruder_dimension(tuned_model, threshold_epsilon=0.7)
cpu_time = time.perf_counter() - start
print(f"MPS: {mps_time:.2f}s")
print(f"CPU: {cpu_time:.2f}s")

"""
MPS: 755.36s
CPU: 757.99s
"""

"""
looks like SVD isn't even running on the GPU with mps
"""