from peft import PeftModel
from utils.intruders_lowrank import reduce_intruder_dimension
from transformers import LlavaForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
import torch
from torch.utils.benchmark import Timer
import os
import json

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


lowrank_svd = Timer(
    stmt="reduce_intruder_dimension(tuned_model)",
    globals={"tuned_model":tuned_model,
             "reduce_intruder_dimension": reduce_intruder_dimension})

#perc_intruder:  # 0=30%, 1=60%, 2=90%
lowrank_svd_30_per = Timer(
    stmt="reduce_intruder_dimension(tuned_model, target_rate=.30)",
    globals={"tuned_model":tuned_model,
             "reduce_intruder_dimension": reduce_intruder_dimension})

lowrank_svd_60_per = Timer(
    stmt="reduce_intruder_dimension(tuned_model, target_rate=.60)",
    globals={"tuned_model":tuned_model,
             "reduce_intruder_dimension": reduce_intruder_dimension})

lowrank_svd_80_per = Timer(
    stmt="reduce_intruder_dimension(tuned_model, target_rate=.80)",
    globals={"tuned_model":tuned_model,
             "reduce_intruder_dimension": reduce_intruder_dimension})

lowrank_svd_100_per = Timer(
    stmt="reduce_intruder_dimension(tuned_model, target_rate=1.)",
    globals={"tuned_model":tuned_model,
             "reduce_intruder_dimension": reduce_intruder_dimension})



full_svd = Timer(
    stmt="reduce_intruder_dimension(tuned_model,use_lowrank=False)",
    globals={"tuned_model":tuned_model,
             "reduce_intruder_dimension": reduce_intruder_dimension})



print(lowrank_svd_30_per.blocked_autorange())
print(lowrank_svd_60_per.blocked_autorange())
print(lowrank_svd_80_per.blocked_autorange())

print(lowrank_svd_100_per.blocked_autorange())
print(full_svd.blocked_autorange())



full_svd_sec = 726.36
lowrank_per = [0.,.18,.3,.6,.8,1.]
lowrank_sec = [248.13, 351.45,456.43,686.86,745.69, 884.35]

