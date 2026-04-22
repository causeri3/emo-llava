import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, Trainer, BatchFeature
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image
from utils.prep_data import get_sample_data_set, get_clean_data
from utils.clean_data import CSV_LABEL_PATH
from datasets import Dataset
from datasets.formatting.formatting import LazyBatch
import math
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import time


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

if DEVICE == "mps":
    FP16=False
else:
    FP16 = True

PROMPT = """USER: <image>
What emotions does this person display? Only use emotions from this list: 
'Peace', 'Affection', 'Esteem', 'Anticipation', 'Engagement', 'Confidence', 'Happiness', 'Pleasure', 'Excitement', 'Surprise', 'Sympathy', 'Doubt/Confusion', 'Disconnection', 'Fatigue', 'Embarrassment', 'Yearning', 'Disapproval', 'Aversion', 'Annoyance', 'Anger', 'Sensitivity', 'Sadness', 'Disquietment', 'Fear', 'Pain', 'Suffering'
ASSISTANT:"""


processor = AutoProcessor.from_pretrained("llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

RANK = 8  # trainable parameters; 4, 8, 16, 32
LORA_CONFIG = LoraConfig(
    r=RANK,  #default 8
    lora_alpha=RANK*2,  # scaling factor for added weight matrix (weight of the learned weights), default 8
    target_modules=["q_proj", "v_proj"], # default=None, q_proj = Query projection, k_proj = Key projection, v_proj = Value projection, o_proj = Output projection
    # lora_dropout=0.05, # default 0.0
    # bias="lora_only", #default "none"
    # task_type="CAUSAL_LM"
)

model = get_peft_model(model, LORA_CONFIG)
model.print_trainable_parameters()

def get_lengths(dataset: Dataset) -> (int, int):
    sample_images = dataset["image"][0]
    answer_tokens = processor.tokenizer(
        "Peace, Affection, Esteem, Anticipation, Engagement, Confidence, Happiness, Pleasure, Excitement, Surprise, Sympathy, Doubt/Confusion, Disconnection, Fatigue, Embarrassment, Yearning, Disapproval, Aversion, Annoyance, Anger, Sensitivity, Sadness, Disquietment, Fear, Pain, Suffering",
        add_special_tokens=False  # Don't add BOS/EOS, just count the text
    )
    answer_length = len(answer_tokens["input_ids"])
    prompt_only = processor(text=PROMPT, images=sample_images, return_tensors="pt")
    prompt_length = prompt_only["input_ids"].shape[1]

    return prompt_length, answer_length+prompt_length

if os.path.exists(CSV_LABEL_PATH):
    train_data, eval_data, test_data = get_clean_data()
else:
    train_data, eval_data, test_data = get_sample_data_set(sample_perc=0.001)

PROMPT_LENGTH, MAX_LENGTH = get_lengths(train_data)


def mask_pad_n_prompt(inputs:BatchFeature) -> BatchFeature:
    """Mask (to ignore prompt and image prediction in loss calculation)"""
    #inputs["labels"][:, :PROMPT_LENGTH] = -100
    inputs["labels"][inputs["attention_mask"] == 0] = -100
    first_real_token = inputs["attention_mask"].argmax(dim=1)  # Shape: [batch_size]
    seq_range = torch.arange(MAX_LENGTH, device=inputs["labels"].device).unsqueeze(0)  # Shape: [1, seq_length]
    prompt_mask = (seq_range >= first_real_token.unsqueeze(1)) & (
                seq_range < (first_real_token + PROMPT_LENGTH).unsqueeze(1))
    inputs["labels"][prompt_mask] = -100
    return inputs

def preprocess_function(examples:LazyBatch) -> BatchFeature:

    images = examples["image"]
    labels = examples["label"]

    texts = [PROMPT + f" {label}" for label in labels]

    inputs = processor(
        text=texts,
        images=images,
        #padding=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt" # pytorch
    )
    inputs["labels"] = inputs["input_ids"].clone()
    inputs = mask_pad_n_prompt(inputs)

    return inputs

def plot_loss(log_history:list) -> None:
    df_logs = pd.DataFrame(log_history)
    plt.figure(figsize=(8,5))
    df_train = df_logs.dropna(subset=["loss"])
    df_eval = df_logs.dropna(subset=["eval_loss"])
    plt.plot(df_train["step"], df_train["loss"], label="Train Loss", marker="o")
    plt.plot(df_eval["step"], df_eval["eval_loss"], label="Evaluation Loss", marker="s")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.legend()
    #plt.grid(True)
    plt.show()
    plt.pause(0.1)


# Preprocess both datasets
train_dataset = train_data.map(
    preprocess_function,
    batched=True,
    remove_columns=train_data.column_names
)

eval_dataset = eval_data.map(
    preprocess_function,
    batched=True,
    remove_columns=eval_data.column_names
)

EPOCHS = 5
BATCH_SIZE = 1 #2
SAVE_STEPS = math.ceil((len(train_data)/BATCH_SIZE)) # close to epoch size

training_args = TrainingArguments(
    output_dir="./llava-emotion-lora",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE, # per gpu
    gradient_accumulation_steps=1, #4,# helper, such that effective batch size is per_device_train_batch_size x gradient_accumulation_steps
    learning_rate=2e-4,
    fp16=FP16,
    logging_steps=SAVE_STEPS//4,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS//2, # about twice per epoch
    save_strategy="steps",#"epoch",
    save_steps=SAVE_STEPS//2,
    save_total_limit=1, # no of saved last checkpoints, apparently in combination with load_best_model_at_end, if best model is not one of teh last save_total_limit+1 (best nodel) will be saved
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # "loss"
    greater_is_better=False,
    remove_unused_columns=False, # necessary fot multimodal models
    push_to_hub=False,
    report_to="none",
)

# can I save checkpoint only if losses decreased - nope
# maybe make save_steps and eval_steps the same

start_time = time.time()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

start_time = time.time()
current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
print("Started Tuning at:", current_time)

trainer.train()

print("Tuning took {:.2f} Sec".format(time.time() - start_time))


#log_history = trainer.state.log_history
last_checkpoint = get_last_checkpoint("llava-emotion-lora")

logs_path = os.path.join(last_checkpoint, "trainer_state.json")
with open(logs_path, "r") as f:
    check_data = json.load(f)
log_history = check_data["log_history"]


plot_loss(log_history)

#processor = AutoProcessor.from_pretrained("llava-1.5-7b-hf")

base_model = LlavaForConditionalGeneration.from_pretrained(
    "llava-1.5-7b-hf",
    torch_dtype=torch.float16,  # consider float32 on MPS if needed
    low_cpu_mem_usage=True,
)
tuned_model = PeftModel.from_pretrained(base_model, check_data["best_model_checkpoint"])

# Save the LoRA adapter
tuned_model.save_pretrained("./llava-emotion-lora-adapter")
processor.save_pretrained("./llava-emotion-lora-adapter")

print("Training complete! Model saved to ./llava-emotion-lora-adapter")




#____________ helper functions _____________________________ #
def test_masking(train_dataset, num_samples=3):
    """Test if masking is working correctly"""
    for idx in range(min(num_samples, len(train_dataset))):
        sample = train_dataset[idx]

        input_ids = sample["input_ids"]
        labels = sample["labels"]

        print(f"\n{'=' * 80}")
        print(f"Sample {idx + 1}")
        print(f"{'=' * 80}")

        # Decode full input
        full_text = processor.decode(input_ids, skip_special_tokens=False)
        print(f"\nFull input text:\n{full_text}")
        import numpy as np
        # Find where labels are not masked
        labels_array = np.array(labels)
        non_masked_indices = np.where(labels_array != -100)[0]
        trained_tokens = labels_array[non_masked_indices]
        trained_text = processor.decode(trained_tokens.tolist(), skip_special_tokens=False)
        masked_count = np.sum(labels_array == -100)

        print(f"\nMasked tokens (prompt - should be ignored): {masked_count}")
        print(f"Non-masked tokens (answer - should be learned): {len(non_masked_indices)}")
        print(f"\nText being trained on (what model learns to predict):\n{trained_text}")

        # Show the split point
        first_non_masked = non_masked_indices[0].item()
        masked_text = processor.decode(input_ids[:first_non_masked], skip_special_tokens=False)
        print(f"\nMasked portion:\n{masked_text}")


test_masking(train_dataset)


# Inference
def predict_emotion(image_path):
    """Run inference on a face image"""
    model.eval()
    image = Image.open(image_path).convert("RGB")

    prompt = "USER: <image>\nWhat emotion is this person showing? Answer with a single word.\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)

    response = processor.decode(outputs[0], skip_special_tokens=True)
    return response
