# TUNE.GPT2.QLORA

```bash
{
    "MODEL": "gpt2-xl",
    "TRAIN_FILE": "MICRO.PERSONA.MINI.DECODER.json",
    "OUTPUT_DIR": "MINI.DECODER.GPT2.XL.QLORA",
    "BATCH_SIZE": "4",    
    "NUM_SAMPLES": "10",
    "OVERWRITE": "True",
    "EPOCHS": "1",
    "LRATE": "2e-4",
    "STEPS": "4",
    "SAVE_STEPS": "500",
    "LOAD_4BIT": "True",
    "LOAD_8BIT": "False",
    "FULLTUNE": "False",
    "MAXSEQ": "1024",
    "OPTIM": "adamw"
}
```

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import set_seed
import os,random,sys,json,re

random.seed()
def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system.
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

seed = get_truly_random_seed_through_os()
set_seed(seed)

json_file = sys.argv[1]
with open(json_file,"r") as jf:
    config = json.load(jf)

MODEL = config["MODEL"]
TRAIN_FILE = config["TRAIN_FILE"]
OUTPUT_DIR = config["OUTPUT_DIR"]
OVERWRITE = bool(config["OVERWRITE"])
BATCH_SIZE = int(config['BATCH_SIZE'])
EPOCHS = int(config["EPOCHS"])
LRATE = float(config["LRATE"])
STEPS = int(config["STEPS"])
LOAD_4BIT = config["LOAD_4BIT"].lower() == "true"
LOAD_8BIT = config["LOAD_8BIT"].lower() == "true"
FULLTUNE = config["FULLTUNE"].lower() == "true"
OPTIMIZER = config["OPTIM"]
MAXSEQ= int(config["MAXSEQ"])
if("PERCENT" in config):
    PERCENT = int(config["PERCENT"])
else:
    PERCENT = 100
if("NUM_SAMPLES" in config):
    NUM_SAMPLES = int(config["NUM_SAMPLES"])
else:
    NUM_SAMPLES=0
if("SELECT_OUTPUT" in config):
    SELECT_OUTPUT = config["SELECT_OUTPUT"]
else:
    SELECT_OUTPUT = "output"
if("SHUFFLE" in config):
    os.system("python " + config["SHUFFLE"])

#config["EPOCHS"]= str(EPOCHS)
#with open(json_file,"w") as jf:
#    json.dump(config,jf,indent=4)

dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
dtype = torch.bfloat16
bf16 = True

print("-----------------------------------------------------")
print("Configuration")
print("-----------------------------------------------------")
print("MODEL",MODEL)
print("TRAIN_FILE",TRAIN_FILE)
print("OUTPUT_DIR",OUTPUT_DIR)
print("BATCH_SIZE","AUTO")
print("EPOCHS",EPOCHS)
print("LRATE",LRATE)
print("STEPS",STEPS)
print("LOAD_4BIT",LOAD_4BIT)
print("LOAD_8BIT",LOAD_8BIT)
print("FULLTUNE",FULLTUNE)
print("MAXSEQ",MAXSEQ)
print("-----------------------------------------------------")


dataset = load_dataset("json", data_files=TRAIN_FILE, split="train").shuffle(seed)

if(NUM_SAMPLES==0):
    NUM_SAMPLES = int(len(dataset) * (PERCENT / 100))

# Select only the first X% of the dataset
dataset = dataset.select(range(NUM_SAMPLES))

output_tags = [
    "output",
    "response",
    "answer",
    "assistant",       
]

ignore_tags = [
    "style_guide",
    "ignore",
    "system",
    "select_output",
    "opcode",    
]
input_tags = [
    "instruction",
    "question",
    "input",    
    "context",
    "user",
    "keywords",
    "tweets",
    "hashtags",
    "visualtags",
    "entitytags",
    "attributes",
    "characteristics",
    "features",
    "properties",
    "key_points",
    "key_attributes",
    "key_characteristics",
    "key_features",
    "key_properties",        
    "term",
    "entity",
    "title",
    "definition",
    "description",
    "summary",
    "positive_prompt",
    "negative_prompt",
    "ai_persona_prompt",
    "persona_prompt",
    "detailed_ai_persona_prompt",
    "entity_persona",
]


def select_input(ex):
    for k in input_tags:        
        if(k in ex): return ex[k]
    return None 

def select_output(example):
    return example[SELECT_OUTPUT]
    

# 2. Tokenizer
model_name = MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def to_text(example):
    global SELECT_OUTPUT
    #print(SELECT_OUTPUT)
    ex = dict(example)    
        
    input = {}
    if(SELECT_OUTPUT == "Random"):
        x = random.choice(example.keys())
        while x in input_tags:
            x = random.choice(example.keys())
        SELECT_OUTPUT = x
    
    for k in list(ex.keys()):
        if(ex[k] is None): continue
        if(k != SELECT_OUTPUT):
            input[k] = ex[k]
       
    temp = ''
    for k in list(input.keys()):
        if(type(input[k]) is list):
            temp = temp + k + ": " + ",".join(input[k]) + "\n"
        else:
            temp = temp + k + ": " + input[k] + "\n"
    if(SELECT_OUTPUT != 'output'):
        temp = SELECT_OUTPUT + ": " + temp.strip()
    else:
        temp = temp.strip()
    input = f"""```{temp}```"""
    
    temp = select_output(ex)
    if(type(temp) is list):
        temp = '\n\n'.join(temp)
    temp = temp.strip()
    output = f"""```{temp}```<|endoftext|>"""

    #print("### Prompt:")
    #print(input)
    ##print("### Response:")
    #print(output)
    
    text = f"### Prompt:\n\n{input}\n\n### Response:\n\n{output}"
    
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAXSEQ)
    labels = enc["input_ids"].copy()

    # Find "### Response:" token index
    resp_tok_ids = tokenizer("### Response:", add_special_tokens=False)["input_ids"]
    input_ids = enc["input_ids"]

    # Find the boundary (first match of response marker)
    for i in range(len(input_ids) - len(resp_tok_ids)):
        if input_ids[i:i+len(resp_tok_ids)] == resp_tok_ids:
            boundary = i + len(resp_tok_ids)
            break
    else:
        boundary = len(input_ids)

    # Mask everything before the response
    labels[:boundary] = [-100] * boundary
    enc["labels"] = labels
    return enc

train_dataset = dataset.map(to_text)
last_checkpoint = None
last_checkpoint_step = 0




print("-------------------------------------------------------------")

if os.path.isdir(OUTPUT_DIR):
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

if last_checkpoint is not None:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    # Extract the step count from checkpoint path (e.g., "checkpoint-500")
    last_checkpoint_step = int(last_checkpoint.split("-")[-1])
else:
    print("No previous checkpoint found. Training from scratch.")

total_samples = len(train_dataset)
print("Total Samples:",total_samples)
num_gpus = max(1, torch.cuda.device_count())  # Ensure at least 1 (for CPU training)
print("Num GPU:",num_gpus)
print("Batch Size/Device:",BATCH_SIZE)
print("Gradient Steps:", STEPS)
# Compute steps for one epoch based on current dataset size
num_update_steps_per_epoch = total_samples // (
    num_gpus * BATCH_SIZE * STEPS
)

print("Steps: ",num_update_steps_per_epoch)
# Adjust max_steps based on last checkpoint
max_steps = last_checkpoint_step + num_update_steps_per_epoch
print(f"Updated max_steps: {max_steps}")

print("-------------------------------------------------------------")

resume = last_checkpoint is not None


from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,      # nested quantization for more compression
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (better than fp4)
    bnb_4bit_compute_dtype="bfloat16"    # computation dtype: bfloat16 (preferred) or float16
)

# 3. Model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,   # 👈 use the config from above
    device_map="auto",   
)
from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


# 4. Add LoRA adapters (PEFT)
peft_config = LoraConfig(
    r=16,              # rank
    lora_alpha=32,     # scaling
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. Training config
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,    
    num_train_epochs=EPOCHS,
    learning_rate=LRATE,
    bf16=True,
    gradient_accumulation_steps=STEPS,
    optim="paged_adamw_8bit",   # use bnb optimizer
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)



# 6. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=training_args,
    peft_config=peft_config,   # 👈 LoRA config here
    max_seq_length=MAXSEQ,
    packing=True
)


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

checkpoint = None
if resume == True:
    checkpoint = last_checkpoint

trainer_stats = trainer.train(resume_from_checkpoint=checkpoint)

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print("Saving Model....")
#trainer.save(OUTPUT_DIR)
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```
