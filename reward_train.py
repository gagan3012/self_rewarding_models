import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import DPOTrainer
from fire import Fire
from random import randrange

# model_name = "/lustre07/scratch/gagan30/arocr/meta-llama/arabic/outputs/Mistral-7B-Instruct-v0.2-instruct-ar/checkpoint-4000-merged"
# new_model = "inst-ar-dpo-mistral"

def fix_prompt(source, target, dialect):
    prompt_formatted = f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: 
Translate from English to {dialect}:
{source}
Assistant:
{target}
'''
    return prompt_formatted

def DPO(model_name="/lustre07/scratch/gagan30/arocr/meta-llama/arabic/MegaArabic-instruct-ar/checkpoint-5400-merged", new_model="MegaArabic-instruct-dpo-ar", dataset_name=""):

    dataset = load_dataset("parquet", data_files=dataset_name, 
                        cache_dir="/lustre07/scratch/gagan30/arocr/cache",
                            split="train", num_proc=4)

    
    # new_model = model_name.replace("sft", "dpo")
    # new_model = new_model + f"-v{turns}"

    print(f"dataset size: {len(dataset)}")
    print(dataset[randrange(len(dataset))])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def format_instruction_dpo(sample):
        sample['chosen']= tokenizer.apply_chat_template(sample['chosen'], tokenize=False, add_generation_prompt=True)
        sample['rejected']= tokenizer.apply_chat_template(sample['rejected'], tokenize=False, add_generation_prompt=True)

        # sample['chosen']= fix_prompt(sample['chosen'])
        # sample['rejected']= fix_prompt(sample['rejected'])

        # sample['prompt']= fix_prompt(sample['source'], sample['target_positive'], dialect)
        # sample['chosen']= fix_prompt(sample['source'], sample['target_positive'], dialect)
        # sample['rejected']= fix_prompt(sample['source'], sample['target_negative'], dialect)

        return sample
    
    dataset = dataset.map(format_instruction_dpo)
    print(dataset[randrange(len(dataset))])


    dataset = dataset.shuffle(seed=42)

    # dataset = dataset.train_test_split(test_size=0.001)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    model.config.use_cache = False

    # Reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True
    )
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        save_strategy="steps",
        evaluation_strategy="no",
        logging_steps=1,
        output_dir=new_model,
        optim="paged_adamw_32bit",
        warmup_steps=100,
        bf16=True,
        report_to="wandb",
    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        beta=0.1,
        max_prompt_length=256,
        max_length=256,
    )

    # Fine-tune model with DPO
    dpo_trainer.train()

    dpo_trainer.model.save_pretrained(f"{new_model}/final_checkpoint")
    tokenizer.save_pretrained(f"{new_model}/final_checkpoint")

    # Flush memory
    del dpo_trainer, model, ref_model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload model in FP16 (instead of NF4)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, model_id=f"{new_model}/final_checkpoint")
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)

    # Flush memory
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    Fire(DPO)

    