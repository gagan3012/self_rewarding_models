import os
import gc
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, SFTTrainer
import bitsandbytes as bnb
from fire import Fire
from random import randrange
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl, Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

def SFT(model_name_org="/lustre07/scratch/gagan30/arocr/meta-llama/models/Mistral-7B-Instruct-v0.2"):

    dataset = load_dataset("parquet",data_files="/lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/ultrafeedback_binarized/train_srwm.parquet", 
                           cache_dir="/lustre07/scratch/gagan30/arocr/cache", 
                           split="train", 
                           num_proc=4)

    # print(dataset[0])

    print(f"dataset size: {len(dataset)}")
    # print(dataset[randrange(len(dataset))])

    base_model_id = f"{model_name_org}"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def format_instruction_sft(sample):
        sample['text']= tokenizer.apply_chat_template(sample['messages'], tokenize=False, add_generation_prompt=True)
        # sample['text']= fix_prompt(sample['source'], sample['target'], dialect)
        return sample
        
    dataset = dataset.map(format_instruction_sft)

    # print(dataset[randrange(len(dataset))])

    dataset = dataset.shuffle(seed=42)

    dataset = dataset.train_test_split(test_size=0.001)

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map="auto", use_cache=False)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    print_trainable_parameters(model)

    print(model)

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    model = accelerator.prepare_model(model)

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    print(model)

    class SavePeftModelCallback(TrainerCallback):
        def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
        ):
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            kwargs["model"].save_pretrained(checkpoint_folder)

            pytorch_model_path = os.path.join(
                checkpoint_folder, "pytorch_model.bin")
            torch.save({}, pytorch_model_path)
            return control

    model_name = base_model_id.split("/")[-1]
    output_dir = f"/lustre07/scratch/gagan30/arocr/meta-llama/self_rewarding_models/{model_name}-sft"

    args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            # learning_rate=2.5e-5, # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_32bit",
            logging_steps=1,              # When to start reporting loss
            save_strategy="steps",       # Save the model checkpoint every logging step
            save_steps=100,                # Save checkpoints every 50 steps
            evaluation_strategy="steps", # Evaluate the model every logging step
            eval_steps=100,               # Evaluate and save checkpoints every 50 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to="wandb",           # Comment this out if you don't want to use weights & baises
            dataloader_pin_memory=True,                           
            dataloader_num_workers=4,
            logging_first_step=True,
            lr_scheduler_type="cosine",
            seed=42,
    )

    max_seq_length = 2048

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,  # You can specify the maximum sequence length here
        tokenizer=tokenizer,
        args=args,
        packing=True,
        eval_dataset=eval_dataset,
        neftune_noise_alpha=5,
        callbacks=[SavePeftModelCallback()],
    )
    trainer.train()

    trainer.model.save_pretrained(f"{output_dir}/final_checkpoint")
    tokenizer.save_pretrained(f"{output_dir}/final_checkpoint")

    # Flush memory
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload model in FP16 (instead of NF4)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, model_id=f"{output_dir}/final_checkpoint")
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Flush memory
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

if "__main__" == __name__:
    Fire(SFT)