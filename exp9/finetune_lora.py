"""Optional LoRA fine-tuning template. Use only with a prepared JSONL dataset and suitable GPU."""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset", default="data/train.jsonl")
    p.add_argument("--out", default="checkpoints/lora-adapter")
    p.add_argument("--epochs", type=float, default=1)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    if "text" not in dataset.column_names:
        raise ValueError('JSONL rows must contain a "text" field, e.g. {"text":"User: ... Assistant: ..."}')
    peft_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    training_args = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=2e-4, logging_steps=5, save_strategy="epoch",
        report_to="none", fp16=__import__("torch").cuda.is_available()
    )
    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=dataset,
        peft_config=peft_config, processing_class=tokenizer
    )
    trainer.train()
    trainer.model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("Saved LoRA adapter to", args.out)


if __name__ == "__main__":
    main()
