#!/usr/bin/env python3
"""
Medical Model Fine-tuning with LoRA
===================================
Fine-tune a language model for medical Q&A using LoRA adapters
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalFineTuner:
    def __init__(self, base_model_name="distilgpt2", output_dir="./medical-lora-model"):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
        
    def setup_lora_config(self):
        """Setup LoRA configuration for efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Low rank adaptation dimension
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.05,  # LoRA dropout
            target_modules=["c_attn", "c_proj"],  # Target modules for DistilGPT2
            bias="none",
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied")
        
    def load_and_prepare_dataset(self, data_file="medical_training_data.json"):
        """Load and tokenize the medical training dataset"""
        logger.info(f"Loading dataset from {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        # Format data for training
        formatted_data = []
        for item in data:
            # Create prompt-response format
            prompt = f"Medical Question: {item['instruction']}\n\nMedical Answer: {item['output']}"
            formatted_data.append({"text": prompt})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
        return tokenized_dataset
        
    def fine_tune_model(self, dataset):
        """Fine-tune the model with LoRA"""
        logger.info("Starting LoRA fine-tuning...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=10,
            logging_steps=10,
            save_steps=50,
            evaluation_strategy="no",
            save_strategy="epoch",
            load_best_model_at_end=False,
            push_to_hub=False,
            report_to=None,
            dataloader_drop_last=False,
            learning_rate=3e-4,
            fp16=False,  # Use float32 for CPU training
            remove_unused_columns=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=None,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Fine-tuning completed! Model saved to {self.output_dir}")
        
    def test_model(self):
        """Test the fine-tuned model"""
        logger.info("Testing fine-tuned model...")
        
        # Load the fine-tuned model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map="auto"
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, self.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        
        # Test questions
        test_questions = [
            "What are the symptoms of a fast heartbeat and nausea?",
            "What causes high blood pressure?",
            "What should I do if I have chest pain?"
        ]
        
        for question in test_questions:
            prompt = f"Medical Question: {question}\n\nMedical Answer:"
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n=== Question: {question} ===")
            print(f"Response: {response[len(prompt):]}")
            print("-" * 50)

def main():
    """Main fine-tuning process"""
    print("🔬 Starting Medical Model Fine-tuning with LoRA...")
    
    # Initialize fine-tuner
    fine_tuner = MedicalFineTuner()
    
    # Setup model and tokenizer
    fine_tuner.setup_model_and_tokenizer()
    
    # Setup LoRA configuration
    fine_tuner.setup_lora_config()
    
    # Load and prepare dataset
    dataset = fine_tuner.load_and_prepare_dataset()
    
    # Fine-tune the model
    fine_tuner.fine_tune_model(dataset)
    
    # Test the model
    fine_tuner.test_model()
    
    print("✅ Medical model fine-tuning completed successfully!")

if __name__ == "__main__":
    main()
