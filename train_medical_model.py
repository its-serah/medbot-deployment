#!/usr/bin/env python3
"""
Medical Chatbot Model Training Script
====================================

Fine-tune GPT-2 small model on medical Q&A data using LoRA (Low-Rank Adaptation)
for efficient training and deployment.
"""

import json
import logging
import os
import torch
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalModelTrainer:
    """
    Medical chatbot model trainer using LoRA fine-tuning.
    """
    
    def __init__(
        self,
        base_model_name: str = "gpt2",
        training_data_path: str = "medical_training_data.json",
        output_dir: str = "medbot-finetuned",
        max_length: int = 512
    ):
        """
        Initialize the medical model trainer.
        
        Args:
            base_model_name: Base model to fine-tune (default: gpt2)
            training_data_path: Path to training data JSON file
            output_dir: Directory to save the fine-tuned model
            max_length: Maximum sequence length for training
        """
        self.base_model_name = base_model_name
        self.training_data_path = training_data_path
        self.output_dir = output_dir
        self.max_length = max_length
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def load_training_data(self) -> List[Dict[str, str]]:
        """Load training data from JSON file."""
        logger.info(f"Loading training data from {self.training_data_path}")
        
        if not os.path.exists(self.training_data_path):
            raise FileNotFoundError(f"Training data file not found: {self.training_data_path}")
        
        with open(self.training_data_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} training examples")
        return data
    
    def prepare_model_and_tokenizer(self):
        """Load and prepare the base model and tokenizer."""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            padding_side="right",  # Important for causal LM
            trust_remote_code=True
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True
        )
        
        # Move model to device if not using device_map
        if self.device.type == "cpu":
            self.model = self.model.to(self.device)
        
        # Resize token embeddings if necessary
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora(self):
        """Set up LoRA (Low-Rank Adaptation) for efficient fine-tuning."""
        logger.info("Setting up LoRA configuration")
        
        # LoRA configuration for GPT-2
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,  # LoRA dropout
            target_modules=["c_attn", "c_proj"],  # GPT-2 specific modules
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info("LoRA setup complete")
    
    def create_prompt(self, input_text: str, output_text: str) -> str:
        """
        Create a medical prompt in a conversational format.
        
        Args:
            input_text: User's medical question
            output_text: Assistant's medical response
            
        Returns:
            Formatted prompt string
        """
        return f"Patient: {input_text}\\n\\nMedical Assistant: {output_text}{self.tokenizer.eos_token}"
    
    def preprocess_data(self, raw_data: List[Dict[str, str]]) -> Dataset:
        """
        Preprocess the raw training data into a tokenized dataset.
        
        Args:
            raw_data: List of input-output pairs
            
        Returns:
            Tokenized dataset ready for training
        """
        logger.info("Preprocessing training data")
        
        # Create prompts
        prompts = []
        for example in raw_data:
            prompt = self.create_prompt(example["input"], example["output"])
            prompts.append(prompt)
        
        # Tokenize all prompts
        def tokenize_function(examples):
            # Tokenize the prompts
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",  # Pad to max_length for consistent batching
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # For causal language modeling, labels are the same as input_ids
            # Convert tensors back to lists for dataset compatibility
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            # Convert back to lists for dataset storage
            return {
                "input_ids": tokenized["input_ids"].tolist(),
                "attention_mask": tokenized["attention_mask"].tolist(), 
                "labels": tokenized["labels"].tolist()
            }
        
        # Create dataset
        dataset = Dataset.from_dict({"text": prompts})
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info(f"Dataset preprocessed: {len(dataset)} examples")
        return dataset
    
    def train_model(self):
        """Train the medical model using the prepared dataset."""
        logger.info("Starting model training")
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,  # Start with 3 epochs
            per_device_train_batch_size=4,  # Small batch size for stability
            gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
            warmup_steps=100,
            weight_decay=0.01,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            eval_strategy="no",  # No evaluation for now
            learning_rate=5e-5,  # Conservative learning rate
            lr_scheduler_type="cosine",
            fp16=self.device.type == "cuda",  # Use fp16 only on GPU
            dataloader_drop_last=True,
            report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
            run_name=f"medbot-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            remove_unused_columns=False,
        )
        
        # Simple data collator - let the tokenizer handle padding during batching
        def data_collator(features):
            # features is a list of dicts with 'input_ids', 'attention_mask', 'labels'
            batch = {}
            for key in features[0].keys():
                batch[key] = torch.tensor([f[key] for f in features])
            return batch
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Training started...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving the trained model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training metadata
        metadata = {
            "base_model": self.base_model_name,
            "training_data": self.training_data_path,
            "num_examples": len(self.dataset),
            "max_length": self.max_length,
            "training_completed": datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, "training_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training completed! Model saved to: {self.output_dir}")
    
    def run_training(self):
        """Run the complete training pipeline."""
        logger.info("Starting medical model training pipeline")
        
        try:
            # Load training data
            raw_data = self.load_training_data()
            
            # Prepare model and tokenizer
            self.prepare_model_and_tokenizer()
            
            # Set up LoRA
            self.setup_lora()
            
            # Preprocess data
            self.dataset = self.preprocess_data(raw_data)
            
            # Train the model
            self.train_model()
            
            logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    """Main training function."""
    # Initialize wandb (optional)
    if os.getenv("WANDB_API_KEY"):
        wandb.init(
            project="medbot-training",
            name=f"medbot-gpt2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    
    # Initialize trainer
    trainer = MedicalModelTrainer(
        base_model_name="gpt2",  # Using GPT-2 small (124M parameters)
        training_data_path="medical_training_data.json",
        output_dir="./medbot-finetuned",
        max_length=512
    )
    
    # Run training
    trainer.run_training()
    
    print("\\n" + "="*60)
    print("🎉 MEDICAL MODEL TRAINING COMPLETED! 🎉")
    print("="*60)
    print(f"✅ Model saved to: ./medbot-finetuned")
    print(f"✅ Training data: {len(trainer.dataset)} examples")
    print(f"✅ Base model: GPT-2 (124M parameters)")
    print(f"✅ Fine-tuning: LoRA (Low-Rank Adaptation)")
    print("="*60)
    print("Your custom medical model is ready for deployment!")


if __name__ == "__main__":
    main()
