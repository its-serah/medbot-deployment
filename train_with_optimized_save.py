#!/usr/bin/env python3
"""
Complete Training Script with Optimized Model Saving
====================================================

This shows exactly how to add the optimized saving to your training pipeline.
Copy this code into your actual training script where you save your model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import os

def train_and_save_model():
    """
    Example training function with optimized saving.
    Replace this with your actual training code.
    """
    
    print("🚀 Starting training with optimized saving...")
    
    # Example: Load your base model (replace with your actual model)
    model_name = "microsoft/DialoGPT-small"  # Replace with your model
    
    print(f"📥 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("🔧 Model loaded, ready for training...")
    
    # === YOUR TRAINING CODE GOES HERE ===
    # Replace this section with your actual fine-tuning code
    print("💡 [Simulation] Training would happen here...")
    print("   - Load your medical dataset")
    print("   - Set up training arguments") 
    print("   - Run trainer.train()")
    print("   - Training complete!")
    
    # === OPTIMIZED SAVING (ADD THIS TO YOUR CODE) ===
    save_directory = "./medbot_model_optimized"
    
    print(f"💾 Saving optimized model to {save_directory}...")
    
    # Create directory
    os.makedirs(save_directory, exist_ok=True)
    
    # Save model with optimizations
    print("  📦 Saving model with half precision...")
    model.save_pretrained(
        save_directory,
        torch_dtype=torch.float16,  # 🎯 Use half precision (saves ~50% space)
        safe_serialization=True     # 🎯 Use safer, smaller format
    )
    
    # Save tokenizer
    print("  🔤 Saving tokenizer...")
    tokenizer.save_pretrained(save_directory)
    
    # Check file sizes
    print("  📊 Model files created:")
    total_size = 0
    for root, dirs, files in os.walk(save_directory):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path)
            total_size += size
            print(f"    {file}: {size / (1024*1024):.1f} MB")
    
    print(f"  ✅ Total model size: {total_size / (1024*1024):.1f} MB")
    print(f"  🎉 Model saved successfully to {save_directory}")
    
    return save_directory


def test_optimized_loading(model_path):
    """Test that the optimized model loads correctly."""
    
    print(f"\n🧪 Testing optimized model loading from {model_path}...")
    
    try:
        # Test loading with our MedBotModel
        from model import MedBotModel
        
        medbot = MedBotModel(model_path=model_path)
        
        # Test a simple question
        test_question = "What is diabetes?"
        response = medbot.generate_response(test_question)
        
        print(f"  ❓ Test question: {test_question}")
        print(f"  ✅ Model response: {response[:100]}...")
        print("  🎉 Optimized model loading works!")
        
    except Exception as e:
        print(f"  ❌ Loading test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 MEDBOT TRAINING WITH OPTIMIZED SAVING")
    print("=" * 60)
    
    # Run the training and saving
    saved_model_path = train_and_save_model()
    
    # Test the saved model
    test_optimized_loading(saved_model_path)
    
    print("\n" + "=" * 60)
    print("✨ SUMMARY: What you need to add to your training code:")
    print("=" * 60)
    print("""
# Add these lines after your training is complete:

# 1. Save model with optimizations
model.save_pretrained(
    "./medbot_model",
    torch_dtype=torch.float16,  # Half precision = 50% smaller
    safe_serialization=True     # Better format
)

# 2. Save tokenizer
tokenizer.save_pretrained("./medbot_model")

# That's it! Your model will be optimized for Railway deployment.
    """)
    
    print("🚀 Ready for Railway deployment!")
