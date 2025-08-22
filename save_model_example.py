#!/usr/bin/env python3
"""
Example: How to Save Your Model with Optimizations
==================================================

This script shows you exactly how to use the optimized saving method
when you're training or fine-tuning your medical model.

Use this code in your training script after your model is trained.
"""

from model import MedBotModel
import torch

def save_your_trained_model(model, tokenizer, save_directory="./medbot_model"):
    """
    Example function showing how to save your trained model with optimizations.
    
    Args:
        model: Your trained/fine-tuned model
        tokenizer: Your tokenizer
        save_directory: Where to save the model
    """
    
    print("Saving your model with optimizations...")
    
    # Method 1: Use the optimized saving method from MedBotModel
    MedBotModel.save_model_optimized(
        model=model,
        tokenizer=tokenizer, 
        save_path=save_directory,
        save_adapter_only=True  # Set to True if you used LoRA/PEFT (much smaller!)
    )
    
    print(f"✅ Model saved to {save_directory}")


def alternative_direct_save(model, tokenizer, save_directory="./medbot_model"):
    """
    Alternative: Save directly with the same optimizations.
    
    This is the code you can add directly to your training script.
    """
    
    print("Saving model directly with optimizations...")
    
    # Save the model with optimizations
    model.save_pretrained(
        save_directory,
        torch_dtype=torch.float16,  # Use half precision (saves ~50% space)
        safe_serialization=True     # Use safer, more efficient format
    )
    
    # Save the tokenizer
    tokenizer.save_pretrained(save_directory)
    
    print(f"✅ Model saved to {save_directory}")


if __name__ == "__main__":
    print("📝 This is an example script showing optimized model saving.")
    print("\n🔧 In your actual training code, add this after training:")
    print("""
# When saving your model, use these optimizations:
model.save_pretrained("./medbot_model", 
                     torch_dtype=torch.float16,  # Use half precision
                     safe_serialization=True)    # Smaller file format

# Also save your tokenizer:
tokenizer.save_pretrained("./medbot_model")

# For LoRA/PEFT models, this will automatically save only the adapter weights!
    """)
    
    print("\n💡 Benefits of these optimizations:")
    print("✅ torch_dtype=torch.float16 → Reduces model size by ~50%")
    print("✅ safe_serialization=True → Uses safetensors format (smaller, safer)")
    print("✅ LoRA adapters → Only saves adapter weights, not full model")
    print("✅ Better for Railway deployment → Smaller files = faster uploads")
