#!/usr/bin/env python3
"""
🎯 EXACTLY WHAT TO ADD TO YOUR TRAINING CODE
============================================

Copy these lines into your training script after your model is trained.
"""

# ============================================================================
# ADD THIS TO YOUR TRAINING SCRIPT (after training is complete):
# ============================================================================

def save_optimized_model(model, tokenizer, save_path="./medbot_model"):
    """
    🎯 COPY THIS FUNCTION INTO YOUR TRAINING CODE
    
    Then call it after training:
    save_optimized_model(your_trained_model, your_tokenizer)
    """
    import torch
    import os
    
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    
    print(f"💾 Saving optimized model to {save_path}...")
    
    # 🎯 KEY OPTIMIZATION 1: Save model with half precision + safe format
    model.save_pretrained(
        save_path,
        torch_dtype=torch.float16,  # 50% smaller file size!
        safe_serialization=True     # Better, safer format
    )
    
    # 🎯 KEY OPTIMIZATION 2: Save tokenizer  
    tokenizer.save_pretrained(save_path)
    
    print("✅ Optimized model saved!")
    
    # Optional: Check file sizes
    total_size = sum(
        os.path.getsize(os.path.join(root, file))
        for root, dirs, files in os.walk(save_path)
        for file in files
    )
    print(f"📊 Total size: {total_size / (1024*1024):.1f} MB")


# ============================================================================
# OR EVEN SIMPLER - JUST ADD THESE 2 LINES TO YOUR EXISTING CODE:
# ============================================================================

# Replace your current model.save_pretrained() call with this:
"""
model.save_pretrained(
    "./medbot_model", 
    torch_dtype=torch.float16,  # Add this line
    safe_serialization=True     # Add this line
)
tokenizer.save_pretrained("./medbot_model")  # Don't forget tokenizer
"""

print("=" * 60)
print("🎯 WHAT TO ADD TO YOUR TRAINING CODE:")
print("=" * 60)
print("""
# Instead of:
# model.save_pretrained("./medbot_model")

# Use this:
model.save_pretrained(
    "./medbot_model",
    torch_dtype=torch.float16,  # ← ADD THIS (50% smaller)
    safe_serialization=True     # ← ADD THIS (better format)
)
tokenizer.save_pretrained("./medbot_model")  # ← ADD THIS (save tokenizer)
""")

print("=" * 60)
print("🚀 BENEFITS FOR RAILWAY DEPLOYMENT:")
print("=" * 60)
print("✅ 50% smaller model files")
print("✅ Faster upload to Railway") 
print("✅ Less memory usage")
print("✅ Better performance")
print("✅ Safer file format")

if __name__ == "__main__":
    print("\n🎉 That's it! Just add those 2 parameters to your save_pretrained() call!")
