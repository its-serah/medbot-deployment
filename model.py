"""
MedBot Model Module
===================

This module provides the MedBotModel class that handles loading and inference
from your fine-tuned medical LLM. It includes fallback mechanisms and 
optimizations for deployment scenarios.

The model supports:
- Fine-tuned GPT-based models with LoRA adapters
- Quantized models for efficient inference
- Fallback to rule-based responses when model fails
- Optimized text generation with medical context
"""

import os
import re
import logging
from typing import Optional, Dict, Any

# Try to import ML dependencies, fall back gracefully if not available
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        BitsAndBytesConfig, pipeline
    )
    from peft import PeftModel
    ML_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ML dependencies not available: {e}")
    ML_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)

class MedBotModel:
    """
    Medical Question-Answering Model Wrapper.
    
    This class handles loading your fine-tuned model and generating responses
    to medical questions. It includes fallback mechanisms and optimizations.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the MedBot model.
        
        Args:
            model_path: Path to your fine-tuned model directory
        """
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if not ML_AVAILABLE:
            logger.warning("ML dependencies not available, using fallback mode only")
            self._setup_fallback()
            return
            
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        
        # Model configuration
        self.max_length = 512
        self.max_new_tokens = 256
        self.temperature = 0.7
        self.top_p = 0.9
        
        # Try to load the model
        model_paths = [
            model_path,
            "./models",
            "/app/models",
            "./saved_model",
            "/app/saved_model"
        ]
        
        for path in model_paths:
            if path and self._try_load_model(path):
                break
        else:
            logger.warning("Could not load fine-tuned model, using fallback")
            self._setup_fallback()
    
    def _try_load_model(self, model_path: str) -> bool:
        """
        Try to load the model from the specified path.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if not os.path.exists(model_path):
                return False
                
            logger.info(f"Attempting to load model from {model_path}")
            
            # Configure quantization for efficient inference
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ) if self.device == "cuda" else None
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model_name = self._detect_base_model(model_path)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Load LoRA adapter if available
            adapter_path = os.path.join(model_path, "adapter_model.bin")
            if os.path.exists(adapter_path):
                logger.info("Loading LoRA adapter")
                self.model = PeftModel.from_pretrained(self.model, model_path)
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def _detect_base_model(self, model_path: str) -> str:
        """
        Detect the base model name from config files.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            str: Base model name
        """
        # Try to read from config
        config_file = os.path.join(model_path, "config.json")
        if os.path.exists(config_file):
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    if "_name_or_path" in config:
                        return config["_name_or_path"]
            except:
                pass
        
        # Fallback to common medical models
        return "microsoft/DialoGPT-medium"  # Or your preferred base model
    
    def _setup_fallback(self):
        """Setup fallback rule-based responses."""
        self.knowledge_base = {
            "diabetes": "Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). There are two main types: Type 1 (autoimmune) and Type 2 (insulin resistance). Management typically involves medication, diet, and lifestyle changes. Please consult with a healthcare provider for personalized advice.",
            
            "hypertension": "High blood pressure (hypertension) occurs when the force of blood against artery walls is consistently too high. It can be managed through lifestyle changes like diet, exercise, stress management, and medication when necessary. Regular monitoring is important.",
            
            "asthma": "Asthma is a respiratory condition causing inflammation and narrowing of airways, leading to breathing difficulties. Common triggers include allergens, exercise, and stress. Treatment typically involves inhalers and avoiding known triggers.",
            
            "covid": "COVID-19 is caused by the SARS-CoV-2 virus. Symptoms can range from mild to severe. Prevention includes vaccination, masking, and good hygiene. Consult healthcare providers for current guidelines and treatment options.",
            
            "heart": "Heart conditions can vary widely. Common issues include coronary artery disease, heart failure, and arrhythmias. Maintaining a healthy lifestyle with regular exercise, proper diet, and regular check-ups is important for heart health."
        }
    
    def generate_response(self, question: str) -> str:
        """
        Generate a response to the medical question.
        
        Args:
            question: User's medical question
            
        Returns:
            str: Generated response
        """
        if not question or not question.strip():
            return "Please provide a medical question so I can assist you."
        
        # Clean and prepare the question
        question = question.strip()
        
        if self.pipeline:
            return self._generate_with_model(question)
        else:
            return self._generate_with_fallback(question)
    
    def _generate_with_model(self, question: str) -> str:
        """Generate response using the fine-tuned model."""
        try:
            # Create a medical prompt template
            prompt = f"""You are a helpful medical AI assistant. Answer the following medical question accurately and concisely. Always remind users to consult healthcare professionals for medical advice.

Question: {question}

Answer:"""
            
            # Generate response
            response = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            answer = generated_text.split("Answer:")[-1].strip()
            
            # Clean up the response
            answer = self._clean_response(answer)
            
            # Add medical disclaimer if not present
            if "healthcare" not in answer.lower() and "doctor" not in answer.lower():
                answer += "\n\nPlease consult with a healthcare professional for personalized medical advice."
            
            return answer
            
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            return self._generate_with_fallback(question)
    
    def _generate_with_fallback(self, question: str) -> str:
        """Generate response using rule-based fallback."""
        question_lower = question.lower()
        
        # Search for keywords in the knowledge base
        for condition, response in self.knowledge_base.items():
            if condition in question_lower:
                return response
        
        # Generic fallback response
        return """I understand you have a medical question, but I don't have specific information about this topic in my current knowledge base. 

For accurate medical information and advice, I recommend:
1. Consulting with a licensed healthcare provider
2. Contacting your doctor or medical clinic
3. Seeking emergency medical attention if this is urgent

Remember, AI assistants should not replace professional medical consultation."""
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the model response."""
        # Remove any remaining prompt artifacts
        response = re.sub(r'^(Answer:|Response:|A:)', '', response).strip()
        
        # Remove repeated patterns
        lines = response.split('\n')
        clean_lines = []
        seen_lines = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                clean_lines.append(line)
                seen_lines.add(line)
        
        # Limit response length
        response = '\n'.join(clean_lines)
        if len(response) > 1000:
            response = response[:1000] + "..."
        
        return response
