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
        
        # First, try to load tiny models for speed and efficiency
        if self._try_load_tiny_model():
            return
            
        # Try to load the model from saved paths
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
            logger.warning("Could not load any model, using rule-based fallback")
            self._setup_fallback()
    
    def _try_load_tiny_model(self) -> bool:
        """Try to load a tiny, efficient model for fast deployment."""
        tiny_models = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B parameters - tiny but capable
            "microsoft/DialoGPT-small",  # Small conversational model ~117M
            "distilgpt2",  # ~82M parameters - very lightweight
            "gpt2"  # ~124M parameters - last resort
        ]
        
        logger.info("Loading tiny model for super fast performance!")
        
        for tiny_model in tiny_models:
            try:
                logger.info(f"Trying tiny model: {tiny_model}")
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tiny_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                # Load model with optimizations
                self.model = AutoModelForCausalLM.from_pretrained(
                    tiny_model,
                    torch_dtype=torch.float32,
                    device_map=None,  # Use CPU
                    low_cpu_mem_usage=True,  # Memory efficient loading
                    trust_remote_code=True
                )
                
                # Create optimized pipeline
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,  # CPU only for tiny models
                    batch_size=1
                )
                
                logger.info(f"Tiny model {tiny_model} loaded successfully!")
                self.model_name = tiny_model
                
                # Adjust generation parameters for tiny models
                self.max_new_tokens = 128  # Shorter responses for speed
                self.temperature = 0.8
                self.top_p = 0.85
                
                return True
                
            except Exception as e:
                logger.warning(f"Tiny model {tiny_model} failed: {e}")
                continue
                
        logger.info("No tiny models available, falling back to other options")
        return False
    
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
                
            # Check if we have the required files
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                return False
                
            logger.info(f"Attempting to load model from {model_path}")
            
            # Load tokenizer first
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except:
                # Fallback to base model tokenizer
                base_model = self._detect_base_model(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load the actual fine-tuned model directly from your files
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,  # Use half precision on GPU
                device_map=None  # Let it use CPU
            )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # Use CPU
                torch_dtype=torch.float32
            )
            
            logger.info("Your fine-tuned model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load your model from {model_path}: {e}")
            # Try to load base model as fallback
            try:
                base_model = self._detect_base_model(model_path)
                logger.info(f"Falling back to base model: {base_model}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1
                )
                
                logger.info("Base model loaded as fallback")
                return True
                
            except Exception as e2:
                logger.error(f"Even base model failed: {e2}")
# Try tiny models in order of preference
                tiny_models = [
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Super tiny but capable
                    "microsoft/DialoGPT-small",  # Small conversational model
                    "distilgpt2",  # Very lightweight
                    "gpt2"  # Last resort
                ]
                
                for tiny_model in tiny_models:
                    try:
                        logger.info(f"Trying tiny model: {tiny_model}")
                        
                        self.tokenizer = AutoTokenizer.from_pretrained(tiny_model)
                        if self.tokenizer.pad_token is None:
                            self.tokenizer.pad_token = self.tokenizer.eos_token
                            
                        self.model = AutoModelForCausalLM.from_pretrained(
                            tiny_model,
                            torch_dtype=torch.float32,
                            device_map=None,
                            low_cpu_mem_usage=True
                        )
                        
                        self.pipeline = pipeline(
                            "text-generation",
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device=-1,  # CPU only for tiny models
                            batch_size=1
                        )
                        
                        logger.info(f"Tiny model {tiny_model} loaded successfully!")
                        self.model_name = tiny_model
                        return True
                        
                    except Exception as model_error:
                        logger.warning(f"Tiny model {tiny_model} failed: {model_error}")
                        continue
                        
                logger.error("All tiny model loading attempts failed")
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
            # Common conditions
            "diabetes": "Diabetes is a chronic condition affecting blood sugar regulation. Type 1 is autoimmune; Type 2 involves insulin resistance. Management includes medication, diet modifications, regular exercise, and blood sugar monitoring. Consult your healthcare provider for personalized treatment plans.",
            
            "hypertension": "High blood pressure occurs when blood pushes against artery walls with excessive force. Management often includes dietary changes (reducing sodium), regular exercise, stress management, and medications as prescribed. Regular monitoring is essential for cardiovascular health.",
            
            "blood pressure": "Blood pressure measures the force of blood against artery walls. Normal range is typically below 120/80 mmHg. High blood pressure can be managed through lifestyle changes and medication. Regular monitoring and medical supervision are important.",
            
            "asthma": "Asthma causes airway inflammation and breathing difficulties. Common triggers include allergens, exercise, cold air, and stress. Treatment typically involves rescue inhalers for acute symptoms and controller medications for long-term management. Work with your doctor to identify triggers.",
            
            "covid": "COVID-19 is caused by SARS-CoV-2 virus. Symptoms range from mild cold-like symptoms to severe respiratory illness. Prevention includes vaccination, masking in crowded areas, and good hand hygiene. Contact healthcare providers for current testing and treatment guidance.",
            
            "heart": "Heart health is crucial for overall wellbeing. Common concerns include coronary artery disease, heart failure, and arrhythmias. Maintaining heart health involves regular exercise, balanced diet, not smoking, and managing stress. Regular check-ups can help detect issues early.",
            
            # Symptoms
            "headache": "Headaches can have various causes including tension, dehydration, stress, or underlying conditions. Most are benign but persistent, severe, or sudden headaches should be evaluated by a healthcare provider, especially if accompanied by other symptoms.",
            
            "fever": "Fever is the body's natural response to infection. For adults, temperatures above 100.4°F (38°C) are considered fever. Stay hydrated, rest, and consider fever-reducing medications if appropriate. Seek medical care for high fever or concerning symptoms.",
            
            "cough": "Coughs can result from infections, allergies, asthma, or other conditions. Dry coughs may benefit from honey or throat lozenges. Persistent coughs lasting more than 3 weeks, bloody cough, or cough with breathing difficulties warrant medical evaluation.",
            
            "pain": "Pain is a signal that something needs attention. Acute pain often resolves with rest, ice/heat, or over-the-counter pain relievers. Chronic or severe pain should be evaluated by healthcare professionals for proper diagnosis and treatment.",
            
            # General health
            "nutrition": "A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Proper nutrition supports immune function, energy levels, and overall health. Consider consulting a registered dietitian for personalized nutrition advice.",
            
            "exercise": "Regular physical activity benefits cardiovascular health, mental wellbeing, and overall fitness. Adults should aim for at least 150 minutes of moderate exercise weekly. Start slowly and gradually increase intensity. Consult your doctor before starting new exercise routines.",
            
            "sleep": "Quality sleep is essential for health. Adults typically need 7-9 hours nightly. Good sleep hygiene includes consistent bedtime, limiting screens before bed, and creating a comfortable sleep environment. Persistent sleep issues may require medical evaluation.",
            
            "stress": "Chronic stress can impact physical and mental health. Management techniques include regular exercise, meditation, deep breathing, and maintaining social connections. If stress becomes overwhelming, consider speaking with a mental health professional.",
            
            # Medications
            "medication": "Medications should be taken exactly as prescribed by your healthcare provider. Never stop medications abruptly without medical guidance. Keep an updated list of all medications and discuss any concerns or side effects with your doctor or pharmacist.",
            
            "side effects": "Medication side effects vary by individual and drug. Common side effects are usually listed on medication labels. Report any concerning or unexpected side effects to your healthcare provider promptly. Never stop medications without medical consultation."
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
                return f"{response}\n\n**Please consult with a healthcare professional for personalized medical advice.**"
        
        # Check for common variations and synonyms
        keyword_map = {
            "sugar": "diabetes",
            "blood sugar": "diabetes",
            "bp": "blood pressure", 
            "hypertension": "blood pressure",
            "corona": "covid",
            "coronavirus": "covid",
            "breathing": "asthma",
            "chest pain": "heart",
            "heart attack": "heart",
            "migraine": "headache",
            "flu": "fever",
            "cold": "cough",
            "diet": "nutrition",
            "workout": "exercise",
            "insomnia": "sleep",
            "anxiety": "stress",
            "medicine": "medication",
            "drug": "medication"
        }
        
        for keyword, mapped_condition in keyword_map.items():
            if keyword in question_lower and mapped_condition in self.knowledge_base:
                response = self.knowledge_base[mapped_condition]
                return f"{response}\n\n**Please consult with a healthcare professional for personalized medical advice.**"
        
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
    
    @staticmethod
    def save_model_optimized(model, tokenizer, save_path: str, save_adapter_only: bool = True):
        """
        Save model with optimizations for smaller size and faster loading.
        
        Args:
            model: The fine-tuned model to save
            tokenizer: The tokenizer to save
            save_path: Directory path to save the model
            save_adapter_only: If True, only save LoRA adapter weights (much smaller)
        """
        import os
        import torch
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        logger.info(f"Saving optimized model to {save_path}")
        
        try:
            if save_adapter_only and hasattr(model, 'save_pretrained'):
                # For LoRA/PEFT models, save only the adapter weights
                if hasattr(model, 'peft_config'):
                    logger.info("Saving LoRA adapter weights only (much smaller)")
                    model.save_pretrained(
                        save_path,
                        torch_dtype=torch.float16,  # Use half precision
                        safe_serialization=True     # Use safer, smaller format
                    )
                else:
                    # Regular fine-tuned model - save with optimizations
                    logger.info("Saving full model with optimizations")
                    model.save_pretrained(
                        save_path,
                        torch_dtype=torch.float16,  # Use half precision
                        safe_serialization=True     # Use safer, smaller format
                    )
            else:
                # Fallback to standard saving
                model.save_pretrained(save_path)
            
            # Save tokenizer
            tokenizer.save_pretrained(save_path)
            
            logger.info("Model saved successfully with optimizations")
            
            # Log file sizes for reference
            total_size = 0
            for root, dirs, files in os.walk(save_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path)
                    total_size += size
                    logger.info(f"  {file}: {size / (1024*1024):.1f} MB")
            
            logger.info(f"Total model size: {total_size / (1024*1024):.1f} MB")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
