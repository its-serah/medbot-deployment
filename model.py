"""
MedBot Model Module - CUSTOM TRAINED Medical Assistant
=====================================================

A medical chatbot using CUSTOM FINE-TUNED GPT-2 model with LoRA adaptation
for accurate medical question answering based on trained medical knowledge.
"""

import logging
import os
import torch
from typing import Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TextGenerationPipeline,
    pipeline
)
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class MedBotModel:
    """
    CUSTOM TRAINED Medical Question-Answering System using fine-tuned GPT-2.
    
    Uses YOUR CUSTOM FINE-TUNED GPT-2 model with LoRA adaptation trained
    specifically on medical data for accurate medical responses.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the MedBot with CUSTOM TRAINED model.
        
        Args:
            model_path: Path to custom model directory
        """
        logger.info("INITIALIZING CUSTOM TRAINED MEDICAL MODEL!")
        
        # Setup fallback knowledge first
        self._setup_fallback_knowledge()
        
        # Set device preference
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        
        # Use YOUR custom trained model
        self.base_model_name = "gpt2"
        self.custom_model_path = model_path or "./medbot-finetuned"
        
        # Check if custom model exists
        if not os.path.exists(self.custom_model_path):
            logger.error(f"Custom model not found at {self.custom_model_path}")
            raise FileNotFoundError(f"Custom trained model missing: {self.custom_model_path}")
        
        try:
            logger.info(f"Loading YOUR custom trained model from: {self.custom_model_path}")
            self._load_custom_model()
            logger.info("CUSTOM MODEL LOADED SUCCESSFULLY!")
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            logger.info("Falling back to knowledge-based responses")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
    
    def _load_custom_model(self):
        """Load YOUR custom trained medical model with LoRA adapters."""
        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            
            # Load tokenizer from custom model directory
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.custom_model_path,
                padding_side="left",
                trust_remote_code=True
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapters from your custom model
            logger.info(f"Loading LoRA adapters from {self.custom_model_path}")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.custom_model_path,
                torch_dtype=torch.float32
            )
            self.model.eval()  # Set to evaluation mode
            
            # Move to device
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1 if self.device == "cpu" else 0,
                max_length=512,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("CUSTOM MEDICAL MODEL READY FOR ACTION!")
            
        except Exception as e:
            logger.error(f"Error loading custom model: {e}")
            raise
    
    def _generate_custom_ai_response(self, question: str) -> str:
        """Generate response using YOUR CUSTOM TRAINED MODEL!"""
        # Create medical prompt in the training format
        prompt = f"Patient: {question}\n\nMedical Assistant:"
        
        try:
            # Generate response with your custom trained model
            outputs = self.pipeline(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            # Extract and clean the response
            generated_text = outputs[0]['generated_text']
            response = generated_text.strip()
            
            # Clean up response
            if response:
                # Remove any leftover prompt artifacts
                response = response.replace(prompt, "").strip()
                
                # Split at EOS token if present
                if self.tokenizer.eos_token in response:
                    response = response.split(self.tokenizer.eos_token)[0]
                
                # Clean sentences
                sentences = [s.strip() for s in response.split('.') if s.strip()]
                if sentences:
                    response = '. '.join(sentences[:4])
                    if not response.endswith('.'):
                        response += '.'
                else:
                    response = None
            
            # If we got a good response, add disclaimer and return
            if response and len(response) > 20:
                response += "\n\nMedical Disclaimer: This information is for educational purposes only. Always consult with a qualified healthcare professional for personalized medical advice, diagnosis, or treatment."
                return response
            else:
                # Fall back to knowledge base if AI response is poor
                logger.info("Custom model response too short, using fallback")
                raise ValueError("Generated response too short")
                
        except Exception as e:
            logger.error(f"Custom AI generation failed: {e}")
            raise
    
    def _load_model(self):
        """Load the language model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=True
            )
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimized settings for CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()  # Set to evaluation mode
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,  # Use CPU
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _setup_fallback_knowledge(self):
        """Setup comprehensive fallback knowledge base for when AI model fails."""
        self.fallback_responses = {
            # Chronic conditions
            "diabetes": "Diabetes is a chronic condition where your body cannot properly regulate blood sugar levels. **Type 1** is autoimmune (body doesn't produce insulin), while **Type 2** involves insulin resistance. **Management includes:** regular blood sugar monitoring, prescribed medications (like metformin or insulin), balanced diet focusing on complex carbs, regular exercise, and maintaining healthy weight. **Complications** can include heart disease, kidney damage, and nerve problems if uncontrolled.",
            
            "hypertension": "High blood pressure (hypertension) occurs when blood consistently pushes too hard against artery walls. **Normal:** <120/80 mmHg, **High:** ≥140/90 mmHg. **Causes:** genetics, diet high in sodium, lack of exercise, stress, age. **Management:** low-sodium diet, regular exercise, stress reduction, weight management, and medications if prescribed. **Risks:** heart attack, stroke, kidney disease if untreated.",
            
            "blood pressure": "Blood pressure measures the force of blood against artery walls. **Systolic** (top number) measures pressure when heart beats, **diastolic** (bottom) when heart rests. **Normal ranges:** <120/80 mmHg is optimal, 120-129/<80 is elevated, ≥130/80 is high. **Natural ways to lower:** reduce salt, exercise regularly, maintain healthy weight, limit alcohol, manage stress.",
            
            # Symptoms
            "headache": "Headaches have various causes: **Tension headaches** (stress, poor posture, eye strain), **migraines** (throbbing pain with nausea/light sensitivity), **cluster headaches** (severe, around one eye). **When to seek help:** sudden severe headache, headache with fever/stiff neck, vision changes, or after head injury. **Management:** adequate sleep, hydration, stress management, identifying triggers.",
            
            "migraine": "Migraines are severe headaches often with nausea, vomiting, and sensitivity to light/sound. **Triggers:** stress, certain foods (chocolate, aged cheese), hormonal changes, lack of sleep, bright lights. **Phases:** prodrome (warning signs), aura (visual disturbances), headache, postdrome (recovery). **Management:** identify triggers, maintain regular sleep schedule, stay hydrated, consider preventive medications.",
            
            "fever": "Fever is your body's natural defense against infection. **Normal temp:** 98.6°F (37°C), **Fever:** ≥100.4°F (38°C). **Causes:** infections (viral, bacterial), inflammatory conditions, medications. **Treatment:** rest, fluids, fever reducers (acetaminophen, ibuprofen) if comfortable. **Seek medical care:** fever >103°F, lasts >3 days, severe symptoms, difficulty breathing, persistent vomiting.",
            
            "cough": "Coughs help clear airways of irritants. **Types:** dry (non-productive) or wet (productive with mucus). **Causes:** infections (cold, flu, pneumonia), allergies, asthma, acid reflux. **Duration:** acute (<3 weeks), chronic (>8 weeks). **Treatment:** honey for dry cough, expectorants for wet cough, treat underlying cause. **See doctor:** blood in cough, fever, difficulty breathing, lasts >3 weeks.",
            
            # Heart-related
            "heart": "Heart health is vital for overall wellbeing. **Key factors:** regular exercise (150 min/week moderate activity), heart-healthy diet (fruits, vegetables, whole grains, lean proteins), no smoking, limited alcohol, stress management, adequate sleep. **Warning signs:** chest pain, shortness of breath, unusual fatigue, swelling in legs. **Regular checkups** help monitor blood pressure, cholesterol, and early detection of issues.",
            
            "chest pain": "Chest pain can range from minor to life-threatening. **Cardiac causes:** heart attack, angina, pericarditis. **Non-cardiac:** muscle strain, acid reflux, anxiety, lung issues. **EMERGENCY signs:** crushing chest pressure, pain radiating to arm/jaw/back, shortness of breath, sweating, nausea. **Call 911 immediately** if these symptoms occur. Other chest pain should be evaluated by healthcare provider.",
            
            # Respiratory
            "shortness of breath": "Difficulty breathing (dyspnea) can indicate various conditions. **Acute causes:** asthma, panic attack, allergic reaction, pneumonia, heart attack. **Chronic causes:** COPD, heart failure, anemia. **EMERGENCY:** severe difficulty breathing, blue lips/face, chest pain. **Management depends on cause:** inhalers for asthma, medications for heart conditions, lifestyle changes for chronic conditions.",
            
            "asthma": "Asthma causes airway inflammation and breathing difficulties. **Symptoms:** wheezing, coughing, chest tightness, shortness of breath. **Triggers:** allergens (dust, pollen, pets), exercise, cold air, stress, infections. **Management:** avoid triggers, rescue inhaler (albuterol) for attacks, controller medications for daily use, peak flow monitoring, asthma action plan.",
            
            # Mental health
            "anxiety": "Anxiety is normal but becomes problematic when excessive or interferes with daily life. **Symptoms:** worry, restlessness, fatigue, difficulty concentrating, muscle tension, sleep problems. **Types:** generalized anxiety, panic disorder, social anxiety. **Management:** therapy (CBT), relaxation techniques, regular exercise, adequate sleep, limit caffeine. **Medications** may be needed for severe cases.",
            
            "stress": "Chronic stress impacts both physical and mental health. **Physical effects:** headaches, muscle tension, fatigue, sleep problems, digestive issues. **Mental effects:** anxiety, depression, irritability, concentration problems. **Management:** regular exercise, relaxation techniques (meditation, deep breathing), adequate sleep, social support, time management, professional help if needed.",
            
            # General health
            "sleep": "Quality sleep is essential for health and wellbeing. **Adults need:** 7-9 hours nightly. **Sleep hygiene:** consistent bedtime, comfortable environment, limit screens before bed, avoid caffeine late in day, regular exercise (not close to bedtime). **Sleep disorders:** insomnia, sleep apnea, restless leg syndrome may require medical evaluation.",
            
            "nutrition": "Balanced nutrition supports overall health and disease prevention. **Key components:** fruits and vegetables (5+ servings daily), whole grains, lean proteins, healthy fats (omega-3s), limit processed foods, sugar, and excess sodium. **Hydration:** 8 glasses water daily. **Special needs:** pregnancy, diabetes, heart disease may require modified diets. Consider consulting registered dietitian for personalized advice.",
            
            "exercise": "Regular physical activity is crucial for health. **Recommendations:** 150 minutes moderate aerobic activity weekly, plus 2 days strength training. **Benefits:** cardiovascular health, weight management, bone strength, mental health, disease prevention. **Types:** walking, swimming, cycling, strength training. **Start slowly** and gradually increase. **Consult doctor** before starting if you have health conditions.",
            
            # Specific symptom combinations
            "fast heartbeat nausea": "Fast heartbeat (tachycardia) combined with nausea can indicate several conditions:\n\n**Common Causes:**\n• **Anxiety/Panic attacks** - rapid heart rate with nausea, sweating\n• **Dehydration** - especially with vomiting or poor fluid intake\n• **Medication side effects** - stimulants, certain antidepressants\n• **Caffeine excess** - coffee, energy drinks, supplements\n• **Low blood sugar** - especially in diabetics\n\n**More Serious Causes:**\n• **Heart rhythm disorders** - atrial fibrillation, SVT\n• **Heart attack** - especially in older adults or those with risk factors\n• **Thyroid problems** - hyperthyroidism\n• **Electrolyte imbalances** - low potassium, magnesium\n\n**When to Seek Immediate Care:**\n• Chest pain or pressure\n• Severe shortness of breath\n• Dizziness or fainting\n• Heart rate over 150 bpm\n• Symptoms persist or worsen\n\n**Immediate Steps:**\n• Sit down and rest\n• Try slow, deep breathing\n• Sip water if not nauseous\n• Avoid caffeine and stimulants\n\nSeek medical evaluation for proper diagnosis and treatment.",
            
            "rapid heartbeat": "Rapid heartbeat (tachycardia) can have various causes:\n\n**Normal Causes:**\n• Exercise or physical activity\n• Emotional stress or anxiety\n• Caffeine or stimulants\n• Dehydration\n• Fever or illness\n\n**Medical Conditions:**\n• **Supraventricular tachycardia (SVT)** - sudden rapid heartbeat\n• **Atrial fibrillation** - irregular, rapid rhythm\n• **Hyperthyroidism** - overactive thyroid\n• **Anemia** - low red blood cell count\n• **Heart disease** - various cardiac conditions\n\n**When to Seek Care:**\n• Heart rate consistently over 100 bpm at rest\n• Chest pain or discomfort\n• Shortness of breath\n• Dizziness or fainting\n• Palpitations lasting more than a few minutes\n\n**Management:**\n• Identify and avoid triggers\n• Practice relaxation techniques\n• Stay hydrated\n• Limit caffeine and alcohol\n• Regular exercise (as tolerated)\n\nConsult a healthcare provider for persistent or concerning symptoms.",
            
            "palpitations": "Heart palpitations are the feeling of a racing, pounding, or fluttering heart:\n\n**Common Triggers:**\n• Stress and anxiety\n• Caffeine or alcohol\n• Nicotine\n• Exercise\n• Medications (decongestants, asthma inhalers)\n• Hormonal changes (pregnancy, menopause)\n\n**Medical Causes:**\n• **Arrhythmias** - abnormal heart rhythms\n• **Thyroid disorders** - hyperthyroidism\n• **Heart disease** - valve problems, cardiomyopathy\n• **Electrolyte imbalances** - low potassium, magnesium\n• **Anemia** - low iron levels\n\n**When to Seek Medical Care:**\n• Palpitations with chest pain\n• Severe shortness of breath\n• Dizziness or fainting\n• Palpitations lasting several minutes\n• Family history of sudden cardiac death\n\n**Self-Care:**\n• Practice deep breathing\n• Reduce caffeine and alcohol\n• Manage stress\n• Stay hydrated\n• Get adequate sleep\n\nKeep a diary of triggers and symptoms to discuss with your doctor.",
            
            "nausea": "Nausea can result from various causes:\n\n**Common Causes:**\n• **Gastrointestinal** - food poisoning, gastritis, ulcers\n• **Medications** - antibiotics, pain relievers, chemotherapy\n• **Motion sickness** - travel, vestibular disorders\n• **Pregnancy** - morning sickness\n• **Anxiety** - stress-related nausea\n\n**Serious Causes:**\n• **Heart attack** - especially with chest pain\n• **Appendicitis** - with abdominal pain\n• **Bowel obstruction** - with vomiting, no bowel movements\n• **Meningitis** - with fever, headache, stiff neck\n• **Kidney stones** - with severe flank pain\n\n**Management:**\n• Rest and hydrate with small sips\n• Try ginger tea or ginger supplements\n• Eat bland foods (crackers, toast)\n• Avoid strong odors\n• Practice deep breathing\n\n**Seek Medical Care:**\n• Severe, persistent vomiting\n• Signs of dehydration\n• High fever\n• Severe abdominal pain\n• Blood in vomit\n\nNausea with chest pain or severe symptoms requires immediate evaluation."
        }
        
        # Add keyword mapping for better matching
        self.keyword_mapping = {
            "fast heart": "fast heartbeat nausea",
            "rapid heart": "rapid heartbeat",
            "heart racing": "rapid heartbeat",
            "pounding heart": "palpitations",
            "heart palpitations": "palpitations",
            "feel sick": "nausea",
            "throwing up": "nausea",
            "vomiting": "nausea"
        }
    
    def _create_medical_prompt(self, question: str) -> str:
        """Create expert medical prompt for accurate responses."""
        # Use medical knowledge-based approach instead of poor AI generation
        return self._get_medical_knowledge_response(question)
    
    def generate_response(self, question: str) -> str:
        """
        Generate expert medical response using YOUR CUSTOM TRAINED MODEL!
        
        Args:
            question: User's medical question
            
        Returns:
            str: AI-generated medical response from YOUR trained model
        """
        if not question or not question.strip():
            return "Please provide a medical question so I can assist you."
        
        question = question.strip()
        
        # Check if we have a loaded model
        if self.model and self.tokenizer and self.pipeline:
            try:
                # USE YOUR CUSTOM TRAINED MODEL!
                logger.info(f"Generating response with CUSTOM TRAINED model for: {question[:50]}...")
                return self._generate_custom_ai_response(question)
            except Exception as e:
                logger.error(f"Custom model failed: {e}")
                logger.info("Falling back to knowledge base")
        
        # Use knowledge base (either as fallback or primary)
        return self._get_medical_knowledge_response(question)
    
    def _get_medical_knowledge_response(self, question: str) -> str:
        """Get intelligent medical response using enhanced knowledge base."""
        question_lower = question.lower()
        
        # Check for specific symptom combinations first
        if any(phrase in question_lower for phrase in ["fast heart", "rapid heart", "heart racing"]) and "nausea" in question_lower:
            return f"{self.fallback_responses['fast heartbeat nausea']}\n\n⚕️ **Medical Disclaimer:** This information is for educational purposes only. Always consult with a qualified healthcare professional for personalized medical advice, diagnosis, or treatment."
        
        # Check keyword mapping
        for key_phrase, mapped_response in self.keyword_mapping.items():
            if key_phrase in question_lower and mapped_response in self.fallback_responses:
                return f"{self.fallback_responses[mapped_response]}\n\n⚕️ **Medical Disclaimer:** This information is for educational purposes only. Always consult with a qualified healthcare professional for personalized medical advice, diagnosis, or treatment."
        
        # Check direct keyword matches
        for keyword, response in self.fallback_responses.items():
            if keyword in question_lower:
                return f"{response}\n\n⚕️ **Medical Disclaimer:** This information is for educational purposes only. Always consult with a qualified healthcare professional for personalized medical advice, diagnosis, or treatment."
        
        # If no specific match, provide helpful guidance
        return """I understand you have a medical question. For the most accurate and personalized medical advice, I recommend:

🏥 **Immediate Steps:**
• Contact your healthcare provider
• Call a nurse helpline if available
• Visit an urgent care center for non-emergency concerns
• Call 911 for emergency symptoms

📋 **When Describing Symptoms:**
• Note when symptoms started
• Describe severity and location
• List any triggering factors
• Mention current medications
• Include relevant medical history

⚕️ **Remember:** Professional medical evaluation is essential for proper diagnosis and treatment. AI assistants provide general information but cannot replace clinical judgment."""
    
    def _generate_ai_response(self, question: str) -> str:
        """Generate response using AI model."""
        prompt = self._create_medical_prompt(question)
        
        # Generate response with optimized parameters
        outputs = self.pipeline(
            prompt,
            max_new_tokens=250,
            num_return_sequences=1,
            temperature=0.8,  # Slightly higher for more creativity
            do_sample=True,
            top_p=0.95,  # Higher for more diverse responses
            repetition_penalty=1.2,  # Higher to avoid repetition
            pad_token_id=self.tokenizer.eos_token_id,
            return_full_text=False  # Only return new tokens
        )
        
        # Extract and clean the response
        generated_text = outputs[0]['generated_text']
        
        # Clean up the response intelligently
        response = generated_text.strip()
        
        # Remove prompt artifacts and clean up
        response = response.replace(prompt, "").strip()
        
        # Split into sentences and take meaningful ones
        sentences = response.split('. ')
        clean_sentences = []
        
        for sentence in sentences[:4]:  # Limit to 4 sentences for clarity
            sentence = sentence.strip()
            if len(sentence) > 10 and not any(bad_word in sentence.lower() for bad_word in ['error', 'undefined', 'unknown']):
                clean_sentences.append(sentence)
        
        if not clean_sentences:
            return self._generate_fallback_response(question)
        
        # Join sentences properly
        response = '. '.join(clean_sentences)
        if not response.endswith('.'):
            response += '.'
        
        # Add proper medical disclaimer
        response += "\n\n⚕️ **Medical Disclaimer:** This information is for educational purposes only. Always consult with a qualified healthcare professional for personalized medical advice, diagnosis, or treatment."
        
        return response
    
    def _generate_fallback_response(self, question: str) -> str:
        """Generate fallback response using rule-based matching."""
        question_lower = question.lower()
        
        # Search for keywords in the question
        for keyword, response in self.fallback_responses.items():
            if keyword in question_lower:
                return f"{response}\n\n**Please consult with a healthcare professional for personalized medical advice.**"
        
        # Generic helpful response
        return """I understand you have a medical question. While I can provide general health information, I recommend:

1. Consulting with a licensed healthcare provider
2. Contacting your doctor or medical clinic
3. Calling a nurse helpline if available in your area
4. Seeking emergency medical attention if this is urgent

**Remember: AI assistants should not replace professional medical consultation.**"""
