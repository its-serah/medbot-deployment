"""
MedBot Model Module - Lightweight Medical Knowledge Base
========================================================

A simple, fast medical knowledge base without heavy ML dependencies.
Provides reliable medical information through rule-based responses.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

class MedBotModel:
    """
    Lightweight Medical Question-Answering System.
    
    Uses rule-based responses for fast, reliable medical information.
    No ML dependencies required - perfect for production deployment.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the MedBot with medical knowledge base.
        
        Args:
            model_path: Ignored - for compatibility only
        """
        logger.info("Initializing MedBot with medical knowledge base")
        self._setup_medical_knowledge()
        
    def _setup_medical_knowledge(self):
        """Setup comprehensive medical knowledge base."""
        self.knowledge_base = {
            # Common conditions
            "diabetes": "Diabetes is a chronic condition affecting blood sugar regulation. Type 1 is autoimmune; Type 2 involves insulin resistance. Management includes medication, diet modifications, regular exercise, and blood sugar monitoring. Consult your healthcare provider for personalized treatment plans.",
            
            "hypertension": "High blood pressure occurs when blood pushes against artery walls with excessive force. Management often includes dietary changes (reducing sodium), regular exercise, stress management, and medications as prescribed. Regular monitoring is essential for cardiovascular health.",
            
            "blood pressure": "Blood pressure measures the force of blood against artery walls. Normal range is typically below 120/80 mmHg. High blood pressure can be managed through lifestyle changes and medication. Regular monitoring and medical supervision are important.",
            
            "asthma": "Asthma causes airway inflammation and breathing difficulties. Common triggers include allergens, exercise, cold air, and stress. Treatment typically involves rescue inhalers for acute symptoms and controller medications for long-term management. Work with your doctor to identify triggers.",
            
            "covid": "COVID-19 is caused by SARS-CoV-2 virus. Symptoms range from mild cold-like symptoms to severe respiratory illness. Prevention includes vaccination, masking in crowded areas, and good hand hygiene. Contact healthcare providers for current testing and treatment guidance.",
            
            "coronavirus": "COVID-19 is caused by SARS-CoV-2 virus. Symptoms can range from mild to severe respiratory illness. Prevention includes vaccination, proper hygiene, and following public health guidelines. Seek medical attention if you develop severe symptoms.",
            
            "heart": "Heart health is crucial for overall wellbeing. Common concerns include coronary artery disease, heart failure, and arrhythmias. Maintaining heart health involves regular exercise, balanced diet, not smoking, and managing stress. Regular check-ups can help detect issues early.",
            
            "heart disease": "Heart disease encompasses various cardiovascular conditions. Risk factors include high blood pressure, high cholesterol, smoking, diabetes, and family history. Prevention involves healthy lifestyle choices, regular exercise, and proper medical care.",
            
            # Symptoms
            "headache": "Headaches can have various causes including tension, dehydration, stress, or underlying conditions. Most are benign but persistent, severe, or sudden headaches should be evaluated by a healthcare provider, especially if accompanied by other symptoms.",
            
            "migraine": "Migraines are severe headaches often accompanied by nausea, vomiting, and sensitivity to light and sound. Triggers can include stress, certain foods, hormonal changes, and sleep patterns. Treatment options include medications and lifestyle modifications.",
            
            "fever": "Fever is the body's natural response to infection. For adults, temperatures above 100.4°F (38°C) are considered fever. Stay hydrated, rest, and consider fever-reducing medications if appropriate. Seek medical care for high fever or concerning symptoms.",
            
            "cough": "Coughs can result from infections, allergies, asthma, or other conditions. Dry coughs may benefit from honey or throat lozenges. Persistent coughs lasting more than 3 weeks, bloody cough, or cough with breathing difficulties warrant medical evaluation.",
            
            "pain": "Pain is a signal that something needs attention. Acute pain often resolves with rest, ice/heat, or over-the-counter pain relievers. Chronic or severe pain should be evaluated by healthcare professionals for proper diagnosis and treatment.",
            
            "chest pain": "Chest pain can have various causes, from muscle strain to serious heart conditions. Sudden, severe chest pain, especially with shortness of breath, sweating, or nausea, requires immediate medical attention. Don't ignore persistent chest discomfort.",
            
            "shortness of breath": "Difficulty breathing can indicate various conditions from anxiety to serious heart or lung problems. Sudden onset of severe shortness of breath requires immediate medical attention. Gradual onset should be evaluated by a healthcare provider.",
            
            # General health topics
            "nutrition": "A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Proper nutrition supports immune function, energy levels, and overall health. Consider consulting a registered dietitian for personalized nutrition advice.",
            
            "diet": "A healthy diet emphasizes whole foods, fruits, vegetables, lean proteins, and whole grains while limiting processed foods, excess sugar, and unhealthy fats. Proper nutrition is fundamental to good health and disease prevention.",
            
            "exercise": "Regular physical activity benefits cardiovascular health, mental wellbeing, and overall fitness. Adults should aim for at least 150 minutes of moderate exercise weekly. Start slowly and gradually increase intensity. Consult your doctor before starting new exercise routines.",
            
            "workout": "Regular exercise is essential for physical and mental health. Start with activities you enjoy and gradually build up intensity and duration. Always warm up before exercising and cool down afterward. Stay hydrated and listen to your body.",
            
            "sleep": "Quality sleep is essential for health. Adults typically need 7-9 hours nightly. Good sleep hygiene includes consistent bedtime, limiting screens before bed, and creating a comfortable sleep environment. Persistent sleep issues may require medical evaluation.",
            
            "insomnia": "Insomnia involves difficulty falling asleep or staying asleep. Causes can include stress, anxiety, medical conditions, or poor sleep habits. Sleep hygiene improvements and relaxation techniques can help. Persistent insomnia should be evaluated by a healthcare provider.",
            
            "stress": "Chronic stress can impact physical and mental health. Management techniques include regular exercise, meditation, deep breathing, and maintaining social connections. If stress becomes overwhelming, consider speaking with a mental health professional.",
            
            "anxiety": "Anxiety is a normal response to stress, but persistent or excessive anxiety can interfere with daily life. Symptoms may include worry, restlessness, and physical symptoms. Treatment options include therapy, medication, and self-care strategies.",
            
            # Medications and treatments
            "medication": "Medications should be taken exactly as prescribed by your healthcare provider. Never stop medications abruptly without medical guidance. Keep an updated list of all medications and discuss any concerns or side effects with your doctor or pharmacist.",
            
            "side effects": "Medication side effects vary by individual and drug. Common side effects are usually listed on medication labels. Report any concerning or unexpected side effects to your healthcare provider promptly. Never stop medications without medical consultation.",
            
            "vaccine": "Vaccines are safe and effective tools for preventing serious diseases. They work by training your immune system to recognize and fight specific infections. Follow recommended vaccination schedules and discuss any concerns with your healthcare provider.",
            
            "vaccination": "Vaccinations protect both individuals and communities from serious diseases. They undergo rigorous testing for safety and effectiveness. Stay up to date with recommended vaccines based on your age, health conditions, and risk factors.",
            
            # Common health concerns
            "weight loss": "Healthy weight loss involves creating a moderate calorie deficit through balanced diet and regular exercise. Aim for 1-2 pounds per week. Avoid extreme diets or rapid weight loss. Consult healthcare providers for personalized weight management plans.",
            
            "obesity": "Obesity increases risk for various health conditions including diabetes, heart disease, and certain cancers. Treatment involves lifestyle changes including diet modification, increased physical activity, and sometimes medical intervention. Professional support can be helpful.",
            
            "cold": "Common cold symptoms include runny nose, cough, and congestion. Most colds resolve in 7-10 days with rest, fluids, and supportive care. See a healthcare provider if symptoms worsen or persist, or if you develop high fever.",
            
            "flu": "Influenza causes fever, body aches, cough, and fatigue. Annual flu vaccination is recommended. Treatment focuses on rest, fluids, and symptom management. Antiviral medications may help if started early. Seek medical care for severe symptoms.",
        }
        
        # Keyword mapping for better question matching
        self.keyword_map = {
            "sugar": "diabetes",
            "blood sugar": "diabetes",
            "bp": "blood pressure",
            "high blood pressure": "hypertension",
            "corona": "covid",
            "covid-19": "covid",
            "breathing": "shortness of breath",
            "heart attack": "chest pain",
            "flu": "fever",
            "influenza": "flu",
            "medicine": "medication",
            "drug": "medication",
            "weight": "weight loss",
            "tired": "sleep",
            "fatigue": "sleep",
            "headaches": "headache",
            "migraines": "migraine",
        }
    
    def generate_response(self, question: str) -> str:
        """
        Generate a response to the medical question.
        
        Args:
            question: User's medical question
            
        Returns:
            str: Generated medical response with disclaimer
        """
        if not question or not question.strip():
            return "Please provide a medical question so I can assist you."
        
        # Clean and prepare the question
        question = question.strip().lower()
        
        # Search for direct matches in knowledge base
        for condition, response in self.knowledge_base.items():
            if condition in question:
                return f"{response}\n\n**Please consult with a healthcare professional for personalized medical advice.**"
        
        # Search using keyword mapping
        for keyword, mapped_condition in self.keyword_map.items():
            if keyword in question and mapped_condition in self.knowledge_base:
                response = self.knowledge_base[mapped_condition]
                return f"{response}\n\n**Please consult with a healthcare professional for personalized medical advice.**"
        
        # Generic helpful response for unmatched questions
        return """I understand you have a medical question, but I don't have specific information about this topic in my current knowledge base. 

For accurate medical information and advice, I recommend:
1. Consulting with a licensed healthcare provider
2. Contacting your doctor or medical clinic  
3. Calling a nurse helpline if available in your area
4. Seeking emergency medical attention if this is urgent

**Remember: AI assistants should not replace professional medical consultation.**"""
