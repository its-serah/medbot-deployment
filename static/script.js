// MedBot JavaScript
class MedBot {
    constructor() {
        this.chatMessages = document.getElementById('chatMessages');
        this.questionForm = document.getElementById('questionForm');
        this.questionInput = document.getElementById('questionInput');
        this.sendButton = document.getElementById('sendButton');
        this.charCount = document.getElementById('charCount');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.errorModal = document.getElementById('errorModal');
        this.errorMessage = document.getElementById('errorMessage');
        
        this.init();
    }
    
    init() {
        // Event listeners
        this.questionForm.addEventListener('submit', this.handleSubmit.bind(this));
        this.questionInput.addEventListener('input', this.handleInputChange.bind(this));
        this.questionInput.addEventListener('keydown', this.handleKeyDown.bind(this));
        
        // Auto-resize textarea
        this.questionInput.addEventListener('input', this.autoResize.bind(this));
        
        // Focus on input
        this.questionInput.focus();
        
        console.log('MedBot initialized');
    }
    
    handleSubmit(e) {
        e.preventDefault();
        
        const question = this.questionInput.value.trim();
        if (!question) {
            this.showError('Please enter a medical question.');
            return;
        }
        
        if (question.length > 1000) {
            this.showError('Question is too long. Please keep it under 1000 characters.');
            return;
        }
        
        this.askQuestion(question);
    }
    
    handleInputChange() {
        const length = this.questionInput.value.length;
        this.charCount.textContent = length;
        
        // Update character counter color
        if (length > 900) {
            this.charCount.style.color = 'var(--danger-color)';
        } else if (length > 700) {
            this.charCount.style.color = 'var(--warning-color)';
        } else {
            this.charCount.style.color = 'var(--text-secondary)';
        }
        
        // Update send button state
        this.sendButton.disabled = !this.questionInput.value.trim();
    }
    
    handleKeyDown(e) {
        // Send on Ctrl/Cmd + Enter
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            this.questionForm.dispatchEvent(new Event('submit'));
        }
    }
    
    autoResize() {
        const textarea = this.questionInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    async askQuestion(question) {
        // Add user message
        this.addMessage(question, 'user');
        
        // Clear input
        this.questionInput.value = '';
        this.charCount.textContent = '0';
        this.handleInputChange();
        
        // Show loading
        this.showLoading(true);
        
        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! status: ${response.status}`);
            }
            
            // Add bot response
            this.addMessage(data.response, 'bot');
            
        } catch (error) {
            console.error('Error asking question:', error);
            this.showError(error.message || 'Failed to get response from MedBot. Please try again.');
            
            // Add error message to chat
            this.addMessage(
                'I apologize, but I encountered an error while processing your question. Please try again later or consult a healthcare professional for medical advice.',
                'bot'
            );
            
        } finally {
            this.showLoading(false);
            this.questionInput.focus();
        }
    }
    
    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Format content with paragraphs
        const paragraphs = content.split('\n').filter(p => p.trim());
        paragraphs.forEach(paragraph => {
            const p = document.createElement('p');
            p.textContent = paragraph;
            messageContent.appendChild(p);
        });
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        this.chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }
    
    showLoading(show) {
        this.loadingOverlay.style.display = show ? 'flex' : 'none';
        this.sendButton.disabled = show;
        
        if (show) {
            document.body.style.overflow = 'hidden';
        } else {
            document.body.style.overflow = '';
        }
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorModal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    }
    
    closeErrorModal() {
        this.errorModal.style.display = 'none';
        document.body.style.overflow = '';
        this.questionInput.focus();
    }
}

// Global function for modal close (called from HTML)
function closeErrorModal() {
    if (window.medBot) {
        window.medBot.closeErrorModal();
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.medBot = new MedBot();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && window.medBot) {
        window.medBot.questionInput.focus();
    }
});

// Service worker registration for PWA (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}
