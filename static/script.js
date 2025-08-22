// MedBot Advanced JavaScript
class MedBot {
    constructor() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageForm = document.getElementById('messageForm');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.charCount = document.getElementById('charCount');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.errorToast = document.getElementById('errorToast');
        this.errorMessage = document.getElementById('errorMessage');
        this.thinkingStatus = document.getElementById('thinkingStatus');
        
        this.isTyping = false;
        this.thinkingMessages = [
            'Analyzing medical data...',
            'Consulting knowledge base...',
            'Processing your question...',
            'Generating AI response...',
            'Validating medical information...'
        ];
        
        this.currentThinkingIndex = 0;
        this.thinkingInterval = null;
        
        this.init();
    }
    
    init() {
        // Event listeners
        this.messageForm.addEventListener('submit', this.handleSubmit.bind(this));
        this.messageInput.addEventListener('input', this.handleInputChange.bind(this));
        this.messageInput.addEventListener('keydown', this.handleKeyDown.bind(this));
        
        // Focus on input
        this.messageInput.focus();
        
        // Add welcome animation
        this.animateWelcome();
        
        console.log('🤖 MedBot AI initialized');
    }
    
    animateWelcome() {
        const quickBtns = document.querySelectorAll('.quick-btn');
        quickBtns.forEach((btn, index) => {
            btn.style.opacity = '0';
            btn.style.transform = 'translateY(20px)';
            setTimeout(() => {
                btn.style.transition = 'all 0.4s ease-out';
                btn.style.opacity = '1';
                btn.style.transform = 'translateY(0)';
            }, index * 100 + 600);
        });
    }
    
    handleSubmit(e) {
        e.preventDefault();
        
        if (this.isTyping) return;
        
        const question = this.messageInput.value.trim();
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
        const length = this.messageInput.value.length;
        this.charCount.textContent = length;
        
        const charCounter = document.querySelector('.char-counter');
        if (length > 900) {
            charCounter.style.color = '#E63946';
        } else if (length > 700) {
            charCounter.style.color = '#FF9500';
        } else {
            charCounter.style.color = '#64748B';
        }
        
        this.sendBtn.disabled = !this.messageInput.value.trim() || this.isTyping;
        this.autoResize();
    }
    
    handleKeyDown(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            this.handleSubmit(e);
        }
    }
    
    autoResize() {
        const textarea = this.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    async askQuestion(question) {
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.style.transition = 'all 0.3s ease-out';
            welcomeMessage.style.opacity = '0';
            welcomeMessage.style.transform = 'translateY(-20px)';
            setTimeout(() => welcomeMessage.remove(), 300);
        }
        
        await this.addMessage(question, 'user');
        
        this.messageInput.value = '';
        this.charCount.textContent = '0';
        this.handleInputChange();
        
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
                throw new Error(data.error || 'HTTP error! status: ' + response.status);
            }
            
            await this.addMessage(data.response, 'bot', true);
            
        } catch (error) {
            console.error('Error asking question:', error);
            this.showError(error.message || 'Failed to get response from MedBot.');
            
            await this.addMessage(
                'I apologize, but I encountered an error. Please try again later or consult a healthcare professional.',
                'bot'
            );
            
        } finally {
            this.showLoading(false);
            this.messageInput.focus();
        }
    }
    
    async addMessage(content, type, animate = false) {
        return new Promise((resolve) => {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message-bubble ' + type;
            messageDiv.style.opacity = '0';
            messageDiv.style.transform = 'translateY(20px)';
            
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            if (type === 'user') {
                avatar.innerHTML = '<i class="fas fa-user"></i>';
            } else {
                avatar.innerHTML = '<i class="fas fa-robot"></i>';
            }
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            
            this.messagesContainer.appendChild(messageDiv);
            
            setTimeout(() => {
                messageDiv.style.transition = 'all 0.4s ease-out';
                messageDiv.style.opacity = '1';
                messageDiv.style.transform = 'translateY(0)';
            }, 50);
            
            if (animate && type === 'bot') {
                this.typeMessage(content, messageContent, resolve);
            } else {
                this.displayMessage(content, messageContent);
                setTimeout(resolve, 400);
            }
            
            this.scrollToBottom();
        });
    }
    
    displayMessage(content, container) {
        const formattedContent = this.formatContent(content);
        container.innerHTML = formattedContent;
    }
    
    async typeMessage(content, container, callback) {
        const formattedContent = this.formatContent(content);
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.innerHTML = '<span>MedBot is typing</span><div class="typing-dots"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
        container.appendChild(typingDiv);
        
        await this.delay(1000);
        
        typingDiv.remove();
        container.innerHTML = formattedContent;
        
        container.style.opacity = '0';
        setTimeout(() => {
            container.style.transition = 'opacity 0.5s ease-out';
            container.style.opacity = '1';
        }, 50);
        
        callback();
    }
    
    formatContent(content) {
        let formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
        
        formatted = '<p>' + formatted + '</p>';
        
        formatted = formatted.replace(
            /(⚕️.*?Medical Disclaimer.*?treatment\.)/gi,
            '<div class="medical-disclaimer">$1</div>'
        );
        
        return formatted;
    }
    
    scrollToBottom() {
        setTimeout(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }, 100);
    }
    
    showLoading(show) {
        this.isTyping = show;
        
        if (show) {
            this.loadingOverlay.classList.remove('hidden');
            this.sendBtn.disabled = true;
            this.startThinkingAnimation();
        } else {
            this.loadingOverlay.classList.add('hidden');
            this.sendBtn.disabled = false;
            this.stopThinkingAnimation();
        }
    }
    
    startThinkingAnimation() {
        this.currentThinkingIndex = 0;
        this.updateThinkingStatus();
        
        this.thinkingInterval = setInterval(() => {
            this.currentThinkingIndex = (this.currentThinkingIndex + 1) % this.thinkingMessages.length;
            this.updateThinkingStatus();
        }, 2000);
    }
    
    stopThinkingAnimation() {
        if (this.thinkingInterval) {
            clearInterval(this.thinkingInterval);
            this.thinkingInterval = null;
        }
    }
    
    updateThinkingStatus() {
        if (this.thinkingStatus) {
            this.thinkingStatus.style.opacity = '0';
            setTimeout(() => {
                this.thinkingStatus.textContent = this.thinkingMessages[this.currentThinkingIndex];
                this.thinkingStatus.style.opacity = '1';
            }, 200);
        }
    }
    
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorToast.classList.remove('hidden');
        
        setTimeout(() => {
            this.closeErrorToast();
        }, 5000);
        
        this.errorToast.style.animation = 'shake 0.5s ease-out';
        setTimeout(() => {
            this.errorToast.style.animation = '';
        }, 500);
    }
    
    closeErrorToast() {
        this.errorToast.classList.add('hidden');
        this.messageInput.focus();
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Quick question function for buttons
function askQuickQuestion(question) {
    if (window.medBot) {
        window.medBot.messageInput.value = question;
        window.medBot.handleInputChange();
        window.medBot.messageInput.focus();
        
        setTimeout(() => {
            window.medBot.handleSubmit(new Event('submit'));
        }, 500);
    }
}

// Global function for error toast close
function closeErrorToast() {
    if (window.medBot) {
        window.medBot.closeErrorToast();
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.medBot = new MedBot();
    
    // Add CSS for additional animations
    const style = document.createElement('style');
    style.textContent = `
        .medical-disclaimer {
            background: linear-gradient(135deg, rgba(230, 57, 70, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
            border-left: 4px solid #E63946;
            padding: 12px 16px;
            margin: 16px 0;
            border-radius: 8px;
            font-size: 0.875rem;
            color: #334155;
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
        
        .message-bubble {
            animation: messageSlideIn 0.4s ease-out;
        }
        
        @keyframes messageSlideIn {
            0% {
                transform: translateY(20px);
                opacity: 0;
            }
            100% {
                transform: translateY(0);
                opacity: 1;
            }
        }
    `;
    document.head.appendChild(style);
});

console.log('🚀 MedBot Advanced Interface Ready!');
