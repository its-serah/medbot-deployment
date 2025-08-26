// Main JavaScript for Purple AI Portfolio

document.addEventListener('DOMContentLoaded', function() {
    
    // Hide preloader when page loads
    window.addEventListener('load', function() {
        document.body.classList.add('loaded');
        setTimeout(function() {
            const preloader = document.querySelector('.preloader');
            if (preloader) {
                preloader.style.display = 'none';
            }
        }, 300);
    });

    // Navbar scroll effect
    window.addEventListener('scroll', function() {
        const navbar = document.getElementById('navbar');
        if (window.scrollY > 100) {
            navbar.style.background = 'rgba(14, 14, 14, 0.98)';
            navbar.style.boxShadow = '0 2px 20px rgba(155, 81, 224, 0.1)';
        } else {
            navbar.style.background = 'rgba(14, 14, 14, 0.95)';
            navbar.style.boxShadow = 'none';
        }
    });

    // Scroll to top button
    const scrollUpBtn = document.querySelector('.scroll-up');
    
    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            scrollUpBtn.classList.add('show');
        } else {
            scrollUpBtn.classList.remove('show');
        }
    });

    scrollUpBtn.addEventListener('click', function(e) {
        e.preventDefault();
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('.nav-link[href^="#"], .dropdown-item[href^="#"], .scroll-down[href^="#"]');
    
    navLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });

                // Close mobile menu if open
                const navbarCollapse = document.querySelector('.navbar-collapse');
                if (navbarCollapse.classList.contains('show')) {
                    navbarCollapse.classList.remove('show');
                }
            }
        });
    });

    // Mobile menu toggle
    const navbarToggle = document.querySelector('.navbar-menu');
    const navbarCollapse = document.querySelector('.navbar-collapse');
    
    if (navbarToggle) {
        navbarToggle.addEventListener('click', function() {
            navbarCollapse.classList.toggle('show');
            
            // Animate hamburger menu
            const btnLines = this.querySelectorAll('.btn-line');
            if (navbarCollapse.classList.contains('show')) {
                btnLines[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
                btnLines[1].style.opacity = '0';
                btnLines[2].style.transform = 'rotate(-45deg) translate(7px, -6px)';
            } else {
                btnLines[0].style.transform = 'none';
                btnLines[1].style.opacity = '1';
                btnLines[2].style.transform = 'none';
            }
        });
    }

    // Add active class to current section in navigation
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-link, .dropdown-item');
    
    window.addEventListener('scroll', function() {
        let current = '';
        
        sections.forEach(function(section) {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (scrollY >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });
        
        navItems.forEach(function(item) {
            item.classList.remove('active');
            if (item.getAttribute('href') === '#' + current) {
                item.classList.add('active');
            }
        });
    });

    // Contact form submission
    const contactForm = document.getElementById('contact-form');
    
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(this);
            const name = formData.get('name');
            const email = formData.get('email');
            const subject = formData.get('subject');
            const message = formData.get('message');
            
            // Simple form validation
            if (!name || !email || !subject || !message) {
                showNotification('Please fill in all fields', 'error');
                return;
            }
            
            if (!isValidEmail(email)) {
                showNotification('Please enter a valid email address', 'error');
                return;
            }
            
            // Simulate form submission (replace with actual form handling)
            showNotification('Message sent successfully!', 'success');
            this.reset();
        });
    }

    // Email validation function
    function isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    // Notification system
    function showNotification(message, type) {
        // Remove existing notifications
        const existingNotification = document.querySelector('.notification');
        if (existingNotification) {
            existingNotification.remove();
        }
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
            <span>${message}</span>
            <button class="notification-close">&times;</button>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? 'var(--theme-color)' : '#e74c3c'};
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
            z-index: 10000;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            transform: translateX(400px);
            transition: transform 0.3s ease;
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Close button functionality
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.style.cssText = `
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
            margin-left: 10px;
        `;
        
        closeBtn.addEventListener('click', () => {
            notification.style.transform = 'translateX(400px)';
            setTimeout(() => notification.remove(), 300);
        });
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.transform = 'translateX(400px)';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
    }

    // Animate elements on scroll
    function animateOnScroll() {
        const animatedElements = document.querySelectorAll('.resume-item, .single-plan, .single-service, .single-review');
        
        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });
        
        animatedElements.forEach(function(element) {
            element.style.opacity = '0';
            element.style.transform = 'translateY(30px)';
            element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(element);
        });
    }
    
    // Initialize scroll animations
    animateOnScroll();

    // Parallax effect for header background
    window.addEventListener('scroll', function() {
        const header = document.querySelector('.header');
        if (header && window.scrollY < window.innerHeight) {
            const scrolled = window.scrollY;
            const parallaxSpeed = scrolled * 0.5;
            header.style.transform = `translateY(${parallaxSpeed}px)`;
        }
    });

    // Add typing cursor blink effect
    const typingText = document.querySelector('.typing_text');
    if (typingText) {
        // This will be handled by the Typed.js library in the HTML
        // Enhanced cursor effects
        const cursor = document.querySelector('.typed-cursor');
        if (cursor) {
            cursor.style.animation = 'blink 1s infinite, colorShift 3s infinite';
        }
    }
    
    // Enhanced scroll animations with stagger effect
    function enhancedScrollAnimations() {
        const elements = document.querySelectorAll('.resume-item, .single-plan, .single-service, .single-review');
        
        elements.forEach((element, index) => {
            element.style.setProperty('--stagger-delay', index);
            element.classList.add('fade-in-up', 'stagger-animation');
        });
    }
    
    // Initialize enhanced animations
    enhancedScrollAnimations();
    
    // Add hover effects to cards
    const cards = document.querySelectorAll('.single-plan, .single-service, .single-review, .contact-item');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-15px) scale(1.02)';
            this.style.zIndex = '10';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            this.style.zIndex = 'auto';
        });
    });
    
    // Add floating effect to icons
    const icons = document.querySelectorAll('.plan-icon, .service-icon');
    icons.forEach(icon => {
        icon.style.animation = 'float 3s ease-in-out infinite';
    });
    
    // Enhanced button interactions
    const buttons = document.querySelectorAll('.button-header, .contact-form button, .btn');
    buttons.forEach(button => {
        button.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-3px) scale(1.05)';
            this.style.boxShadow = '0 15px 35px rgba(155, 81, 224, 0.4)';
        });
        
        button.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            this.style.boxShadow = '';
        });
    });
    
    // Parallax effect for background elements
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.3;
        
        const titleBgs = document.querySelectorAll('.title-bg');
        titleBgs.forEach(bg => {
            bg.style.transform = `translate(-50%, -50%) translateY(${rate}px)`;
        });
    });

    // Social icons hover effect enhancement
    const socialIcons = document.querySelectorAll('.list-social-icons a');
    socialIcons.forEach(function(icon) {
        icon.addEventListener('mouseenter', function() {
            this.style.transform = 'translateX(5px) scale(1.1)';
        });
        
        icon.addEventListener('mouseleave', function() {
            this.style.transform = 'translateX(0) scale(1)';
        });
    });

    // Add ripple effect to buttons
    function addRippleEffect() {
        const buttons = document.querySelectorAll('.button-header, .contact-form button');
        
        buttons.forEach(function(button) {
            button.addEventListener('click', function(e) {
                const ripple = document.createElement('span');
                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.cssText = `
                    position: absolute;
                    border-radius: 50%;
                    background: rgba(255,255,255,0.3);
                    transform: scale(0);
                    animation: ripple 0.6s linear;
                    left: ${x}px;
                    top: ${y}px;
                    width: ${size}px;
                    height: ${size}px;
                `;
                
                this.style.position = 'relative';
                this.style.overflow = 'hidden';
                this.appendChild(ripple);
                
                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });
    }
    
    // Add ripple animation CSS
    const style = document.createElement('style');
    style.textContent = `
        @keyframes ripple {
            to {
                transform: scale(4);
                opacity: 0;
            }
        }
        
        .notification-close:hover {
            opacity: 0.7;
        }
    `;
    document.head.appendChild(style);
    
    // Initialize ripple effects
    addRippleEffect();

    console.log('Purple AI Portfolio - JavaScript loaded successfully! 🟣');
});

// Utility functions
window.purplePortfolio = {
    // Scroll to specific section
    scrollToSection: function(sectionId) {
        const element = document.getElementById(sectionId);
        if (element) {
            const offsetTop = element.offsetTop - 80;
            window.scrollTo({
                top: offsetTop,
                behavior: 'smooth'
            });
        }
    },
    
    // Get current active section
    getCurrentSection: function() {
        const sections = document.querySelectorAll('section[id]');
        let current = '';
        
        sections.forEach(function(section) {
            const sectionTop = section.offsetTop;
            if (window.scrollY >= (sectionTop - 200)) {
                current = section.getAttribute('id');
            }
        });
        
        return current;
    }
};
