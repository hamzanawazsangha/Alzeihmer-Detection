/**
 * NeuroScan AI - 3D Effects and Animations
 * Advanced visual effects for immersive user experience
 */

// ========================================
// PARTICLE SYSTEM
// ========================================

function createParticles() {
    const particlesContainer = document.getElementById('particles');
    if (!particlesContainer) return;
    
    const particleCount = 50;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 20 + 's';
        particle.style.animationDuration = (15 + Math.random() * 10) + 's';
        particle.style.opacity = Math.random() * 0.5 + 0.2;
        particlesContainer.appendChild(particle);
    }
}

// ========================================
// HEADER SCROLL EFFECT
// ========================================

function initHeaderScroll() {
    const header = document.querySelector('.main-header');
    if (!header) return;
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });
}

// ========================================
// 3D CARD TILT EFFECT
// ========================================

function init3DCards() {
    document.querySelectorAll('.card-3d, .feature-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const rotateX = (y - centerY) / 20;
            const rotateY = (centerX - x) / 20;
            
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px) scale(1.02)`;
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) translateY(0) scale(1)';
        });
    });
}

// ========================================
// SMOOTH SCROLL
// ========================================

function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ========================================
// INTERSECTION OBSERVER FOR ANIMATIONS
// ========================================

function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.feature-card, .card-3d, .stats-card').forEach(el => {
        observer.observe(el);
    });
}

// ========================================
// BUTTON RIPPLE EFFECT
// ========================================

function initRippleEffect() {
    document.querySelectorAll('.btn').forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
}

// ========================================
// PARALLAX EFFECT
// ========================================

function initParallax() {
    const parallaxElements = document.querySelectorAll('[data-parallax]');
    
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        
        parallaxElements.forEach(el => {
            const speed = el.dataset.parallax || 0.5;
            const yPos = -(scrolled * speed);
            el.style.transform = `translateY(${yPos}px)`;
        });
    });
}

// ========================================
// LOADING ANIMATION
// ========================================

function showLoading(container) {
    const loadingHTML = `
        <div class="loading-container" style="text-align: center; padding: 3rem;">
            <div class="loading-spinner"></div>
            <div class="loading-dots" style="margin-top: 1rem;">
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
            </div>
            <p style="color: var(--gray-600); margin-top: 1rem; font-weight: 500;">
                Analyzing MRI scan...
            </p>
        </div>
    `;
    container.innerHTML = loadingHTML;
}

function hideLoading(container) {
    const loadingContainer = container.querySelector('.loading-container');
    if (loadingContainer) {
        loadingContainer.style.opacity = '0';
        setTimeout(() => loadingContainer.remove(), 300);
    }
}

// ========================================
// PROGRESS BAR ANIMATION
// ========================================

function animateProgressBar(element, targetWidth, duration = 1000) {
    let start = null;
    const startWidth = 0;
    
    function animate(timestamp) {
        if (!start) start = timestamp;
        const progress = Math.min((timestamp - start) / duration, 1);
        
        const easeOutCubic = 1 - Math.pow(1 - progress, 3);
        const currentWidth = startWidth + (targetWidth - startWidth) * easeOutCubic;
        
        element.style.width = currentWidth + '%';
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    requestAnimationFrame(animate);
}

// ========================================
// COUNTER ANIMATION
// ========================================

function animateCounter(element, target, duration = 2000) {
    let start = null;
    const startValue = 0;
    
    function animate(timestamp) {
        if (!start) start = timestamp;
        const progress = Math.min((timestamp - start) / duration, 1);
        
        const easeOutQuad = 1 - (1 - progress) * (1 - progress);
        const currentValue = Math.floor(startValue + (target - startValue) * easeOutQuad);
        
        element.textContent = currentValue;
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        } else {
            element.textContent = target;
        }
    }
    
    requestAnimationFrame(animate);
}

// ========================================
// TOAST NOTIFICATIONS
// ========================================

function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        position: fixed;
        bottom: 2rem;
        right: 2rem;
        padding: 1rem 1.5rem;
        background: ${type === 'success' ? 'var(--success)' : type === 'error' ? 'var(--error)' : 'var(--primary)'};
        color: white;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-3d-lg);
        z-index: 10000;
        animation: slideInRight 0.3s ease-out;
        font-weight: 500;
    `;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ========================================
// MOBILE MENU TOGGLE
// ========================================

function initMobileMenu() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
        
        // Close menu when clicking on a link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            });
        });
    }
}

// ========================================
// IMAGE PREVIEW WITH 3D EFFECT
// ========================================

function initImagePreview() {
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    
    if (fileInput && previewContainer && previewImage) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewContainer.style.display = 'block';
                    previewContainer.style.animation = 'fadeInUp 0.5s ease-out';
                };
                reader.readAsDataURL(file);
            }
        });
    }
}

// ========================================
// KEYBOARD SHORTCUTS
// ========================================

function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K: Focus search/upload
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const fileInput = document.getElementById('fileInput');
            if (fileInput) fileInput.click();
        }
        
        // Escape: Close modals/reset
        if (e.key === 'Escape') {
            const activeModal = document.querySelector('.modal.active');
            if (activeModal) activeModal.classList.remove('active');
        }
    });
}

// ========================================
// INITIALIZE ALL EFFECTS
// ========================================

function initAll3DEffects() {
    // Core effects
    createParticles();
    initHeaderScroll();
    init3DCards();
    initSmoothScroll();
    initScrollAnimations();
    initRippleEffect();
    initParallax();
    initMobileMenu();
    initImagePreview();
    initKeyboardShortcuts();
    
    console.log('✨ 3D Effects initialized successfully');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll3DEffects);
} else {
    initAll3DEffects();
}

// Export functions for use in other scripts
window.NeuroScan3D = {
    showLoading,
    hideLoading,
    animateProgressBar,
    animateCounter,
    showToast,
    createParticles
};

