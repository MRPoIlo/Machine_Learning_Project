(function () {
    'use strict';

    const navbar = document.getElementById('navbar');
    const navToggle = document.getElementById('navToggle');
    const navMenu = document.getElementById('navMenu');
    const navLinks = document.querySelectorAll('.navbar__link');
    const dropdowns = document.querySelectorAll('.navbar__dropdown');
    const parallaxLayers = document.querySelectorAll('[data-parallax]');
    const sections = document.querySelectorAll('section[id], header[id]');

    const isMobile = () => window.matchMedia('(max-width: 768px)').matches;

    function onScroll() {
        const y = window.scrollY;

        if (navbar) {
            navbar.classList.toggle('is-scrolled', y > 24);
        }

        parallaxLayers.forEach(layer => {
            const speed = parseFloat(layer.dataset.parallax) || 0.2;
            const parent = layer.closest('.parallax-section, .hero, .phase-hero');
            const offset = parent ? parent.offsetTop : 0;
            layer.style.transform = `translate3d(0, ${(y - offset) * speed}px, 0)`;
        });
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navToggle.classList.toggle('is-open');
            navMenu.classList.toggle('is-open');
        });
    }

    dropdowns.forEach(dropdown => {
        const trigger = dropdown.querySelector('.navbar__link--has-sub');
        if (!trigger) return;

        trigger.addEventListener('click', e => {
            if (isMobile()) {
                e.preventDefault();
                dropdowns.forEach(d => { if (d !== dropdown) d.classList.remove('is-open'); });
                dropdown.classList.toggle('is-open');
                trigger.setAttribute('aria-expanded', dropdown.classList.contains('is-open'));
            }
        });
    });

    document.addEventListener('click', e => {
        if (isMobile()) return;
        if (!e.target.closest('.navbar__dropdown')) {
            dropdowns.forEach(d => d.classList.remove('is-open'));
        }
    });

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (link.classList.contains('navbar__link--has-sub') && isMobile()) return;
            navToggle?.classList.remove('is-open');
            navMenu?.classList.remove('is-open');
            dropdowns.forEach(d => d.classList.remove('is-open'));
        });
    });

    function updateActiveLink() {
        if (sections.length === 0) return;
        const scrollPos = window.scrollY + 140;
        let activeId = null;

        sections.forEach(section => {
            const top = section.offsetTop;
            const height = section.offsetHeight;
            if (scrollPos >= top && scrollPos < top + height) {
                activeId = section.id;
            }
        });

        navLinks.forEach(link => {
            if (link.classList.contains('navbar__link--has-sub')) return;
            const target = link.getAttribute('href') || '';
            const hashIndex = target.indexOf('#');
            const linkAnchor = hashIndex >= 0 ? target.slice(hashIndex + 1) : '';
            link.classList.toggle('is-active', linkAnchor && linkAnchor === activeId);
        });
    }

    window.addEventListener('scroll', updateActiveLink, { passive: true });
    updateActiveLink();

    const revealTargets = [
        ...document.querySelectorAll('.about__card'),
        ...document.querySelectorAll('.phases__item'),
        ...document.querySelectorAll('.team__card'),
        ...document.querySelectorAll('.section__header'),
        ...document.querySelectorAll('.phase-block')
    ];

    revealTargets.forEach((el, i) => {
        el.setAttribute('data-reveal', '');
        el.style.transitionDelay = `${(i % 4) * 80}ms`;
    });

    const io = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-revealed');
                io.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15, rootMargin: '0px 0px -60px 0px' });

    revealTargets.forEach(el => io.observe(el));

    const yearEl = document.getElementById('year');
    if (yearEl) {
        yearEl.textContent = new Date().getFullYear();
    }

    document.querySelectorAll('.rain-layer').forEach(layer => {
        const dropCount = window.innerWidth < 768 ? 18 : 36;
        const fragment = document.createDocumentFragment();
        for (let i = 0; i < dropCount; i++) {
            const drop = document.createElement('span');
            drop.className = 'rain-drop';
            const left = Math.random() * 100;
            const duration = 2.8 + Math.random() * 3.6;
            const delay = -Math.random() * duration;
            const width = 1 + Math.random() * 1.4;
            const height = 10 + Math.random() * 14;
            const alpha = 0.18 + Math.random() * 0.32;
            drop.style.left = left.toFixed(2) + '%';
            drop.style.width = width.toFixed(2) + 'px';
            drop.style.height = height.toFixed(2) + 'px';
            drop.style.animationDuration = duration.toFixed(2) + 's';
            drop.style.animationDelay = delay.toFixed(2) + 's';
            drop.style.setProperty('--drop-alpha', alpha.toFixed(2));
            fragment.appendChild(drop);
        }
        layer.appendChild(fragment);
    });
})();
