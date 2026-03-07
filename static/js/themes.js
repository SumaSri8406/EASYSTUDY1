/**
 * EASYSUDY Global Theme Engine
 * Handles Light, Dark, and Neon modes.
 */

function setTheme(themeName) {
    // 1. Remove existing theme hooks
    document.documentElement.classList.remove('dark');
    document.documentElement.removeAttribute('data-theme');

    // 2. Apply new theme
    if (themeName === 'dark') {
        document.documentElement.classList.add('dark');
    } else if (themeName === 'neon') {
        document.documentElement.setAttribute('data-theme', 'neon');
        // Some components might need 'dark' class as a fallback for Tailwind
        document.documentElement.classList.add('dark');
    }

    // 3. Persist setting
    localStorage.setItem('easystudy-theme', themeName);

    // 4. Update UI Buttons (if they exist on the page, like in Settings)
    document.querySelectorAll('[data-set-theme]').forEach(btn => {
        if (btn.dataset.setTheme === themeName) {
            btn.classList.add('active', 'border-primary', 'ring-2', 'ring-primary/20');
        } else {
            btn.classList.remove('active', 'border-primary', 'ring-2', 'ring-primary/20');
        }
    });
}

// Initialize theme immediately to prevent FOIT (Flash of Incorrect Theme)
(function () {
    const savedTheme = localStorage.getItem('easystudy-theme') || 'light';
    if (savedTheme === 'dark') {
        document.documentElement.classList.add('dark');
    } else if (savedTheme === 'neon') {
        document.documentElement.setAttribute('data-theme', 'neon');
        document.documentElement.classList.add('dark');
    }
})();

// Re-run attribution after DOM content is loaded
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('easystudy-theme') || 'light';
    setTheme(savedTheme);

    // Attach click listeners to any theme buttons found on the page
    document.querySelectorAll('[data-set-theme]').forEach(btn => {
        btn.addEventListener('click', () => {
            setTheme(btn.dataset.setTheme);
        });
    });
});
