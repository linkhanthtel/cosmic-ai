(function (global) {
    var THEME_KEY = 'cosmic-theme';
    var LANG_KEY = 'cosmic-lang';

    function getTheme() {
        return localStorage.getItem(THEME_KEY) || '';
    }

    function resolveTheme(stored) {
        if (stored === 'light' || stored === 'dark') {
            return stored;
        }
        return global.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }

    function getLanguage() {
        var lang = localStorage.getItem(LANG_KEY);
        return lang === 'my' ? 'my' : 'en';
    }

    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
    }

    function applyLanguage(lang) {
        var normalized = lang === 'my' ? 'my' : 'en';
        document.documentElement.setAttribute('lang', normalized);
        localStorage.setItem(LANG_KEY, normalized);
        global.dispatchEvent(new CustomEvent('cosmic:languagechange', { detail: { lang: normalized } }));
    }

    function setTheme(theme) {
        if (theme !== 'light' && theme !== 'dark') {
            return;
        }
        localStorage.setItem(THEME_KEY, theme);
        applyTheme(theme);
        global.dispatchEvent(new CustomEvent('cosmic:themechange', { detail: { theme: theme } }));
    }

    function setLanguage(lang) {
        applyLanguage(lang);
    }

    function init() {
        applyTheme(resolveTheme(getTheme()));
        applyLanguage(getLanguage());
    }

    global.CosmicPrefs = {
        THEME_KEY: THEME_KEY,
        LANG_KEY: LANG_KEY,
        getTheme: getTheme,
        resolveTheme: resolveTheme,
        getLanguage: getLanguage,
        applyTheme: applyTheme,
        applyLanguage: applyLanguage,
        setTheme: setTheme,
        setLanguage: setLanguage,
        init: init,
    };

    init();
})(window);
