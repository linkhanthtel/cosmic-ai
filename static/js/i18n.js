(function (global) {
    var STRINGS = {
        en: {
            'nav.chat': 'Chat',
            'nav.settings': 'Settings',
            'nav.openMenu': 'Open menu',
            'nav.closeMenu': 'Close menu',
            'nav.menu': 'Menu',
            'settings.title': 'Settings',
            'settings.subtitle': 'Customize your Cosmic AI experience.',
            'settings.appearance': 'Appearance',
            'settings.appearanceHint': 'Choose how Cosmic AI looks on your device.',
            'settings.theme': 'Theme',
            'settings.themeLight': 'Light',
            'settings.themeDark': 'Dark',
            'settings.language': 'Language',
            'settings.languageHint': 'Choose your preferred language for the interface.',
            'settings.languageEn': 'English',
            'settings.languageMy': 'Burmese',
            'settings.profile': 'Profile',
            'settings.profileHint': 'More profile customization is coming soon.',
            'settings.comingSoon': 'Coming soon',
            'chat.newChat': 'New Chat',
            'chat.clear': 'Clear',
            'chat.welcomeTitle': 'Cosmic AI',
            'chat.welcomeSubtitle': 'Ask me anything',
            'chat.placeholder': 'Type your message...',
            'chat.send': 'Send',
            'chat.attach': 'Attach image or video',
            'chat.messageInput': 'Message input',
            'chat.removeAttachment': 'Remove attachment',
            'chat.noResponse': 'No response.',
            'chat.errorPrefix': 'Error:',
            'chat.requestFailed': 'Request failed ({status})',
        },
        my: {
            'nav.chat': 'စကားပြော',
            'nav.settings': 'ဆက်တင်',
            'nav.openMenu': 'မီနူးဖွင့်',
            'nav.closeMenu': 'မီနူးပိတ်',
            'nav.menu': 'မီနူး',
            'settings.title': 'ဆက်တင်',
            'settings.subtitle': 'Cosmic AI အတွေ့အကြုံကို စိတ်ကြိုက်ပြင်ဆင်လိုက်ပါ',
            'settings.appearance': 'အပြင်အဆင်',
            'settings.appearanceHint': 'Cosmic AI ကို သင့်စက်တွင် မည်သို့ပြသမည်ကို ရွေးချယ်ပါ။',
            'settings.theme': 'Theme',
            'settings.themeLight': 'အလင်း',
            'settings.themeDark': 'အမှောင်',
            'settings.language': 'ဘာသာစကား',
            'settings.languageHint': 'အင်္ဂါရပ်အတွက် သင်နှစ်သက်သော ဘာသာစကားကို ရွေးချယ်ပါ။',
            'settings.languageEn': 'အင်္ဂလိပ်',
            'settings.languageMy': 'မြန်မာ',
            'settings.profile': 'Profile',
            'settings.profileHint': 'Profile စိတ်ကြိုက်ပြင်ဆင်မှုများ မကြာမီ ရရှိပါမည်။',
            'settings.comingSoon': 'မကြာမီ ရရှိပါမည်',
            'chat.newChat': 'စကားပြောအသစ်',
            'chat.clear': 'ရှင်းလင်းမည်',
            'chat.welcomeTitle': 'Cosmic AI',
            'chat.welcomeSubtitle': 'မိမိသိချင်သည်ကို မေးမြန်းနိုင်ပါသည်',
            'chat.placeholder': 'သင့်စာကို ရိုက်ထည့်ပါ...',
            'chat.send': 'ပို့မည်',
            'chat.attach': 'ပုံ သို့မဟုတ် ဗီဒီယို ပူးတွဲမည်',
            'chat.messageInput': 'မက်ဆေ့ခ်ျ ရိုက်ထည့်ရန်',
            'chat.removeAttachment': 'ပူးတွဲဖိုင်ကို ဖယ်ရှားမည်',
            'chat.noResponse': 'တုံ့ပြန်ချက် မရှိပါ။',
            'chat.errorPrefix': 'အမှား:',
            'chat.requestFailed': 'တောင်းဆိုမှု မအောင်မြင်ပါ ({status})',
        },
    };

    function getLanguage() {
        return global.CosmicPrefs ? global.CosmicPrefs.getLanguage() : 'en';
    }

    function t(key, vars) {
        var lang = getLanguage();
        var table = STRINGS[lang] || STRINGS.en;
        var text = table[key] || STRINGS.en[key] || key;
        if (vars) {
            Object.keys(vars).forEach(function (name) {
                text = text.replace('{' + name + '}', String(vars[name]));
            });
        }
        return text;
    }

    function apply(root) {
        var scope = root || document;
        scope.querySelectorAll('[data-i18n]').forEach(function (el) {
            el.textContent = t(el.getAttribute('data-i18n'));
        });
        scope.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
            el.setAttribute('placeholder', t(el.getAttribute('data-i18n-placeholder')));
        });
        scope.querySelectorAll('[data-i18n-aria]').forEach(function (el) {
            el.setAttribute('aria-label', t(el.getAttribute('data-i18n-aria')));
        });
        scope.querySelectorAll('[data-i18n-title]').forEach(function (el) {
            el.setAttribute('title', t(el.getAttribute('data-i18n-title')));
        });
    }

    global.CosmicI18n = {
        t: t,
        apply: apply,
    };

    global.addEventListener('cosmic:languagechange', function () {
        apply(document);
    });
})(window);
