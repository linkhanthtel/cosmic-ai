(function (global) {
    var listEl = null;

    function apiError(data, status) {
        if (global.CosmicI18n) {
            return global.CosmicI18n.t('chat.requestFailed', { status: status });
        }
        return 'Request failed (' + status + ')';
    }

    function parseJson(response) {
        return response.json().then(function (data) {
            if (!response.ok) {
                var detail = data && (data.detail || data.error || data.message);
                throw new Error(typeof detail === 'string' ? detail : apiError(data, response.status));
            }
            return data;
        });
    }

    function listConversations() {
        return fetch('/conversations').then(parseJson);
    }

    function getConversation(id) {
        return fetch('/conversations/' + encodeURIComponent(id)).then(parseJson);
    }

    function deleteConversation(id) {
        return fetch('/conversations/' + encodeURIComponent(id), { method: 'DELETE' }).then(parseJson);
    }

    function escapeHtml(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function activeIdFromUrl() {
        var params = new URLSearchParams(global.location.search);
        return params.get('c') || '';
    }

    function renderList(conversations, activeId) {
        if (!listEl) return;
        listEl.innerHTML = '';
        var items = conversations || [];

        if (!items.length) {
            var empty = document.createElement('p');
            empty.className = 'recent-empty';
            empty.setAttribute('data-i18n', 'nav.noRecent');
            empty.textContent = global.CosmicI18n ? global.CosmicI18n.t('nav.noRecent') : 'No recent chats';
            listEl.appendChild(empty);
            return;
        }

        items.forEach(function (item) {
            var row = document.createElement('div');
            row.className = 'recent-item' + (item.id === activeId ? ' active' : '');

            var link = document.createElement('a');
            link.className = 'recent-item-link';
            link.href = '/chat?c=' + encodeURIComponent(item.id);
            link.title = item.title || '';
            link.textContent = item.title || (global.CosmicI18n ? global.CosmicI18n.t('chat.newChat') : 'New Chat');
            link.addEventListener('click', function () {
                if (global.innerWidth <= 768) global.setSidebarOpen(false);
            });

            var remove = document.createElement('button');
            remove.type = 'button';
            remove.className = 'recent-item-delete';
            remove.setAttribute('data-i18n-aria', 'nav.deleteChat');
            remove.setAttribute('aria-label', global.CosmicI18n ? global.CosmicI18n.t('nav.deleteChat') : 'Delete chat');
            remove.textContent = '×';
            remove.addEventListener('click', function (event) {
                event.preventDefault();
                event.stopPropagation();
                var ok = global.confirm(
                    global.CosmicI18n ? global.CosmicI18n.t('nav.deleteConfirm') : 'Delete this chat?'
                );
                if (!ok) return;
                deleteConversation(item.id).then(function () {
                    if (typeof global.onConversationDeleted === 'function') {
                        global.onConversationDeleted(item.id);
                    } else if (activeIdFromUrl() === item.id) {
                        global.location.href = '/chat';
                    } else {
                        refresh(activeIdFromUrl());
                    }
                }).catch(function () {});
            });

            row.appendChild(link);
            row.appendChild(remove);
            listEl.appendChild(row);
        });
    }

    function refresh(activeId) {
        listEl = document.getElementById('recentChats');
        if (!listEl) return Promise.resolve();
        var current = activeId === undefined ? activeIdFromUrl() : activeId;
        return listConversations()
            .then(function (data) {
                renderList(data.conversations || [], current || '');
            })
            .catch(function () {
                renderList([], current || '');
            });
    }

    global.CosmicChats = {
        list: listConversations,
        get: getConversation,
        remove: deleteConversation,
        refresh: refresh,
        activeIdFromUrl: activeIdFromUrl,
        escapeHtml: escapeHtml,
    };

    document.addEventListener('DOMContentLoaded', function () {
        refresh();
    });

    global.addEventListener('cosmic:languagechange', function () {
        refresh(activeIdFromUrl());
    });
})(window);
