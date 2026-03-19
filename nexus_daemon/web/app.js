const chatHistory = document.getElementById('chat-history');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const memoryList = document.getElementById('memory-list');
const refreshBtn = document.getElementById('refresh-memories');
const modelSelect = document.getElementById('model-select');

const statEpisodes = document.getElementById('stat-episodes');
const statRooms = document.getElementById('stat-rooms');

// Generate a random session ID for this user tab
const sessionId = 'web_session_' + Math.random().toString(36).substring(7);

async function fetchStats() {
    try {
        const res = await fetch('/stats');
        const data = await res.json();
        statEpisodes.innerText = data.episodes_count;
        statRooms.innerText = data.palace_rooms;
    } catch (e) {
        console.error('Failed to fetch stats', e);
    }
}

async function fetchMemories() {
    try {
        const res = await fetch('/api/memories');
        const data = await res.json();
        memoryList.innerHTML = '';
        
        data.memories.reverse().forEach(mem => {
            const el = document.createElement('div');
            el.className = 'memory-item';
            el.innerHTML = `
                <div class="content">${mem.content}</div>
                <div class="meta">Confidence: ${(mem.confidence * 100).toFixed(0)}%</div>
                <button class="delete-btn" onclick="deleteMemory('${mem.id}')" title="Forget Memory">×</button>
            `;
            memoryList.appendChild(el);
        });
        
        fetchStats();
    } catch (e) {
        console.error('Failed to fetch memories', e);
    }
}

// Ensure globally accessible
window.deleteMemory = async function(id) {
    try {
        await fetch(`/api/memories/${id}`, { method: 'DELETE' });
        fetchMemories(); // Refresh list after deletion
    } catch (e) {
        console.error('Failed to delete memory', e);
    }
}

modelSelect.addEventListener('change', async (e) => {
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ model_provider: e.target.value })
        });
    } catch (e) {
        console.error('Failed to change model', e);
    }
});

function appendMessage(text, sender) {
    const msg = document.createElement('div');
    msg.className = `message ${sender}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    
    if (sender === 'ai') {
        bubble.innerHTML = marked.parse(text);
    } else {
        bubble.innerText = text;
    }
    
    msg.appendChild(bubble);
    chatHistory.appendChild(msg);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function showTyping() {
    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator message ai';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = '<span></span><span></span><span></span>';
    chatHistory.appendChild(indicator);
    indicator.style.display = 'flex';
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function hideTyping() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    chatInput.value = '';
    appendMessage(text, 'human');
    showTyping();
    
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ session_id: sessionId, message: text })
        });
        const data = await res.json();
        hideTyping();
        appendMessage(data.response, 'ai');
        
        // Background consolidation triggers automatically on backend, refresh memories after a brief delay
        setTimeout(fetchMemories, 1500);
    } catch (e) {
        hideTyping();
        appendMessage("Error communicating with NEXUS Daemon.", 'ai');
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

refreshBtn.addEventListener('click', fetchMemories);

// Initial fetches
fetchStats();
fetchMemories();

// --- Live Log Streaming ---
const logPanel = document.getElementById('log-panel');
const logOutput = document.getElementById('log-output');
const toggleLogsBtn = document.getElementById('toggle-logs');
const clearLogsBtn = document.getElementById('clear-logs');
let logEventSource = null;

toggleLogsBtn.addEventListener('click', () => {
    const isOpen = logPanel.classList.toggle('open');
    toggleLogsBtn.classList.toggle('active', isOpen);

    if (isOpen && !logEventSource) {
        logEventSource = new EventSource('/api/logs');
        logEventSource.onmessage = (e) => {
            const line = document.createTextNode(e.data + '\n');
            logOutput.appendChild(line);
            // Auto-scroll to bottom
            logOutput.scrollTop = logOutput.scrollHeight;
        };
        logEventSource.onerror = () => {
            logOutput.textContent += '[Connection to log stream lost. Refresh to reconnect.]\n';
            logEventSource.close();
            logEventSource = null;
        };
    } else if (!isOpen && logEventSource) {
        logEventSource.close();
        logEventSource = null;
    }
});

clearLogsBtn.addEventListener('click', () => {
    logOutput.textContent = '';
});

// ── Tab Switching ──
let cachedData = null;

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        if (btn.dataset.tab === 'data' && !cachedData) {
            loadDataTab();
        }
    });
});

document.querySelectorAll('.sub-tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.sub-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.sub-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('sub-' + btn.dataset.sub).classList.add('active');
    });
});

// ── Data Tab Logic ──
async function loadDataTab() {
    try {
        const res = await fetch('/api/data');
        cachedData = await res.json();
        renderDataTab(cachedData);
    } catch (e) {
        console.error('Failed to load data', e);
    }
}

document.getElementById('refresh-data').addEventListener('click', async () => {
    cachedData = null;
    await loadDataTab();
});

document.getElementById('data-search').addEventListener('input', (e) => {
    if (!cachedData) return;
    const q = e.target.value.toLowerCase();
    renderDataTab(cachedData, q);
});

function renderDataTab(data, filter = '') {
    renderPalace(data.palace_memories, filter);
    renderEpisodes(data.episodes, filter);
    renderChatLog(data.chat_history, filter);
}

function renderPalace(memories, filter) {
    const tbody = document.getElementById('palace-tbody');
    const rows = memories.filter(m => !filter || m.content.toLowerCase().includes(filter));
    if (!rows.length) { tbody.innerHTML = `<tr><td colspan="5" class="empty-state">No memories found.</td></tr>`; return; }
    tbody.innerHTML = rows.map(m => {
        const statusClass = m.status === 'active' ? 'badge-active' : 'badge-archived';
        return `<tr>
            <td>${escHtml(m.content)}</td>
            <td>${(m.confidence * 100).toFixed(0)}%</td>
            <td><span class="badge ${statusClass}">${m.status}</span></td>
            <td>${escHtml(m.modality)}</td>
            <td><button class="row-delete-btn" onclick="deleteAndRefresh('${m.id}')">Delete</button></td>
        </tr>`;
    }).join('');
}

function renderEpisodes(episodes, filter) {
    const tbody = document.getElementById('episodes-tbody');
    const rows = episodes.filter(e => !filter || e.content.toLowerCase().includes(filter));
    if (!rows.length) { tbody.innerHTML = `<tr><td colspan="4" class="empty-state">No episodes found.</td></tr>`; return; }
    tbody.innerHTML = rows.map(ep => {
        const consBadge = ep.consolidated ? `<span class="badge badge-yes">Yes</span>` : `<span class="badge badge-no">No</span>`;
        return `<tr>
            <td>${escHtml(ep.content)}</td>
            <td>${ep.salience ?? '—'}</td>
            <td>${consBadge}</td>
            <td style="font-size:0.7rem;color:var(--text-secondary)">${ep.id.substring(0,12)}...</td>
        </tr>`;
    }).join('');
}

function renderChatLog(chatLog, filter) {
    const tbody = document.getElementById('chatlog-tbody');
    const rows = chatLog.filter(c => !filter || (c.content||'').toLowerCase().includes(filter));
    if (!rows.length) { tbody.innerHTML = `<tr><td colspan="3" class="empty-state">No chat history found.</td></tr>`; return; }
    tbody.innerHTML = rows.map(c => {
        const roleParts = c.role.toLowerCase();
        const roleClass = roleParts.includes('ai') || roleParts.includes('assistant') ? 'badge-ai' : 'badge-human';
        const roleLabel = roleParts.includes('ai') || roleParts.includes('assistant') ? 'AI' : 'Human';
        return `<tr>
            <td><span class="badge ${roleClass}">${roleLabel}</span></td>
            <td>${escHtml(c.content)}</td>
            <td style="font-size:0.75rem;color:var(--text-secondary);white-space:nowrap">${c.timestamp || '—'}</td>
        </tr>`;
    }).join('');
}

window.deleteAndRefresh = async function(id) {
    await fetch(`/api/memories/${id}`, { method: 'DELETE' });
    cachedData = null;
    fetchMemories();
    await loadDataTab();
};

function escHtml(str) {
    return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
