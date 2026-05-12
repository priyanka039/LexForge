// ─────────────────────────────────────────────────────────────────────────────
// LEXFORGE FRONTEND · app.js · v3.2
// All functionality preserved. Theme switcher added.
// ─────────────────────────────────────────────────────────────────────────────

const SCREEN_TITLES = {
  dashboard:'Chambers', research:'Research Mode', argument:'Argument Builder',
  opposition:"Devil's Advocate", debate:'Debate Simulation',
  sessions:'Instruction Log', case:'Brief', session:'Saved Work'
};

// API base URL — always use same origin (works on localhost AND ngrok)
const API = window.location.origin;

window._lastResearch   = null;
window._lastIrac       = null;
window._lastOpposition = null;
window._lastDebate     = null;
window._lastPrecedents = [];

let _activeCaseId = null;
let _cases = [];

/** Where to return when leaving a saved session (set when opening one) */
let _sessionReturn = { screen: 'dashboard', params: {} };

/** Picks list view to return to after closing Saved Work */
function captureSessionReturnTarget() {
  const active = document.querySelector('.screen.active');
  const sid    = active?.id?.replace('screen-', '') || '';

  if (sid === 'sessions') return { screen: 'sessions', params: {} };
  if (sid === 'case' && _activeCaseId)
    return { screen: 'case', params: { caseId: _activeCaseId } };
  if (sid === 'dashboard') return { screen: 'dashboard', params: {} };

  if (['research', 'argument', 'opposition', 'debate'].includes(sid))
    return { screen: 'sessions', params: {} };

  if (sid === 'session') return { ..._sessionReturn };

  return { screen: 'dashboard', params: {} };
}

function sessionBackButtonLabel(ret) {
  if (ret.screen === 'sessions') return '← Instruction Log';
  if (ret.screen === 'case') return '← Back to brief';
  return '← Chambers';
}

function backFromSavedSession() {
  navigate(_sessionReturn.screen, _sessionReturn.params);
}

// ── THEME SYSTEM ─────────────────────────────────────────────────────────────
const THEMES = {
  chambers: { label:'Royal Blue', swatch:'swatch-chambers' },
  midnight: { label:'Midnight Judicial', swatch:'swatch-midnight' },
  crimson:  { label:'Crimson',  swatch:'swatch-crimson'  },
  forest:   { label:'Emerald',   swatch:'swatch-forest'   },
  slate:    { label:'Barrister Slate', swatch:'swatch-slate' },
  copper:   { label:'Teal Bench',     swatch:'swatch-copper' },
  auburn:   { label:'Auburn Brief',   swatch:'swatch-auburn' },
};

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme === 'chambers' ? '' : theme);
  // update active swatch
  document.querySelectorAll('.theme-swatch').forEach(s => {
    s.classList.toggle('active', s.dataset.theme === theme);
  });
  // update label
  const lbl = document.getElementById('current-theme-name');
  if (lbl) lbl.textContent = THEMES[theme]?.label || '';
  localStorage.setItem('lf-theme', theme);
}

function initTheme() {
  const saved = localStorage.getItem('lf-theme') || 'chambers';
  applyTheme(THEMES[saved] ? saved : 'chambers');
}

// ── AUTO-EXPAND TEXTAREAS ─────────────────────────────────────────────────────
function autoExpand(el) {
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = el.scrollHeight + 'px';
}
document.addEventListener('input', e => {
  if (e.target.tagName === 'TEXTAREA') autoExpand(e.target);
});

// ── SIDEBAR TOGGLE ────────────────────────────────────────────────────────────
function toggleSidebar() {
  const sb   = document.getElementById('sidebar');
  const main = document.getElementById('main-area');
  const btn  = document.getElementById('sb-open-btn');
  const collapsed = sb.classList.toggle('collapsed');
  main.classList.toggle('expanded', collapsed);
  if (btn) btn.style.display = collapsed ? 'flex' : 'none';
}

// ── MODAL HELPERS ─────────────────────────────────────────────────────────────
function openModal(id) {
  const m = document.getElementById(id);
  if (m) m.classList.add('open');
}
function closeModal(id) {
  const m = document.getElementById(id);
  if (m) m.classList.remove('open');
}
function openExportModal() { buildExportList(); openModal('modal-export'); }

// ── NAVIGATION ────────────────────────────────────────────────────────────────
function navigate(screen, params = {}) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const target = document.getElementById('screen-' + screen);
  if (target) target.classList.add('active');

  document.querySelectorAll('.sidebar .nav-item').forEach(n => n.classList.remove('active'));
  const navMap = { dashboard:0, research:1, argument:2, opposition:3, debate:4 };
  const navItems = document.querySelectorAll('.sidebar .nav-item');
  if (screen === 'session') {
    const back = _sessionReturn?.screen;
    if (back === 'sessions') {
      navItems[0]?.classList.add('active');
    } else {
      const idx = back !== undefined ? navMap[back] : undefined;
      if (idx !== undefined) navItems[idx]?.classList.add('active');
    }
  } else if (screen === 'sessions') {
    navItems[0]?.classList.add('active');
  } else if (navMap[screen] !== undefined) {
    navItems[navMap[screen]]?.classList.add('active');
  }

  document.getElementById('topbar-title').textContent = SCREEN_TITLES[screen] || screen;

  if (screen === 'sessions')                       loadSessions();
  if (screen === 'case' && params.caseId)          loadCaseView(params.caseId);
  if (screen === 'session' && params.sessionId)    loadSessionView(params.sessionId);
  populateCaseDropdowns();
}

// ── ARGUMENT BUILDER TABS ─────────────────────────────────────────────────────
function switchArgTab(idx) {
  [0,1,2].forEach(i => {
    const t = document.getElementById('arg-tab-' + i);
    if (t) t.classList.toggle('active', i === idx);
  });
  document.getElementById('arg-input').style.display      = idx === 0 ? 'block' : 'none';
  document.getElementById('arg-output').style.display     = idx === 1 ? 'block' : 'none';
  document.getElementById('arg-precedents').style.display = idx === 2 ? 'block' : 'none';
}

function switchSubTab(el, target) {
  el.closest('.page').querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  ['arg-input','arg-output','arg-precedents'].forEach(t => {
    const e = document.getElementById(t);
    if (e) e.style.display = 'none';
  });
  const targetEl = document.getElementById(target);
  if (targetEl) targetEl.style.display = 'block';
}

// ── LOADING HELPERS ───────────────────────────────────────────────────────────
function setLoading(btn, text) {
  if (!btn) return;
  btn.disabled = true;
  btn.dataset.orig = btn.innerHTML;
  btn.innerHTML = `<span class="loading-dot"></span><span class="loading-dot"></span><span class="loading-dot"></span> ${text}`;
}
function clearLoading(btn) {
  if (!btn) return;
  btn.disabled = false;
  btn.innerHTML = btn.dataset.orig || btn.innerHTML;
}

// ── DATES & DISPLAY (API may return SQLite UTC without timezone) ────────────────
function parseAppDate(ts) {
  if (ts == null || ts === '') return null;
  const s = String(ts).trim();
  if (!s) return null;
  if (/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}/.test(s)) {
    return new Date(s.replace(' ', 'T') + 'Z');
  }
  const d = new Date(s);
  return isNaN(d.getTime()) ? null : d;
}

function timeAgoFromMs(deltaMs) {
  const sec = Math.max(0, Math.floor(deltaMs / 1000));
  if (sec < 60) return 'Just now';
  if (sec < 3600) return Math.floor(sec / 60) + 'm ago';
  if (sec < 86400) return Math.floor(sec / 3600) + 'h ago';
  return Math.floor(sec / 86400) + 'd ago';
}

function timeAgo(ts) {
  const d = parseAppDate(ts);
  if (!d) return '—';
  return timeAgoFromMs(Date.now() - d.getTime());
}

function formatSessionDateTime(ts) {
  const d = parseAppDate(ts);
  if (!d) return '—';
  return d.toLocaleString('en-IN', {
    day: 'numeric', month: 'short', year: 'numeric',
    hour: 'numeric', minute: '2-digit', hour12: true,
  });
}

function formatSessionWhen(ts) {
  const d = parseAppDate(ts);
  if (!d) return '—';
  return `${formatSessionDateTime(ts)} · ${timeAgoFromMs(Date.now() - d.getTime())}`;
}

function escapeHtml(text) {
  if (text == null) return '';
  const el = document.createElement('div');
  el.textContent = String(text);
  return el.innerHTML;
}

function langLabel(code) {
  const c = (code || 'auto').toLowerCase();
  if (c === 'hi') return 'हिन्दी';
  if (c === 'en') return 'Legal English';
  return 'Auto (match input)';
}

function buildSavedInputSection(session) {
  const inp = session.input_data || {};
  const blocks = [];
  const push = (label, val) => {
    if (val == null || String(val).trim() === '') return;
    blocks.push({ label, val: String(val) });
  };
  if (session.session_type === 'research') {
    push('Your question', inp.query);
    push('Reply language', langLabel(inp.response_language));
    if (inp.use_internet != null) push('Indian Kanoon live search', inp.use_internet ? 'On' : 'Off');
  } else if (session.session_type === 'argument') {
    push('Facts / instructions', inp.facts);
    push('Court', inp.jurisdiction);
    push('Area of law', inp.area_of_law);
    push('Appearing for', inp.client_position);
    push('Further instructions', inp.extra_context);
    push('Reply language', langLabel(inp.response_language));
  } else if (session.session_type === 'opposition') {
    push('Submission stress-tested', inp.argument);
    push('Bench persona id', inp.judge_persona);
    push('Reply language', langLabel(inp.response_language));
  } else if (session.session_type === 'debate') {
    push('Case summary / dispute', inp.case_summary);
    push("Petitioner's submissions", inp.plaintiff_position);
    push("Respondent's anticipated stand", inp.defense_position);
    push('Court', inp.jurisdiction);
    push('Bench persona id', inp.judge_persona);
    push('Reply language', langLabel(inp.response_language));
  }
  if (!blocks.length) return '';
  const rows = blocks.map(b => `
    <div class="saved-instr-block">
      <div class="saved-instr-label">${escapeHtml(b.label)}</div>
      <div class="saved-input-pane" dir="auto">${escapeHtml(b.val)}</div>
    </div>`).join('');
  return `
    <div class="card mb-24 saved-instructions-card">
      <div class="eyebrow" style="margin-bottom:6px">Original instructions</div>
      <div class="card-title">Your saved input</div>
      <div class="card-sub">Exact text as you provided — for review alongside the output below.</div>
      ${rows}
    </div>`;
}

// ── NOTIFICATION ──────────────────────────────────────────────────────────────
let _notifHideTimer = null;

async function showNotif(title, desc, icon = '✓') {
  const bar = document.getElementById('notif-bar');
  if (!bar) return;
  if (_notifHideTimer) clearTimeout(_notifHideTimer);
  document.getElementById('notif-icon').textContent  = icon;
  document.getElementById('notif-title').textContent = title;
  document.getElementById('notif-desc').textContent  = desc;
  bar.classList.add('show');
  _notifHideTimer = setTimeout(() => bar.classList.remove('show'), 7000);
  try {
    if (typeof Notification !== 'undefined') {
      if (Notification.permission === 'granted') {
        new Notification('LexForge — ' + title, { body: desc });
      } else if (Notification.permission === 'default') {
        const p = await Notification.requestPermission();
        if (p === 'granted') new Notification('LexForge — ' + title, { body: desc });
      }
    }
  } catch (_) { /* ignore */ }
}

function showJobStartedNotif(title) {
  showNotif(title, 'Processing on the server — we will notify you again when the result is ready.', '⚙');
}

// ── PROCESSING BANNER ─────────────────────────────────────────────────────────
function showProcessingBanner(container, msg = 'Preparing your brief...') {
  container.innerHTML = `
    <div class="processing-wrap">
      <div class="processing-icon">⚙</div>
      <div class="processing-title">${msg}</div>
      <div class="processing-time"><span>ESTIMATED · 10 MINUTES</span></div>
      <div class="processing-note">You may navigate away — a notification will appear when the transcript is ready. Do not close this browser window.</div>
    </div>`;
}

// ── PRECEDENTS RENDERER ───────────────────────────────────────────────────────
function renderPrecedentsHtml(precedents) {
  if (!precedents || !precedents.length) {
    return '<div style="font-family:var(--mono);font-size:11px;color:var(--w20);padding:8px;font-style:italic">No precedents found.</div>';
  }
  return '<div class="prec-list">' + precedents.map((p, i) => {
    const court = (p.court && p.court !== 'Unknown') ? p.court : 'Indian Court';
    const year  = (p.year  && p.year  !== 'Unknown') ? p.year  : '';
    const bind  = p.binding === 'Binding' ? 'binding' : 'persuasive';
    return `<div class="prec-item">
      <div class="prec-rank">0${i+1}</div>
      <div class="prec-body">
        <div class="prec-name">${p.case_name}<span class="binding-tag ${bind}">${p.binding || 'Persuasive'}</span></div>
        <div class="prec-meta">${[court, year].filter(Boolean).join(' · ')}</div>
        <div class="prec-snip">${p.snippet || ''}</div>
      </div>
    </div>`;
  }).join('') + '</div>';
}

// ── API STATUS ────────────────────────────────────────────────────────────────
async function loadApiStatus() {
  try {
    const data = await (await fetch(`${API}/`)).json();
    const sc = document.getElementById('stat-corpus');
    if (sc) sc.textContent = data.library_size;
    const sm = document.getElementById('stat-model');
    if (sm) {
      const n = (data.model || 'qwen3:8b').split(':')[0];
      sm.textContent = n.charAt(0).toUpperCase() + n.slice(1);
    }
  } catch {
    const sc = document.getElementById('stat-corpus');
    if (sc) { sc.textContent = 'Offline'; sc.style.color = '#f87171'; }
  }
}

// ── CASES MANAGEMENT ──────────────────────────────────────────────────────────
async function loadCases() {
  try {
    const data = await (await fetch(`${API}/api/cases`)).json();
    _cases = data.cases || [];
    renderSidebarCases();
    populateCaseDropdowns();
    renderDashCasesGrid();
    renderChambersDate();
    const s1 = document.getElementById('stat-cases');
    if (s1) s1.textContent = _cases.filter(c => c.status === 'active').length;
    const s2 = document.getElementById('stat-cases2');
    if (s2) s2.textContent = _cases.filter(c => c.status === 'active').length;
  } catch(e) { console.error('Load cases failed:', e); }
}

function renderSidebarCases() {
  const el = document.getElementById('sidebar-cases');
  if (!el) return;
  if (!_cases.length) {
    el.innerHTML = '<div style="font-family:var(--mono);font-size:10px;color:var(--w20);padding:8px">No briefs on docket</div>';
    return;
  }
  const colors = ['#7eaaee','#c8a55a','#f87171','#4ade80','#fbbf24','#a78bfa'];
  const briefIcon = (stroke) => `<span class="matter-icon" style="--matter-stroke:${stroke}" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M10 22V8a2 2 0 012-2h7.5a2.5 2.5 0 012.5 2.5v11a2 2 0 01-2 2z"/><path d="M6 22H8a2 2 0 002-2V4a2 2 0 00-2-2H4a2 2 0 00-2 2v16a2 2 0 002 2h2z"/><path d="M10 12h8"/></svg></span>`;
  el.innerHTML = _cases.map((cas, i) => `
    <div class="matter-item ${_activeCaseId === cas.id ? 'active' : ''}" onclick="openCaseView(${cas.id})">
      ${briefIcon(colors[i % colors.length])}
      <div class="history-name" title="${cas.name}">${cas.name}</div>
      <div class="history-count">${cas.session_count || 0}</div>
    </div>`).join('');
}

function renderDashCasesGrid() {
  const grid = document.getElementById('dash-cases-grid');
  if (!grid) return;
  const colors = ['#7eaaee','#c8a55a','#f87171','#4ade80','#fbbf24','#a78bfa'];
  const activeCases = _cases.filter(c => c.status !== 'closed');
  if (!activeCases.length) {
    grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1">
      <div class="empty-icon">⚖</div>
      <div class="empty-title">No Briefs on Docket</div>
      <div class="empty-desc">Click "+ New Brief" to open your first matter.</div>
    </div>`;
    return;
  }
  grid.innerHTML = activeCases.map((cas, i) => {
    const col  = colors[i % colors.length];
    const area = cas.area_of_law || 'General';
    const cnt  = cas.session_count || 0;
    const upd  = cas.updated_at || cas.created_at;
    const date = upd ? formatSessionWhen(upd) : '';
    return `<div style="background:var(--s1);border:1px solid var(--gold-8);border-radius:4px;padding:20px 22px;cursor:pointer;transition:all 0.2s;position:relative;overflow:hidden"
      onclick="openCaseView(${cas.id})"
      onmouseenter="this.style.borderColor='rgba(193,128,33,.25)';this.style.transform='translateY(-2px)'"
      onmouseleave="this.style.borderColor='';this.style.transform=''">
      <div style="position:absolute;top:0;left:0;right:0;height:3px;background:${col}"></div>
      <div style="display:flex;justify-content:space-between;margin-bottom:8px">
        <div style="font-family:var(--mono);font-size:9.5px;color:var(--w40);letter-spacing:0.14em;text-transform:uppercase">${area}</div>
        <div style="font-family:var(--mono);font-size:9.5px;color:var(--w40)">${cnt} instruction${cnt !== 1 ? 's' : ''}</div>
      </div>
      <div style="font-family:var(--legal);font-size:16px;font-weight:700;font-style:italic;color:var(--cream);line-height:1.25;margin-bottom:3px">${cas.name}</div>
      ${cas.client_name ? `<div style="font-size:12px;color:var(--w40);margin-bottom:6px">Client: ${cas.client_name}</div>` : ''}
      <div style="font-size:11px;color:var(--w40);font-family:var(--mono);margin-top:10px;margin-bottom:10px;line-height:1.45">Updated ${date}</div>
      <div style="display:flex;gap:6px">
        <button onclick="event.stopPropagation();navigate('argument');setTimeout(()=>{const s=document.getElementById('arg-case-select');if(s)s.value='${cas.id}';},150)"
          style="font-size:11px;padding:5px 11px;background:none;border:1px solid var(--gold-8);border-radius:6px;color:var(--w60);cursor:pointer">+ Submission</button>
        <button onclick="event.stopPropagation();navigate('research');setTimeout(()=>{const s=document.getElementById('research-case-select');if(s)s.value='${cas.id}';},150)"
          style="font-size:11px;padding:5px 11px;background:none;border:1px solid var(--gold-8);border-radius:6px;color:var(--w60);cursor:pointer">+ Search</button>
      </div>
    </div>`;
  }).join('');
}

function populateCaseDropdowns() {
  ['arg-case-select','opp-case-select','debate-case-select','research-case-select'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const current = el.value;
    el.innerHTML = '<option value="">— Save to a matter (optional) —</option>' +
      _cases.map(c => `<option value="${c.id}">${c.name}</option>`).join('');
    el.value = current;
  });
}

async function createCase() {
  const name   = document.getElementById('new-case-name').value.trim();
  const client = document.getElementById('new-case-client').value.trim();
  const area   = document.getElementById('new-case-area').value;
  const desc   = document.getElementById('new-case-desc').value.trim();
  if (!name) { alert('Please enter a matter name.'); return; }
  try {
    const res  = await fetch(`${API}/api/cases`, {
      method:'POST', headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},
      body: JSON.stringify({name, client_name:client, area_of_law:area, description:desc})
    });
    const data = await res.json();
    closeModal('modal-case');
    ['new-case-name','new-case-client','new-case-desc'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = '';
    });
    await loadCases();
    showNotif('Brief Opened', `"${data.case.name}" created.`);
  } catch(e) { alert('Failed to create matter: ' + e.message); }
}

function openCaseView(caseId) {
  _activeCaseId = caseId;
  renderSidebarCases();
  navigate('case', { caseId });
}

async function loadCaseView(caseId) {
  const header = document.getElementById('case-header');
  const list   = document.getElementById('case-sessions-list');
  header.innerHTML = '<div style="color:var(--w40);font-size:13px">Loading...</div>';
  try {
    const [caseRes, sessRes] = await Promise.all([
      fetch(`${API}/api/cases/${caseId}`),
      fetch(`${API}/api/cases/${caseId}/sessions`)
    ]);
    const caseData = await caseRes.json();
    const sessData = await sessRes.json();
    const c = caseData.case;
    header.innerHTML = `
      <div class="case-header-card animate-in">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:16px">
          <div>
            <div style="font-family:var(--mono);font-size:10px;color:var(--gold);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:6px">${c.area_of_law} · ${c.status.toUpperCase()}</div>
            <div style="font-family:var(--legal);font-size:26px;font-weight:700;font-style:italic;color:var(--cream);margin-bottom:4px">${c.name}</div>
            ${c.client_name ? `<div style="font-size:13px;color:var(--w40)">Client: ${c.client_name}</div>` : ''}
            ${c.description ? `<div style="font-family:var(--legal);font-size:13px;color:var(--w60);margin-top:8px;line-height:1.6">${c.description}</div>` : ''}
          </div>
          <div style="display:flex;gap:8px">
            <button class="btn btn-ghost" style="font-size:12px" onclick="navigate('research');setTimeout(()=>{document.getElementById('research-case-select').value='${caseId}'},100)">+ Search Precedents</button>
            <button class="btn btn-gold" style="font-size:12px" onclick="navigate('argument');setTimeout(()=>{document.getElementById('arg-case-select').value='${caseId}'},100)">+ Draft Submissions</button>
          </div>
        </div>
        <div class="stat-row" style="margin-bottom:0">
          <div class="stat-chip"><div class="stat-value" style="font-size:20px">${sessData.total}</div><div class="stat-label">Total Entries</div></div>
          <div class="stat-chip"><div class="stat-value" style="font-size:20px">${sessData.sessions.filter(s=>s.session_type==='research').length}</div><div class="stat-label">Searches</div></div>
          <div class="stat-chip"><div class="stat-value" style="font-size:20px">${sessData.sessions.filter(s=>s.session_type==='argument').length}</div><div class="stat-label">Submissions</div></div>
          <div class="stat-chip"><div class="stat-value" style="font-size:20px">${sessData.sessions.filter(s=>s.session_type==='debate').length}</div><div class="stat-label">Hearings</div></div>
        </div>
      </div>`;
    renderSessionCards(sessData.sessions, list);
  } catch(e) {
    header.innerHTML = `<div class="card" style="border-color:rgba(192,57,43,0.3)"><div style="color:#f87171">Error loading matter: ${e.message}</div></div>`;
  }
}

// SVGs match dashboard "What are you working on today?" feature cards
function sessionIconMarkup(type) {
  const map = {
    research: `<svg class="session-type-svg" width="18" height="18" viewBox="0 0 19 19" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><circle cx="8" cy="8" r="5.5" stroke="currentColor" stroke-width="1.7"/><line x1="12.2" y1="12.2" x2="17" y2="17" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>`,
    argument: `<svg class="session-type-svg" width="15" height="19" viewBox="0 0 16 19" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><rect x="1" y="1" width="14" height="17" rx="1.5" stroke="currentColor" stroke-width="1.7"/><line x1="4" y1="6" x2="12" y2="6" stroke="currentColor" stroke-width="1.7"/><line x1="4" y1="9.5" x2="12" y2="9.5" stroke="currentColor" stroke-width="1.7"/><line x1="4" y1="13" x2="9.5" y2="13" stroke="currentColor" stroke-width="1.7"/></svg>`,
    opposition: `<svg class="session-type-svg" width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M8 7H4v4M16 17h4v-4" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 8L4 12l3 4M17 16l3-4-3-4" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    debate: `<svg class="session-type-svg" width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12 4v16M5 9l7-3 7 3M5 15l7 3 7-3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
  };
  return map[type] || map.research;
}

/** Strip common markdown from session titles for card preview */
function plainTitlePreview(t) {
  if (!t) return '';
  let s = String(t);
  s = s.replace(/\*\*([^*]+)\*\*/g, '$1');
  s = s.replace(/\*([^*]+)\*/g, '$1');
  return s.replace(/\s+/g, ' ').trim();
}

// ── SESSIONS ──────────────────────────────────────────────────────────────────
const SESSION_LABELS = {
  research:   { label:'Precedent Research',    color:'#6090c8' },
  argument:   { label:'Written Submissions',   color:'var(--gold)' },
  opposition: { label:'Opposition Analysis',   color:'#c07060' },
  debate:     { label:'Adversarial Hearing',   color:'#6abf6a' },
};

function renderSessionCards(sessions, container) {
  if (!sessions.length) {
    container.innerHTML = `<div class="empty-state"><div class="empty-icon">📄</div><div class="empty-title">No entries yet</div><div class="empty-desc">Your research, submissions, and simulations will appear here.</div></div>`;
    return;
  }
  container.innerHTML = sessions.map(s => {
    const meta = SESSION_LABELS[s.session_type] || SESSION_LABELS.research;
    const icon = sessionIconMarkup(s.session_type);
    const titlePreview = plainTitlePreview(s.title);
    return `
      <div class="session-card" onclick="openSessionView(${s.id})">
        <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:8px">
          <div class="session-icon-wrap" style="color:${meta.color}">${icon}</div>
          <div style="min-width:0;flex:1">
            <div class="session-type" style="color:${meta.color}">${meta.label}</div>
          </div>
          <div class="session-meta" style="margin-left:auto;text-align:right;max-width:min(52%,220px);flex-shrink:0">${formatSessionWhen(s.created_at)}</div>
        </div>
        <div class="session-title">${titlePreview}</div>
        ${s.notes ? `<div style="font-size:11px;color:var(--w40);margin-top:6px;font-style:italic">${s.notes.slice(0,80)}${s.notes.length > 80 ? '...' : ''}</div>` : ''}
        <div class="session-actions">
          <button class="btn btn-ghost" style="font-size:11px;padding:5px 10px" onclick="event.stopPropagation();openSessionView(${s.id})">Open</button>
          <button class="btn btn-ghost" style="font-size:11px;padding:5px 10px" onclick="event.stopPropagation();downloadSession(${s.id})">↓ PDF</button>
          <button class="btn btn-ghost" style="font-size:11px;padding:5px 10px;color:#f87171;border-color:rgba(248,113,113,.3)" onclick="event.stopPropagation();deleteSession(${s.id},this)">Delete</button>
        </div>
      </div>`;
  }).join('');
}

async function loadSessions(type = null) {
  const container = document.getElementById('sessions-list');
  if (container) container.innerHTML = '<div style="color:var(--w40);font-size:13px;padding:12px">Loading...</div>';
  try {
    const url  = `${API}/api/sessions${type ? `?session_type=${type}` : ''}`;
    const data = await (await fetch(url)).json();
    if (container) renderSessionCards(data.sessions || [], container);

    const activity = document.getElementById('activity-list');
    if (activity) renderSessionCards((data.sessions || []).slice(0,5), activity);

    const today  = (data.sessions || []).filter(s => {
      const d = parseAppDate(s.created_at);
      return d && d.toDateString() === new Date().toDateString();
    }).length;
    const statEl = document.getElementById('stat-sessions');
    if (statEl) statEl.textContent = today;
  } catch(e) {
    if (container) container.innerHTML = `<div class="card" style="border-color:rgba(192,57,43,0.3)"><div style="color:#f87171">Error: ${e.message}</div></div>`;
  }
}

function openSessionView(sessionId) {
  _sessionReturn = captureSessionReturnTarget();
  navigate('session', { sessionId });
}

async function loadSessionView(sessionId) {
  const container = document.getElementById('session-view-content');
  container.innerHTML = '<div style="color:var(--w40);font-size:13px;padding:12px">Loading saved session...</div>';
  try {
    const data    = await (await fetch(`${API}/api/sessions/${sessionId}`)).json();
    const session = data.session;
    const meta    = SESSION_LABELS[session.session_type] || SESSION_LABELS.research;
    const output  = session.output_data || {};
    let contentHtml = '';

    if (session.session_type === 'research') {
      const answer = output.answer || '—';
      const precs  = output.precedents || [];
      const live   = output.live_results || [];
      contentHtml = `
        <div class="grid-2" style="gap:20px;align-items:start">
          <div class="card">
            <div class="eyebrow" style="margin-bottom:4px">Research Memorandum</div>
            <div class="card-title">Authorities &amp; Analysis</div>
            <div class="card-sub">Based on ${output.total_sources || 0} sources</div>
            <div class="answer-body" style="margin-top:16px">
              ${marked.parse(
                answer
                  .replace(/\[SOURCE (\d+)\]/g, '<span class="cite">SOURCE $1</span>')
                  .replace(/\[LIVE SOURCE (\d+)\]/g, '<span class="cite-live">LIVE $1</span>')
              )}
            </div>
          </div>
          <div class="card">
            <div class="card-title">Precedents Found</div>
            ${renderPrecedentsHtml(precs)}
            ${live.length ? `<div class="divider"></div>
              <div style="font-family:var(--mono);font-size:10px;color:#6abf6a;margin-bottom:10px;text-transform:uppercase;letter-spacing:0.1em">🟢 Live from Indian Kanoon</div>
              ${live.map(r => `<div class="prec-item">
                <div class="prec-body">
                  <div class="prec-name">${r.title}<span class="live-badge">Live</span></div>
                  <div class="prec-meta">${r.court} · ${r.year}</div>
                  <div class="prec-snip">${r.snippet || ''}</div>
                  <a href="${r.url}" target="_blank" style="font-size:10px;color:var(--gold)">View on Indian Kanoon →</a>
                </div>
              </div>`).join('')}` : ''}
          </div>
        </div>`;

    } else if (session.session_type === 'argument') {
      const args = output.arguments || [];
      contentHtml = args.map((arg, idx) => {
        const irac = arg.irac || {};
        return `
          <div style="margin-bottom:32px">
            <div style="font-family:var(--mono);font-size:10px;color:var(--gold);letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px">Issue ${idx+1} · ${arg.area_of_law} · ${(arg.priority||'medium').toUpperCase()} PRIORITY</div>
            <div style="font-family:var(--legal);font-size:20px;font-weight:700;color:var(--cream);margin-bottom:16px">${arg.issue_title}</div>
            ${['issue','rule','application','conclusion'].map(k => {
              const labels  = {issue:'Issue — Legal Question',rule:'Rule — Applicable Law',application:'Application — Law to Facts',conclusion:'Conclusion — Outcome & Remedy'};
              const letters = {issue:'I',rule:'R',application:'A',conclusion:'C'};
              return `<div class="irac-block"><div class="irac-hdr"><div class="irac-letter">${letters[k]}</div><div><div class="irac-kind">${k.toUpperCase()}</div><div class="irac-title">${labels[k]}</div></div></div><div class="irac-body">${(irac[k]||'—').replace(/\[SOURCE (\d+)\]/g,'<span class="cite">SOURCE $1</span>')}</div></div>`;
            }).join('')}
          </div>`;
      }).join('');

    } else if (session.session_type === 'opposition') {
      const an = output.analysis || {};
      const rc = {HIGH:'risk-HIGH',MODERATE:'risk-MODERATE',LOW:'risk-LOW'}[an.risk_level] || 'risk-MODERATE';
      contentHtml = `
        <div class="card mb-24">
          <div class="flex items-center justify-between mb-16">
            <div><div class="card-title">Risk Assessment</div><div class="card-sub">Stress-test: opposing counsel &amp; Bench anticipation</div></div>
            <span class="risk-badge ${rc}">${an.risk_level || 'MODERATE'} RISK</span>
          </div>
          <div class="stat-row" style="margin-bottom:0">
            <div class="stat-chip"><div class="stat-value" style="font-size:20px;color:#fbbf24">${(an.weaknesses||[]).length}</div><div class="stat-label">Weaknesses</div></div>
            <div class="stat-chip"><div class="stat-value" style="font-size:20px;color:#f87171">${(an.weaknesses||[]).filter(w=>w.severity==='HIGH').length}</div><div class="stat-label">High Risk</div></div>
          </div>
        </div>
        <div class="grid-2">
          <div class="card">
            <div class="card-title">Weaknesses</div>
            ${(an.weaknesses||[]).map(w=>`<div class="weakness"><div class="weakness-id" style="color:${w.severity==='HIGH'?'#f87171':'#fbbf24'}">${w.id}</div><div><div style="font-size:11px;font-family:var(--mono);color:${w.severity==='HIGH'?'#f87171':'#fbbf24'};margin-bottom:4px">${w.severity} RISK</div><div class="weakness-text">${w.description}</div></div></div>`).join('')}
          </div>
          <div class="card">
            <div class="card-title">Counter-Arguments</div>
            <div class="debate-body">${(an.counter_arguments||[]).map(c=>`<div class="debate-point">${c.point}${(c.source||c.authority)?`<span class="citation-tag">${c.source||c.authority}</span>`:''}</div>`).join('')}</div>
          </div>
        </div>`;

    } else if (session.session_type === 'debate') {
      const sm = output.summary || {};
      const rc = {HIGH:'risk-HIGH',MODERATE:'risk-MODERATE',LOW:'risk-LOW'}[sm.risk_level] || 'risk-MODERATE';
      const renderPts = pts => (pts||[]).map(p=>`<div class="debate-point">${p.point||p}${p.citation?`<span class="citation-tag">${p.citation}</span>`:''}</div>`).join('');
      const jn = output.judge_persona?.name || '';
      const jline = [jn, output.jurisdiction].filter(Boolean).join(' · ');
      contentHtml = `
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
          <div style="font-family:var(--mono);font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:var(--w40)">Round 1</div>
          <div style="flex:1;height:1px;background:rgba(193,128,33,.1)"></div>
          <div style="font-family:var(--serif);font-size:14px;color:var(--gold);font-style:italic">Opening Submissions</div>
        </div>
        <div class="debate-grid mb-24">
          <div class="debate-side debate-p"><div class="debate-hdr">PETITIONER</div><div class="debate-body">${renderPts(output.round1?.plaintiff)}</div></div>
          <div class="debate-vs">vs</div>
          <div class="debate-side debate-d"><div class="debate-hdr">RESPONDENT</div><div class="debate-body">${renderPts(output.round1?.defense)}</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
          <div style="font-family:var(--mono);font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:var(--w40)">Round 2</div>
          <div style="flex:1;height:1px;background:rgba(193,128,33,.1)"></div>
          <div style="font-family:var(--serif);font-size:14px;color:var(--gold);font-style:italic">Rebuttal round · Sur-rebuttal</div>
        </div>
        <div class="debate-grid mb-24">
          <div class="debate-side debate-p"><div class="debate-hdr">PETITIONER — REBUTTAL</div><div class="debate-body">${renderPts(output.round2?.plaintiff)}</div></div>
          <div class="debate-vs">vs</div>
          <div class="debate-side debate-d"><div class="debate-hdr">RESPONDENT — SUR-REBUTTAL</div><div class="debate-body">${renderPts(output.round2?.defense)}</div></div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;margin-top:8px">
          <div style="font-family:var(--mono);font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:var(--w40)">Round 3</div>
          <div style="flex:1;height:1px;background:rgba(193,128,33,.1)"></div>
          <div style="font-family:var(--serif);font-size:14px;color:var(--gold);font-style:italic">Judicial Observations</div>
        </div>
        <div class="card" style="border-color:rgba(193,128,33,0.3)">
          <div class="flex items-center gap-8 mb-16">
            <div><div class="card-title">Judicial Observations</div>
            <div class="card-sub">${jline ? escapeHtml(jline) + ' · ' : ''}Assessment, likely outcome &amp; strategy</div></div>
            <span class="risk-badge ${rc}" style="margin-left:auto">Overall: ${sm.risk_level||'MODERATE'} Risk</span>
          </div>
          ${sm.judicial_observation ? `<div style="background:var(--void);border-left:3px solid var(--gold);padding:13px 16px;border-radius:4px;margin-bottom:18px"><div style="font-family:var(--mono);font-size:9.5px;color:var(--gold);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:7px">From the Bench</div><div style="font-family:var(--legal);font-size:17px;font-style:italic;color:var(--cream);line-height:1.6;text-align:justify">"${escapeHtml(sm.judicial_observation)}"</div></div>` : ''}
          <div style="font-family:var(--legal);font-size:12pt;line-height:2;color:var(--w60);text-align:justify">
            <strong style="color:var(--cream)">Assessment:</strong> ${sm.overall_assessment||'—'}<br/><br/>
            <strong style="color:var(--cream)">Likely Outcome:</strong> ${sm.likely_outcome||'—'}<br/><br/>
            <strong style="color:var(--cream)">Strategy:</strong> ${sm.strategic_recommendation||'—'}
          </div>
        </div>`;
    }

    const savedHtml = buildSavedInputSection(session);
    const backLbl    = sessionBackButtonLabel(_sessionReturn);

    const voiceRole = ({
      research:   'researcher',
      argument:   'petitioner',
      opposition: 'opposition',
      debate:     'judge',
    })[session.session_type] || 'default';

    container.innerHTML = `
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:24px;flex-wrap:wrap">
        <button type="button" class="btn btn-ghost session-back-btn" style="font-size:12px" onclick="backFromSavedSession()">${backLbl}</button>
        <div style="flex:1;min-width:0">
          <div style="font-family:var(--mono);font-size:10px;color:${meta.color};letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px;line-height:1.45">${meta.label}<br/><span style="color:var(--w40)">${formatSessionWhen(session.created_at)}</span></div>
          <div style="font-family:var(--legal);font-size:22px;font-weight:700;font-style:italic;color:var(--cream);word-break:break-word">${escapeHtml(plainTitlePreview(session.title))}</div>
        </div>
        <div class="result-actions" style="margin-bottom:0">
          <button class="btn btn-ghost voice-listen-btn" type="button"
                  data-target="session-output-content" data-role="${voiceRole}"
                  data-label-idle="Read aloud" data-label-playing="Stop reading">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5L6 9H3v6h3l5 4V5z"/><path d="M15.5 8.5a5 5 0 010 7M19 5a9 9 0 010 14"/></svg>
            <span class="voice-btn-label">Read aloud</span>
          </button>
          <button class="btn btn-ghost" style="font-size:12px" onclick="downloadSession(${session.id})">↓ PDF</button>
        </div>
      </div>
      ${savedHtml}
      <div class="mb-16" style="font-family:var(--mono);font-size:10px;letter-spacing:0.14em;text-transform:uppercase;color:var(--w40)">Generated output</div>
      <div id="session-output-content">${contentHtml}</div>
      <div class="notes-box">
        <label>Counsel's Notes</label>
        <textarea placeholder="Record your observations, instructions, or hearing notes..." onblur="saveNotes(${session.id},this.value)">${session.notes || ''}</textarea>
      </div>
      <div style="margin-top:16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <div style="font-family:var(--mono);font-size:10px;color:var(--w40);text-transform:uppercase;letter-spacing:0.1em">Assign to Matter:</div>
        <select class="form-select" style="width:220px" onchange="assignSessionToCase(${session.id},this.value)">
          <option value="">— Not assigned —</option>
          ${_cases.map(c=>`<option value="${c.id}" ${session.case_id===c.id?'selected':''}>${c.name}</option>`).join('')}
        </select>
      </div>`;

  } catch(e) {
    container.innerHTML = `<div class="card" style="border-color:rgba(192,57,43,0.3)"><div style="color:#f87171">Error loading session: ${e.message}</div></div>`;
  }
}

async function saveNotes(sessionId, notes) {
  try {
    await fetch(`${API}/api/sessions/${sessionId}/notes`, {
      method:'PATCH', headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},
      body: JSON.stringify({notes})
    });
  } catch(e) { console.error('Save notes failed', e); }
}

async function assignSessionToCase(sessionId, caseId) {
  try {
    await fetch(`${API}/api/sessions/${sessionId}/case`, {
      method:'PATCH', headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},
      body: JSON.stringify({case_id: caseId ? parseInt(caseId) : null})
    });
    await loadCases();
  } catch(e) { console.error('Assign case failed', e); }
}

async function deleteSession(sessionId, btn) {
  if (!confirm('Delete this session? This cannot be undone.')) return;
  try {
    await fetch(`${API}/api/sessions/${sessionId}`, {method:'DELETE'});
    btn.closest('.session-card').remove();
    await loadSessions();
  } catch(e) { alert('Delete failed: ' + e.message); }
}

// ── UPLOAD LIBRARY ────────────────────────────────────────────────────────────
let _filesToUpload = [];

function handleDrop(e) {
  e.preventDefault();
  const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.pdf'));
  addToUploadQueue(files);
}
function handleFileSelect(e) {
  addToUploadQueue(Array.from(e.target.files).filter(f => f.name.endsWith('.pdf')));
  e.target.value = '';
}
function addToUploadQueue(files) {
  files.forEach(f => { if (!_filesToUpload.find(x => x.name === f.name)) _filesToUpload.push(f); });
  renderUploadQueue();
}
function renderUploadQueue() {
  const q   = document.getElementById('upload-queue');
  const btn = document.getElementById('upload-btn');
  if (!_filesToUpload.length) { q.innerHTML=''; btn.style.display='none'; return; }
  btn.style.display = 'inline-flex';
  btn.textContent   = `Upload ${_filesToUpload.length} File${_filesToUpload.length > 1 ? 's' : ''}`;
  q.innerHTML = _filesToUpload.map((f, i) => `
    <div class="upload-item" id="upitem-${i}">
      <span>📄</span>
      <span class="upload-item-name">${f.name}</span>
      <span class="upload-item-size">${(f.size/1024).toFixed(0)} KB</span>
      <span class="upload-item-status" id="upstat-${i}" style="color:var(--w40)">Pending</span>
      <button onclick="_filesToUpload.splice(${i},1);renderUploadQueue()" style="background:none;border:none;color:var(--w40);cursor:pointer">✕</button>
    </div>`).join('');
}

async function uploadFiles() {
  const btn = document.getElementById('upload-btn');
  btn.disabled=true; btn.textContent='Uploading...';
  let ok = 0;
  for (let i=0; i < _filesToUpload.length; i++) {
    const f  = _filesToUpload[i];
    const se = document.getElementById(`upstat-${i}`);
    const ie = document.getElementById(`upitem-${i}`);
    if (se) { se.textContent='Uploading...'; se.style.color='#7eaaee'; }
    const fd = new FormData(); fd.append('file', f);
    try {
      const data = await (await fetch(`${API}/api/corpus/upload`,{method:'POST',body:fd})).json();
      if (data.status === 'success') {
        if (se) { se.textContent=`✓ ${data.chunk_count} sections`; se.style.color='#4ade80'; }
        if (ie) ie.style.borderColor='rgba(22,163,74,0.3)';
        const sc = document.getElementById('stat-corpus');
        if (sc) sc.textContent = data.corpus_size;
        ok++;
      } else if (data.status === 'already_exists') {
        if (se) { se.textContent='Already in library'; se.style.color='#fbbf24'; }
      } else { throw new Error(data.detail || 'Failed'); }
    } catch(e) { if (se) { se.textContent='Failed'; se.style.color='#f87171'; } }
  }
  btn.disabled=false; btn.textContent = ok > 0 ? `✓ ${ok} Added` : 'Retry';
  _filesToUpload = [];
  if (ok > 0) showNotif('Authorities Added', `${ok} judgment${ok > 1 ? 's' : ''} indexed.`);
}

async function loadLibraryList() {
  const el = document.getElementById('library-list');
  el.style.display='block';
  el.innerHTML='<div style="color:var(--w40);font-size:12px">Loading...</div>';
  try {
    const data = await (await fetch(`${API}/api/corpus/list`)).json();
    if (!data.cases?.length) {
      el.innerHTML='<div style="color:var(--w40);font-size:12px;padding:8px 0">No documents in library yet.</div>';
      return;
    }
    el.innerHTML = `<div style="font-family:var(--mono);font-size:10px;color:var(--w40);margin-bottom:10px">${data.total_cases} documents · ${data.total_chunks} sections indexed</div>` +
      data.cases.map(c => `
        <div class="corpus-item">
          <div class="corpus-body">
            <div class="corpus-name">${c.case_name}</div>
            <div class="corpus-meta">${c.court} · ${c.year}</div>
          </div>
          <button style="background:none;border:none;color:var(--w40);cursor:pointer;font-size:14px" onclick="deleteFromLibrary('${c.case_file}',this)" title="Remove">✕</button>
        </div>`).join('');
  } catch(e) { el.innerHTML=`<div style="color:#f87171;font-size:12px">Error: ${e.message}</div>`; }
}

async function deleteFromLibrary(caseFile, btn) {
  if (!confirm(`Remove "${caseFile}" from your library?`)) return;
  btn.disabled=true; btn.textContent='...';
  try {
    const data = await (await fetch(`${API}/api/corpus/delete/${encodeURIComponent(caseFile)}`,{method:'DELETE'})).json();
    btn.closest('.corpus-item').remove();
    const sc = document.getElementById('stat-corpus');
    if (sc) sc.textContent = data.corpus_size;
  } catch(e) { btn.disabled=false; btn.textContent='✕'; alert('Remove failed: '+e.message); }
}

// ── EXPORT ────────────────────────────────────────────────────────────────────
/** Matches backend PDF headings/footer (`document_locale`): `hi` vs `en`; `auto` → `en`. */
function exportDocumentLocale(reportType, sessionInput) {
  let v = '';
  if (sessionInput?.response_language) v = sessionInput.response_language;
  else {
    const id =
      reportType === 'research' ? 'research-lang' :
      reportType === 'argument' ? 'arg-lang' :
      reportType === 'opposition' ? 'opp-lang' :
      reportType === 'debate' ? 'debate-lang' : null;
    v = id ? (document.getElementById(id)?.value || '') : '';
  }
  if ((v || '').toLowerCase() === 'hi') return 'hi';
  return 'en';
}

function buildExportList() {
  const list = document.getElementById('export-options-list');
  const opts = [];
  if (window._lastResearch)   opts.push({type:'research',   icon:'🔍', label:'Precedent Research',  desc:`Query: ${(window._lastResearch.query||'').slice(0,50)}...`});
  if (window._lastIrac)       opts.push({type:'argument',   icon:'📋', label:'Written Submissions', desc:`${window._lastIrac.total_issues||0} issues`});
  if (window._lastOpposition) opts.push({type:'opposition', icon:'⇌',  label:'Opposition Analysis', desc:`Risk: ${window._lastOpposition.analysis?.risk_level||'Unknown'}`});
  if (window._lastDebate)     opts.push({type:'debate',     icon:'⚖',  label:'Adversarial Hearing', desc:`${window._lastDebate.jurisdiction||''}`});
  if (!opts.length) {
    list.innerHTML='<div style="text-align:center;padding:24px;color:var(--w40)"><div style="font-size:28px;opacity:0.4;margin-bottom:8px">📄</div><div>Run any feature first, then export.</div></div>';
    return;
  }
  list.innerHTML = opts.map(o => `
    <div class="export-option" onclick="downloadReport('${o.type}')">
      <div class="export-option-icon">${o.icon}</div>
      <div><div class="export-option-title">${o.label}</div><div class="export-option-desc">${o.desc}</div></div>
      <div class="export-option-arrow">↓ PDF</div>
    </div>`).join('');
}

async function downloadSession(sessionId) {
  try {
    const data    = await (await fetch(`${API}/api/sessions/${sessionId}`)).json();
    const session = data.session;
    const output  = session.output_data || {};
    let p = {report_type:session.session_type, title:session.title, jurisdiction:'Indian Courts'};
    if (session.session_type==='research')    { p.query=session.input_data?.query; p.answer=output.answer; p.precedents=output.precedents||[]; }
    else if (session.session_type==='argument'){ p.facts=session.input_data?.facts; p.jurisdiction=session.input_data?.jurisdiction||''; p.arguments=(output.arguments||[]).map(a=>({issue_title:a.issue_title,area_of_law:a.area_of_law,priority:a.priority,irac:a.irac,precedents:a.precedents})); }
    else if (session.session_type==='opposition'){ p.argument=session.input_data?.argument; p.risk_level=output.analysis?.risk_level; p.weaknesses=output.analysis?.weaknesses||[]; p.counter_args=output.analysis?.counter_arguments||[]; p.strategy=output.analysis?.strategy_recommendations||[]; }
    else if (session.session_type==='debate')  { p.jurisdiction=session.input_data?.jurisdiction||''; p.round1=output.round1; p.round2=output.round2; p.summary=output.summary; }
    p.document_locale = exportDocumentLocale(session.session_type, session.input_data);
    await triggerDownload(p);
  } catch(e) { alert('Export failed: '+e.message); }
}

async function downloadReport(type) {
  const listEl   = document.getElementById('export-options-list');
  const loadEl   = document.getElementById('export-loading');
  listEl.style.display='none'; loadEl.style.display='block';
  loadEl.innerHTML=`<div style="text-align:center;padding:24px"><div style="font-size:24px;animation:breathe 2s ease-in-out infinite">⚙</div><div style="font-family:var(--serif);font-size:20px;font-style:italic;color:var(--cream);margin-top:12px">Generating PDF...</div></div>`;
  let p = {report_type:type, title:'', jurisdiction:'Indian Courts'};
  if (type==='research' && window._lastResearch)   { const d=window._lastResearch; p.title=`Legal Research: ${(d.query||'').slice(0,60)}`; p.query=d.query; p.answer=d.answer; p.precedents=d.precedents||[]; }
  if (type==='argument' && window._lastIrac)       { const d=window._lastIrac; p.title=`Argument`; p.facts=d.facts; p.jurisdiction=d.jurisdiction||''; p.arguments=(d.arguments||[]).map(a=>({issue_title:a.issue_title,area_of_law:a.area_of_law,priority:a.priority,irac:a.irac,precedents:a.precedents})); }
  if (type==='opposition' && window._lastOpposition){ const d=window._lastOpposition; const an=d.analysis||{}; p.title='Case Test Analysis'; p.argument=d.argument; p.risk_level=an.risk_level; p.weaknesses=an.weaknesses||[]; p.counter_args=an.counter_arguments||[]; p.strategy=an.strategy_recommendations||[]; }
  if (type==='debate' && window._lastDebate)       { const d=window._lastDebate; p.title=`Court Simulation`; p.jurisdiction=d.jurisdiction; p.round1=d.round1; p.round2=d.round2; p.summary=d.summary; }
  p.document_locale = exportDocumentLocale(type, null);
  await triggerDownload(p);
  closeModal('modal-export');
  listEl.style.display='block'; loadEl.style.display='none';
}

async function triggerDownload(payload) {
  try {
    const res = await fetch(`${API}/api/export/report`,{method:'POST',headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},body:JSON.stringify(payload)});
    if (!res.ok) throw new Error((await res.json()).detail || 'Export failed');
    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `LexForge_${payload.report_type}_${new Date().toISOString().slice(0,10)}.pdf`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);
  } catch(e) { alert('Download failed: '+e.message); }
}

// ── RESEARCH ──────────────────────────────────────────────────────────────────
async function doResearch() {
  const query  = document.getElementById('research-query').value.trim();
  const useNet = document.getElementById('use-internet').checked;
  const caseId = document.getElementById('research-case-select')?.value || null;
  const respLang = document.getElementById('research-lang')?.value || 'auto';
  const btn    = document.querySelector('#screen-research .btn-gold');
  const area   = document.getElementById('research-results');
  if (!query) { alert('Please enter a legal question.'); return; }

  setLoading(btn, 'Searching...');
  showJobStartedNotif('Research started');
  showProcessingBanner(area, 'Searching your library and Indian Kanoon...');

  try {
    const res = await fetch(`${API}/api/research`, {
      method:'POST', headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},
      body: JSON.stringify({query, top_k:4, case_id:caseId?parseInt(caseId):null, use_internet:useNet, response_language:respLang})
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    window._lastResearch = data;

    const live = data.live_results || [];
    area.innerHTML = `
      <div class="result-actions" id="research-actions">
        <button class="btn btn-ghost voice-listen-btn" type="button"
                data-target="research-answer-body" data-role="researcher"
                data-label-idle="Read aloud" data-label-playing="Stop reading">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5L6 9H3v6h3l5 4V5z"/><path d="M15.5 8.5a5 5 0 010 7M19 5a9 9 0 010 14"/></svg>
          <span class="voice-btn-label">Read aloud</span>
        </button>
        <button class="btn btn-ghost" type="button" onclick="openExportModal()">
          <span style="font-family:var(--mono);font-weight:600">↓ PDF</span>
        </button>
        <button class="btn btn-ghost" type="button" onclick="navigator.clipboard.writeText(document.getElementById('research-answer-body')?.innerText||'').then(()=>{this.textContent='Copied!';setTimeout(()=>this.textContent='Copy',1500)})">Copy</button>
      </div>
      <div class="grid-2" style="gap:20px;align-items:start">
        <div class="card animate-in">
          <div class="eyebrow" style="margin-bottom:4px">Research Memorandum</div>
          <div class="card-title">Authorities &amp; Analysis</div>
          <div class="card-sub" style="font-size:13px">
            <span style="color:#7eaaee;font-weight:600">${(data.precedents||[]).length} from your record</span>
            ${live.length ? `&nbsp;·&nbsp;<span style="color:#6abf6a;font-weight:600">${live.length} live from Indian Kanoon</span>` : ''}
          </div>
          <div class="answer-body" id="research-answer-body" style="margin-top:16px">
            ${marked.parse(
              (data.answer||'—')
                .replace(/\[SOURCE (\d+)\]/g, '<span class="cite">SOURCE $1</span>')
                .replace(/\[LIVE SOURCE (\d+)\]/g, '<span class="cite-live">LIVE $1</span>')
            )}
          </div>
          <div class="divider"></div>
          <div style="font-family:var(--mono);font-size:10px;color:#6abf6a">✓ Filed to Instruction Log</div>
        </div>
        <div>
          <div class="card animate-in" style="margin-bottom:16px">
            <div class="eyebrow" style="margin-bottom:4px">Your Record</div>
            <div class="card-title">Library Authorities</div>
            <div class="scroll-area">${renderPrecedentsHtml(data.precedents||[])}</div>
          </div>
          ${live.length ? `<div class="card animate-in">
            <div class="eyebrow" style="margin-bottom:4px">Indian Kanoon Live</div>
            <div class="card-title">Recent Judgments <span class="live-badge">Updated</span></div>
            <div class="card-sub">Retrieved live from indiankanoon.org</div>
            <div class="scroll-area">${live.map(r=>`
              <div class="prec-item">
                <div class="prec-body">
                  <div class="prec-name">${r.title}<span class="live-badge">Live</span></div>
                  <div class="prec-meta">${r.court} · ${r.year}</div>
                  <div class="prec-snip">${r.snippet||''}</div>
                  <a href="${r.url}" target="_blank" style="font-size:10px;color:var(--gold);margin-top:4px;display:inline-block">View on Indian Kanoon →</a>
                </div>
              </div>`).join('')}</div>
          </div>` : ''}
        </div>
      </div>
      ${data.weak_result ? `<div class="upload-prompt"><div class="upload-prompt-title">⚠ Thin Library Coverage</div><div class="upload-prompt-desc">Your library has limited material on this point. Upload relevant judgments to strengthen the research base.</div><button class="btn btn-ghost" style="font-size:12px" onclick="openModal('modal-upload')">Add More Judgments</button></div>` : ''}`;

    if (data.disclaimer) {
      const d = document.createElement('div');
      d.style.cssText='background:rgba(193,128,33,.06);border:1px solid rgba(193,128,33,.2);border-radius:2px;padding:13px 16px;margin-top:16px;font-family:var(--mono);font-size:11px;color:var(--gold-d);letter-spacing:.04em';
      d.innerHTML = '<strong style="color:var(--gold)">Verification required:</strong> ' + data.disclaimer;
      area.appendChild(d);
    }
    showNotif('Research Memorandum Ready', `${data.total_sources} source${data.total_sources!==1?'s':''} retrieved.`);
    await loadSessions();

  } catch(err) {
    area.innerHTML = `<div class="card" style="border-color:rgba(192,57,43,0.3)"><div class="card-title" style="color:#f87171">Error</div><div class="card-sub">${err.message}</div></div>`;
  } finally { clearLoading(btn); }
}

// ── ARGUMENT BUILDER ──────────────────────────────────────────────────────────
function _argSetPipe(step, status, text) {
  const dot = document.getElementById(`pp-${step}`);
  const st  = document.getElementById(`ps-${step}`);
  if (dot) { dot.className=`pip-dot ${status}`; if(status==='done') dot.textContent='✓'; }
  if (st)  st.textContent = text;
}
function _argRenderIracBlocks(irac) {
  const labels  = {issue:'Point in Issue',rule:'Applicable Law &amp; Precedents',application:'Application to Facts',conclusion:'Conclusion &amp; Relief'};
  const letters = {issue:'I',rule:'R',application:'A',conclusion:'C'};
  return ['issue','rule','application','conclusion'].map(k =>
    `<div class="irac-block"><div class="irac-hdr"><div class="irac-letter">${letters[k]}</div><div><div class="irac-kind">${k.toUpperCase()}</div><div class="irac-title">${labels[k]}</div></div></div><div class="irac-body">${(irac[k]||'—').replace(/\[SOURCE (\d+)\]/g,'<span class="cite">SOURCE $1</span>')}</div></div>`
  ).join('');
}
function _argIssuePlaceholder(idx, totalHint, issueTitle, area, priority) {
  const rc = priority==='high'?'risk-HIGH':priority==='medium'?'risk-MODERATE':'risk-LOW';
  const label = totalHint ? `Issue ${idx+1} of ${totalHint}` : `Issue ${idx+1}`;
  return `
    <div class="arg-issue-card animate-in" id="arg-issue-${idx}" data-state="pending">
      <div class="arg-issue-hdr">
        <div>
          <div class="arg-issue-eyebrow">${label}${area ? ` · ${escapeHtml(area)}` : ''}</div>
          <div class="arg-issue-title">${escapeHtml(issueTitle || 'Identifying issue...')}</div>
        </div>
        <span class="risk-badge ${rc}">${(priority||'medium').toUpperCase()} PRIORITY</span>
      </div>
      <div class="arg-issue-stage">
        <span class="arg-stage-dot"></span>
        <span class="arg-stage-text">Queued</span>
      </div>
      <div class="arg-issue-body" style="display:none"></div>
    </div>`;
}
function _argSetIssueStage(idx, label, state) {
  const card = document.getElementById(`arg-issue-${idx}`);
  if (!card) return;
  card.dataset.state = state || 'active';
  const t = card.querySelector('.arg-stage-text');
  if (t) t.textContent = label;
}
function _argFillIssueBody(idx, arg) {
  const card = document.getElementById(`arg-issue-${idx}`);
  if (!card) return;
  card.dataset.state = 'done';
  const stage = card.querySelector('.arg-issue-stage');
  if (stage) stage.remove();
  const body = card.querySelector('.arg-issue-body');
  if (!body) return;
  const cases = (arg.precedents||[]).map(p => `<span class="citation-tag">${escapeHtml(p.case_name||'Case')}</span>`).join(' ');
  body.style.display = 'block';
  body.innerHTML = `
    ${_argRenderIracBlocks(arg.irac || {})}
    ${cases ? `<div class="arg-issue-cases">Cases referenced: ${cases}</div>` : ''}`;
  // Update title (in case the saved title is more precise than the placeholder)
  const titleEl = card.querySelector('.arg-issue-title');
  if (titleEl && arg.issue_title) titleEl.textContent = arg.issue_title;
}

async function doArgument() {
  const facts  = document.getElementById('arg-facts').value.trim();
  const juris  = document.getElementById('arg-jurisdiction').value;
  const area   = document.getElementById('arg-area').value;
  const pos    = document.getElementById('arg-position').value;
  const caseId = document.getElementById('arg-case-select').value;
  const extra  = document.getElementById('arg-extra')?.value?.trim() || '';
  const respLang = document.getElementById('arg-lang')?.value || 'auto';
  const btn    = document.querySelector('#arg-input .btn-gold');

  if (!facts) { alert('Please describe the facts of your case.'); return; }

  switchArgTab(1);

  const out = document.getElementById('arg-output');
  showJobStartedNotif('Submissions drafting started');

  // Reset the pipeline strip
  ['issues','search','irac','done'].forEach((s,i) => _argSetPipe(s,'pending', i===0?'Pending':'Pending'));
  _argSetPipe('issues','active','Identifying issues...');

  // Initial scaffold so the user sees structure right away
  out.innerHTML = `
    <div class="arg-stream-shell">
      <div class="arg-actions" id="arg-actions" style="display:none"></div>
      <div class="arg-stream-header">
        <div class="arg-stream-eyebrow" id="arg-stream-eyebrow">Drafting · ${escapeHtml(juris)}</div>
        <div class="arg-stream-title">Written Submissions</div>
        <div class="arg-stream-status" id="arg-stream-status">
          <span class="arg-stage-dot active"></span>
          <span>Reading instructions and identifying legal issues...</span>
        </div>
      </div>
      <div id="arg-issues-grid"></div>
      <div id="arg-stream-disclaimer"></div>
    </div>`;

  setLoading(btn, 'Building...');

  const setStreamStatus = (msg, state) => {
    const el = document.getElementById('arg-stream-status');
    if (!el) return;
    el.innerHTML = `<span class="arg-stage-dot ${state||'active'}"></span><span>${escapeHtml(msg)}</span>`;
  };

  // Per-issue argument list as it is being assembled (for the final state).
  const args = [];
  let totalIssues = 0;
  let aggregated  = null;
  let weakResult  = false;

  try {
    const res = await fetch(`${API}/api/argument/stream`, {
      method:'POST',
      headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},
      body: JSON.stringify({
        facts, jurisdiction:juris, area_of_law:area, client_position:pos,
        extra_context: extra, response_language: respLang,
        case_id: caseId ? parseInt(caseId) : null,
      })
    });
    if (!res.ok || !res.body) throw new Error(`Server error ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const t = line.trim();
        if (!t) continue;
        let evt;
        try { evt = JSON.parse(t); } catch(_) { continue; }

        if (evt.kind === 'stage') {
          _argSetPipe(evt.stage, evt.status, evt.message || (evt.status==='done'?'Done':'Working...'));
          if (evt.message) setStreamStatus(evt.message, evt.status);
        }
        else if (evt.kind === 'issues') {
          totalIssues = (evt.issues || []).length;
          const grid = document.getElementById('arg-issues-grid');
          grid.innerHTML = (evt.issues || []).map((iss, idx) =>
            _argIssuePlaceholder(idx, totalIssues, iss.issue, iss.area_of_law, iss.priority)
          ).join('');
          _argSetIssueStage(0, 'Searching authorities...', 'active');
          setStreamStatus(`Drafting submissions for ${totalIssues} issue${totalIssues!==1?'s':''}...`, 'active');
        }
        else if (evt.kind === 'issue_progress') {
          const phase = evt.phase === 'searching' ? 'Searching authorities...' : 'Drafting IRAC submissions...';
          _argSetIssueStage(evt.index, phase, 'active');
        }
        else if (evt.kind === 'issue_done') {
          args[evt.index] = evt.argument;
          _argFillIssueBody(evt.index, evt.argument);
          // Activate next issue
          if (evt.index + 1 < totalIssues) {
            _argSetIssueStage(evt.index + 1, 'Searching authorities...', 'active');
          }
        }
        else if (evt.kind === 'complete') {
          aggregated = {
            facts, jurisdiction:juris, client_position:pos,
            total_issues:    args.length,
            arguments:       args,
            all_precedents:  evt.all_precedents || [],
            disclaimer:      evt.disclaimer || null,
            session_id:      evt.session_id,
          };
          weakResult = (aggregated.all_precedents.length === 0);
        }
        else if (evt.kind === 'error') {
          throw new Error(evt.message || 'Error during drafting');
        }
      }
    }

    if (!aggregated) throw new Error('Stream ended without a complete event');
    window._lastIrac       = aggregated;
    window._lastPrecedents = aggregated.all_precedents || [];

    // Final action toolbar at the top of the output (Read aloud, PDF, etc.)
    const actions = document.getElementById('arg-actions');
    if (actions) {
      actions.style.display = 'flex';
      actions.innerHTML = `
        <button class="btn btn-ghost voice-listen-btn" type="button"
                data-target="arg-issues-grid" data-role="petitioner"
                data-label-idle="Read aloud" data-label-playing="Stop reading">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5L6 9H3v6h3l5 4V5z"/><path d="M15.5 8.5a5 5 0 010 7M19 5a9 9 0 010 14"/></svg>
          <span class="voice-btn-label">Read aloud</span>
        </button>
        <button class="btn btn-ghost" type="button" onclick="openExportModal()">
          <span style="font-family:var(--mono);font-weight:600">↓ PDF</span>
        </button>
        <button class="btn btn-ghost" type="button" onclick="switchArgTab(2);renderPrecedentsTab()">All Authorities</button>
        <button class="btn btn-ghost" type="button" onclick="document.getElementById('opp-argument').value=window._lastIrac?.arguments?.map(a=>a.irac?.rule||'').join(' ')||'';navigate('opposition')">Devil's Advocate</button>`;
    }

    // Replace status with final summary
    const status = document.getElementById('arg-stream-status');
    if (status) status.innerHTML =
      `<span class="arg-stage-dot done"></span><span>${aggregated.total_issues} legal issue${aggregated.total_issues!==1?'s':''} addressed · filed to Instruction Log</span>`;

    if (aggregated.disclaimer) {
      const dWrap = document.getElementById('arg-stream-disclaimer');
      if (dWrap) dWrap.innerHTML = `
        <div class="arg-disclaimer">
          <strong>Verification required:</strong> ${escapeHtml(aggregated.disclaimer)}
        </div>`;
    }
    if (weakResult) {
      const dWrap = document.getElementById('arg-stream-disclaimer');
      if (dWrap) dWrap.insertAdjacentHTML('beforeend',
        `<div class="upload-prompt"><div class="upload-prompt-title">⚠ Few Precedents Found</div><div class="upload-prompt-desc">Your argument would be stronger with more relevant cases. Upload related judgments for more specific citations.</div><button class="btn btn-ghost" style="font-size:12px" onclick="openModal('modal-upload')">Add More Judgments</button></div>`);
    }

    showNotif('Submissions Drafted', `${aggregated.total_issues} legal issue${aggregated.total_issues!==1?'s':''} addressed.`);
    await loadSessions();

  } catch(err) {
    out.innerHTML = `<div class="card" style="border-color:rgba(192,57,43,0.3)"><div class="card-title" style="color:#f87171">Error</div><div class="card-sub">${escapeHtml(err.message || 'Unknown error')}</div></div>`;
  } finally { clearLoading(btn); }
}

function renderPrecedentsTab() {
  const el   = document.getElementById('arg-precedents');
  const prec = window._lastPrecedents || [];
  if (!prec.length) {
    el.innerHTML = '<div class="empty-state"><div class="empty-icon">📚</div><div class="empty-title">No cases loaded yet</div><div class="empty-desc">Build your argument first, then view cases here.</div></div>';
    return;
  }
  el.innerHTML = `
    <div class="card animate-in">
      <div class="card-title">All Precedents Cited</div>
      <div class="card-sub">${prec.length} cases across ${window._lastIrac?.total_issues||'all'} legal issues</div>
      ${prec.map((p,i) => {
        const court = (p.court&&p.court!=='Unknown')?p.court:'Indian Court';
        const year  = (p.year &&p.year !=='Unknown')?p.year :'';
        return `<div class="prec-item"><div class="prec-rank">0${i+1}</div><div class="prec-body"><div class="prec-name">${p.case_name}<span class="binding-tag ${p.binding==='Binding'?'binding':'persuasive'}">${p.binding||'Persuasive'}</span></div><div class="prec-meta">${[court,year,`Score: ${p.score}`].filter(Boolean).join(' · ')}</div><div class="prec-snip">${p.snippet||''}</div></div></div>`;
      }).join('')}
    </div>`;
}

// ── DEVIL'S ADVOCATE ──────────────────────────────────────────────────────────
async function doOpposition() {
  const argument     = document.getElementById('opp-argument').value.trim();
  const caseId       = document.getElementById('opp-case-select').value;
  const judgePersona = document.getElementById('opp-persona')?.value || 'strict_proceduralist';
  const respLang = document.getElementById('opp-lang')?.value || 'auto';
  const btn          = document.querySelector('#screen-opposition .btn-gold');
  const area         = document.getElementById('opp-results');
  if (!argument) { alert('Please enter your argument to test.'); return; }

  setLoading(btn, 'Testing...');
  showJobStartedNotif("Devil's Advocate running");
  showProcessingBanner(area, 'Identifying weaknesses in your argument...');

  try {
    const res = await fetch(`${API}/api/opposition`, {
      method:'POST', headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},
      body: JSON.stringify({argument, judge_persona:judgePersona, case_id:caseId?parseInt(caseId):null, response_language:respLang})
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    window._lastOpposition = data;
    const an = data.analysis || {};
    const rc = {HIGH:'risk-HIGH',MODERATE:'risk-MODERATE',LOW:'risk-LOW'}[an.risk_level] || 'risk-MODERATE';

    area.innerHTML = `
      <div class="result-actions">
        <button class="btn btn-ghost voice-listen-btn" type="button"
                data-target="opp-output-body" data-role="opposition"
                data-label-idle="Read aloud" data-label-playing="Stop reading">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5L6 9H3v6h3l5 4V5z"/><path d="M15.5 8.5a5 5 0 010 7M19 5a9 9 0 010 14"/></svg>
          <span class="voice-btn-label">Read aloud</span>
        </button>
        <button class="btn btn-ghost" type="button" onclick="openExportModal()">
          <span style="font-family:var(--mono);font-weight:600">↓ PDF</span>
        </button>
      </div>
      <div id="opp-output-body">
      <div class="card mb-24">
        <div class="flex items-center justify-between mb-16">
          <div><div class="card-title">Exposure Assessment</div><div class="card-sub">Stress-test: opposing counsel's attack lines &amp; questions the Bench may put</div></div>
          <span class="risk-badge ${rc}">${an.risk_level||'MODERATE'} RISK</span>
        </div>
        <div class="stat-row" style="margin-bottom:0">
          <div class="stat-chip"><div class="stat-value" style="font-size:20px;color:#fbbf24">${(an.weaknesses||[]).length}</div><div class="stat-label">Weaknesses Found</div></div>
          <div class="stat-chip"><div class="stat-value" style="font-size:20px;color:#f87171">${(an.weaknesses||[]).filter(w=>w.severity==='HIGH').length}</div><div class="stat-label">Critical Risks</div></div>
        </div>
      </div>
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Vulnerabilities</div>
          <div class="card-sub">Points opposing counsel will attack</div>
          ${(an.weaknesses||[]).map(w=>`<div class="weakness" style="${w.severity!=='HIGH'?'border-color:rgba(217,119,6,0.18);background:rgba(217,119,6,0.06)':''}"><div class="weakness-id" style="color:${w.severity==='HIGH'?'#f87171':'#fbbf24'}">${w.id}</div><div><div style="font-size:12px;font-family:var(--mono);margin-bottom:4px;color:${w.severity==='HIGH'?'#f87171':'#fbbf24'}">${w.severity} RISK</div><div class="weakness-text">${w.description}</div></div></div>`).join('')||'<div style="color:var(--w40);padding:8px 0;font-size:13px">No significant weaknesses found.</div>'}
        </div>
        <div class="card">
          <div class="card-title">Anticipated Rejoinder</div>
          <div class="card-sub">Counter-submissions opposing counsel will advance</div>
          <div class="debate-body">${(an.counter_arguments||[]).map(c=>`<div class="debate-point">${c.point}${(c.source||c.authority)?`<span class="citation-tag">${c.source||c.authority}</span>`:''}</div>`).join('')}</div>
          ${data.contrary_precedents?.length?`<div class="divider"></div><div style="font-family:var(--mono);font-size:10px;color:var(--gold);margin-bottom:10px;text-transform:uppercase;letter-spacing:0.1em">Cases That May Be Used Against You</div>${data.contrary_precedents.map(p=>`<div class="prec-item"><div class="prec-rank">→</div><div class="prec-body"><div class="prec-name">${p.case_name}</div><div class="prec-meta">${[p.court,p.year].filter(x=>x&&x!=='Unknown').join(' · ')}</div><div class="prec-snip">${p.snippet}</div></div></div>`).join('')}`:''}
        </div>
      </div>
      ${(an.bench_questions||[]).length ? `
        <div class="card mt-24" style="border-color:rgba(200,165,90,0.2)">
          <div style="font-family:var(--mono);font-size:10px;color:var(--gold);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:4px">Bench Anticipation · ${data.judge_persona?.name||'Simulated Bench'}</div>
          <div class="card-title">Questions the Bench Will Put to You</div>
          <div style="margin-top:14px;display:flex;flex-direction:column;gap:10px">
            ${(an.bench_questions||[]).map((q,qi)=>`
              <div style="background:var(--void);border:1px solid rgba(239,68,68,0.18);border-radius:9px;padding:14px 16px">
                <div style="display:flex;gap:10px;align-items:flex-start">
                  <div style="font-family:var(--mono);font-size:10px;color:var(--gold);flex-shrink:0;margin-top:3px">Q${qi+1}</div>
                  <div>
                    <div style="font-family:var(--legal);font-size:16px;font-style:italic;color:var(--cream);line-height:1.55;margin-bottom:6px">"${q.question}"</div>
                    <div style="font-size:12.5px;color:var(--w40)">${q.implication}</div>
                  </div>
                </div>
              </div>`).join('')}
          </div>
        </div>` : ''}
      <div class="card mt-24">
        <div class="card-label" style="margin-bottom:5px">Before the Next Date</div>
        <div class="card-title">Priority Actions</div>
        <ol style="margin-top:14px;padding-left:22px;display:flex;flex-direction:column;gap:12px">
          ${(an.priority_actions||[]).length
            ? (an.priority_actions||[]).map(a=>`<li style="font-family:var(--legal);font-size:12pt;line-height:1.8;color:var(--w60)">${a}</li>`).join('')
            : (an.strategy_recommendations||[]).map(s=>`<li style="font-family:var(--legal);font-size:12pt;line-height:1.8;color:var(--w60)"><strong style="font-style:normal;color:${s.type==='DO'?'#6abf6a':'#e08070'}">${s.type==='DO'?'Do:':'Avoid:'}</strong> ${s.advice}</li>`).join('')
          }
        </ol>
      </div>
      </div>`;

    showNotif("Devil's Advocate Complete", `${(an.weaknesses||[]).length} vulnerabilities · ${(an.bench_questions||[]).length} bench questions.`, '⇌');
    await loadSessions();

  } catch(err) {
    area.innerHTML = `<div class="card" style="border-color:rgba(192,57,43,0.3)"><div class="card-title" style="color:#f87171">Error</div><div class="card-sub">${err.message}</div></div>`;
  } finally { clearLoading(btn); }
}

// ── DEBATE SIMULATION ─────────────────────────────────────────────────────────
async function doDebate() {
  const summary      = document.getElementById('debate-summary').value.trim();
  const plaintiff    = document.getElementById('debate-plaintiff').value.trim();
  const defense      = document.getElementById('debate-defense').value.trim();
  const juris        = document.getElementById('debate-jurisdiction').value;
  const caseId       = document.getElementById('debate-case-select').value;
  const judgePersona = document.getElementById('debate-persona')?.value || 'strict_proceduralist';
  const respLang = document.getElementById('debate-lang')?.value || 'auto';
  const btn          = document.querySelector('#screen-debate .btn-gold');
  const area         = document.getElementById('debate-results');
  if (!summary) { alert('Please describe the case dispute.'); return; }

  setLoading(btn, 'Commencing hearing...');
  showJobStartedNotif('Court simulation started');
  showProcessingBanner(area, 'Commencing adversarial hearing...');

  try {
    const res = await fetch(`${API}/api/debate`, {
      method:'POST', headers:{'Content-Type':'application/json','ngrok-skip-browser-warning':'true'},
      body: JSON.stringify({
        case_summary:summary, jurisdiction:juris, plaintiff_position:plaintiff, defense_position:defense,
        judge_persona:judgePersona, case_id:caseId?parseInt(caseId):null, response_language:respLang,
      })
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    const data = await res.json();
    window._lastDebate = data;

    const sm = data.summary || {};
    const rc = {HIGH:'risk-HIGH',MODERATE:'risk-MODERATE',LOW:'risk-LOW'}[sm.risk_level] || 'risk-MODERATE';
    const renderPts = pts => (pts||[]).map(p=>`<div class="debate-point">${p.point||p}${p.citation?`<span class="citation-tag">${p.citation}</span>`:''}</div>`).join('');

    area.innerHTML = `
      <div class="result-actions">
        <button class="btn btn-ghost voice-debate-btn" type="button"
                data-target="debate-output-body"
                data-label-idle="Play hearing" data-label-playing="Stop hearing">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="6,4 20,12 6,20" fill="currentColor"/></svg>
          <span class="voice-btn-label">Play hearing</span>
        </button>
        <button class="btn btn-ghost voice-listen-btn" type="button"
                data-target="debate-output-body" data-role="judge"
                data-label-idle="Read summary" data-label-playing="Stop reading">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5L6 9H3v6h3l5 4V5z"/><path d="M15.5 8.5a5 5 0 010 7M19 5a9 9 0 010 14"/></svg>
          <span class="voice-btn-label">Read summary</span>
        </button>
        <button class="btn btn-ghost" type="button" onclick="openExportModal()">
          <span style="font-family:var(--mono);font-weight:600">↓ PDF</span>
        </button>
      </div>
      <div id="debate-output-body">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
        <div style="font-family:var(--mono);font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:var(--w40)">Round 1</div>
        <div style="flex:1;height:1px;background:rgba(193,128,33,.1)"></div>
        <div style="font-family:var(--serif);font-size:14px;color:var(--gold);font-style:italic">Opening Submissions</div>
      </div>
      <div class="debate-grid mb-24">
        <div class="debate-side debate-p"><div class="debate-hdr">PETITIONER</div><div class="debate-body">${renderPts(data.round1?.plaintiff)}</div></div>
        <div class="debate-vs">vs</div>
        <div class="debate-side debate-d"><div class="debate-hdr">RESPONDENT</div><div class="debate-body">${renderPts(data.round1?.defense)}</div></div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
        <div style="font-family:var(--mono);font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:var(--w40)">Round 2</div>
        <div style="flex:1;height:1px;background:rgba(193,128,33,.1)"></div>
        <div style="font-family:var(--serif);font-size:14px;color:var(--gold);font-style:italic">Rebuttal round · Sur-rebuttal</div>
      </div>
      <div class="debate-grid mb-24">
        <div class="debate-side debate-p"><div class="debate-hdr">PETITIONER — REBUTTAL</div><div class="debate-body">${renderPts(data.round2?.plaintiff)}</div></div>
        <div class="debate-vs">vs</div>
        <div class="debate-side debate-d"><div class="debate-hdr">RESPONDENT — SUR-REBUTTAL</div><div class="debate-body">${renderPts(data.round2?.defense)}</div></div>
      </div>
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;margin-top:8px">
        <div style="font-family:var(--mono);font-size:10px;letter-spacing:0.15em;text-transform:uppercase;color:var(--w40)">Round 3</div>
        <div style="flex:1;height:1px;background:rgba(193,128,33,.1)"></div>
        <div style="font-family:var(--serif);font-size:14px;color:var(--gold);font-style:italic">Judicial Observations</div>
      </div>
      <div class="card" style="border-color:rgba(193,128,33,0.3);background:rgba(193,128,33,0.04)">
        <div class="flex items-center gap-8 mb-16">
          <div><div class="card-title text-gold">Judicial Observations</div><div class="card-sub">Bench: ${escapeHtml(data.judge_persona?.name || '')} · ${escapeHtml(juris)} · Assessment, likely outcome &amp; strategy</div></div>
          <span class="risk-badge ${rc}" style="margin-left:auto">Overall: ${sm.risk_level||'MODERATE'} Risk</span>
        </div>
        ${sm.judicial_observation ? `<div style="background:var(--void);border-left:3px solid var(--gold);padding:13px 16px;border-radius:4px;margin-bottom:18px"><div style="font-family:var(--mono);font-size:9.5px;color:var(--gold);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:7px">From the Bench</div><div style="font-family:var(--legal);font-size:17px;font-style:italic;color:var(--cream);line-height:1.6;text-align:justify">"${escapeHtml(sm.judicial_observation)}"</div></div>` : ''}
        <div style="font-family:var(--legal);font-size:12pt;line-height:2;color:var(--w60);text-align:justify">
          <strong style="color:var(--cream)">Assessment:</strong> ${sm.overall_assessment||'—'}<br/><br/>
          <strong style="color:var(--cream)">Likely Outcome:</strong> ${sm.likely_outcome||'—'}<br/><br/>
          <strong style="color:var(--cream)">Strategic Advice:</strong> ${sm.strategic_recommendation||'—'}
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:18px">
          <button class="btn btn-ghost" onclick="navigate('opposition')">Devil's Advocate</button>
          <button class="btn btn-ghost" onclick="navigate('argument')">Revise Submissions</button>
        </div>
      </div>
      ${(data.precedents||[]).length?`<div class="card mt-24"><div class="card-title">Cases Referenced</div><div class="card-sub">Precedents used in this simulation</div><div class="scroll-area">${renderPrecedentsHtml(data.precedents)}</div></div>`:''}
      </div>`;

    showNotif('Moot Court Complete', 'Opening, rebuttal & sur-rebuttal complete — judicial observations filed.');
    await loadSessions();

  } catch(err) {
    area.innerHTML = `<div class="card" style="border-color:rgba(192,57,43,0.3)"><div class="card-title" style="color:#f87171">Error</div><div class="card-sub">${err.message}</div></div>`;
  } finally { clearLoading(btn); }
}

// ── CHAMBERS INIT ─────────────────────────────────────────────────────────────
function renderChambersDate() {
  const now  = new Date();
  const hr   = now.getHours();
  const greet= hr < 12 ? 'Good morning' : hr < 17 ? 'Good afternoon' : 'Good evening';
  const el   = document.getElementById('dash-greeting');
  if (el) el.textContent = greet + ', Advocate.';
}

// ── RERUN SESSION ─────────────────────────────────────────────────────────────
function rerunSession(sessionId) {
  fetch(`${API}/api/sessions/${sessionId}`)
    .then(r => r.json())
    .then(data => {
      const s     = data.session;
      const input = s.input_data || {};
      if (s.session_type === 'research') {
        navigate('research');
        setTimeout(() => {
          document.getElementById('research-query').value = input.query || '';
          if (input.use_internet != null) document.getElementById('use-internet').checked = !!input.use_internet;
          const rl = document.getElementById('research-lang');
          if (rl && input.response_language) rl.value = input.response_language;
        }, 100);
      } else if (s.session_type === 'argument') {
        navigate('argument');
        setTimeout(() => {
          const f = document.getElementById('arg-facts'); f.value = input.facts || ''; autoExpand(f);
          if (input.jurisdiction) document.getElementById('arg-jurisdiction').value = input.jurisdiction;
          if (input.area_of_law) document.getElementById('arg-area').value = input.area_of_law;
          if (input.client_position) document.getElementById('arg-position').value = input.client_position;
          const ex = document.getElementById('arg-extra'); if (ex) { ex.value = input.extra_context || ''; autoExpand(ex); }
          const al = document.getElementById('arg-lang'); if (al && input.response_language) al.value = input.response_language;
        }, 100);
      } else if (s.session_type === 'opposition') {
        navigate('opposition');
        setTimeout(() => {
          document.getElementById('opp-argument').value = input.argument || '';
          if (input.judge_persona) document.getElementById('opp-persona').value = input.judge_persona;
          const ol = document.getElementById('opp-lang'); if (ol && input.response_language) ol.value = input.response_language;
        }, 100);
      } else if (s.session_type === 'debate') {
        navigate('debate');
        setTimeout(() => {
          const s2 = document.getElementById('debate-summary'); s2.value = input.case_summary || ''; autoExpand(s2);
          const p = document.getElementById('debate-plaintiff'); p.value = input.plaintiff_position || ''; autoExpand(p);
          const d = document.getElementById('debate-defense'); d.value = input.defense_position || ''; autoExpand(d);
          if (input.jurisdiction) document.getElementById('debate-jurisdiction').value = input.jurisdiction;
          if (input.judge_persona) document.getElementById('debate-persona').value = input.judge_persona;
          const dl = document.getElementById('debate-lang'); if (dl && input.response_language) dl.value = input.response_language;
        }, 100);
      }
    });
}

// ── DOM READY ─────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  // Theme init
  initTheme();

  // Theme swatch click handlers
  document.querySelectorAll('.theme-swatch').forEach(swatch => {
    swatch.addEventListener('click', () => {
      applyTheme(swatch.dataset.theme);
    });
  });

  // Enter key on research
  const rq = document.getElementById('research-query');
  if (rq) rq.addEventListener('keydown', e => { if (e.key === 'Enter') doResearch(); });

  if (Notification.permission === 'default') Notification.requestPermission();

  renderChambersDate();
  document.querySelectorAll('textarea').forEach(t => autoExpand(t));

  await loadApiStatus();
  await loadCases();
  await loadSessions();
});