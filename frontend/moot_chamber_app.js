/**
 * MootChamberApp — UI controller for the Moot Chamber.
 * Screens: setup → chamber → debrief.
 */

class MootChamberApp {
  constructor() {
    this.config = {
      caseName:         '',
      sideArguing:      'petitioner',
      caseStatement:    '',
      statutes:         '',
      courtLevel:       'high_court',
      judgePersonality: 'sinha',
      experienceLevel:  'junior',
      language:         'en-IN',
      matterId:         null,
      silentCitation:   true,
      weaknessAlerts:   true,
      showTranscript:   true,
    };

    this.client    = null;
    this.sessionId = null;

    this._timerInt      = null;
    this._timerSec      = 0;
    this._exchangeCount = 0;
    this._citationCount = 0;
    this._flagCount     = 0;
    this._matterSavedAtSetup = false;

    this.JUDGES = {
      verma:        { name: 'JUSTICE A.K. VERMA',      display: 'Justice Verma',        style: 'The Constitutional Philosopher' },
      mehta:        { name: 'JUSTICE S.K. MEHTA',      display: 'Justice Mehta',        style: 'The Technocrat' },
      krishnaswamy: { name: 'JUSTICE M. KRISHNASWAMY', display: 'Justice Krishnaswamy', style: 'The Activist' },
      sinha:        { name: 'JUSTICE R.P. SINHA',      display: 'Justice Sinha',        style: 'The Skeptic' },
      kaul:         { name: 'JUSTICE P. KAUL',         display: 'Justice Kaul',         style: 'The Pragmatist' },
    };
    this.COURTS = {
      district:   'IN THE COURT OF DISTRICT & SESSIONS JUDGE',
      high_court: 'IN THE HIGH COURT',
      supreme:    'IN THE SUPREME COURT OF INDIA',
    };
  }

  /* ── INIT ───────────────────────────────────────────────────── */

  init() {
    this._initSelectors();
    this._initVizBars();
    this._loadMeta();
    this._loadMatters();
    this._bindControls();
  }

  _initSelectors() {
    this._bindGroup('side-selector',  '.mc-radio',       'sideArguing');
    this._bindGroup('court-selector', '.mc-card-choice', 'courtLevel');
    this._bindGroup('judge-selector', '.mc-judge-card',  'judgePersonality');
    this._bindGroup('level-selector', '.mc-card-choice', 'experienceLevel');

    const on = (id, fn) => { const el = document.getElementById(id); if (el) el.addEventListener('change', fn); };
    on('opt-citation',   (e) => { this.config.silentCitation = e.target.checked; });
    on('opt-weakness',   (e) => { this.config.weaknessAlerts = e.target.checked; });
    on('opt-transcript', (e) => { this.config.showTranscript = e.target.checked; });
    on('language-select',(e) => { this.config.language = e.target.value; });
    on('matter-select',  (e) => { this.config.matterId = e.target.value ? parseInt(e.target.value, 10) : null; });
  }

  _bindGroup(containerId, selector, key) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.querySelectorAll(selector).forEach((el) => {
      el.addEventListener('click', () => {
        container.querySelectorAll(selector).forEach((x) => x.classList.remove('active'));
        el.classList.add('active');
        this.config[key] = el.dataset.value;
      });
    });
  }

  async _loadMeta() {
    try {
      const res = await fetch('/api/moot/meta');
      if (!res.ok) return;
      const meta = await res.json();
      const sel = document.getElementById('language-select');
      if (sel && Array.isArray(meta.languages)) {
        sel.innerHTML = '';
        meta.languages.forEach((l) => {
          const opt = document.createElement('option');
          opt.value = l.code;
          opt.textContent = l.label;
          sel.appendChild(opt);
        });
        sel.value = this.config.language;
      }
      const note = document.getElementById('setup-footnote');
      if (note && Array.isArray(meta.providers)) {
        const cloud = meta.providers.filter((p) => !p.startsWith('ollama'));
        note.textContent = cloud.length
          ? 'Bench reasoning: ' + cloud[0].split(':')[0] + ' · Voice: Sarvam'
          : 'No cloud key found in .env — the bench will use the local model (slower responses).';
      }
    } catch (_) { /* setup still works with defaults */ }
  }

  async _loadMatters() {
    try {
      const res = await fetch('/api/cases');
      if (!res.ok) return;
      const data = await res.json();
      const matters = Array.isArray(data) ? data : (data.cases || []);
      const sel = document.getElementById('matter-select');
      if (!sel) return;
      matters.forEach((m) => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name || ('Matter ' + m.id);
        sel.appendChild(opt);
      });
    } catch (_) { /* optional */ }
  }

  _bindControls() {
    const click = (id, fn) => { const el = document.getElementById(id); if (el) el.addEventListener('click', fn); };
    click('enter-chamber-btn', () => this._onEnterChamber());
    click('record-btn',        () => this._toggleRecording());
    click('end-session-btn',   () => this._endSession());
    click('download-btn',      () => this._downloadTranscript());
    click('new-session-btn',   () => { this._resetForNewSession(); this._transition('setup'); });
    click('save-matter-btn',   () => this._saveToMatter());
  }

  /* ── SETUP → CHAMBER ────────────────────────────────────────── */

  _onEnterChamber() {
    const get = (id) => (document.getElementById(id) ? document.getElementById(id).value.trim() : '');
    this.config.caseName      = get('case-name');
    this.config.caseStatement = get('case-statement');
    this.config.statutes      = get('statutes');

    if (!this.config.caseName) {
      const el = document.getElementById('case-name');
      el.classList.add('mc-invalid');
      el.focus();
      setTimeout(() => el.classList.remove('mc-invalid'), 2000);
      return;
    }

    this._matterSavedAtSetup = !!this.config.matterId;
    this._transition('chamber');
    this._startSession();
  }

  _startSession() {
    this._updateBenchUI();
    this._startTimer();

    this.client = new MootVoiceClient({
      onTranscript:    (text)         => this._onTranscript(text),
      onAgentResponse: (data)         => this._onAgentResponse(data),
      onAgentStatus:   (agent, st)    => this._onAgentStatus(agent, st),
      onStats:         (stats)        => this._onStats(stats),
      onStateChange:   (state)        => this._onVoiceState(state),
      onError:         (msg)          => this._onVoiceError(msg),
    });

    const sessionConfig = {
      case_name:                this.config.caseName,
      side_arguing:             this.config.sideArguing,
      case_statement:           this.config.caseStatement,
      relevant_statutes:        this.config.statutes.split(',').map((s) => s.trim()).filter(Boolean),
      court_level:              this.config.courtLevel,
      judge_personality:        this.config.judgePersonality,
      experience_level:         this.config.experienceLevel,
      language:                 this.config.language,
      matter_id:                this.config.matterId,
      silent_citation_checking: this.config.silentCitation,
      weakness_alerts:          this.config.weaknessAlerts,
      show_transcript:          this.config.showTranscript,
    };

    this.client.connect(sessionConfig)
      .then(() => {
        this.sessionId = this.client.sessionId;
        this._setLive(true);
      })
      .catch((err) => this._onVoiceError(err.message || 'Could not reach the chamber.'));
  }

  _updateBenchUI() {
    const judge = this.JUDGES[this.config.judgePersonality] || this.JUDGES.sinha;
    const set = (id, text) => { const el = document.getElementById(id); if (el) el.textContent = text; };
    set('bench-court-label', this.COURTS[this.config.courtLevel] || 'IN THE HIGH COURT');
    set('bench-judge-name',  judge.name);
    set('bench-judge-style', judge.style);
    set('floor-case-name',   (this.config.caseName || 'MATTER UNDER ARGUMENT').toUpperCase());
  }

  /* ── CHAMBER ────────────────────────────────────────────────── */

  async _toggleRecording() {
    if (!this.client) return;
    if (this.client.isSpeaking) {
      // Tap while the bench speaks = "May I interject, My Lord" —
      // cuts the judge off and reopens the floor to counsel.
      this.client.interrupt();
      if (!this.client.isRecording) await this.client.startRecording();
      return;
    }
    if (this.client.isRecording) {
      this.client.stopRecording();
      this._setRecordBtn('\u25CF Resume Arguing', false);
    } else {
      await this.client.startRecording();
    }
  }

  _setRecordBtn(text, recording) {
    const btn = document.getElementById('record-btn');
    if (!btn) return;
    btn.textContent = text;
    btn.classList.toggle('recording', !!recording);
  }

  _initVizBars() {
    const container = document.getElementById('viz-bars');
    if (!container) return;
    container.innerHTML = '';
    for (let i = 0; i < 24; i++) {
      const bar = document.createElement('div');
      bar.className = 'mc-viz-bar';
      bar.style.setProperty('--bar-height', (8 + Math.floor(Math.random() * 22)) + 'px');
      bar.style.setProperty('--i', i);
      container.appendChild(bar);
    }
  }

  _onVoiceState(state) {
    const vizBars = document.getElementById('viz-bars');
    if (vizBars) {
      vizBars.className = 'mc-viz-bars';
      if (state === 'listening')  vizBars.classList.add('listening');
      if (state === 'processing') vizBars.classList.add('processing');
      if (state === 'responding') vizBars.classList.add('speaking');
    }

    const labels = {
      idle:       'Ready',
      connecting: 'Entering the chamber\u2026',
      connected:  'The bench is seated',
      recording:  'The bench is listening',
      listening:  'Hearing you\u2026',
      processing: 'Submission taken on record\u2026',
      responding: 'The bench speaks',
      stopped:    'Paused',
    };
    const vizStatus = document.getElementById('viz-status');
    if (vizStatus) vizStatus.textContent = labels[state] || state;

    if (state === 'recording' || state === 'listening') {
      this._setRecordBtn('\u25A0 Arguing\u2026', true);
    } else if (state === 'responding') {
      this._setRecordBtn('\u25CF Bench Speaking \u2014 tap to interject', false);
    } else if (state === 'connected') {
      this._setRecordBtn('\u25CF Start Arguing', false);
    }
  }

  _onTranscript(text) {
    if (!text || !text.trim()) return;
    if (this.config.showTranscript) {
      this._appendTranscript('counsel', 'Counsel', text);
    }
    this._exchangeCount++;
    this._updateStats();
  }

  _onAgentStatus(agent, status) {
    if (agent === 'stt') return;
    this._setAgentBar(agent, status === 'processing' ? 'processing' : '');
  }

  _onAgentResponse(data) {
    const agent     = data.agent || '';
    const text      = data.text || '';
    const spoken    = data.spoken || '';
    const citations = data.citations || [];
    const metadata  = data.metadata || {};
    const agentKey  = agent.toLowerCase();

    this._setAgentBar(agentKey, 'active');
    setTimeout(() => this._setAgentBar(agentKey, ''), 1200);

    if (agent === 'Judge') {
      const judge = this.JUDGES[metadata.judge_personality] || this.JUDGES[this.config.judgePersonality];
      this._appendTranscript('judge', judge ? judge.display : 'The Court', text);
    } else if (agent === 'Weakness' && text) {
      this._appendTranscript('system', null, text);
      this._flagCount++;
    } else if (agent === 'Citation' && text) {
      this._appendTranscript('system', null, text);
      this._addCitationFlag(text);
      this._flagCount++;
    } else if (agent === 'Precedent') {
      if (metadata.cases && metadata.cases.length) {
        this._addCasesSurfaced(metadata.cases);
      }
      if (spoken) this._appendTranscript('agent', 'Researcher', spoken);
    } else if (agent === 'Counter' && text) {
      this._appendTranscript('agent', 'Opposing Counsel', text);
    }

    this._updateStats();
  }

  _onStats(stats) {
    if (!stats) return;
    this._exchangeCount = stats.exchanges      != null ? stats.exchanges      : this._exchangeCount;
    this._citationCount = stats.citations_used != null ? stats.citations_used : this._citationCount;
    this._flagCount     = stats.flags          != null ? stats.flags          : this._flagCount;
    this._updateStats();
  }

  _onVoiceError(msg) {
    this._appendTranscript('system', null, msg);
  }

  _appendTranscript(type, label, text) {
    const feed = document.getElementById('transcript-feed');
    if (!feed) return;
    const empty = feed.querySelector('.mc-transcript-empty');
    if (empty) empty.remove();

    const entry = document.createElement('div');
    entry.className = 'mc-entry-' + type;

    if (type !== 'system' && label) {
      const labelEl = document.createElement('span');
      labelEl.className = 'mc-entry-label';
      labelEl.textContent = label;
      entry.appendChild(labelEl);
    }
    const textEl = document.createElement('div');
    textEl.className = type === 'system' ? '' : 'mc-entry-text';
    textEl.textContent = text;
    entry.appendChild(textEl);

    feed.appendChild(entry);
    requestAnimationFrame(() => { feed.scrollTop = feed.scrollHeight; });
  }

  _setAgentBar(agentKey, cls) {
    const bar = document.querySelector('[data-agent="' + agentKey + '"]');
    if (!bar) return;
    bar.classList.remove('active', 'processing');
    if (cls) bar.classList.add(cls);
  }

  _addCasesSurfaced(cases) {
    const feed = document.getElementById('cases-feed');
    if (!feed) return;
    const empty = feed.querySelector('.mc-cases-empty');
    if (empty) empty.remove();

    cases.forEach((c) => {
      const item = document.createElement('div');
      item.className = 'mc-case-item';

      const nameEl = document.createElement('span');
      nameEl.className = 'mc-case-name';
      nameEl.textContent = c.case_name || 'Unknown';
      item.appendChild(nameEl);

      if (c.year || c.court) {
        const citEl = document.createElement('span');
        citEl.className = 'mc-case-citation';
        citEl.textContent = [c.year, c.court].filter(Boolean).join(' · ');
        item.appendChild(citEl);
      }
      if (c.holding) {
        const holdEl = document.createElement('span');
        holdEl.className = 'mc-case-holding';
        holdEl.textContent = c.holding.slice(0, 140);
        item.appendChild(holdEl);
      }
      feed.appendChild(item);
    });
  }

  _addCitationFlag(text) {
    const feed = document.getElementById('citation-flags-feed');
    if (!feed) return;
    const empty = feed.querySelector('.mc-cases-empty');
    if (empty) empty.remove();
    const flag = document.createElement('div');
    flag.className = 'mc-flag-item';
    flag.textContent = text;
    feed.appendChild(flag);
  }

  _updateStats() {
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
    set('stat-exchanges', this._exchangeCount);
    set('stat-citations', this._citationCount);
    set('stat-flags',     this._flagCount);
  }

  _startTimer() {
    this._timerSec = 0;
    clearInterval(this._timerInt);
    this._timerInt = setInterval(() => {
      this._timerSec++;
      const h = Math.floor(this._timerSec / 3600);
      const m = Math.floor((this._timerSec % 3600) / 60);
      const s = this._timerSec % 60;
      const pad = (n) => String(n).padStart(2, '0');
      const el = document.getElementById('session-timer');
      if (el) el.textContent = pad(h) + ':' + pad(m) + ':' + pad(s);
    }, 1000);
  }

  _stopTimer() {
    clearInterval(this._timerInt);
    this._setLive(false);
  }

  _setLive(active) {
    const dot = document.getElementById('live-dot');
    if (dot) dot.classList.toggle('active', !!active);
  }

  _endSession() {
    if (this.client) this.client.disconnect();
    this._stopTimer();
    this._transition('debrief');
    this._loadDebrief();
  }

  _downloadTranscript() {
    const feed = document.getElementById('transcript-feed');
    if (!feed) return;
    const entries = feed.querySelectorAll('.mc-entry-judge, .mc-entry-counsel, .mc-entry-agent, .mc-entry-system');
    const lines = [];
    entries.forEach((entry) => {
      const label = entry.querySelector('.mc-entry-label');
      const text  = entry.querySelector('.mc-entry-text') || entry.lastChild;
      const who   = label ? label.textContent.toUpperCase() : 'NOTE';
      lines.push(who + ': ' + (text ? text.textContent : ''));
    });
    const content =
      'MOOT CHAMBER \u2014 RECORD OF PROCEEDINGS\n' +
      (this.config.caseName || 'Untitled matter') + '\n' +
      new Date().toLocaleString() + '\n\n' +
      lines.join('\n\n');
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'moot_transcript_' + Date.now() + '.txt';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  /* ── DEBRIEF ────────────────────────────────────────────────── */

  async _loadDebrief() {
    if (!this.sessionId) { this._renderDebriefFallback(); return; }
    try {
      const res = await fetch('/api/moot/session/' + this.sessionId + '/debrief');
      if (!res.ok) throw new Error('debrief unavailable');
      this._renderDebrief(await res.json());
    } catch (_) {
      this._renderDebriefFallback();
    }
  }

  _renderDebrief(data) {
    const score    = data.score    || {};
    const feedback = data.feedback || {};

    // Summary + meta
    const summaryEl = document.getElementById('debrief-summary');
    if (summaryEl) {
      summaryEl.textContent = feedback.overall_summary || 'Session complete.';
      const dur  = data.duration_seconds || 0;
      const meta = document.createElement('div');
      meta.className = 'mc-debrief-meta';
      meta.innerHTML =
        '<span>Duration <strong>' + Math.floor(dur / 60) + 'm ' + (dur % 60) + 's</strong></span>' +
        '<span>Exchanges <strong>' + (data.exchange_count || 0) + '</strong></span>' +
        '<span>Citations <strong>' + ((data.citations_used || []).length) + '</strong></span>' +
        '<span>Cases surfaced <strong>' + ((data.cases_surfaced || []).length) + '</strong></span>';
      summaryEl.appendChild(meta);
    }

    // Scorecard
    const scorecardEl = document.getElementById('debrief-scorecard');
    if (scorecardEl) {
      scorecardEl.innerHTML = '';
      const dims = [
        { key: 'structure',      label: 'Structure' },
        { key: 'authority',      label: 'Authority' },
        { key: 'responsiveness', label: 'Responsiveness' },
        { key: 'precision',      label: 'Precision' },
        { key: 'coherence',      label: 'Coherence' },
      ];
      dims.forEach((dim) => {
        const val = Number(score[dim.key] || 0);
        const row = document.createElement('div');
        row.className = 'mc-score-row';
        row.innerHTML =
          '<span class="mc-score-label">' + dim.label + '</span>' +
          '<div class="mc-score-track"><div class="mc-score-fill ' + dim.key + '" data-pct="' + (val * 10) + '%"></div></div>' +
          '<span class="mc-score-value">' + val.toFixed(1) + '</span>';
        scorecardEl.appendChild(row);
      });
      const overallEl = document.createElement('div');
      overallEl.className = 'mc-overall-score';
      overallEl.innerHTML =
        '<span class="mc-overall-number">' + Number(score.overall || 0).toFixed(1) + '</span>' +
        '<span class="mc-overall-label">Overall / 10</span>';
      scorecardEl.appendChild(overallEl);

      // Staggered fill animation
      const fills = scorecardEl.querySelectorAll('.mc-score-fill');
      fills.forEach((fill, i) => {
        setTimeout(() => { fill.style.width = fill.dataset.pct; }, 150 + i * 120);
      });
    }

    // Dimension notes
    const notesEl = document.getElementById('debrief-notes');
    if (notesEl) {
      notesEl.innerHTML = '<h3>Assessment</h3>';
      [
        ['Structure',      feedback.structure_note],
        ['Authority',      feedback.authority_note],
        ['Precision',      feedback.precision_note],
        ['Responsiveness', feedback.responsiveness_note],
      ].forEach(([label, note]) => {
        if (!note) return;
        const div = document.createElement('div');
        div.className = 'mc-note-item';
        const strong = document.createElement('strong');
        strong.textContent = label;
        div.appendChild(strong);
        div.appendChild(document.createTextNode(note));
        notesEl.appendChild(div);
      });
    }

    // Weaknesses
    const wkEl = document.getElementById('debrief-weaknesses');
    if (wkEl) {
      wkEl.innerHTML = '';
      if ((data.weaknesses || []).length) {
        wkEl.innerHTML = '<h3>Weaknesses Flagged</h3>';
        data.weaknesses.forEach((w, i) => {
          const item = document.createElement('div');
          item.className = 'mc-weakness-item';
          item.textContent = (i + 1) + '. ' + w;
          wkEl.appendChild(item);
        });
      }
    }

    // Cases to know
    const casesEl = document.getElementById('debrief-cases');
    if (casesEl) {
      casesEl.innerHTML = '';
      const toKnow = (feedback.cases_to_know || []);
      if (toKnow.length) {
        casesEl.innerHTML = '<h3>Cases You Should Know</h3>';
        toKnow.forEach((c) => {
          const item = document.createElement('div');
          item.className = 'mc-case-know-item';
          item.textContent = c;
          casesEl.appendChild(item);
        });
      }
    }

    // Full transcript (collapsible)
    const tEl = document.getElementById('debrief-transcript');
    if (tEl) {
      tEl.innerHTML = '';
      const transcript = data.transcript || [];
      if (transcript.length) {
        tEl.innerHTML = '<h3>Record of Proceedings</h3>';
        const toggle = document.createElement('button');
        toggle.className = 'mc-collapsible-toggle';
        toggle.textContent = 'Show Full Record';
        const body = document.createElement('div');
        body.className = 'mc-transcript-full';
        transcript.forEach((h) => {
          const p = document.createElement('p');
          const strong = document.createElement('strong');
          const role = (h.role || 'unknown').toUpperCase();
          strong.textContent = (role === 'MOOTER' ? 'COUNSEL' : role) + ': ';
          p.appendChild(strong);
          p.appendChild(document.createTextNode(h.text || ''));
          body.appendChild(p);
        });
        toggle.addEventListener('click', () => {
          body.classList.toggle('expanded');
          toggle.textContent = body.classList.contains('expanded') ? 'Hide Record' : 'Show Full Record';
        });
        tEl.appendChild(toggle);
        tEl.appendChild(body);
      }
    }

    // Save-to-matter button state
    const saveBtn = document.getElementById('save-matter-btn');
    if (saveBtn) {
      if (this._matterSavedAtSetup) {
        saveBtn.textContent = 'Saved to Matter \u2713';
        saveBtn.disabled = true;
      } else if (!this.config.matterId) {
        saveBtn.style.display = 'none';
      }
    }
  }

  _renderDebriefFallback() {
    const summaryEl = document.getElementById('debrief-summary');
    if (summaryEl) {
      summaryEl.textContent =
        'Session complete \u2014 ' + this._exchangeCount +
        ' exchange(s). The detailed debrief is unavailable for this session.';
    }
  }

  async _saveToMatter() {
    if (!this.sessionId || !this.config.matterId) return;
    const btn = document.getElementById('save-matter-btn');
    try {
      const res = await fetch(
        '/api/moot/session/' + this.sessionId + '/save-to-matter?matter_id=' + this.config.matterId,
        { method: 'POST' }
      );
      if (btn) btn.textContent = res.ok ? 'Saved \u2713' : 'Save failed';
      if (res.ok && btn) btn.disabled = true;
    } catch (_) {
      if (btn) btn.textContent = 'Save failed';
    }
  }

  /* ── UTILITIES ──────────────────────────────────────────────── */

  _transition(name) {
    document.querySelectorAll('.mc-screen').forEach((s) => s.classList.remove('active'));
    const target = document.getElementById('screen-' + name);
    if (target) target.classList.add('active');
  }

  _resetForNewSession() {
    this._exchangeCount = 0;
    this._citationCount = 0;
    this._flagCount     = 0;
    this._timerSec      = 0;
    this.sessionId      = null;
    this.client         = null;
    this._matterSavedAtSetup = false;

    const feed = document.getElementById('transcript-feed');
    if (feed) feed.innerHTML = '<div class="mc-transcript-empty">The bench is being seated. Press <strong>Start Arguing</strong> when ready.</div>';
    const casesFeed = document.getElementById('cases-feed');
    if (casesFeed) casesFeed.innerHTML = '<div class="mc-cases-empty">Cases surfaced during argument will appear here. Ask the bench: \u201ccases on\u2026\u201d</div>';
    const flagsFeed = document.getElementById('citation-flags-feed');
    if (flagsFeed) flagsFeed.innerHTML = '<div class="mc-cases-empty">No flags.</div>';
    const saveBtn = document.getElementById('save-matter-btn');
    if (saveBtn) { saveBtn.style.display = ''; saveBtn.disabled = false; saveBtn.textContent = 'Save to Matter'; }
    const timerEl = document.getElementById('session-timer');
    if (timerEl) timerEl.textContent = '00:00:00';

    this._updateStats();
    this._initVizBars();
  }
}

/* ── BOOT ───────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const app = new MootChamberApp();
  app.init();
  window._mootApp = app;
});
