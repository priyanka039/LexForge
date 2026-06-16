/**
 * MootChamberApp — UI controller for the Moot Chamber.
 *
 * Screen 1 (setup):   data-driven from GET /api/moot/meta + /api/cases
 * Screen 2 (chamber): MootVoiceClient lifecycle, transcript, agent
 *                     activity, stats, timer, typed fallback
 * Screen 3 (debrief): GET /api/moot/session/{id}/debrief → scorecard
 */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);

  class MootChamberApp {
    constructor() {
      this.config = {
        case_name: '', side_arguing: 'petitioner', case_statement: '',
        relevant_statutes: '', court_level: 'high_court',
        judge_personality: 'sinha', experience_level: 'junior',
        language: 'en-IN', matter_id: null,
        silent_citation_checking: true, weakness_alerts: true, show_transcript: true,
      };
      this.meta = null;
      this.client = null;
      this.sessionId = null;
      this.benchInfo = null;
      this._timer = null;
      this._timerSec = 0;
      this._stats = { exchanges: 0, citations: 0, flags: 0 };
    }

    async init() {
      this._initVizBars();
      this._bindChamberControls();
      this._bindDebriefControls();
      $('enter-chamber-btn').addEventListener('click', () => this._onEnterChamber());
      await Promise.all([this._loadMeta(), this._loadMatters()]);
    }

    /* ══ SETUP SCREEN ═══════════════════════════════════════════ */

    async _loadMeta() {
      try {
        const res = await fetch('/api/moot/meta');
        this.meta = await res.json();
      } catch (_) {
        this.meta = null;
      }
      if (!this.meta) return;

      // Sides
      const sideRow = $('side-selector');
      this.meta.sides.forEach((s, i) => {
        sideRow.appendChild(this._pill(s, s.charAt(0).toUpperCase() + s.slice(1), i === 0, (v) => {
          this.config.side_arguing = v;
        }, sideRow));
      });

      // Courts
      const courtDescs = {
        district:   'Fundamental skills. Strict procedure.',
        high_court: 'Constitutional arguments. Bench intervention.',
        supreme:    'Constitutional morality. Jurisprudential depth.',
      };
      const courtRow = $('court-selector');
      this.meta.courts.forEach((c) => {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'mc-card-choice' + (c.id === this.config.court_level ? ' active' : '');
        card.dataset.value = c.id;
        card.innerHTML =
          `<span class="mc-card-title">${this._esc(c.court_name)}</span>` +
          `<span class="mc-card-desc">${courtDescs[c.id] || ''}</span>`;
        card.addEventListener('click', () => {
          courtRow.querySelectorAll('.mc-card-choice').forEach(x => x.classList.remove('active'));
          card.classList.add('active');
          this.config.court_level = c.id;
        });
        courtRow.appendChild(card);
      });

      // Judges
      const judgeRow = $('judge-selector');
      this.meta.judges.forEach((j) => {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'mc-judge-card' + (j.id === this.config.judge_personality ? ' active' : '');
        card.dataset.value = j.id;
        card.innerHTML =
          `<span class="mc-judge-name">${this._esc(j.name)}</span>` +
          `<span class="mc-judge-style">${this._esc(j.style)}</span>` +
          `<span class="mc-judge-desc">${this._esc(j.desc)}</span>` +
          `<span class="mc-judge-tag">Best for: ${this._esc(j.best_for)}</span>`;
        card.addEventListener('click', () => {
          judgeRow.querySelectorAll('.mc-judge-card').forEach(x => x.classList.remove('active'));
          card.classList.add('active');
          this.config.judge_personality = j.id;
        });
        judgeRow.appendChild(card);
      });

      // Experience levels
      const levelRow = $('level-selector');
      this.meta.levels.forEach((l) => {
        const card = document.createElement('button');
        card.type = 'button';
        card.className = 'mc-card-choice' + (l.id === this.config.experience_level ? ' active' : '');
        card.dataset.value = l.id;
        card.innerHTML =
          `<span class="mc-card-title">${this._esc(l.label)}</span>` +
          `<span class="mc-card-desc">${this._esc(l.desc)}</span>`;
        card.addEventListener('click', () => {
          levelRow.querySelectorAll('.mc-card-choice').forEach(x => x.classList.remove('active'));
          card.classList.add('active');
          this.config.experience_level = l.id;
        });
        levelRow.appendChild(card);
      });

      // Languages
      const langRow = $('language-selector');
      this.meta.languages.forEach((l) => {
        langRow.appendChild(this._pill(l.code, l.label, l.code === 'en-IN', (v) => {
          this.config.language = v;
        }, langRow));
      });
    }

    _pill(value, label, active, onPick, row) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'mc-radio' + (active ? ' active' : '');
      btn.dataset.value = value;
      btn.textContent = label;
      btn.addEventListener('click', () => {
        row.querySelectorAll('.mc-radio').forEach(x => x.classList.remove('active'));
        btn.classList.add('active');
        onPick(value);
      });
      return btn;
    }

    async _loadMatters() {
      try {
        const res = await fetch('/api/cases');
        if (!res.ok) return;
        const data = await res.json();
        const matters = Array.isArray(data) ? data : (data.cases || []);
        const sel = $('matter-select');
        matters.forEach((m) => {
          const opt = document.createElement('option');
          opt.value = m.id;
          opt.textContent = m.name || ('Matter ' + m.id);
          sel.appendChild(opt);
        });
      } catch (_) { /* matters unavailable — dropdown stays empty */ }
    }

    _onEnterChamber() {
      this.config.case_name = $('case-name').value.trim();
      this.config.case_statement = $('case-statement').value.trim();
      this.config.relevant_statutes = $('statutes').value.trim();
      this.config.matter_id = $('matter-select').value || null;
      this.config.silent_citation_checking = $('opt-citation').checked;
      this.config.weakness_alerts = $('opt-weakness').checked;
      this.config.show_transcript = $('opt-transcript').checked;

      if (!this.config.case_name) {
        const el = $('case-name');
        el.classList.add('mc-invalid');
        el.focus();
        setTimeout(() => el.classList.remove('mc-invalid'), 2000);
        return;
      }

      this._transition('chamber');
      this._startSession();
    }

    /* ══ CHAMBER SCREEN ═════════════════════════════════════════ */

    _bindChamberControls() {
      $('record-btn').addEventListener('click', () => {
        if (this.client && this.client.isRecording) this._pauseArguing();
        else this._resumeArguing();
      });
      $('end-session-btn').addEventListener('click', () => this._endSession());
      $('download-btn').addEventListener('click', () => this._downloadTranscript());
      $('typed-form').addEventListener('submit', (e) => {
        e.preventDefault();
        const input = $('typed-input');
        const text = input.value.trim();
        if (!text || !this.client) return;
        this.client.sendText(text);
        input.value = '';
      });
    }

    _initVizBars() {
      const wrap = $('viz-bars');
      wrap.innerHTML = '';
      for (let i = 0; i < 24; i++) {
        const bar = document.createElement('div');
        bar.className = 'mc-viz-bar';
        bar.style.setProperty('--bar-height', (8 + Math.floor(Math.random() * 20)) + 'px');
        bar.style.setProperty('--i', i);
        wrap.appendChild(bar);
      }
      wrap.classList.add('idle');
    }

    _startSession() {
      this._resetChamber();
      this.client = new MootVoiceClient({
        onEvent: (type, data) => this._onServerEvent(type, data),
        onState: (state) => this._onVoiceState(state),
      });

      this.client.connect(this.config)
        .then((info) => {
          this.sessionId = this.client.sessionId;
          this.benchInfo = info || {};
          this._paintBench();
          this._startTimer();
          return this.client.startRecording();
        })
        .then(() => this._setRecordBtn(true))
        .catch((err) => {
          this._appendEntry('system', null, err.message || 'Could not open the session.');
        });
    }

    _paintBench() {
      const info = this.benchInfo || {};
      if (info.court_label) $('bench-court-label').textContent = info.court_label;
      if (info.judge_short) $('bench-judge-name').textContent = info.judge_short;
      if (info.judge_addr)  $('bench-judge-addr').textContent = 'Address the bench as \u201c' + info.judge_addr + '\u201d';
      $('floor-case-name').textContent = (this.config.case_name || 'MATTER UNDER ARGUMENT').toUpperCase();
    }

    async _pauseArguing() {
      this.client && this.client.stopRecording();
      this._setRecordBtn(false);
    }

    async _resumeArguing() {
      if (!this.client) return;
      try {
        await this.client.startRecording();
        this._setRecordBtn(true);
      } catch (_) { /* error already surfaced via onEvent */ }
    }

    _setRecordBtn(recording) {
      const btn = $('record-btn');
      if (recording) {
        btn.innerHTML = '&#9632; Stop';
        btn.classList.add('recording');
      } else {
        btn.innerHTML = '&#9679; Start Arguing';
        btn.classList.remove('recording');
      }
    }

    /* ── server events ─────────────────────────────────────────── */

    _onServerEvent(type, data) {
      switch (type) {
        case 'transcript':
          this._appendEntry('counsel', 'Counsel', data.text);
          break;

        case 'agent_status':
          this._setAgentBar(data.agent, data.status);
          break;

        case 'agent_response':
          this._renderAgentResponse(data);
          break;

        case 'citation_flag':
          this._appendEntry('system', null, data.text);
          (data.flags || []).forEach(f => this._addFlagItem(
            '\u26a0 ' + f.citation + (f.issue === 'year_out_of_range' ? ' \u2014 check the year' : ' \u2014 check the format')
          ));
          break;

        case 'citations_used':
          this._stats.citations = data.total || this._stats.citations;
          this._paintStats();
          break;

        case 'exchange_done':
          this._stats.exchanges = data.exchanges;
          this._stats.citations = data.citations;
          this._stats.flags = data.flags;
          this._paintStats();
          break;

        case 'error':
          this._appendEntry('system', null, data.message || 'Something went wrong.');
          break;
      }
    }

    _renderAgentResponse(data) {
      const agent = (data.agent || '').toLowerCase();
      const meta = data.metadata || {};

      if (agent === 'judge') {
        const judgeName = meta.judge_name || 'The Court';
        this._appendEntry('judge', judgeName, data.text);
      } else if (agent === 'counter') {
        this._appendEntry('counter', 'Counsel for the ' + (meta.opposing_side || 'other side'), data.text);
      } else if (agent === 'precedent') {
        this._appendEntry('researcher', 'Research', data.text);
        this._addCases(data.cases || []);
      } else if (agent === 'weakness' && data.text) {
        this._appendEntry('system', null, data.text);
      }
    }

    _onVoiceState(state) {
      const viz = $('viz-bars');
      viz.className = 'mc-viz-bars';
      const map = {
        idle: 'idle', connecting: 'idle', connected: 'idle', stopped: 'idle', error: 'idle',
        recording: 'idle', listening: 'listening', processing: 'processing', speaking: 'speaking',
      };
      viz.classList.add(map[state] || 'idle');

      const labels = {
        idle: 'Ready', connecting: 'Entering the chamber\u2026', connected: 'The bench is seated',
        recording: 'The bench is listening', listening: 'Hearing you\u2026',
        processing: 'On record \u2014 considering\u2026', speaking: 'The floor speaks',
        stopped: 'Session ended', error: 'Microphone unavailable',
      };
      $('viz-status').textContent = labels[state] || state;

      if (state === 'stopped' || state === 'error') this._setRecordBtn(false);
    }

    /* ── transcript + panels ───────────────────────────────────── */

    _appendEntry(kind, label, text) {
      if (!text) return;
      const feed = $('transcript-feed');
      const empty = feed.querySelector('.mc-transcript-empty');
      if (empty) empty.remove();

      const entry = document.createElement('div');
      entry.className = 'mc-entry ' + kind;
      if (kind !== 'system' && label) {
        const lab = document.createElement('span');
        lab.className = 'mc-entry-label';
        lab.textContent = label;
        entry.appendChild(lab);
      }
      const body = document.createElement('div');
      body.className = 'mc-entry-text';
      body.textContent = text;
      entry.appendChild(body);
      feed.appendChild(entry);
      requestAnimationFrame(() => { feed.scrollTop = feed.scrollHeight; });
    }

    _setAgentBar(agent, status) {
      const bar = document.querySelector('.mc-agent-bar[data-agent="' + agent + '"]');
      if (!bar) return;
      bar.classList.remove('processing', 'done');
      if (status === 'processing') {
        bar.classList.add('processing');
      } else if (status === 'done') {
        bar.classList.add('done');
        setTimeout(() => bar.classList.remove('done'), 1200);
      }
    }

    _addCases(cases) {
      if (!cases.length) return;
      const feed = $('cases-feed');
      const empty = feed.querySelector('.mc-cases-empty');
      if (empty) empty.remove();
      cases.forEach((c) => {
        const item = document.createElement('div');
        item.className = 'mc-case-item';
        const name = document.createElement('span');
        name.className = 'mc-case-name';
        name.textContent = c.case_name || 'Unknown';
        item.appendChild(name);
        if (c.court || c.year) {
          const cit = document.createElement('span');
          cit.className = 'mc-case-citation';
          cit.textContent = [c.court, c.year].filter(Boolean).join(' \u00b7 ');
          item.appendChild(cit);
        }
        if (c.holding) {
          const hold = document.createElement('span');
          hold.className = 'mc-case-holding';
          hold.textContent = c.holding;
          item.appendChild(hold);
        }
        feed.appendChild(item);
      });
    }

    _addFlagItem(text) {
      const feed = $('citation-flags-feed');
      const empty = feed.querySelector('.mc-cases-empty');
      if (empty) empty.remove();
      const flag = document.createElement('div');
      flag.className = 'mc-flag-item';
      flag.textContent = text;
      feed.appendChild(flag);
    }

    _paintStats() {
      $('stat-exchanges').textContent = this._stats.exchanges;
      $('stat-citations').textContent = this._stats.citations;
      $('stat-flags').textContent = this._stats.flags;
    }

    _startTimer() {
      this._timerSec = 0;
      $('live-dot').classList.add('active');
      this._timer = setInterval(() => {
        this._timerSec++;
        const h = String(Math.floor(this._timerSec / 3600)).padStart(2, '0');
        const m = String(Math.floor((this._timerSec % 3600) / 60)).padStart(2, '0');
        const s = String(this._timerSec % 60).padStart(2, '0');
        $('session-timer').textContent = h + ':' + m + ':' + s;
      }, 1000);
    }

    _stopTimer() {
      clearInterval(this._timer);
      $('live-dot').classList.remove('active');
    }

    _downloadTranscript() {
      const entries = [...document.querySelectorAll('#transcript-feed .mc-entry')];
      const lines = entries.map((e) => {
        const label = e.querySelector('.mc-entry-label');
        const text = e.querySelector('.mc-entry-text');
        const who = label ? label.textContent.toUpperCase() : 'NOTE';
        return who + ': ' + (text ? text.textContent : '');
      });
      const head = 'MOOT CHAMBER \u2014 RECORD OF PROCEEDINGS\n' +
        (this.config.case_name || 'Untitled matter') + '\n' +
        new Date().toLocaleString() + '\n\n';
      const blob = new Blob([head + lines.join('\n\n')], { type: 'text/plain;charset=utf-8' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'moot_transcript_' + Date.now() + '.txt';
      a.click();
      URL.revokeObjectURL(a.href);
    }

    _endSession() {
      if (this.client) this.client.disconnect();
      this._stopTimer();
      this._transition('debrief');
      this._loadDebrief();
    }

    _resetChamber() {
      this._stats = { exchanges: 0, citations: 0, flags: 0 };
      this._paintStats();
      $('session-timer').textContent = '00:00:00';
      $('transcript-feed').innerHTML =
        '<div class="mc-transcript-empty">The bench has assembled. Press <strong>Start Arguing</strong> and open your submissions.</div>';
      $('cases-feed').innerHTML =
        '<div class="mc-cases-empty">Ask the floor for authority \u2014 \u201cany case on\u2026\u201d \u2014 and it will appear here.</div>';
      $('citation-flags-feed').innerHTML = '<div class="mc-cases-empty">No flags.</div>';
      document.querySelectorAll('.mc-agent-bar').forEach(b => b.classList.remove('processing', 'done'));
    }

    /* ══ DEBRIEF SCREEN ═════════════════════════════════════════ */

    _bindDebriefControls() {
      $('new-session-btn').addEventListener('click', () => {
        this.sessionId = null;
        this.client = null;
        $('save-matter-btn').textContent = 'Save to Matter';
        this._transition('setup');
      });
      $('save-matter-btn').addEventListener('click', () => this._saveToMatter());
    }

    async _loadDebrief() {
      const summaryEl = $('debrief-summary');
      summaryEl.textContent = 'Preparing the record\u2026';
      if (!this.sessionId) { this._debriefFallback(); return; }
      try {
        const res = await fetch('/api/moot/session/' + this.sessionId + '/debrief');
        if (!res.ok) throw new Error();
        this._renderDebrief(await res.json());
      } catch (_) {
        this._debriefFallback();
      }
    }

    _renderDebrief(data) {
      const score = data.score || {};
      const fb = data.feedback || {};

      const mins = Math.floor((data.duration_seconds || 0) / 60);
      const secs = (data.duration_seconds || 0) % 60;
      $('debrief-summary').innerHTML =
        this._esc(fb.overall_summary || 'Session complete.') +
        '<span class="mc-debrief-meta">' +
        this._esc(data.case_name || '') + ' \u00b7 ' + mins + 'm ' + secs + 's \u00b7 ' +
        (data.exchange_count || 0) + ' exchange(s) \u00b7 ' +
        (data.citations_used || []).length + ' citation(s) \u00b7 ' +
        (data.cases_surfaced || []).length + ' case(s) surfaced</span>';

      // Scorecard
      const card = $('debrief-scorecard');
      card.innerHTML = '';
      const dims = [
        { key: 'structure',      label: 'Structure',      note: fb.structure_note },
        { key: 'authority',      label: 'Authority',      note: fb.authority_note },
        { key: 'responsiveness', label: 'Responsiveness', note: fb.responsiveness_note },
        { key: 'precision',      label: 'Precision',      note: fb.precision_note },
        { key: 'coherence',      label: 'Coherence',      note: null },
      ];
      dims.forEach((d) => {
        const val = Number(score[d.key] || 0);
        const row = document.createElement('div');
        row.className = 'mc-score-row';
        row.innerHTML =
          '<span class="mc-score-label">' + d.label + '</span>' +
          '<div class="mc-score-track"><div class="mc-score-fill ' + d.key + '" data-pct="' + (val * 10) + '%"></div></div>' +
          '<span class="mc-score-value">' + val.toFixed(1) + '</span>';
        card.appendChild(row);
        if (d.note) {
          const note = document.createElement('div');
          note.className = 'mc-score-note';
          note.textContent = d.note;
          card.appendChild(note);
        }
      });
      const overall = document.createElement('div');
      overall.className = 'mc-overall-score';
      overall.innerHTML =
        '<span class="mc-overall-number">' + Number(score.overall || 0).toFixed(1) + '</span>' +
        '<span class="mc-overall-label">Overall / 10</span>';
      card.appendChild(overall);
      setTimeout(() => {
        card.querySelectorAll('.mc-score-fill').forEach((f, i) => {
          setTimeout(() => { f.style.width = f.dataset.pct; }, i * 130);
        });
      }, 150);

      // Weaknesses
      const wk = $('debrief-weaknesses');
      wk.innerHTML = '';
      if ((data.weaknesses || []).length) {
        wk.innerHTML = '<h3>Areas of Improvement</h3>';
        data.weaknesses.forEach((w, i) => {
          const item = document.createElement('div');
          item.className = 'mc-weakness-item';
          item.textContent = (i + 1) + '. ' + w;
          wk.appendChild(item);
        });
      }

      // Cases to know
      const cs = $('debrief-cases');
      cs.innerHTML = '';
      if ((fb.cases_to_know || []).length) {
        cs.innerHTML = '<h3>Cases You Should Know</h3>';
        fb.cases_to_know.forEach((c) => {
          const item = document.createElement('div');
          item.className = 'mc-case-know-item';
          item.textContent = c;
          cs.appendChild(item);
        });
      }

      // Full transcript
      const tr = $('debrief-transcript');
      tr.innerHTML = '';
      if ((data.transcript || []).length) {
        tr.innerHTML = '<h3>Record of Proceedings</h3>';
        const toggle = document.createElement('button');
        toggle.className = 'mc-collapsible-toggle';
        toggle.textContent = 'Show Full Record';
        const body = document.createElement('div');
        body.className = 'mc-transcript-full';
        const roleLabels = {
          mooter: 'COUNSEL', judge: 'THE BENCH', counter: 'OPPOSING COUNSEL',
          researcher: 'RESEARCH', system: 'NOTE',
        };
        data.transcript.forEach((h) => {
          const p = document.createElement('p');
          const strong = document.createElement('strong');
          strong.textContent = (roleLabels[h.role] || 'NOTE') + ': ';
          p.appendChild(strong);
          p.appendChild(document.createTextNode(h.text || ''));
          body.appendChild(p);
        });
        toggle.addEventListener('click', () => {
          body.classList.toggle('expanded');
          toggle.textContent = body.classList.contains('expanded') ? 'Hide Record' : 'Show Full Record';
        });
        tr.appendChild(toggle);
        tr.appendChild(body);
      }

      // Save button visibility
      $('save-matter-btn').style.display = this.config.matter_id ? '' : 'none';
    }

    _debriefFallback() {
      $('debrief-summary').textContent =
        'Session complete. ' + this._stats.exchanges + ' exchange(s) on record. The full debrief could not be retrieved.';
      $('debrief-scorecard').innerHTML = '';
      $('save-matter-btn').style.display = 'none';
    }

    async _saveToMatter() {
      if (!this.sessionId || !this.config.matter_id) return;
      const btn = $('save-matter-btn');
      btn.textContent = 'Saving\u2026';
      try {
        const res = await fetch(
          '/api/moot/session/' + this.sessionId + '/save-to-matter?matter_id=' + this.config.matter_id,
          { method: 'POST' }
        );
        btn.textContent = res.ok ? 'Saved \u2713' : 'Save failed';
      } catch (_) {
        btn.textContent = 'Save failed';
      }
    }

    /* ══ UTIL ═══════════════════════════════════════════════════ */

    _transition(name) {
      document.querySelectorAll('.mc-screen').forEach(s => s.classList.remove('active'));
      const target = $('screen-' + name);
      if (target) target.classList.add('active');
    }

    _esc(s) {
      const div = document.createElement('div');
      div.textContent = s == null ? '' : String(s);
      return div.innerHTML;
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    const app = new MootChamberApp();
    app.init();
    window._mootApp = app;
  });
})();
