// ─────────────────────────────────────────────────────────────────────────────
// LEXFORGE · voice.js  ·  Sarvam AI voice layer (frontend)
//
//   - mic buttons → MediaRecorder → /api/voice/transcribe → fill input
//   - explicit "Read aloud" buttons (.voice-listen-btn) inside action toolbars
//     → /api/voice/speak → Web Audio playback
//   - .voice-debate-btn → alternating petitioner / opposition / judge voices
//   - silent multi-language output via /api/voice/speak (lang param)
//   - floating settings bar (mic toggle, language, volume)
//
// Pure additive layer. The page renders its own .voice-listen-btn / .voice-debate-btn
// markup inside any action toolbar — this file binds behaviour by class.
// ─────────────────────────────────────────────────────────────────────────────

(function () {
  'use strict';

  const API = window.location.origin;

  // ── State ────────────────────────────────────────────────────────────────
  const State = {
    languages:     [{ code: 'en-IN', label: 'English (India)' }],
    lang:          localStorage.getItem('lexforge_voice_lang') || 'en-IN',
    micEnabled:    localStorage.getItem('lexforge_voice_mic')  !== 'off',
    volume:        Math.max(0, Math.min(1, parseFloat(localStorage.getItem('lexforge_voice_vol') || '1'))),
    audioUnlocked: false,
    audioCtx:      null,
    currentSrc:    null,    // currently playing AudioBufferSourceNode
    currentBtn:    null,    // currently active listen/debate button
    currentToken:  0,       // monotonic token to invalidate stale plays
    recorder:      null,
    recorderBtn:   null,
    recorderChunks:[],
    recorderStream:null,
    debateQueueId: 0,       // identifies an in-flight debate auto-play
  };

  // ── Helpers ──────────────────────────────────────────────────────────────
  function persistLang(v) {
    State.lang = v;
    try { localStorage.setItem('lexforge_voice_lang', v); } catch (_) {}
  }
  function persistMic(on) {
    State.micEnabled = !!on;
    try { localStorage.setItem('lexforge_voice_mic', on ? 'on' : 'off'); } catch (_) {}
    document.body.classList.toggle('voice-mic-disabled', !on);
  }
  function persistVol(v) {
    State.volume = v;
    try { localStorage.setItem('lexforge_voice_vol', String(v)); } catch (_) {}
  }

  // SVGs reused for button state swaps
  const ICON_SPEAKER = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M11 5L6 9H3v6h3l5 4V5z"/><path d="M15.5 8.5a5 5 0 010 7M19 5a9 9 0 010 14"/></svg>';
  const ICON_PAUSE   = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.55" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>';

  // ── Audio plumbing ────────────────────────────────────────────────────────
  function ensureAudioCtx() {
    if (!State.audioCtx) {
      try {
        const Ctor = window.AudioContext || window.webkitAudioContext;
        if (Ctor) State.audioCtx = new Ctor();
      } catch (_) {}
    }
    return State.audioCtx;
  }

  function unlockAudioOnce() {
    if (State.audioUnlocked) return;
    const ctx = ensureAudioCtx();
    if (!ctx) return;
    const tryResume = () => {
      try { ctx.resume && ctx.resume(); } catch (_) {}
      State.audioUnlocked = true;
      ['click', 'touchend', 'keydown'].forEach(ev =>
        window.removeEventListener(ev, tryResume, true));
    };
    ['click', 'touchend', 'keydown'].forEach(ev =>
      window.addEventListener(ev, tryResume, true));
  }

  function setBtnState(btn, state) {
    if (!btn) return;
    btn.classList.remove('is-loading', 'is-playing');
    const labelEl = btn.querySelector('.voice-btn-label');
    const idle = btn.dataset.labelIdle || 'Read aloud';
    const playing = btn.dataset.labelPlaying || 'Stop reading';
    if (state === 'loading') {
      btn.classList.add('is-loading');
      if (labelEl) labelEl.textContent = idle;
    } else if (state === 'playing') {
      btn.classList.add('is-playing');
      if (labelEl) labelEl.textContent = playing;
      const sym = btn.querySelector('svg');
      if (sym && !btn.classList.contains('voice-debate-btn')) {
        // swap speaker → pause for non-debate listen buttons
        const wrap = document.createElement('span');
        wrap.innerHTML = ICON_PAUSE;
        sym.replaceWith(wrap.firstElementChild);
      }
    } else {
      if (labelEl) labelEl.textContent = idle;
      const sym = btn.querySelector('svg');
      if (sym && !btn.classList.contains('voice-debate-btn')) {
        const wrap = document.createElement('span');
        wrap.innerHTML = ICON_SPEAKER;
        sym.replaceWith(wrap.firstElementChild);
      }
    }
  }

  function stopCurrent() {
    State.currentToken++;
    State.debateQueueId++;          // invalidate any debate in-flight
    if (State.currentSrc) {
      try { State.currentSrc.onended = null; State.currentSrc.stop(0); } catch (_) {}
      State.currentSrc = null;
    }
    if (State.currentBtn) setBtnState(State.currentBtn, 'idle');
    State.currentBtn = null;
    document.querySelectorAll('.debate-side.speaking').forEach(el => el.classList.remove('speaking'));
  }

  function base64ToArrayBuffer(b64) {
    try {
      const bin = atob(b64);
      const len = bin.length;
      const buf = new ArrayBuffer(len);
      const view = new Uint8Array(buf);
      for (let i = 0; i < len; i++) view[i] = bin.charCodeAt(i);
      return buf;
    } catch (_) { return null; }
  }

  async function playB64(b64, { onStart, onEnd } = {}) {
    if (!b64) { onEnd && onEnd(); return; }
    unlockAudioOnce();
    const ctx = ensureAudioCtx();
    const buf = base64ToArrayBuffer(b64);
    if (!ctx || !buf) { onEnd && onEnd(); return; }

    let decoded;
    try {
      decoded = await new Promise((res, rej) => {
        try { ctx.decodeAudioData(buf, res, rej); }
        catch (e) { rej(e); }
      });
    } catch (_) { onEnd && onEnd(); return; }

    // Stop only the previously playing source (don't bump tokens — we are inside an active job).
    if (State.currentSrc) {
      try { State.currentSrc.onended = null; State.currentSrc.stop(0); } catch (_) {}
      State.currentSrc = null;
    }
    const src = ctx.createBufferSource();
    src.buffer = decoded;
    const gain = ctx.createGain();
    gain.gain.value = State.volume;
    src.connect(gain).connect(ctx.destination);

    State.currentSrc = src;
    onStart && onStart();

    return new Promise(resolve => {
      src.onended = () => {
        if (State.currentSrc === src) State.currentSrc = null;
        onEnd && onEnd();
        resolve();
      };
      try { src.start(0); }
      catch (_) { onEnd && onEnd(); resolve(); }
    });
  }

  // ── Encode an AudioBuffer (mixed to mono) to a 16-bit PCM WAV Blob ───────
  // Sarvam STT does not accept audio/webm or opus — only WAV / MP3 / AAC /
  // PCM. Chrome's MediaRecorder cannot produce WAV directly, so we decode
  // the captured webm/ogg blob and re-encode it as WAV here in the browser.
  function audioBufferToWav(buffer) {
    const numCh = 1;
    const sr    = buffer.sampleRate;
    const len   = buffer.length;

    const mono = new Float32Array(len);
    const channels = buffer.numberOfChannels || 1;
    for (let ch = 0; ch < channels; ch++) {
      const data = buffer.getChannelData(ch);
      for (let i = 0; i < len; i++) mono[i] += data[i] / channels;
    }

    const dataLen = len * 2;
    const bufOut  = new ArrayBuffer(44 + dataLen);
    const view    = new DataView(bufOut);
    let pos = 0;
    const writeStr = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(pos++, s.charCodeAt(i)); };
    const writeU32 = (v) => { view.setUint32(pos, v, true); pos += 4; };
    const writeU16 = (v) => { view.setUint16(pos, v, true); pos += 2; };

    writeStr('RIFF');
    writeU32(36 + dataLen);
    writeStr('WAVE');
    writeStr('fmt ');
    writeU32(16);
    writeU16(1);                    // PCM
    writeU16(numCh);                // channels
    writeU32(sr);                   // sample rate
    writeU32(sr * numCh * 2);       // byte rate
    writeU16(numCh * 2);            // block align
    writeU16(16);                   // bits per sample
    writeStr('data');
    writeU32(dataLen);

    for (let i = 0; i < len; i++) {
      let s = Math.max(-1, Math.min(1, mono[i]));
      s = s < 0 ? s * 0x8000 : s * 0x7FFF;
      view.setInt16(pos, s, true); pos += 2;
    }
    return new Blob([bufOut], { type: 'audio/wav' });
  }

  async function blobToWav(blob) {
    if (!blob || !blob.size) return null;
    const ctx = ensureAudioCtx();
    if (!ctx) return null;
    let arr;
    try { arr = await blob.arrayBuffer(); } catch (_) { return null; }
    let decoded;
    try {
      decoded = await new Promise((res, rej) => {
        try { ctx.decodeAudioData(arr.slice(0), res, rej); }
        catch (e) { rej(e); }
      });
    } catch (_) { return null; }
    try { return audioBufferToWav(decoded); } catch (_) { return null; }
  }

  // ── Networking ────────────────────────────────────────────────────────────
  async function apiSpeak(text, role) {
    if (!text) return '';
    try {
      const res = await fetch(`${API}/api/voice/speak`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
        body: JSON.stringify({ text: text.slice(0, 4000), role: role || 'default', lang: State.lang || 'en-IN' }),
      });
      if (!res.ok) return '';
      const data = await res.json();
      const b64 = (data && data.audio_b64) || '';
      if (!b64 && data && data.ok === false && data.reason) {
        console.warn('[LexForge voice] No audio from Sarvam:', data.reason, '(see server terminal /api/voice/health)');
      }
      return b64;
    } catch (_) { return ''; }
  }

  async function apiTranscribe(blob) {
    try {
      const fd = new FormData();
      const fname = (blob.type && blob.type.includes('wav')) ? 'audio.wav' : 'audio.webm';
      fd.append('audio', blob, fname);
      const res = await fetch(`${API}/api/voice/transcribe`, {
        method: 'POST',
        headers: { 'ngrok-skip-browser-warning': 'true' },
        body: fd,
      });
      if (!res.ok) return '';
      const data = await res.json();
      return (data && data.transcript) || '';
    } catch (_) { return ''; }
  }

  async function apiLanguages() {
    try {
      const res = await fetch(`${API}/api/voice/languages`);
      if (!res.ok) return null;
      return await res.json();
    } catch (_) { return null; }
  }

  // ── Mic / STT ─────────────────────────────────────────────────────────────
  function pickRecorderMime() {
    if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) return '';
    const candidates = ['audio/wav', 'audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus'];
    for (const m of candidates) if (MediaRecorder.isTypeSupported(m)) return m;
    return '';
  }

  async function startRecording(btn) {
    if (!State.micEnabled) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return;
    if (State.recorder) {
      try { State.recorder.stop(); } catch (_) {}
      return;
    }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (_) { return; }

    const mime = pickRecorderMime();
    let rec;
    try { rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream); }
    catch (_) { try { rec = new MediaRecorder(stream); } catch (__) { return; } }

    State.recorder = rec;
    State.recorderBtn = btn;
    State.recorderChunks = [];
    State.recorderStream = stream;

    rec.addEventListener('dataavailable', e => {
      if (e.data && e.data.size > 0) State.recorderChunks.push(e.data);
    });
    rec.addEventListener('stop', async () => {
      const chunks = State.recorderChunks.slice();
      const recorderType = rec.mimeType || 'audio/webm';
      let blob = new Blob(chunks, { type: recorderType });
      try { State.recorderStream.getTracks().forEach(t => t.stop()); } catch (_) {}
      State.recorder = null;
      State.recorderBtn = null;
      State.recorderChunks = [];
      State.recorderStream = null;
      btn.classList.remove('recording');

      if (!blob.size) return;
      btn.classList.add('loading');

      // Sarvam STT only accepts wav/mp3/aac/pcm — Chrome's MediaRecorder gives
      // us webm/opus by default. Transcode in the browser before uploading.
      if (!/audio\/wave?$|audio\/wav/i.test(blob.type)) {
        const wav = await blobToWav(blob);
        if (wav) blob = wav;
      }

      const transcript = await apiTranscribe(blob);
      btn.classList.remove('loading');
      if (!transcript) return;
      const targetId = btn.dataset.target;
      const tgt = targetId && document.getElementById(targetId);
      if (!tgt) return;
      const cur = (tgt.value || '').trim();
      tgt.value = cur ? (cur + ' ' + transcript) : transcript;
      tgt.dispatchEvent(new Event('input', { bubbles: true }));
      try { tgt.focus(); } catch (_) {}
    });

    btn.classList.add('recording');
    rec.start();
  }

  function stopRecording() {
    if (State.recorder) {
      try { State.recorder.stop(); } catch (_) {}
    }
  }

  // ── Read-aloud (single text → TTS → play) ─────────────────────────────────
  function readableTextFor(btn) {
    const targetId = btn.dataset.target;
    if (!targetId) return '';
    const el = document.getElementById(targetId);
    if (!el) return '';
    return (el.innerText || '').replace(/\s+/g, ' ').trim();
  }

  async function handleListenClick(btn) {
    // toggle off if this same button is already playing
    if (State.currentBtn === btn) { stopCurrent(); return; }
    stopCurrent();
    const role = btn.dataset.role || 'default';
    const text = readableTextFor(btn);
    if (!text) return;

    State.currentBtn = btn;
    setBtnState(btn, 'loading');
    const b64 = await apiSpeak(text, role);
    if (State.currentBtn !== btn) return; // user cancelled while loading
    if (!b64) {
      setBtnState(btn, 'idle');
      State.currentBtn = null;
      return;
    }
    await playB64(b64, {
      onStart: () => setBtnState(btn, 'playing'),
      onEnd:   () => {
        if (State.currentBtn === btn) {
          setBtnState(btn, 'idle');
          State.currentBtn = null;
        }
      },
    });
  }

  // ── Debate auto-play (alternating voices) ─────────────────────────────────
  function collectDebateTurns(root) {
    const turns = [];
    const sides = root.querySelectorAll('.debate-side');
    sides.forEach(side => {
      const role = side.classList.contains('debate-p') ? 'petitioner'
                 : side.classList.contains('debate-d') ? 'opposition'
                 : 'judge';
      const points = side.querySelectorAll('.debate-point');
      const pointTexts = [];
      points.forEach(p => {
        const t = (p.innerText || '').replace(/\s+/g, ' ').trim();
        if (t) pointTexts.push(t);
      });
      const body = pointTexts.join(' ');
      if (body) turns.push({ role, side, text: body });
    });
    // Bench observation card (if rendered)
    const obs = root.querySelector('.text-gold');
    const benchWrap = obs ? obs.closest('.card') : null;
    if (benchWrap) {
      const t = (benchWrap.innerText || '').replace(/\s+/g, ' ').trim();
      if (t) turns.push({ role: 'judge', side: null, text: t });
    }
    return turns;
  }

  async function handleDebateClick(btn) {
    if (State.currentBtn === btn) { stopCurrent(); return; }
    stopCurrent();
    const targetId = btn.dataset.target || 'debate-output-body';
    const root = document.getElementById(targetId);
    if (!root) return;
    const turns = collectDebateTurns(root);
    if (!turns.length) return;
    State.currentBtn = btn;
    setBtnState(btn, 'playing');
    const myQueue = ++State.debateQueueId;

    for (const t of turns) {
      if (myQueue !== State.debateQueueId) break;
      // Loading state per turn
      const b64 = await apiSpeak(t.text, t.role);
      if (myQueue !== State.debateQueueId) break;
      if (!b64) continue;
      document.querySelectorAll('.debate-side.speaking').forEach(el => el.classList.remove('speaking'));
      if (t.side) t.side.classList.add('speaking');
      await playB64(b64);
      if (myQueue !== State.debateQueueId) break;
    }

    if (myQueue === State.debateQueueId) {
      setBtnState(btn, 'idle');
      State.currentBtn = null;
      document.querySelectorAll('.debate-side.speaking').forEach(el => el.classList.remove('speaking'));
    }
  }

  // ── Click delegation ─────────────────────────────────────────────────────
  document.addEventListener('click', e => {
    const mic = e.target.closest('.voice-mic');
    if (mic) {
      e.preventDefault();
      unlockAudioOnce();
      if (!State.micEnabled) return;
      if (State.recorder && State.recorderBtn === mic) stopRecording();
      else if (State.recorder) stopRecording();
      else startRecording(mic);
      return;
    }
    const debateBtn = e.target.closest('.voice-debate-btn');
    if (debateBtn) {
      e.preventDefault();
      unlockAudioOnce();
      handleDebateClick(debateBtn);
      return;
    }
    const listenBtn = e.target.closest('.voice-listen-btn');
    if (listenBtn) {
      e.preventDefault();
      unlockAudioOnce();
      handleListenClick(listenBtn);
      return;
    }
  });

  // ── Floating voice settings bar ───────────────────────────────────────────
  function fillLanguageSelect(sel, list, current) {
    if (!sel) return;
    sel.innerHTML = '';
    list.forEach(l => {
      const opt = document.createElement('option');
      opt.value = l.code;
      opt.textContent = l.label;
      if (l.code === current) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  async function initLanguageSelectors() {
    const data = await apiLanguages();
    if (data && Array.isArray(data.languages) && data.languages.length) {
      State.languages = data.languages;
    }
    const valid = State.languages.some(l => l.code === State.lang);
    if (!valid) State.lang = 'en-IN';
    fillLanguageSelect(document.getElementById('voice-lang-nav'), State.languages, State.lang);
    fillLanguageSelect(document.getElementById('voice-lang-bar'), State.languages, State.lang);
  }

  function wireBar() {
    const bar    = document.getElementById('voice-bar');
    const toggle = document.getElementById('voice-bar-toggle');
    const navSel = document.getElementById('voice-lang-nav');
    const barSel = document.getElementById('voice-lang-bar');
    const micT   = document.getElementById('voice-mic-global');
    const vol    = document.getElementById('voice-volume');
    const volRO  = document.getElementById('voice-vol-readout');

    if (toggle && bar) {
      toggle.addEventListener('click', () => {
        bar.classList.toggle('open');
      });
      document.addEventListener('click', (e) => {
        if (!bar.classList.contains('open')) return;
        if (bar.contains(e.target)) return;
        bar.classList.remove('open');
      });
    }

    const syncLangFromUI = (v) => {
      persistLang(v);
      if (navSel && navSel.value !== v) navSel.value = v;
      if (barSel && barSel.value !== v) barSel.value = v;
    };
    if (navSel) navSel.addEventListener('change', e => syncLangFromUI(e.target.value));
    if (barSel) barSel.addEventListener('change', e => syncLangFromUI(e.target.value));

    if (micT) {
      const paint = () => {
        micT.textContent = State.micEnabled ? 'On' : 'Off';
        micT.classList.toggle('off', !State.micEnabled);
        micT.setAttribute('aria-pressed', State.micEnabled ? 'true' : 'false');
        document.body.classList.toggle('voice-mic-disabled', !State.micEnabled);
      };
      paint();
      micT.addEventListener('click', () => {
        persistMic(!State.micEnabled);
        paint();
        if (!State.micEnabled) stopRecording();
      });
    }

    if (vol) {
      vol.value = String(Math.round(State.volume * 100));
      if (volRO) volRO.textContent = `${Math.round(State.volume * 100)}%`;
      vol.addEventListener('input', e => {
        const v = Math.max(0, Math.min(1, (+e.target.value || 0) / 100));
        persistVol(v);
        if (volRO) volRO.textContent = `${Math.round(v * 100)}%`;
      });
    }
  }

  // ── Boot ─────────────────────────────────────────────────────────────────
  function boot() {
    document.body.classList.toggle('voice-mic-disabled', !State.micEnabled);
    unlockAudioOnce();
    initLanguageSelectors().catch(() => {});
    wireBar();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  window.LexVoice = {
    stop:    stopCurrent,
    setLang: v => persistLang(v),
  };
})();
