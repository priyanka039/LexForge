/**
 * MootVoiceClient
 * Browser-side voice pipeline for the Moot Chamber.
 *
 *  - Captures mic audio via AudioWorklet (16kHz mono PCM)
 *  - Runs energy-based VAD on the main thread:
 *      HALF-DUPLEX — mic hard-muted while the bench speaks (no echo loop);
 *      counsel interrupts by tapping the session button, not by voice
 *      ~1.15s silence → utterance complete → encoded as WAV → sent over WS
 *  - Receives JSON events (transcript, agent responses, stats)
 *  - Receives binary WAV chunks (judge TTS) and plays them in sequence
 *
 * States emitted via onStateChange:
 *   idle | connecting | connected | recording | listening |
 *   processing | responding | stopped
 */

class MootVoiceClient {
  constructor(options = {}) {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this._wsUrl           = options.wsUrl || `${proto}://${location.host}/api/moot/ws`;
    this._onTranscript    = options.onTranscript    || (() => {});
    this._onAgentResponse = options.onAgentResponse || (() => {});
    this._onAgentStatus   = options.onAgentStatus   || (() => {});
    this._onStats         = options.onStats         || (() => {});
    this._onStateChange   = options.onStateChange   || (() => {});
    this._onError         = options.onError         || (() => {});

    // WebSocket
    this._ws        = null;
    this._sessionId = null;

    // Capture
    this._audioCtx    = null;
    this._mediaStream = null;
    this._workletNode = null;
    this._recording   = false;

    // VAD — tuned for 64ms chunks @16kHz
    this._vad = {
      noiseFloor:      0.004,
      baseThreshold:   0.012,
      speaking:        false,
      voicedRun:       0,
      silentRun:       0,
      startChunks:     2,     // 2 voiced chunks (~128ms) → speech start
      endChunks:       18,    // 18 silent chunks (~1.15s) → speech end:
                              // counsel pause mid-submission; don't cut them off
      minVoicedChunks: 7,     // <~450ms of speech → discard as noise
      preRoll:         [],
      preRollMax:      6,     // ~380ms of pre-speech audio kept
      collected:       [],
      voicedTotal:     0,
    };

    // Playback — ONE persistent AudioContext, unlocked during the user's
    // click that starts the session. Per-chunk contexts get created outside
    // a user gesture and Chrome suspends them → silent judge.
    this._playQueue  = [];
    this._playing    = false;
    this._currentSrc = null;
    this._playCtx    = null;
    // HALF-DUPLEX: while the bench speaks (and shortly after), the mic is
    // hard-muted. Without this the judge's loudspeaker audio is picked up,
    // transcribed, attributed to counsel — and the judge argues with itself.
    this._muteUntil  = 0;

    this._state = 'idle';
  }

  /* ── PUBLIC API ─────────────────────────────────────────────── */

  connect(sessionConfig) {
    this._setState('connecting');

    // Create + unlock the playback context NOW — we are inside the user's
    // click ("Enter the Chamber"), so the browser allows audio.
    try {
      if (!this._playCtx) this._playCtx = new AudioContext();
      if (this._playCtx.state === 'suspended') this._playCtx.resume();
    } catch (_) {}

    this._ws = new WebSocket(this._wsUrl);
    this._ws.binaryType = 'arraybuffer';

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Connection timed out')), 8000);

      this._ws.onopen = () => {
        this._wsSend({ type: 'start_session', config: sessionConfig });
      };

      this._ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
          this._enqueueAudio(event.data);
          return;
        }
        let msg;
        try { msg = JSON.parse(event.data); } catch (_) { return; }

        if (msg.event_type === 'session_state' && !this._sessionId) {
          this._sessionId = msg.data && msg.data.session_id;
          clearTimeout(timeout);
          this._setState('connected', msg.data);
          resolve(msg.data);
          return;
        }
        this._handleEvent(msg);
      };

      this._ws.onerror = () => {
        clearTimeout(timeout);
        this._onError('Connection failed. Is the LexForge server running?');
        reject(new Error('WebSocket error'));
      };

      this._ws.onclose = () => {
        this._recording = false;
        this._setState('stopped');
      };
    });
  }

  async startRecording() {
    if (this._recording) return;
    try {
      this._mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount:     1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl:  true,
        },
      });

      this._audioCtx = new AudioContext({ sampleRate: 16000 });
      if (this._audioCtx.state === 'suspended') {
        await this._audioCtx.resume();
      }
      await this._audioCtx.audioWorklet.addModule('/static/moot_audio_worklet.js');

      const source = this._audioCtx.createMediaStreamSource(this._mediaStream);
      this._workletNode = new AudioWorkletNode(this._audioCtx, 'moot-pcm-processor');
      this._workletNode.port.onmessage = (e) => this._onChunk(e.data);
      source.connect(this._workletNode);

      this._recording = true;
      this._setState('recording');
    } catch (err) {
      if (err && err.name === 'NotAllowedError') {
        this._onError('Microphone access denied. Allow microphone access and try again.');
      } else if (err && err.name === 'NotFoundError') {
        this._onError('No microphone detected. Connect one and try again.');
      } else {
        this._onError('Could not start recording: ' + (err && err.message ? err.message : err));
      }
    }
  }

  stopRecording() {
    if (!this._recording) return;
    // Flush an in-flight utterance so trailing words are not lost.
    this._finalizeUtterance(true);

    if (this._workletNode) { try { this._workletNode.disconnect(); } catch (_) {} }
    this._workletNode = null;
    if (this._mediaStream) { this._mediaStream.getTracks().forEach((t) => t.stop()); }
    this._mediaStream = null;
    if (this._audioCtx) { this._audioCtx.close().catch(() => {}); }
    this._audioCtx = null;

    this._recording = false;
    this._setState('stopped');
  }

  interrupt() {
    this._stopPlayback();
    this._wsSend({ type: 'interrupt' });
  }

  disconnect() {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._wsSend({ type: 'end_session' });
      this._ws.close();
    }
    this.stopRecording();
    this._stopPlayback();
    if (this._playCtx) { this._playCtx.close().catch(() => {}); this._playCtx = null; }
    this._ws = null;
  }

  get sessionId()  { return this._sessionId; }
  get isRecording() { return this._recording; }
  get isSpeaking() { return this._playing; }

  /* ── VAD ────────────────────────────────────────────────────── */

  _onChunk(f32) {
    if (!this._recording) return;

    // HARD half-duplex: mic is dead while the bench speaks, plus a short
    // grace window for room reverb. To interrupt the judge, counsel taps
    // the button — exactly like raising a hand in court.
    if (this._playing || Date.now() < this._muteUntil) {
      this._resetVad();
      return;
    }

    const vad = this._vad;

    let sum = 0;
    for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
    const rms = Math.sqrt(sum / f32.length);

    // Track ambient noise floor while not speaking (slow EMA).
    if (!vad.speaking) {
      vad.noiseFloor = vad.noiseFloor * 0.95 + rms * 0.05;
    }
    const threshold = Math.max(vad.baseThreshold, vad.noiseFloor * 3);
    const voiced = rms > threshold;

    if (!vad.speaking) {
      vad.preRoll.push(f32);
      if (vad.preRoll.length > vad.preRollMax) vad.preRoll.shift();

      if (voiced) {
        vad.voicedRun++;
        if (vad.voicedRun >= vad.startChunks) {
          vad.speaking    = true;
          vad.collected   = vad.preRoll.slice();
          vad.preRoll     = [];
          vad.silentRun   = 0;
          vad.voicedTotal = vad.voicedRun;
          this._setState('listening');
        }
      } else {
        vad.voicedRun = 0;
      }
      return;
    }

    // Speaking — collect everything (speech + intra-utterance pauses).
    vad.collected.push(f32);
    if (voiced) {
      vad.voicedTotal++;
      vad.silentRun = 0;
    } else {
      vad.silentRun++;
      if (vad.silentRun >= vad.endChunks) {
        this._finalizeUtterance(false);
      }
    }
  }

  _resetVad() {
    const vad = this._vad;
    vad.speaking    = false;
    vad.voicedRun   = 0;
    vad.silentRun   = 0;
    vad.collected   = [];
    vad.preRoll     = [];
    vad.voicedTotal = 0;
  }

  _finalizeUtterance(force) {
    const vad = this._vad;
    if (!vad.speaking && !force) return;

    const chunks      = vad.collected;
    const voicedTotal = vad.voicedTotal;

    vad.speaking    = false;
    vad.voicedRun   = 0;
    vad.silentRun   = 0;
    vad.collected   = [];
    vad.voicedTotal = 0;

    if (!chunks.length || voicedTotal < vad.minVoicedChunks) {
      if (this._recording) this._setState('recording');
      return;   // too short — cough / chair creak / silence
    }

    const wav = this._encodeWav(chunks, 16000);
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(wav);
      this._setState('processing');
    }
  }

  _encodeWav(chunks, sampleRate) {
    let total = 0;
    for (const c of chunks) total += c.length;

    const dataLen = total * 2;
    const buf     = new ArrayBuffer(44 + dataLen);
    const view    = new DataView(buf);
    let pos = 0;
    const wStr = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(pos++, s.charCodeAt(i)); };
    const w32  = (v) => { view.setUint32(pos, v, true); pos += 4; };
    const w16  = (v) => { view.setUint16(pos, v, true); pos += 2; };

    wStr('RIFF'); w32(36 + dataLen); wStr('WAVE');
    wStr('fmt '); w32(16); w16(1); w16(1);
    w32(sampleRate); w32(sampleRate * 2); w16(2); w16(16);
    wStr('data'); w32(dataLen);

    for (const c of chunks) {
      for (let i = 0; i < c.length; i++) {
        let s = Math.max(-1, Math.min(1, c[i]));
        view.setInt16(pos, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        pos += 2;
      }
    }
    return buf;
  }

  /* ── EVENTS ─────────────────────────────────────────────────── */

  _handleEvent(msg) {
    const data = msg.data || {};
    switch (msg.event_type) {
      case 'transcript':
        if (data.empty) {
          if (this._recording) this._setState('recording');
        } else {
          this._onTranscript(data.text);
        }
        break;
      case 'agent_status':
        this._onAgentStatus(data.agent, data.status);
        break;
      case 'agent_response':
        this._onAgentResponse(data);
        break;
      case 'session_stats':
        this._onStats(data);
        if (!this._playing && this._recording) this._setState('recording');
        break;
      case 'error':
        this._onError(data.message || 'An error occurred.');
        if (this._recording) this._setState('recording');
        break;
      default:
        break;
    }
  }

  _setState(state, data) {
    this._state = state;
    this._onStateChange(state, data);
  }

  /* ── PLAYBACK ───────────────────────────────────────────────── */

  _enqueueAudio(wavBuffer) {
    // Sentences arrive as separate WAVs; mute the mic the moment audio
    // arrives so gaps between sentence chunks never unmute it.
    this._muteUntil = Date.now() + 800;
    this._playQueue.push(wavBuffer);
    if (!this._playing) this._drainQueue();
  }

  async _drainQueue() {
    this._playing = true;
    this._setState('responding');
    while (this._playQueue.length > 0) {
      const buf = this._playQueue.shift();
      await this._playWav(buf);
    }
    this._playing = false;
    // Grace window: room reverb of the judge's last word must not
    // re-enter the transcript as counsel's speech.
    this._muteUntil = Date.now() + 450;
    if (this._recording) this._setState('recording');
  }

  _getPlayCtx() {
    if (!this._playCtx || this._playCtx.state === 'closed') {
      try { this._playCtx = new AudioContext(); } catch (_) { this._playCtx = null; }
    }
    if (this._playCtx && this._playCtx.state === 'suspended') {
      this._playCtx.resume().catch(() => {});
    }
    return this._playCtx;
  }

  _playWav(wavBuffer) {
    return new Promise((resolve) => {
      const ctx = this._getPlayCtx();
      if (!ctx) { resolve(); return; }

      ctx.decodeAudioData(
        wavBuffer.slice(0),
        (audioBuffer) => {
          const src = ctx.createBufferSource();
          src.buffer = audioBuffer;
          src.connect(ctx.destination);
          this._currentSrc = src;
          src.onended = () => {
            this._currentSrc = null;
            resolve();
          };
          try { src.start(0); } catch (_) { this._currentSrc = null; resolve(); }
        },
        () => resolve()
      );
    });
  }

  _stopPlayback() {
    this._playQueue = [];
    try { if (this._currentSrc) this._currentSrc.stop(); } catch (_) {}
    this._currentSrc = null;
    this._playing    = false;
    this._muteUntil  = Date.now() + 450;
    // The persistent context is kept alive for the next reply.
  }

  /* ── WS ─────────────────────────────────────────────────────── */

  _wsSend(obj) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(obj));
    }
  }
}

window.MootVoiceClient = MootVoiceClient;
