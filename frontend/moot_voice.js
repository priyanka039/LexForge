/**
 * MootVoiceClient — browser voice pipeline for the Moot Chamber.
 *
 * Responsibilities:
 *   - Capture mic audio at 16kHz mono via AudioWorklet
 *   - Segment utterances with an adaptive energy VAD
 *     (speech start → collect → 900ms silence → utterance complete)
 *   - Encode each utterance as 16-bit PCM WAV, send as one binary
 *     WebSocket frame (the backend transcribes it with Sarvam)
 *   - Receive JSON events; play base64 WAV TTS sequentially
 *   - Barge-in: if the mooter starts speaking while an agent is
 *     talking, playback stops instantly — like a real courtroom
 *
 * Emits to the UI via callbacks: onEvent(type, data), onState(state).
 * States: idle | connecting | connected | recording | listening |
 *         processing | speaking | stopped | error
 */
(function () {
  'use strict';

  // VAD tuning
  const VAD = {
    START_FRAMES:    3,      // consecutive loud chunks (~190ms) to open an utterance
    SILENCE_MS:      900,    // trailing silence that closes an utterance
    MIN_UTTER_MS:    450,    // shorter than this = discard as noise
    MAX_UTTER_MS:    90000,  // hard cap — force-close marathon submissions
    NOISE_ALPHA:     0.95,   // noise floor EMA
    START_FACTOR:    3.2,    // speech if rms > noiseFloor * factor
    MIN_THRESHOLD:   0.008,  // absolute floor so silence never triggers
  };
  const SAMPLE_RATE = 16000;
  const CHUNK_MS    = 1024 / SAMPLE_RATE * 1000;   // ~64ms per worklet chunk

  class MootVoiceClient {
    constructor(opts = {}) {
      this.onEvent = opts.onEvent || (() => {});
      this.onState = opts.onState || (() => {});
      this._wsUrl  = opts.wsUrl ||
        `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/moot/ws`;

      this._ws = null;
      this._sessionId = null;

      // capture
      this._ctx = null;
      this._stream = null;
      this._worklet = null;
      this._recording = false;

      // VAD state
      this._noiseFloor = 0.004;
      this._loudStreak = 0;
      this._inUtterance = false;
      this._silenceMs = 0;
      this._utterMs = 0;
      this._chunks = [];           // Float32Array chunks of current utterance
      this._preroll = [];          // ring of recent chunks kept before speech start

      // playback
      this._queue = [];
      this._playing = false;
      this._playCtx = null;
      this._playSrc = null;

      this._state = 'idle';
    }

    /* ── public ──────────────────────────────────────────────── */

    connect(config) {
      this._setState('connecting');
      return new Promise((resolve, reject) => {
        let settled = false;
        const timer = setTimeout(() => {
          if (!settled) { settled = true; reject(new Error('Connection timed out. Is the server running?')); }
        }, 8000);

        this._ws = new WebSocket(this._wsUrl);
        this._ws.binaryType = 'arraybuffer';

        this._ws.onopen = () => {
          this._send({ type: 'start_session', config: config });
        };
        this._ws.onmessage = (e) => {
          let ev;
          try { ev = JSON.parse(e.data); } catch (_) { return; }
          if (ev.event_type === 'session_state' && !settled) {
            settled = true;
            clearTimeout(timer);
            this._sessionId = ev.data && ev.data.session_id;
            this._setState('connected');
            resolve(ev.data);
          }
          this._handleEvent(ev);
        };
        this._ws.onerror = () => {
          if (!settled) { settled = true; clearTimeout(timer); reject(new Error('Could not reach the chamber. Is the server running?')); }
          this.onEvent('error', { message: 'Connection error.' });
        };
        this._ws.onclose = () => {
          this._setState('stopped');
          this._recording = false;
        };
      });
    }

    async startRecording() {
      if (this._recording) return;
      try {
        this._stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        this._ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
        if (this._ctx.state === 'suspended') await this._ctx.resume();
        await this._ctx.audioWorklet.addModule('/static/moot_worklet.js');

        const source = this._ctx.createMediaStreamSource(this._stream);
        this._worklet = new AudioWorkletNode(this._ctx, 'moot-pcm-processor');
        this._worklet.port.onmessage = (e) => this._onChunk(e.data);
        source.connect(this._worklet);

        this._resetVad();
        this._recording = true;
        this._setState('recording');
      } catch (err) {
        let msg = 'Could not start the microphone: ' + err.message;
        if (err.name === 'NotAllowedError') msg = 'Microphone access denied. Allow it in the browser and try again.';
        if (err.name === 'NotFoundError')  msg = 'No microphone detected. Connect one and try again.';
        this.onEvent('error', { message: msg });
        this._setState('error');
        throw err;
      }
    }

    stopRecording() {
      if (!this._recording) return;
      // Flush any in-flight utterance before tearing down.
      if (this._inUtterance) this._closeUtterance(true);
      try { this._worklet && this._worklet.disconnect(); } catch (_) {}
      this._worklet = null;
      try { this._stream && this._stream.getTracks().forEach(t => t.stop()); } catch (_) {}
      this._stream = null;
      if (this._ctx) { this._ctx.close().catch(() => {}); this._ctx = null; }
      this._recording = false;
      this._setState(this._ws && this._ws.readyState === WebSocket.OPEN ? 'connected' : 'stopped');
    }

    sendText(text) {
      this._send({ type: 'text_submission', text: text });
    }

    interrupt() {
      this._stopPlayback();
    }

    disconnect() {
      this.stopRecording();
      this._stopPlayback();
      if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        this._send({ type: 'end_session' });
        try { this._ws.close(); } catch (_) {}
      }
      this._ws = null;
    }

    get sessionId()  { return this._sessionId; }
    get isRecording(){ return this._recording; }

    /* ── VAD + capture ───────────────────────────────────────── */

    _resetVad() {
      this._loudStreak = 0;
      this._inUtterance = false;
      this._silenceMs = 0;
      this._utterMs = 0;
      this._chunks = [];
      this._preroll = [];
    }

    _onChunk(f32) {
      if (!this._recording) return;

      let sum = 0;
      for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
      const rms = Math.sqrt(sum / f32.length);

      const threshold = Math.max(this._noiseFloor * VAD.START_FACTOR, VAD.MIN_THRESHOLD);
      const loud = rms > threshold;

      if (!this._inUtterance) {
        // Adapt the noise floor only when not speaking.
        if (!loud) this._noiseFloor = this._noiseFloor * VAD.NOISE_ALPHA + rms * (1 - VAD.NOISE_ALPHA);

        // Keep a short pre-roll so the first syllable isn't clipped.
        this._preroll.push(f32);
        if (this._preroll.length > 6) this._preroll.shift();

        this._loudStreak = loud ? this._loudStreak + 1 : 0;
        if (this._loudStreak >= VAD.START_FRAMES) {
          this._inUtterance = true;
          this._chunks = this._preroll.slice();
          this._preroll = [];
          this._silenceMs = 0;
          this._utterMs = this._chunks.length * CHUNK_MS;
          // Barge-in: counsel speaks, the agent yields the floor.
          if (this._playing) this._stopPlayback();
          this._setState('listening');
        }
        return;
      }

      // In an utterance.
      this._chunks.push(f32);
      this._utterMs += CHUNK_MS;
      this._silenceMs = loud ? 0 : this._silenceMs + CHUNK_MS;

      if (this._silenceMs >= VAD.SILENCE_MS || this._utterMs >= VAD.MAX_UTTER_MS) {
        this._closeUtterance(false);
      }
    }

    _closeUtterance(force) {
      const chunks = this._chunks;
      const durMs = this._utterMs - (force ? 0 : this._silenceMs);
      this._inUtterance = false;
      this._chunks = [];
      this._silenceMs = 0;
      this._utterMs = 0;
      this._loudStreak = 0;

      if (durMs < VAD.MIN_UTTER_MS || !chunks.length) {
        if (this._recording) this._setState('recording');
        return;
      }

      const wav = this._encodeWav(chunks);
      if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        this._ws.send(wav);
        this._setState('processing');
      } else if (this._recording) {
        this._setState('recording');
      }
    }

    _encodeWav(chunks) {
      let total = 0;
      for (const c of chunks) total += c.length;

      const dataLen = total * 2;
      const buf = new ArrayBuffer(44 + dataLen);
      const view = new DataView(buf);
      let pos = 0;
      const wStr = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(pos++, s.charCodeAt(i)); };
      const w32 = (v) => { view.setUint32(pos, v, true); pos += 4; };
      const w16 = (v) => { view.setUint16(pos, v, true); pos += 2; };

      wStr('RIFF'); w32(36 + dataLen); wStr('WAVE');
      wStr('fmt '); w32(16); w16(1); w16(1);
      w32(SAMPLE_RATE); w32(SAMPLE_RATE * 2); w16(2); w16(16);
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

    /* ── server events + playback ────────────────────────────── */

    _handleEvent(ev) {
      const type = ev.event_type;
      const data = ev.data || {};

      if (type === 'agent_response' && data.audio_b64) {
        this._enqueueAudio(data.audio_b64);
      }
      if (type === 'stt_status' && data.status === 'empty' && this._recording) {
        this._setState('recording');
      }
      this.onEvent(type, data);
    }

    _enqueueAudio(b64) {
      const buf = this._b64ToBuf(b64);
      if (!buf) return;
      this._queue.push(buf);
      if (!this._playing) this._drainQueue();
    }

    async _drainQueue() {
      this._playing = true;
      this._setState('speaking');
      while (this._queue.length) {
        const buf = this._queue.shift();
        await this._playWav(buf);
        if (!this._playing) break;   // barge-in mid-queue
      }
      this._playing = false;
      if (this._recording) this._setState('recording');
    }

    _playWav(buf) {
      return new Promise((resolve) => {
        let ctx;
        try { ctx = new AudioContext(); } catch (_) { return resolve(); }
        this._playCtx = ctx;
        ctx.decodeAudioData(
          buf.slice(0),
          (audio) => {
            if (!this._playing) { ctx.close().catch(() => {}); return resolve(); }
            const src = ctx.createBufferSource();
            src.buffer = audio;
            src.connect(ctx.destination);
            this._playSrc = src;
            src.onended = () => {
              ctx.close().catch(() => {});
              if (this._playSrc === src) this._playSrc = null;
              this._playCtx = null;
              resolve();
            };
            try { src.start(0); } catch (_) { resolve(); }
          },
          () => { ctx.close().catch(() => {}); resolve(); }
        );
      });
    }

    _stopPlayback() {
      this._queue = [];
      this._playing = false;
      try { this._playSrc && this._playSrc.stop(); } catch (_) {}
      if (this._playCtx) { this._playCtx.close().catch(() => {}); }
      this._playSrc = null;
      this._playCtx = null;
    }

    /* ── misc ────────────────────────────────────────────────── */

    _b64ToBuf(b64) {
      try {
        const bin = atob(b64);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out.buffer;
      } catch (_) { return null; }
    }

    _send(obj) {
      if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        this._ws.send(JSON.stringify(obj));
      }
    }

    _setState(state) {
      if (this._state === state) return;
      this._state = state;
      this.onState(state);
    }
  }

  window.MootVoiceClient = MootVoiceClient;
})();
