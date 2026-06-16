/**
 * moot_audio_worklet.js
 * AudioWorkletProcessor — runs on the dedicated audio thread.
 *
 * Rules for worklet code:
 *  - no DOM, no fetch, no console.log inside process()
 *  - buffer input and post in chunks: posting every 128-sample frame
 *    floods the main thread
 *
 * Buffers 1024 samples (64ms @ 16kHz) before posting to the main
 * thread, which runs VAD and WAV encoding.
 */

class MootPCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(1024);
    this._idx    = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channel = input[0];
    for (let i = 0; i < channel.length; i++) {
      this._buffer[this._idx++] = channel[i];
      if (this._idx >= 1024) {
        this.port.postMessage(this._buffer.slice(0, 1024));
        this._idx = 0;
      }
    }
    return true;
  }
}

registerProcessor('moot-pcm-processor', MootPCMProcessor);
