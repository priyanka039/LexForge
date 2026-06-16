/**
 * moot_worklet.js — AudioWorkletProcessor for the Moot Chamber.
 * Runs on the dedicated audio thread.
 *
 * Rules in here: no DOM, no fetch, no console.log inside process().
 * Buffers 1024 samples (~64ms at 16kHz) before posting to the main
 * thread — posting every 128-sample frame would flood the port.
 */

class MootPCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = new Float32Array(1024);
    this._idx = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;   // keep alive with no input

    const channel = input[0];               // mono
    for (let i = 0; i < channel.length; i++) {
      this._buffer[this._idx++] = channel[i];
      if (this._idx >= 1024) {
        // Post a copy — the buffer itself is reused.
        this.port.postMessage(this._buffer.slice(0, 1024));
        this._idx = 0;
      }
    }
    return true;
  }
}

registerProcessor('moot-pcm-processor', MootPCMProcessor);
