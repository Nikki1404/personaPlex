def step_if_ready(self) -> Optional[str]:
    # If buffer trimmed, safest is to reset stream cache alignment
    if self._trimmed_since_last_step and self.emitted_frames > 0:
        self.cache = self.engine.model.encoder.get_initial_cache_state(batch_size=1)
        self.cache = self.engine._move_cache_to_device((self.cache[0], self.cache[1], self.cache[2]))
        self.prev_hyp = None
        self.prev_pred = None
        self.emitted_frames = 0
        self._trimmed_since_last_step = False

    text, self.cache, self.prev_hyp, self.prev_pred, self.emitted_frames, t = self.engine.stream_transcribe(
        audio_f32=self.audio,
        cache=self.cache,
        prev_hyp=self.prev_hyp,
        prev_pred_out=self.prev_pred,
        emitted_frames=self.emitted_frames,
        force_flush=False,
    )

    self.utt_preproc += t.preproc_sec
    self.utt_infer += t.infer_sec

    if text is None or text == "":
        return None

    if text == self.current_text:
        return None

    # incremental partial fix
    if text.startswith(self.current_text):
        delta = text[len(self.current_text):].strip()
    else:
        delta = text

    self.current_text = text
    self.chunks += 1

    return delta if delta else None
