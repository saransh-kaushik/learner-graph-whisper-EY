/**
 * Session Studio — client app.
 *
 * Polling rules (one loop per job / batch, never per render):
 *   • chained setTimeout with backoff (resets when status changes)
 *   • paused while the tab is hidden, resumes immediately when visible
 *   • stops on completion, on 404, or after repeated network failures
 *   • status polls are lightweight (?include_result=false); the full
 *     transcript is fetched exactly once when transcription finishes
 */
(() => {
  'use strict';

  const { $, el, icon, api, storage, session, toast, copy, fmt } = window.EY;

  /**
   * Poller: calls `fn` repeatedly. `fn` returns {stop: bool, changed: bool}.
   * Delay grows by `factor` while nothing changes, resets when it does.
   */
  function createPoller(fn, { minDelay = 2000, maxDelay = 15000, factor = 1.5, maxFailures = 5, onFail } = {}) {
    let timer = null;
    let delay = minDelay;
    let running = false;
    let inflight = false;
    let failures = 0;

    async function tick() {
      timer = null;
      if (!running || inflight) return;
      if (document.hidden) return; // resumed by visibilitychange
      inflight = true;
      try {
        const { stop, changed } = await fn();
        failures = 0;
        if (stop) { running = false; return; }
        delay = changed ? minDelay : Math.min(maxDelay, delay * factor);
      } catch (err) {
        failures += 1;
        if (err.status === 404 || failures >= maxFailures) {
          running = false;
          if (onFail) onFail(err);
          return;
        }
        delay = Math.min(maxDelay, delay * 2);
      } finally {
        inflight = false;
      }
      if (running) timer = setTimeout(tick, delay);
    }

    function onVisibility() {
      if (!document.hidden && running && !inflight) {
        clearTimeout(timer);
        tick();
      }
    }
    document.addEventListener('visibilitychange', onVisibility);

    return {
      start({ immediate = false } = {}) {
        running = true; delay = minDelay; failures = 0;
        clearTimeout(timer);
        timer = setTimeout(tick, immediate ? 0 : minDelay);
      },
      stop() { running = false; clearTimeout(timer); timer = null; },
      get running() { return running; },
    };
  }

  const reportUrl = (learnerId) => `/dashboard.html?learner=${encodeURIComponent(learnerId)}`;

  // ── Elements ────────────────────────────────────────────────────────────
  const form = $('transcribe-form');
  const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
  const inputUrl = $('input-url');
  const inputLearner = $('input-learner-id');
  const inputTutor = $('input-tutor-id');
  const inputDate = $('input-session-date');
  const batchUrls = $('batch-urls');
  const autoSpeakers = $('auto-speakers');
  const numSpeakers = $('num-speakers');
  const rangePreprocess = $('range-preprocess');
  const formError = $('form-error');
  const btnSubmit = $('btn-submit');

  const SAMPLE_AUDIO_URL = 'https://d5cn6xo304j6a.cloudfront.net/Session-Recordings/2026/Sep/21/6aade741b42d9b6472aac626/directory1/directory2/93ae7c24bf441383d167cab71c583612_oURY2NvN_0.mp4';
  const PREPROCESS_LABELS = ['Off', 'Sanitize', 'Filter', 'Denoise', 'Normalize'];
  const STATES = ['state-idle', 'state-processing', 'state-error', 'state-batch', 'state-results'];

  let activeTab = 'url';
  let loadedFileBase64 = null;
  let currentResult = null;
  let currentJobId = null;
  let speakerFilter = 'ALL';

  function showState(id) {
    STATES.forEach((s) => $(s).classList.toggle('hidden', s !== id));
  }

  function showFormError(msg) {
    formError.textContent = msg;
    formError.classList.toggle('hidden', !msg);
  }

  // ── Health (every 30 s, skipped while hidden) ───────────────────────────
  async function checkHealth() {
    if (document.hidden) return;
    const badge = $('health-badge');
    try {
      const h = await api('/health');
      badge.className = 'badge is-online';
      $('health-text').textContent = h.pipeline_loaded === false ? 'Loading models' : 'Online';
    } catch (_) {
      badge.className = 'badge is-offline';
      $('health-text').textContent = 'Offline';
    }
  }
  checkHealth();
  setInterval(checkHealth, 30000);
  document.addEventListener('visibilitychange', checkHealth);

  // ── Tabs ────────────────────────────────────────────────────────────────
  function selectTab(name) {
    activeTab = name;
    tabs.forEach((t) => {
      const on = t.dataset.tab === name;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      $(`panel-${t.dataset.tab}`).classList.toggle('hidden', !on);
    });
    const batch = name === 'batch';
    document.querySelectorAll('[data-optional]').forEach((n) => { n.textContent = batch ? '(required)' : '(optional)'; });
    $('session-date-field').classList.toggle('hidden', batch);
    $('kg-help').textContent = batch
      ? 'Every recording is analysed and added to this learner\'s progress report.'
      : 'Add both IDs to analyse the session and update the learner\'s progress report.';
    $('btn-submit-label').textContent = batch ? 'Queue all sessions' : 'Start analysis';
    showFormError('');
  }
  tabs.forEach((t, i) => {
    t.addEventListener('click', () => selectTab(t.dataset.tab));
    t.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      next.focus();
      selectTab(next.dataset.tab);
    });
  });

  $('btn-sample-url').addEventListener('click', () => { inputUrl.value = SAMPLE_AUDIO_URL; });

  // ── File upload ─────────────────────────────────────────────────────────
  const dropzone = $('dropzone');
  const fileInput = $('file-input');
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  ['dragenter', 'dragover'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.add('drag-over');
  }));
  ['dragleave', 'drop'].forEach((ev) => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.remove('drag-over');
  }));
  dropzone.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
  fileInput.addEventListener('change', (e) => { if (e.target.files[0]) handleFile(e.target.files[0]); });
  $('btn-remove-file').addEventListener('click', resetFile);

  function handleFile(file) {
    if (file.size > 150 * 1024 * 1024) {
      showFormError('That file is larger than 150 MB. Upload it somewhere and use its URL instead.');
      return;
    }
    showFormError('');
    $('file-name').textContent = file.name;
    $('file-size').textContent = fmt.bytes(file.size);
    const reader = new FileReader();
    reader.onload = (e) => {
      $('audio-player').src = e.target.result;
      loadedFileBase64 = String(e.target.result).split(',')[1];
      dropzone.classList.add('hidden');
      $('file-preview').classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  function resetFile() {
    fileInput.value = '';
    loadedFileBase64 = null;
    $('audio-player').removeAttribute('src');
    $('file-preview').classList.add('hidden');
    dropzone.classList.remove('hidden');
  }

  // ── Options ─────────────────────────────────────────────────────────────
  autoSpeakers.addEventListener('change', () => {
    numSpeakers.disabled = autoSpeakers.checked;
    if (autoSpeakers.checked) numSpeakers.value = '';
  });
  rangePreprocess.addEventListener('input', () => {
    $('preprocess-val').textContent = PREPROCESS_LABELS[Number(rangePreprocess.value)] || rangePreprocess.value;
  });

  function transcriptionOptions() {
    return {
      num_speakers: autoSpeakers.checked ? null : (parseInt(numSpeakers.value, 10) || null),
      language: $('select-language').value || null,
      translate: $('check-translate').checked,
      preprocess: Number(rangePreprocess.value),
      prompt: $('input-prompt').value.trim() || null,
    };
  }

  // ── Submit ──────────────────────────────────────────────────────────────
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    showFormError('');
    if (activeTab === 'batch') return submitBatch();
    return submitSingle();
  });

  async function submitSingle() {
    const learnerId = inputLearner.value.trim();
    const tutorId = inputTutor.value.trim();
    const payload = { ...transcriptionOptions(), file_url: null, file_string: null };

    if (activeTab === 'url') {
      payload.file_url = inputUrl.value.trim();
      if (!/^https?:\/\//i.test(payload.file_url)) { showFormError('Enter a recording URL starting with http:// or https://.'); inputUrl.focus(); return; }
    } else {
      if (!loadedFileBase64) { showFormError('Choose an audio or video file first.'); return; }
      payload.file_string = loadedFileBase64;
    }
    if (Boolean(learnerId) !== Boolean(tutorId)) {
      showFormError('Add both a learner ID and a tutor ID to update a progress report (or leave both empty).');
      return;
    }
    if (learnerId) { payload.learner_id = learnerId; payload.tutor_id = tutorId; }
    if (inputDate.value) payload.session_date = inputDate.value;

    btnSubmit.disabled = true;
    try {
      const job = await api('/transcribe', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      saveHistory({ id: job.job_id, status: job.status, learner: learnerId || null, at: Date.now() });
      startJob(job.job_id, learnerId || null);
    } catch (err) {
      showFormError(err.message);
    } finally {
      btnSubmit.disabled = false;
    }
  }

  function parseBatchLines(text) {
    const urls = [], dates = [], errors = [];
    text.split('\n').map((l) => l.trim()).filter(Boolean).forEach((line, i) => {
      const m = line.match(/^([^\s,]+)\s*(?:,?\s*(\d{4}-\d{2}-\d{2}))?\s*,?\s*$/);
      if (!m || !/^https?:\/\//i.test(m[1])) { errors.push(`Line ${i + 1} is not a valid URL.`); return; }
      urls.push(m[1]);
      dates.push(m[2] || null);
    });
    return { urls, dates, errors };
  }

  async function submitBatch() {
    const learnerId = inputLearner.value.trim();
    const tutorId = inputTutor.value.trim();
    if (!learnerId || !tutorId) { showFormError('Batch analysis needs a learner ID and a tutor ID.'); (learnerId ? inputTutor : inputLearner).focus(); return; }
    const { urls, dates, errors } = parseBatchLines(batchUrls.value);
    if (errors.length) { showFormError(errors.slice(0, 3).join(' ')); batchUrls.focus(); return; }
    if (!urls.length) { showFormError('Add at least one recording URL.'); batchUrls.focus(); return; }
    if (urls.length > 20) { showFormError('A batch can have at most 20 recordings.'); return; }

    const opts = transcriptionOptions();
    btnSubmit.disabled = true;
    try {
      const batch = await api('/transcribe/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...opts,
          file_urls: urls,
          session_dates: dates.some(Boolean) ? dates : null,
          learner_id: learnerId,
          tutor_id: tutorId,
        }),
      });
      session.set('ey_active_batch', batch.batch_id);
      startBatch(batch);
    } catch (err) {
      showFormError(err.message);
    } finally {
      btnSubmit.disabled = false;
    }
  }

  // ── Single job lifecycle ────────────────────────────────────────────────
  let jobPoller = null;
  let timerInterval = null;

  function setStep(step, state, msg) {
    const li = document.querySelector(`#job-steps [data-step="${step}"]`);
    li.dataset.state = state;
    if (msg) li.querySelector('[data-step-msg]').textContent = msg;
  }

  function startTimer() {
    clearInterval(timerInterval);
    const start = Date.now();
    const tick = () => { $('elapsed-timer').textContent = fmt.time((Date.now() - start) / 1000); };
    tick();
    timerInterval = setInterval(tick, 1000);
  }

  function startJob(jobId, learnerId) {
    if (jobPoller) jobPoller.stop();
    currentJobId = jobId;
    currentResult = null;
    $('active-job-id').textContent = jobId.slice(0, 8);
    $('processing-title').textContent = 'Working on it';
    setStep('transcribe', 'pending', 'Waiting for a free worker…');
    setStep('analyse', learnerId ? 'pending' : 'skipped', learnerId ? 'Starts after transcription' : 'No learner ID — transcript only');
    showState('state-processing');
    startTimer();

    let lastKey = '';
    jobPoller = createPoller(async () => {
      const job = await api(`/jobs/${jobId}?include_result=false`);
      const key = `${job.status}|${job.kg_status}`;
      const changed = key !== lastKey;
      lastKey = key;
      saveHistory({ id: jobId, status: job.status, learner: learnerId });

      if (job.status === 'error') {
        clearInterval(timerInterval);
        showError(job.error || 'Transcription failed.');
        return { stop: true };
      }
      if (job.status === 'queued') setStep('transcribe', 'pending', 'Waiting for a free worker…');
      if (job.status === 'processing') setStep('transcribe', 'active', 'Transcribing speech and separating speakers…');

      if (job.status === 'done' && !currentResult) {
        const full = await api(`/jobs/${jobId}`);
        currentResult = full.result;
        clearInterval(timerInterval);
        displayResults(full.result);
      }
      if (job.status === 'done') renderKgBanner(job, learnerId);

      const kgDone = ['not_requested', 'done', 'error', 'skipped'].includes(job.kg_status);
      return { stop: job.status === 'done' && kgDone, changed };
    }, {
      minDelay: 2000,
      maxDelay: 10000,
      onFail: (err) => {
        clearInterval(timerInterval);
        showError(err.status === 404
          ? 'This job is no longer on the server (it may have restarted). Please submit the recording again.'
          : 'Lost contact with the server. Check your connection and try again.');
      },
    });
    jobPoller.start({ immediate: true });
  }

  function renderKgBanner(job, learnerId) {
    const banner = $('kg-banner');
    banner.replaceChildren();
    if (job.kg_status === 'not_requested' || job.kg_status === 'skipped') { banner.className = 'alert hidden'; return; }
    if (job.kg_status === 'done') {
      banner.className = 'alert alert-success';
      banner.append(icon('check-circle'),
        el('div', {}, el('strong', { text: 'Progress report updated' }),
          el('p', { text: `This session${job.session_date ? ` (${fmt.date(job.session_date)})` : ''} is now part of ${learnerId}'s report.` })),
        el('a', { class: 'btn btn-primary btn-sm', href: reportUrl(learnerId) }, 'View report'));
    } else if (job.kg_status === 'error') {
      banner.className = 'alert alert-danger';
      banner.append(icon('alert'),
        el('div', {}, el('strong', { text: 'Transcript ready, but the progress analysis failed' }),
          el('p', { text: job.kg_error || 'Unknown error' })));
    } else {
      banner.className = 'alert alert-info';
      banner.append(el('div', { class: 'spinner', 'aria-hidden': 'true' }),
        el('div', {}, el('strong', { text: 'Analysing for the progress report…' }),
          el('p', { text: 'Usually under a minute. The transcript below is ready to use.' })));
    }
  }

  function showError(msg) {
    $('error-message-text').textContent = msg;
    showState('state-error');
  }

  $('btn-retry').addEventListener('click', () => showState('state-idle'));
  $('btn-copy-job-id').addEventListener('click', () => { if (currentJobId) copy(currentJobId, 'Job ID copied'); });

  // ── Batch lifecycle ─────────────────────────────────────────────────────
  let batchPoller = null;
  const STATUS_LABEL = {
    queued: ['Queued', ''],
    processing: ['Transcribing', 'badge-primary'],
    analysing: ['Analysing', 'badge-primary'],
    done: ['Done', 'badge-success'],
    error: ['Failed', 'badge-danger'],
  };

  function jobDisplayStatus(j) {
    if (j.status === 'error') return 'error';
    if (j.status !== 'done') return j.status;
    if (j.kg_status === 'error') return 'error';
    if (['done', 'not_requested', 'skipped'].includes(j.kg_status)) return 'done';
    return 'analysing';
  }

  function renderBatch(batch) {
    $('batch-learner').textContent = batch.learner_id;
    const finished = batch.jobs.filter((j) => ['done', 'error'].includes(jobDisplayStatus(j))).length;
    const failed = batch.jobs.filter((j) => jobDisplayStatus(j) === 'error').length;
    const pct = batch.total ? Math.round((finished / batch.total) * 100) : 0;
    $('batch-summary').textContent = `${finished} of ${batch.total} finished${failed ? ` · ${failed} failed` : ''}`;
    $('batch-progress-fill').style.width = `${pct}%`;
    $('batch-progressbar').setAttribute('aria-valuenow', String(pct));

    const badge = $('batch-badge');
    badge.textContent = batch.complete ? (failed ? 'Finished with errors' : 'Complete') : 'Running';
    badge.className = `badge ${batch.complete ? (failed ? 'badge-warning' : 'badge-success') : 'badge-primary'}`;

    const list = $('batch-jobs-list');
    batch.jobs.forEach((j, idx) => {
      let row = list.querySelector(`[data-job="${CSS.escape(j.job_id)}"]`);
      if (!row) {
        row = el('li', { 'data-job': j.job_id },
          el('span', { class: 'job-index', text: String(idx + 1) }),
          el('div', { class: 'job-main' },
            el('span', { class: 'job-url', title: j.file_url || '', text: (j.file_url || '').split('/').pop() || j.job_id }),
            el('span', { class: 'job-meta', text: j.session_date ? `Session date ${fmt.date(j.session_date)}` : '' }),
            el('span', { class: 'job-error hidden' })),
          el('span', { class: 'badge' }));
        list.append(row);
      }
      const st = jobDisplayStatus(j);
      const [label, cls] = STATUS_LABEL[st] || [st, ''];
      const b = row.querySelector('.badge');
      b.textContent = label;
      b.className = `badge ${cls}`;
      const errEl = row.querySelector('.job-error');
      errEl.textContent = st === 'error' ? (j.error || 'Failed') : '';
      errEl.classList.toggle('hidden', st !== 'error');
    });

    $('batch-footer').classList.toggle('hidden', !batch.complete);
    $('batch-report-link').href = reportUrl(batch.learner_id);
  }

  function startBatch(batch) {
    if (batchPoller) batchPoller.stop();
    $('batch-jobs-list').replaceChildren();
    showState('state-batch');
    renderBatch(batch);

    let lastKey = '';
    batchPoller = createPoller(async () => {
      const b = await api(`/batches/${batch.batch_id}`);
      renderBatch(b);
      const key = b.jobs.map((j) => `${j.status}${j.kg_status}`).join();
      const changed = key !== lastKey;
      lastKey = key;
      if (b.complete) {
        session.remove('ey_active_batch');
        toast(b.kg_error || b.error ? 'Batch finished with some errors' : 'Batch complete — progress report updated');
      }
      return { stop: b.complete, changed };
    }, {
      minDelay: 3000,
      maxDelay: 15000,
      onFail: (err) => {
        session.remove('ey_active_batch');
        $('batch-summary').textContent = err.status === 404
          ? 'This batch is no longer on the server (it may have restarted).'
          : 'Lost contact with the server — refresh the page to check again.';
        $('batch-badge').textContent = 'Unknown';
        $('batch-badge').className = 'badge badge-warning';
      },
    });
    batchPoller.start();
  }

  $('btn-batch-dismiss').addEventListener('click', () => {
    if (batchPoller) batchPoller.stop();
    session.remove('ey_active_batch');
    showState('state-idle');
  });

  // Resume a batch after a page reload
  (async () => {
    const id = session.get('ey_active_batch');
    if (!id) return;
    try {
      selectTab('batch');
      startBatch(await api(`/batches/${id}`));
    } catch (_) {
      session.remove('ey_active_batch');
      showState('state-idle');
    }
  })();

  // ── Results ─────────────────────────────────────────────────────────────
  function displayResults(result) {
    showState('state-results');
    const segments = result.segments || [];
    const speakers = [...new Set(segments.map((s) => s.speaker).filter(Boolean))].sort();
    const total = segments.reduce((m, s) => Math.max(m, s.end || 0), 0);

    $('metric-duration').textContent = fmt.duration(total);
    $('metric-speakers').textContent = String(speakers.length || '–');
    $('metric-segments').textContent = String(segments.length);
    $('metric-language').textContent = (result.language || 'auto').toUpperCase();

    speakerFilter = 'ALL';
    const chips = $('speaker-chips');
    chips.replaceChildren();
    const makeChip = (value, label) => el('button', {
      type: 'button', class: 'chip', 'aria-pressed': String(value === speakerFilter),
      onclick: (e) => {
        speakerFilter = value;
        chips.querySelectorAll('.chip').forEach((c) => c.setAttribute('aria-pressed', String(c === e.currentTarget)));
        renderTranscript();
      },
    }, label);
    chips.append(makeChip('ALL', `All (${speakers.length})`), ...speakers.map((s) => makeChip(s, fmt.speaker(s))));

    renderTranscript();
  }

  function highlight(text, query) {
    if (!query) return [text];
    const parts = [];
    const lower = text.toLowerCase();
    let i = 0;
    for (;;) {
      const j = lower.indexOf(query, i);
      if (j === -1) break;
      if (j > i) parts.push(text.slice(i, j));
      parts.push(el('mark', { text: text.slice(j, j + query.length) }));
      i = j + query.length;
    }
    parts.push(text.slice(i));
    return parts;
  }

  function renderTranscript() {
    if (!currentResult) return;
    const query = $('search-transcript').value.trim().toLowerCase();
    const container = $('transcript-container');
    const frag = document.createDocumentFragment();
    let count = 0;
    (currentResult.segments || []).forEach((seg) => {
      if (speakerFilter !== 'ALL' && seg.speaker !== speakerFilter) return;
      const text = seg.text || '';
      if (query && !text.toLowerCase().includes(query)) return;
      count += 1;
      const spk = parseInt(String(seg.speaker || '0').replace(/\D/g, ''), 10) % 6 || 0;
      const time = `${fmt.time(seg.start)} – ${fmt.time(seg.end)}`;
      frag.append(el('li', {
        class: 'segment', tabindex: '0', title: 'Click to copy',
        onclick: () => copy(`[${fmt.speaker(seg.speaker)} ${time}] ${text}`, 'Segment copied'),
        onkeydown: (e) => { if (e.key === 'Enter') e.currentTarget.click(); },
      },
        el('div', { class: 'segment-meta' },
          el('span', { class: `speaker spk-${spk}`, text: fmt.speaker(seg.speaker) }),
          el('span', { class: 'segment-time', text: time })),
        el('p', { class: 'segment-text' }, highlight(text, query))));
    });
    container.replaceChildren(frag);
    $('visible-count').textContent = String(count);
  }

  let searchTimer = null;
  $('search-transcript').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(renderTranscript, 120);
  });

  // ── Exports ─────────────────────────────────────────────────────────────
  function download(content, name, type) {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = el('a', { href: url, download: name });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const segs = () => (currentResult && currentResult.segments) || [];
  const base = () => `transcript_${(currentJobId || 'export').slice(0, 8)}`;

  $('btn-copy-text').addEventListener('click', () => {
    copy(segs().map((s) => `[${fmt.speaker(s.speaker)} ${fmt.time(s.start)}] ${s.text}`).join('\n'), 'Transcript copied');
  });
  $('btn-export-txt').addEventListener('click', () => {
    download(segs().map((s) => `[${fmt.speaker(s.speaker)} ${fmt.time(s.start)} – ${fmt.time(s.end)}]\n${s.text}\n`).join('\n'), `${base()}.txt`, 'text/plain');
  });
  $('btn-export-srt').addEventListener('click', () => {
    download(segs().map((s, i) => `${i + 1}\n${fmt.srt(s.start)} --> ${fmt.srt(s.end)}\n[${fmt.speaker(s.speaker)}] ${s.text}\n`).join('\n'), `${base()}.srt`, 'text/plain');
  });
  $('btn-export-json').addEventListener('click', () => {
    download(JSON.stringify(currentResult, null, 2), `${base()}.json`, 'application/json');
  });

  // ── History (local to this browser) ─────────────────────────────────────
  const HISTORY_KEY = 'whisper_job_history';

  function saveHistory(item) {
    let list = storage.get(HISTORY_KEY, []);
    const idx = list.findIndex((h) => h.id === item.id);
    if (idx >= 0) list[idx] = { ...list[idx], ...item };
    else list.unshift({ at: Date.now(), ...item });
    list = list.slice(0, 20);
    storage.set(HISTORY_KEY, list);
    $('history-count').textContent = String(list.length);
  }
  $('history-count').textContent = String(storage.get(HISTORY_KEY, []).length);

  function renderHistory() {
    const list = storage.get(HISTORY_KEY, []);
    const ul = $('history-list');
    ul.replaceChildren();
    if (!list.length) { ul.append(el('li', { class: 'history-empty', text: 'No jobs yet.' })); return; }
    list.forEach((h) => {
      const [label, cls] = STATUS_LABEL[h.status] || [h.status, ''];
      ul.append(el('li', {},
        el('div', { class: 'history-info' },
          el('code', { text: h.id }),
          el('span', { class: 'help', text: `${h.learner ? `${h.learner} · ` : ''}${h.at ? new Date(h.at).toLocaleString() : ''}` })),
        el('span', { class: `badge ${cls}`, text: label }),
        el('button', {
          type: 'button', class: 'btn btn-secondary btn-sm',
          onclick: () => { closeHistory(); startJob(h.id, h.learner || null); },
        }, 'Open')));
    });
  }

  const modal = $('history-modal');
  let lastFocus = null;
  function openHistory() {
    lastFocus = document.activeElement;
    renderHistory();
    modal.classList.remove('hidden');
    $('btn-close-history').focus();
  }
  function closeHistory() {
    modal.classList.add('hidden');
    if (lastFocus) lastFocus.focus();
  }
  $('btn-history').addEventListener('click', openHistory);
  $('btn-close-history').addEventListener('click', closeHistory);
  modal.addEventListener('click', (e) => { if (e.target === modal) closeHistory(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !modal.classList.contains('hidden')) closeHistory(); });
  $('btn-clear-history').addEventListener('click', () => {
    storage.remove(HISTORY_KEY);
    $('history-count').textContent = '0';
    renderHistory();
  });
})();
