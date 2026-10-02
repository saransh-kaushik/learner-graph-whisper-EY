/**
 * Whisper Diarization Studio — Client JS Application
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── Elements Reference ──────────────────────────────────────────────────
  const healthBadge = document.getElementById('health-badge');
  const healthText = document.getElementById('health-text');
  
  // Tabs & Form
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const transcribeForm = document.getElementById('transcribe-form');
  const inputUrl = document.getElementById('input-url');
  const btnSampleUrl = document.getElementById('btn-sample-url');
  
  // File Dropzone
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const filePreview = document.getElementById('file-preview');
  const fileName = document.getElementById('file-name');
  const fileSize = document.getElementById('file-size');
  const audioPlayer = document.getElementById('audio-player');
  const btnRemoveFile = document.getElementById('btn-remove-file');

  // Options
  const autoSpeakersCheck = document.getElementById('auto-speakers');
  const numSpeakersInput = document.getElementById('num-speakers');
  const selectLanguage = document.getElementById('select-language');
  const checkTranslate = document.getElementById('check-translate');
  const rangePreprocess = document.getElementById('range-preprocess');
  const preprocessVal = document.getElementById('preprocess-val');
  const inputPrompt = document.getElementById('input-prompt');
  const inputLearnerId = document.getElementById('input-learner-id');
  const inputTutorId = document.getElementById('input-tutor-id');
  const btnSubmit = document.getElementById('btn-submit');

  // States & Panels
  const stateIdle = document.getElementById('state-idle');
  const stateProcessing = document.getElementById('state-processing');
  const stateError = document.getElementById('state-error');
  const stateResults = document.getElementById('state-results');

  // Processing elements
  const jobStatusBadge = document.getElementById('job-status-badge');
  const activeJobId = document.getElementById('active-job-id');
  const btnCopyJobId = document.getElementById('btn-copy-job-id');
  const elapsedTimer = document.getElementById('elapsed-timer');
  const processingMsg = document.getElementById('processing-msg');
  const curlCode = document.getElementById('curl-code');
  const btnCopyCurl = document.getElementById('btn-copy-curl');

  // Error elements
  const errorMessageText = document.getElementById('error-message-text');
  const errorTraceback = document.getElementById('error-traceback');
  const btnRetry = document.getElementById('btn-retry');

  // Metrics & Results
  const metricDuration = document.getElementById('metric-duration');
  const metricSpeakers = document.getElementById('metric-speakers');
  const metricSegments = document.getElementById('metric-segments');
  const metricLanguage = document.getElementById('metric-language');
  const speakerChips = document.getElementById('speaker-chips');
  const searchTranscript = document.getElementById('search-transcript');
  const btnClearSearch = document.getElementById('btn-clear-search');
  const visibleCount = document.getElementById('visible-count');
  const transcriptContainer = document.getElementById('transcript-container');

  // Export buttons
  const btnCopyText = document.getElementById('btn-copy-text');
  const btnExportTxt = document.getElementById('btn-export-txt');
  const btnExportSrt = document.getElementById('btn-export-srt');
  const btnExportJson = document.getElementById('btn-export-json');

  // History Modal
  const btnHistory = document.getElementById('btn-history');
  const historyCount = document.getElementById('history-count');
  const historyModal = document.getElementById('history-modal');
  const btnCloseHistory = document.getElementById('btn-close-history');
  const historyList = document.getElementById('history-list');
  const btnClearHistory = document.getElementById('btn-clear-history');

  // ── State Variables ─────────────────────────────────────────────────────
  let activeTab = 'url'; // 'url' or 'file'
  let loadedFileBase64 = null;
  let activeJobIdVal = null;
  let pollInterval = null;
  let timerInterval = null;
  let startTime = null;
  let currentResultData = null;
  let activeSpeakerFilter = 'ALL';

  // Sample Audio URL for quick testing
  const SAMPLE_AUDIO_URL = 'https://d5cn6xo304j6a.cloudfront.net/Session-Recordings/2026/Sep/21/6aade741b42d9b6472aac626/directory1/directory2/93ae7c24bf441383d167cab71c583612_oURY2NvN_0.mp4';

  // Preprocess slider descriptions
  const PREPROCESS_LABELS = [
    '0 - Off',
    '1 - Sanitize Audio',
    '2 - High/Low Pass Filter',
    '3 - Spectral Denoise',
    '4 - RMS Normalization'
  ];

  // ── Init App ────────────────────────────────────────────────────────────
  checkHealth();
  setInterval(checkHealth, 10000);
  updateHistoryBadge();
  initBatch();

  // ── Batch Session Queue ──────────────────────────────────────────────────
  function initBatch() {
    const btnBatchSubmit = document.getElementById('btn-batch-submit');
    const batchProgressArea = document.getElementById('batch-progress-area');
    const batchSummary = document.getElementById('batch-summary');
    const batchJobsList = document.getElementById('batch-jobs-list');

    let batchPollInterval = null;

    btnBatchSubmit.addEventListener('click', async () => {
      const learnerId = document.getElementById('batch-learner-id').value.trim();
      const tutorId   = document.getElementById('batch-tutor-id').value.trim();
      const urlsRaw   = document.getElementById('batch-urls').value.trim();

      if (!learnerId || !tutorId) {
        alert('Please enter both Learner ID and Tutor ID for batch processing.');
        return;
      }
      const urls = urlsRaw.split('\n').map(u => u.trim()).filter(u => u.length > 0);
      if (urls.length === 0) {
        alert('Please enter at least one URL.');
        return;
      }
      if (urls.length > 20) {
        alert('Maximum 20 URLs per batch.');
        return;
      }

      btnBatchSubmit.disabled = true;
      btnBatchSubmit.textContent = 'Queueing...';

      try {
        const res = await fetch('/transcribe/batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ file_urls: urls, learner_id: learnerId, tutor_id: tutorId })
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Batch submit failed.');
        }

        const batch = await res.json();
        batchProgressArea.classList.remove('hidden');
        renderBatchJobs(batch, batchJobsList);
        updateBatchSummary(batch, batchSummary);

        // Poll batch status
        if (batchPollInterval) clearInterval(batchPollInterval);
        batchPollInterval = setInterval(async () => {
          const r = await fetch(`/batches/${batch.batch_id}`);
          const updated = await r.json();
          renderBatchJobs(updated, batchJobsList);
          updateBatchSummary(updated, batchSummary);

          if (updated.queued === 0 && updated.processing === 0) {
            clearInterval(batchPollInterval);
            btnBatchSubmit.disabled = false;
            btnBatchSubmit.textContent = '🚀 Queue All Sessions';
            showToast(`Batch complete! ${updated.done} done, ${updated.error} errors.`);
          }
        }, 2000);

      } catch (err) {
        alert(err.message);
        btnBatchSubmit.disabled = false;
        btnBatchSubmit.textContent = '🚀 Queue All Sessions';
      }
    });
  }

  function renderBatchJobs(batch, container) {
    container.innerHTML = '';
    batch.job_ids.forEach((jid, idx) => {
      const statusColors = { queued: '#f59e0b', processing: '#6366f1', done: '#10b981', error: '#ef4444' };
      // Look up each job's status from per-job endpoint (cached in batch counts for now)
      const jobEl = document.createElement('div');
      jobEl.style.cssText = 'display:flex;justify-content:space-between;align-items:center;padding:0.5rem 0.75rem;background:rgba(255,255,255,0.04);border-radius:6px;font-size:0.8rem;';
      jobEl.innerHTML = `
        <span style="font-family:monospace;color:#94a3b8;">Session ${idx + 1}</span>
        <a href="#" onclick="navigator.clipboard.writeText('${jid}');return false;" style="color:#6366f1;font-family:monospace;font-size:0.72rem;">${jid.slice(0, 8)}…</a>
        <span id="job-status-${jid}" style="padding:2px 8px;border-radius:10px;background:rgba(245,158,11,0.2);color:#f59e0b;font-weight:600;">QUEUED</span>
      `;
      container.appendChild(jobEl);

      // Poll individual job status
      (async () => {
        let done = false;
        while (!done) {
          await new Promise(r => setTimeout(r, 2500));
          try {
            const r = await fetch(`/jobs/${jid}`);
            const j = await r.json();
            const el = document.getElementById(`job-status-${jid}`);
            if (el) {
              const colors = { queued: '#f59e0b', processing: '#818cf8', done: '#34d399', error: '#f87171' };
              const bgs = { queued: 'rgba(245,158,11,0.2)', processing: 'rgba(99,102,241,0.2)', done: 'rgba(16,185,129,0.2)', error: 'rgba(239,68,68,0.2)' };
              el.textContent = j.status.toUpperCase();
              el.style.color = colors[j.status] || '#94a3b8';
              el.style.background = bgs[j.status] || 'transparent';
            }
            if (j.status === 'done' || j.status === 'error') done = true;
          } catch(e) { done = true; }
        }
      })();
    });
  }

  function updateBatchSummary(batch, el) {
    el.textContent = `${batch.done} done · ${batch.processing} processing · ${batch.queued} queued · ${batch.error} errors`;
  }

  // ── Health Check ────────────────────────────────────────────────────────
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        healthBadge.className = 'badge status-online';
        healthText.textContent = 'API Online';
      } else {
        throw new Error('Health status error');
      }
    } catch (e) {
      healthBadge.className = 'badge status-checking';
      healthText.textContent = 'Offline / Connecting';
    }
  }

  // ── Tab Switching ───────────────────────────────────────────────────────
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      
      btn.classList.add('active');
      activeTab = btn.getAttribute('data-tab');
      document.getElementById(`tab-content-${activeTab}`).classList.add('active');
    });
  });

  // Sample URL button
  btnSampleUrl.addEventListener('click', () => {
    inputUrl.value = SAMPLE_AUDIO_URL;
  });

  // ── File Drag and Drop ──────────────────────────────────────────────────
  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileSelect(files[0]);
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
  });

  btnRemoveFile.addEventListener('click', (e) => {
    e.stopPropagation();
    resetFileInput();
  });

  function handleFileSelect(file) {
    if (file.size > 150 * 1024 * 1024) {
      alert('File size exceeds 150MB limit for browser base64 upload.');
      return;
    }

    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      audioPlayer.src = dataUrl;
      // Extract pure base64 (after 'data:*/*;base64,')
      loadedFileBase64 = dataUrl.split(',')[1];
      dropzone.classList.add('hidden');
      filePreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  function resetFileInput() {
    fileInput.value = '';
    loadedFileBase64 = null;
    audioPlayer.src = '';
    filePreview.classList.add('hidden');
    dropzone.classList.remove('hidden');
  }

  // ── Configuration Options Controls ─────────────────────────────────────
  autoSpeakersCheck.addEventListener('change', (e) => {
    numSpeakersInput.disabled = e.target.checked;
    if (e.target.checked) numSpeakersInput.value = '';
  });

  rangePreprocess.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    preprocessVal.textContent = PREPROCESS_LABELS[val] || `${val}`;
  });

  // ── Form Submission & Job Lifecycle ────────────────────────────────────
  transcribeForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    let fileUrl = null;
    let fileString = null;

    if (activeTab === 'url') {
      fileUrl = inputUrl.value.trim();
      if (!fileUrl) {
        alert('Please enter an audio or video URL.');
        return;
      }
    } else {
      if (!loadedFileBase64) {
        alert('Please select or drop an audio file.');
        return;
      }
      fileString = loadedFileBase64;
    }

    // Build payload
    const payload = {
      file_url: fileUrl,
      file_string: fileString,
      num_speakers: autoSpeakersCheck.checked ? null : parseInt(numSpeakersInput.value, 10) || null,
      language: selectLanguage.value || null,
      translate: checkTranslate.checked,
      preprocess: parseInt(rangePreprocess.value, 10),
      prompt: inputPrompt.value.trim() || null,
      learner_id: inputLearnerId.value.trim() || null,
      tutor_id: inputTutorId.value.trim() || null
    };

    btnSubmit.disabled = true;
    showState('processing');
    setProcessingStatus('queued', 'Submitting job request to server...');

    try {
      const response = await fetch('/transcribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to submit transcription job.');
      }

      const jobData = await response.json();
      activeJobIdVal = jobData.job_id;

      // Update UI with Job ID & curl snippet
      activeJobId.textContent = activeJobIdVal;
      curlCode.textContent = `curl http://localhost:8000/jobs/${activeJobIdVal}`;
      
      saveJobToHistory(activeJobIdVal, 'queued');
      startPollingJob(activeJobIdVal);

    } catch (err) {
      showError(err.message || 'Error starting transcription.');
      btnSubmit.disabled = false;
    }
  });

  // Copy buttons
  btnCopyJobId.addEventListener('click', () => {
    if (activeJobIdVal) copyToClipboard(activeJobIdVal, 'Job ID copied!');
  });
  btnCopyCurl.addEventListener('click', () => {
    copyToClipboard(curlCode.textContent, 'Curl command copied!');
  });

  // ── Polling Loop ────────────────────────────────────────────────────────
  function startPollingJob(jobId) {
    if (pollInterval) clearInterval(pollInterval);
    if (timerInterval) clearInterval(timerInterval);

    startTime = Date.now();
    updateTimer();
    timerInterval = setInterval(updateTimer, 1000);

    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/jobs/${jobId}`);
        if (!res.ok) return;

        const jobStatus = await res.json();
        updateJobState(jobStatus);

      } catch (e) {
        console.error('Polling error:', e);
      }
    }, 1500);
  }

  function updateJobState(jobStatus) {
    const status = jobStatus.status;
    setProcessingStatus(status);

    if (status === 'queued') {
      processingMsg.textContent = 'Job accepted into queue. Waiting for GPU executor worker...';
    } else if (status === 'processing') {
      processingMsg.textContent = 'Transcribing speech & running speaker diarization...';
    } else if (status === 'done') {
      stopPolling();
      saveJobToHistory(jobStatus.job_id, 'done');
      displayResults(jobStatus.result);
      btnSubmit.disabled = false;
    } else if (status === 'error') {
      stopPolling();
      saveJobToHistory(jobStatus.job_id, 'error');
      showError(jobStatus.error || 'Job failed with an error.');
      btnSubmit.disabled = false;
    }
  }

  function stopPolling() {
    if (pollInterval) clearInterval(pollInterval);
    if (timerInterval) clearInterval(timerInterval);
    pollInterval = null;
    timerInterval = null;
  }

  function setProcessingStatus(status) {
    jobStatusBadge.className = `badge status-${status}`;
    jobStatusBadge.textContent = status.toUpperCase();
  }

  function updateTimer() {
    if (!startTime) return;
    const elapsedSec = Math.floor((Date.now() - startTime) / 1000);
    const mins = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
    const secs = String(elapsedSec % 60).padStart(2, '0');
    elapsedTimer.textContent = `${mins}:${secs}`;
  }

  // ── Results Rendering ───────────────────────────────────────────────────
  function displayResults(result) {
    currentResultData = result;
    showState('results');

    const segments = result.segments || [];
    
    // Metrics
    let totalDuration = 0;
    const speakersSet = new Set();

    segments.forEach(seg => {
      if (seg.end > totalDuration) totalDuration = seg.end;
      if (seg.speaker) speakersSet.add(seg.speaker);
    });

    metricDuration.textContent = formatDuration(totalDuration);
    metricSpeakers.textContent = speakersSet.size || 'Auto';
    metricSegments.textContent = segments.length;
    metricLanguage.textContent = (result.language || 'auto').toUpperCase();

    // Render Speaker Chips
    renderSpeakerChips(Array.from(speakersSet).sort());

    // Render Transcript Thread
    renderTranscript(segments);
  }

  function renderSpeakerChips(speakers) {
    speakerChips.innerHTML = `<button class="chip active" data-speaker="ALL">All Speakers (${speakers.length})</button>`;
    
    speakers.forEach((spk, idx) => {
      const btn = document.createElement('button');
      btn.className = `chip`;
      btn.setAttribute('data-speaker', spk);
      btn.textContent = spk.replace('SPEAKER_', 'Spk ');
      btn.addEventListener('click', () => {
        document.querySelectorAll('#speaker-chips .chip').forEach(c => c.classList.remove('active'));
        btn.classList.add('active');
        activeSpeakerFilter = spk;
        filterAndRenderTranscript();
      });
      speakerChips.appendChild(btn);
    });

    // All click listener
    speakerChips.children[0].addEventListener('click', () => {
      document.querySelectorAll('#speaker-chips .chip').forEach(c => c.classList.remove('active'));
      speakerChips.children[0].classList.add('active');
      activeSpeakerFilter = 'ALL';
      filterAndRenderTranscript();
    });
  }

  searchTranscript.addEventListener('input', () => {
    if (searchTranscript.value) {
      btnClearSearch.classList.remove('hidden');
    } else {
      btnClearSearch.classList.add('hidden');
    }
    filterAndRenderTranscript();
  });

  btnClearSearch.addEventListener('click', () => {
    searchTranscript.value = '';
    btnClearSearch.classList.add('hidden');
    filterAndRenderTranscript();
  });

  function filterAndRenderTranscript() {
    if (!currentResultData) return;
    renderTranscript(currentResultData.segments || []);
  }

  function renderTranscript(segments) {
    transcriptContainer.innerHTML = '';
    const searchQuery = searchTranscript.value.trim().toLowerCase();

    let count = 0;

    segments.forEach(seg => {
      // Filter by speaker
      if (activeSpeakerFilter !== 'ALL' && seg.speaker !== activeSpeakerFilter) {
        return;
      }

      // Filter by search query
      if (searchQuery && !seg.text.toLowerCase().includes(searchQuery)) {
        return;
      }

      count++;

      const card = document.createElement('div');
      card.className = 'segment-card';

      // Speaker index for color
      const spkNum = parseInt((seg.speaker || '0').replace('SPEAKER_', ''), 10) % 6;
      const spkLabel = seg.speaker ? seg.speaker.replace('SPEAKER_', 'Speaker ') : 'Speaker ?';

      // Format timestamp
      const timeStr = `${formatTime(seg.start)} → ${formatTime(seg.end)}`;

      // Highlight matching text if searching
      let formattedText = seg.text;
      if (searchQuery) {
        const reg = new RegExp(`(${escapeRegExp(searchQuery)})`, 'gi');
        formattedText = formattedText.replace(reg, '<span class="highlight-match">$1</span>');
      }

      card.innerHTML = `
        <div class="segment-header">
          <span class="speaker-pill spk-${spkNum}">${spkLabel}</span>
          <span class="segment-time font-mono">${timeStr}</span>
        </div>
        <p class="segment-text">${formattedText}</p>
      `;

      // Copy individual segment text on click
      card.addEventListener('click', () => {
        copyToClipboard(`[${spkLabel} ${timeStr}]: ${seg.text}`, 'Segment copied!');
      });

      transcriptContainer.appendChild(card);
    });

    visibleCount.textContent = count;
  }

  // ── Exports Handling ───────────────────────────────────────────────────
  btnCopyText.addEventListener('click', () => {
    if (!currentResultData) return;
    const fullText = (currentResultData.segments || [])
      .map(s => `[${s.speaker || 'SPEAKER'} ${formatTime(s.start)}]: ${s.text}`)
      .join('\n');
    copyToClipboard(fullText, 'Full transcript copied to clipboard!');
  });

  btnExportTxt.addEventListener('click', () => {
    if (!currentResultData) return;
    const txt = (currentResultData.segments || [])
      .map(s => `[${s.speaker || 'SPEAKER'} ${formatTime(s.start)} - ${formatTime(s.end)}]\n${s.text}\n`)
      .join('\n');
    downloadFile(txt, `transcript_${activeJobIdVal || 'export'}.txt`, 'text/plain');
  });

  btnExportSrt.addEventListener('click', () => {
    if (!currentResultData) return;
    const srt = (currentResultData.segments || []).map((s, idx) => {
      return `${idx + 1}\n${formatSrtTime(s.start)} --> ${formatSrtTime(s.end)}\n[${s.speaker || 'Speaker'}] ${s.text}\n`;
    }).join('\n');
    downloadFile(srt, `transcript_${activeJobIdVal || 'export'}.srt`, 'text/plain');
  });

  btnExportJson.addEventListener('click', () => {
    if (!currentResultData) return;
    const jsonStr = JSON.stringify(currentResultData, null, 2);
    downloadFile(jsonStr, `result_${activeJobIdVal || 'export'}.json`, 'application/json');
  });

  // ── UI States Helper ────────────────────────────────────────────────────
  function showState(state) {
    stateIdle.classList.add('hidden');
    stateProcessing.classList.add('hidden');
    stateError.classList.add('hidden');
    stateResults.classList.add('hidden');

    if (state === 'idle') stateIdle.classList.remove('hidden');
    if (state === 'processing') stateProcessing.classList.remove('hidden');
    if (state === 'error') stateError.classList.remove('hidden');
    if (state === 'results') stateResults.classList.remove('hidden');
  }

  function showError(msg) {
    showState('error');
    errorMessageText.textContent = msg;
    errorTraceback.textContent = msg;
  }

  btnRetry.addEventListener('click', () => {
    showState('idle');
  });

  // ── History LocalStorage Helper ─────────────────────────────────────────
  function saveJobToHistory(jobId, status) {
    let history = getHistory();
    const existingIdx = history.findIndex(h => h.id === jobId);
    const item = { id: jobId, status: status, date: new Date().toLocaleTimeString() };

    if (existingIdx >= 0) {
      history[existingIdx] = item;
    } else {
      history.unshift(item);
    }
    if (history.length > 20) history = history.slice(0, 20);
    localStorage.setItem('whisper_job_history', JSON.stringify(history));
    updateHistoryBadge();
  }

  function getHistory() {
    try {
      return JSON.parse(localStorage.getItem('whisper_job_history')) || [];
    } catch (e) {
      return [];
    }
  }

  function updateHistoryBadge() {
    const list = getHistory();
    historyCount.textContent = list.length;
  }

  btnHistory.addEventListener('click', () => {
    renderHistoryModal();
    historyModal.classList.remove('hidden');
  });
  btnCloseHistory.addEventListener('click', () => historyModal.classList.add('hidden'));

  btnClearHistory.addEventListener('click', () => {
    localStorage.removeItem('whisper_job_history');
    renderHistoryModal();
    updateHistoryBadge();
  });

  function renderHistoryModal() {
    const list = getHistory();
    historyList.innerHTML = '';

    if (list.length === 0) {
      historyList.innerHTML = '<p class="empty-history">No past jobs recorded yet.</p>';
      return;
    }

    list.forEach(item => {
      const el = document.createElement('div');
      el.className = 'history-item';
      el.innerHTML = `
        <div class="history-info">
          <code>${item.id}</code>
          <span class="history-date">Submitted: ${item.date}</span>
        </div>
        <div style="display: flex; gap: 0.5rem; align-items: center;">
          <span class="badge status-${item.status}">${item.status}</span>
          <button class="btn btn-xs btn-secondary">Load</button>
        </div>
      `;

      el.querySelector('button').addEventListener('click', () => {
        historyModal.classList.add('hidden');
        activeJobIdVal = item.id;
        showState('processing');
        startPollingJob(item.id);
      });

      historyList.appendChild(el);
    });
  }

  // ── General Utilities ───────────────────────────────────────────────────
  function copyToClipboard(text, toastMsg = 'Copied!') {
    navigator.clipboard.writeText(text).then(() => {
      showToast(toastMsg);
    });
  }

  function showToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'toast-notification';
    toast.style.cssText = `
      position: fixed; bottom: 20px; right: 20px;
      background: #6366f1; color: white; padding: 0.6rem 1.2rem;
      border-radius: 8px; font-size: 0.85rem; font-weight: 600;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3); z-index: 1000;
      animation: fadeIn 0.3s ease;
    `;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 2500);
  }

  function downloadFile(content, fileName, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    a.click();
    URL.revokeObjectURL(url);
  }

  function formatTime(seconds) {
    if (isNaN(seconds)) return '00:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function formatSrtTime(seconds) {
    const date = new Date(seconds * 1000);
    const hh = String(Math.floor(seconds / 3600)).padStart(2, '0');
    const mm = String(date.getUTCMinutes()).padStart(2, '0');
    const ss = String(date.getUTCSeconds()).padStart(2, '0');
    const ms = String(date.getUTCMilliseconds()).padStart(3, '0');
    return `${hh}:${mm}:${ss},${ms}`;
  }

  function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
});
