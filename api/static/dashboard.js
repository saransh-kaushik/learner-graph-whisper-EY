/**
 * Progress report page — renders /kg/learners/{id}/report for the learner.
 * All text from the API is inserted with textContent (see common.js `el`).
 */
(() => {
  'use strict';

  const { $, el, icon, api, storage, fmt } = window.EY;

  const STATES = ['state-empty', 'state-loading', 'state-error', 'state-report'];
  const METRIC_ORDER = ['words_per_turn', 'talk_time_ratio', 'code_switch_ratio', 'speaking_rate_wpm', 'average_words_per_sentence'];
  const TREND = {
    improving: ['Improving', 'badge-success', 'check-circle'],
    declining: ['Needs attention', 'badge-warning', 'alert'],
    stable: ['Steady', '', 'minus'],
    insufficient_data: ['Needs 3+ sessions', '', 'clock'],
  };
  const GRAMMAR_STATUS = {
    persistent: ['Still frequent', 'badge-warning'],
    increasing: ['More frequent', 'badge-danger'],
    new: ['New', 'badge-info'],
    fading: ['Less often', 'badge-success'],
    resolved: ['Not seen recently', 'badge-success'],
  };
  const LEVEL_GUIDE = [
    ['A1–A2', 'Basic, everyday words.'],
    ['B1', 'Common at work — words most colleagues use daily.'],
    ['B2', 'Professional — precise words for meetings, calls and email.'],
    ['C1', 'Advanced — nuanced words that make you sound polished.'],
    ['C2', 'Expert — rare, sophisticated vocabulary.'],
  ];
  const GOAL_STEPS = ['Mentioned', 'Practised', 'Achieved'];
  const GOAL_INDEX = { mentioned: 0, practiced: 1, demonstrated: 2 };
  const KIND_LABEL = { phrase: 'phrase', phrasal_verb: 'phrasal verb', idiom: 'idiom', collocation: 'phrase' };

  const selectLearner = $('learner-select');
  const selectWindow = $('window-select');
  const btnRefresh = $('btn-refresh');
  const btnPrint = $('btn-print');

  let learnerId = null;
  let report = null;
  let charts = [];
  let vtab = 'new';
  let loadToken = 0;

  function showState(id) {
    STATES.forEach((s) => $(s).classList.toggle('hidden', s !== id));
  }

  // ── Learners & URL state ────────────────────────────────────────────────
  const params = new URLSearchParams(location.search);
  if (['3', '5', '10'].includes(params.get('n'))) selectWindow.value = params.get('n');

  async function loadLearners() {
    try {
      const learners = await api('/kg/learners');
      selectLearner.replaceChildren(el('option', { value: '', text: learners.length ? 'Select a learner…' : 'No learners yet' }));
      learners.forEach((l) => {
        const label = `${l.name || l.id} · ${fmt.plural(l.session_count, 'session')}`;
        selectLearner.append(el('option', { value: l.id, text: label }));
      });
      const wanted = params.get('learner');
      if (wanted && learners.some((l) => l.id === wanted)) {
        selectLearner.value = wanted;
        load(wanted);
      } else if (wanted) {
        showError(`No learner with ID "${wanted}" was found.`);
      }
    } catch (err) {
      selectLearner.replaceChildren(el('option', { value: '', text: 'Could not load learners' }));
      showError(err.message);
    }
  }

  function syncUrl() {
    const p = new URLSearchParams();
    if (learnerId) p.set('learner', learnerId);
    if (selectWindow.value !== '5') p.set('n', selectWindow.value);
    history.replaceState(null, '', `${location.pathname}${p.toString() ? `?${p}` : ''}`);
  }

  selectLearner.addEventListener('change', () => {
    if (selectLearner.value) load(selectLearner.value);
    else { learnerId = null; syncUrl(); showState('state-empty'); }
  });
  selectWindow.addEventListener('change', () => { if (learnerId) load(learnerId); else syncUrl(); });
  btnRefresh.addEventListener('click', () => { if (learnerId) load(learnerId, { refresh: true }); });
  btnPrint.addEventListener('click', () => window.print());
  $('btn-retry').addEventListener('click', () => { if (learnerId) load(learnerId); else loadLearners(); });

  async function load(id, { refresh = false } = {}) {
    learnerId = id;
    syncUrl();
    const token = ++loadToken;
    showState('state-loading');
    btnRefresh.disabled = true;
    btnPrint.disabled = true;
    try {
      const q = new URLSearchParams({ n: selectWindow.value });
      if (refresh) q.set('refresh', 'true');
      const data = await api(`/kg/learners/${encodeURIComponent(id)}/report?${q}`);
      if (token !== loadToken) return; // a newer request superseded this one
      report = data;
      render(data);
      showState('state-report');
      btnRefresh.disabled = false;
      btnPrint.disabled = false;
      if ($('tutor-tools').open) loadGraph();
    } catch (err) {
      if (token !== loadToken) return;
      showError(err.status === 404 ? 'This learner was not found.' : err.message);
      btnRefresh.disabled = false;
    }
  }

  function showError(msg) {
    $('error-text').textContent = msg || 'Please try again in a moment.';
    showState('state-error');
  }

  // ── Render ──────────────────────────────────────────────────────────────
  function render(r) {
    const n = r.narrative || {};
    renderHero(r, n);
    renderMetrics(r);
    renderWins(r, n);
    renderFocus(r, n);
    renderPlan(r, n);
    renderVocab(r);
    renderGrammar(r);
    renderStrengths(r);
    renderGoals(r);
    renderTimeline(r);
    const when = r.generated_at ? new Date(r.generated_at).toLocaleString() : fmt.date(r.report_date);
    $('report-foot').textContent = `Generated ${when}. ${
      n.source === 'fallback' ? 'Summary written automatically (AI summary unavailable).'
        : 'Summary written with AI from your session recordings; numbers are measured directly.'}`;
  }

  function renderHero(r, n) {
    const name = r.learner_name || r.learner_id;
    document.title = `${name} · Progress Report · EnglishYaari`;
    $('hero-learner').textContent = `${name} · Progress report`;
    $('hero-headline').textContent = n.headline || 'Your spoken English progress';
    $('hero-summary').textContent = n.summary || '';

    const lv = r.cefr_level || {};
    const box = $('level-box');
    box.replaceChildren();
    if (lv.value) {
      box.append(
        el('div', { class: 'level-top' }, el('span', { class: 'level-code', text: lv.value }), el('span', { class: 'level-name', text: lv.label || '' })),
        el('p', { class: 'level-desc', text: lv.description || '' }),
        el('p', { class: 'level-src', text: lv.source === 'tutor' ? 'Level set by your tutor' : `Estimated from your sessions · ${lv.confidence} confidence` }),
      );
    } else {
      box.append(el('p', { class: 'level-name', text: 'Speaking level' }), el('p', { class: 'level-desc', text: 'Shown after your first analysed session.' }));
    }

    const p = r.period || {};
    const sa = r.sessions_analyzed || { valid: 0, total: 0 };
    const meta = $('hero-meta');
    meta.replaceChildren();
    if (p.start) {
      const range = p.start === p.end ? fmt.date(p.start) : `${fmt.date(p.start, { day: 'numeric', month: 'short' })} – ${fmt.date(p.end)}`;
      meta.append(el('span', {}, icon('calendar', 'icon-sm'), range));
    }
    meta.append(el('span', {}, icon('message', 'icon-sm'), `${fmt.plural(sa.valid, 'session')} analysed`));
    if (p.learner_minutes) meta.append(el('span', {}, icon('mic', 'icon-sm'), `${Math.round(p.learner_minutes)} min of you speaking`));

    const dq = r.data_quality || [];
    const note = $('data-quality');
    note.replaceChildren();
    note.classList.toggle('hidden', !dq.length);
    if (dq.length) {
      note.append(icon('alert'), el('span', {
        text: `${dq.length} of ${sa.total} recordings had no clear speech from you, so ${dq.length === 1 ? 'it was' : 'they were'} left out of this report.`,
      }));
    }
  }

  function formatMetric(m, v) {
    if (v == null) return ['–', ''];
    if (m.unit === '%') return [`${Math.round(v * 100)}`, '%'];
    if (m.unit === 'wpm') return [`${Math.round(v)}`, 'wpm'];
    return [v.toFixed(1), m.unit];
  }

  function renderMetrics(r) {
    charts.forEach((c) => c.destroy());
    charts = [];
    const grid = $('metric-grid');
    grid.replaceChildren();
    const valid = (r.sessions_analyzed || {}).valid || 0;
    $('glance-sub').textContent = valid >= 3
      ? 'Your recent sessions compared with earlier ones in this report.'
      : 'Changes over time appear once you have 3 or more analysed sessions.';

    const metrics = r.metrics || {};
    METRIC_ORDER.filter((k) => metrics[k] && metrics[k].current != null).forEach((key) => {
      const m = metrics[key];
      const [value, unit] = formatMetric(m, m.current);
      const [tLabel, tCls, tIcon] = TREND[m.trend] || TREND.insufficient_data;
      const canvas = el('canvas', { 'aria-hidden': 'true' });
      let change = null;
      if (m.change_pct != null && m.trend !== 'insufficient_data') {
        const dir = m.change_pct > 0 ? 'up' : 'down';
        change = el('span', { class: 'metric-help', text: `${dir} ${Math.abs(m.change_pct).toFixed(0)}% vs your earlier sessions` });
      }
      grid.append(el('div', { class: 'metric' },
        el('span', { class: 'metric-label', text: m.label }),
        el('span', { class: 'metric-value' }, value, unit ? el('small', { class: unit === '%' ? 'tight' : '', text: unit }) : null),
        el('span', { class: `badge trend ${tCls}` }, icon(tIcon), tLabel),
        change,
        el('div', { class: 'metric-spark' }, canvas),
        el('span', { class: 'metric-help', text: m.help })));
      drawSpark(canvas, m);
    });
    if (!grid.children.length) grid.append(el('p', { class: 'muted-empty', text: 'No measurements yet.' }));
  }

  function drawSpark(canvas, m) {
    if (!window.Chart) return;
    const points = (m.history || []).filter((h) => h.value != null);
    if (points.length < 2) { canvas.parentElement.remove(); return; }
    const scale = m.unit === '%' ? 100 : 1;
    charts.push(new window.Chart(canvas, {
      type: 'line',
      data: {
        labels: points.map((h) => fmt.date(h.date, { day: 'numeric', month: 'short' })),
        datasets: [{
          data: points.map((h) => h.value * scale),
          borderColor: '#6d28d9',
          backgroundColor: 'rgba(109, 40, 217, 0.08)',
          fill: true, tension: 0.35, borderWidth: 2, pointRadius: 2.5, pointBackgroundColor: '#6d28d9',
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
          legend: { display: false },
          tooltip: { displayColors: false, callbacks: { label: (ctx) => `${Math.round(ctx.parsed.y * 10) / 10}${m.unit === '%' ? '%' : ` ${m.unit}`}` } },
        },
        scales: { x: { display: false }, y: { display: false } },
      },
    }));
  }

  function renderWins(r, n) {
    const list = $('wins-list');
    list.replaceChildren();
    (n.wins || []).forEach((w) => list.append(el('li', { class: 'win' },
      el('span', { class: 'win-icon' }, icon('check')),
      el('div', {}, el('strong', { text: w.title }), w.detail ? el('p', { text: w.detail }) : null,
        w.evidence ? el('p', { class: 'evidence', text: w.evidence }) : null))));
    if (!list.children.length) {
      list.append(el('li', { class: 'muted-empty', text: 'Improvements show up here after a few sessions. Keep going.' }));
    }

    const improved = (r.grammar || {}).improved || [];
    $('improved-grammar').classList.toggle('hidden', !improved.length);
    $('improved-grammar-tags').replaceChildren(...improved.map((p) => el('span', {
      class: 'badge badge-success', title: p.status === 'resolved' ? 'Not seen in your recent sessions' : 'Happening less often',
    }, icon('check'), p.label)));

    const indep = (r.vocabulary || {}).now_independent || [];
    $('independent-words').classList.toggle('hidden', !indep.length);
    $('independent-words-tags').replaceChildren(...indep.map((w) => el('span', { class: 'badge badge-primary', title: w.meaning || '' }, w.word)));
  }

  function compareRows(you, better) {
    return el('div', { class: 'compare' },
      you ? el('div', { class: 'compare-row compare-you' }, el('span', { class: 'compare-label', text: 'You said' }), el('span', { text: you })) : null,
      better ? el('div', { class: 'compare-row compare-better' }, el('span', { class: 'compare-label', text: 'Better' }), el('span', { text: better })) : null);
  }

  function renderFocus(r, n) {
    const box = $('focus-list');
    box.replaceChildren();
    let items = n.focus_areas || [];
    if (!items.length) {
      items = (r.top_priority_errors || []).slice(0, 2).map((p) => ({
        title: p.label, why_it_matters: '',
        your_sentence: p.examples[0] ? p.examples[0].wrong : '',
        better_sentence: p.examples[0] ? p.examples[0].correct : '',
        tip: p.examples[0] ? p.examples[0].explanation : '',
      }));
    }
    items.forEach((f) => box.append(el('article', { class: 'focus' },
      el('h3', { text: f.title }),
      f.why_it_matters ? el('p', { class: 'why', text: f.why_it_matters }) : null,
      compareRows(f.your_sentence, f.better_sentence),
      f.tip ? el('p', { class: 'tip' }, icon('star', 'icon-sm'), el('span', { text: f.tip })) : null)));
    if (!items.length) box.append(el('p', { class: 'muted-empty', text: 'No recurring mistakes in these sessions. Nice work.' }));
  }

  function renderPlan(r, n) {
    const list = $('plan-list');
    list.replaceChildren();
    const key = (i) => `ey_plan_${r.learner_id}_${r.generated_at || r.report_date}_${i}`;
    (n.practice_plan || []).forEach((t, i) => {
      const done = storage.get(key(i), false);
      const box = el('input', { type: 'checkbox' });
      box.checked = done;
      const li = el('li', { class: `task${done ? ' is-done' : ''}` },
        el('div', { class: 'task-head' }, el('span', { class: 'task-num', 'aria-hidden': 'true' }), el('span', { class: 'task-title', text: t.title })),
        el('div', { class: 'task-meta' },
          el('span', { class: 'badge' }, icon('clock'), `${t.minutes || 5} min`),
          t.situation ? el('span', { class: 'badge badge-primary', text: t.situation }) : null),
        (t.steps || []).length ? el('ol', {}, t.steps.map((s) => el('li', { text: s }))) : null,
        el('label', { class: 'checkbox' }, box, 'Done'));
      box.addEventListener('change', () => {
        storage.set(key(i), box.checked);
        li.classList.toggle('is-done', box.checked);
      });
      list.append(li);
    });
    if (!list.children.length) list.append(el('li', { class: 'muted-empty', text: 'Your practice plan appears after your first analysed session.' }));
    const ms = $('milestone');
    ms.replaceChildren();
    ms.classList.toggle('hidden', !n.next_milestone);
    if (n.next_milestone) ms.append(el('strong', { text: 'Next milestone: ' }), n.next_milestone);
  }

  // ── Vocabulary ──────────────────────────────────────────────────────────
  function levelBadge(w) {
    if (!w.level) return null;
    return el('span', { class: 'badge badge-primary', title: 'Word level' }, w.level_hint ? `${w.level} · ${w.level_hint}` : w.level);
  }

  function wordCard(w, mode) {
    const kind = KIND_LABEL[w.kind];
    const lines = [];
    if (mode === 'new') {
      if (w.your_sentence) lines.push(['You said', w.your_sentence, 'ok']);
      if (w.example) lines.push(['Use it at work', w.example, '']);
    } else if (mode === 'fix') {
      if (w.your_sentence) lines.push(['You said', w.your_sentence, 'bad']);
      if (w.correction) lines.push(['Better', w.correction, 'ok']);
    } else {
      if (w.tutor_sentence) lines.push(['Your tutor said', w.tutor_sentence, '']);
      if (w.example) lines.push(['Use it at work', w.example, '']);
    }
    return el('article', { class: 'word' },
      el('div', { class: 'word-head' },
        el('span', {}, el('span', { class: 'word-term', text: w.word }), kind ? el('span', { class: 'word-kind', text: ` · ${kind}` }) : null),
        levelBadge(w)),
      w.meaning ? el('p', { class: 'word-meaning', text: w.meaning }) : null,
      lines.map(([label, text, cls]) => el('p', { class: `word-line ${cls}` }, el('b', { text: label }), text)));
  }

  const VOCAB_EMPTY = {
    new: 'New words and phrases you use correctly will appear here.',
    fix: 'No misused words in these sessions.',
    try: 'Words your tutor suggests will appear here until you start using them.',
  };

  function renderVocab(r, mode = vtab) {
    const v = r.vocabulary || {};
    const groups = { new: v.new_words || [], fix: v.words_to_fix || [], try: v.words_to_try || [] };
    const totals = v.totals || {};
    document.querySelector('[data-count="new"]').textContent = String(groups.new.length);
    document.querySelector('[data-count="fix"]').textContent = String(groups.fix.length);
    document.querySelector('[data-count="try"]').textContent = String(groups.try.length);
    $('vocab-sub').textContent = totals.new_words
      ? `${fmt.plural(totals.new_words, 'new word')} or phrase${totals.new_words === 1 ? '' : 's'} used correctly in this period.`
      : 'Words and phrases from your sessions, with what they mean and how to use them at work.';

    const panel = $('vpanel');
    panel.replaceChildren();
    const modes = mode === 'all' ? ['new', 'fix', 'try'] : [mode];
    modes.forEach((m) => {
      if (mode === 'all' && groups[m].length) {
        panel.append(el('h3', { class: 'sub-title', style: 'grid-column: 1 / -1', text: { new: 'New', fix: 'To fix', try: 'To try' }[m] }));
      }
      groups[m].forEach((w) => panel.append(wordCard(w, m)));
      if (!groups[m].length && mode !== 'all') panel.append(el('p', { class: 'muted-empty', text: VOCAB_EMPTY[m] }));
    });
  }

  const vtabs = Array.from(document.querySelectorAll('[data-vtab]'));
  function selectVtab(name) {
    vtab = name;
    vtabs.forEach((b) => {
      const on = b.dataset.vtab === name;
      b.setAttribute('aria-selected', String(on));
      b.tabIndex = on ? 0 : -1;
    });
    if (report) renderVocab(report);
  }
  vtabs.forEach((b, i) => {
    b.addEventListener('click', () => selectVtab(b.dataset.vtab));
    b.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const next = vtabs[(i + (e.key === 'ArrowRight' ? 1 : vtabs.length - 1)) % vtabs.length];
      next.focus();
      selectVtab(next.dataset.vtab);
    });
  });

  $('level-guide-list').append(...LEVEL_GUIDE.flatMap(([lvl, desc]) => [el('dt', { text: lvl }), el('dd', { text: desc })]));

  // ── Grammar ─────────────────────────────────────────────────────────────
  function bars(p) {
    const values = (p.series || []).map((s) => s.value);
    const max = Math.max(...values, 0.0001);
    const unit = p.unit === 'per 100 words' ? 'per 100 words' : 'times';
    return el('div', {
      class: 'bars', role: 'img',
      'aria-label': `By session: ${(p.series || []).map((s) => `${fmt.date(s.date, { day: 'numeric', month: 'short' })} ${s.value}`).join(', ')} ${unit}`,
    }, (p.series || []).map((s) => el('span', {
      class: `bar${s.value ? '' : ' zero'}`,
      style: `height:${Math.max(6, (s.value / max) * 100)}%`,
      title: `${fmt.date(s.date, { day: 'numeric', month: 'short' })}: ${s.value} ${unit}`,
    })));
  }

  function renderGrammar(r) {
    const g = r.grammar || {};
    const rows = [...(g.needs_work || []), ...(g.new || []), ...(g.improved || [])];
    const tbody = $('grammar-table').querySelector('tbody');
    tbody.replaceChildren();
    const perWords = rows.some((p) => p.unit === 'per 100 words');
    $('grammar-sub').textContent = perWords
      ? 'Bars show mistakes per 100 words you spoke, oldest session on the left.'
      : 'Bars show mistakes per session, oldest session on the left.';
    rows.forEach((p) => {
      const [label, cls] = GRAMMAR_STATUS[p.status] || [p.status, ''];
      const ex = (p.examples || [])[0];
      tbody.append(el('tr', {},
        el('td', { class: 'area', text: p.label }),
        el('td', {}, el('span', { class: `badge ${cls}`, text: label })),
        el('td', {}, bars(p)),
        el('td', {}, ex ? [el('span', { class: 'example-wrong', text: ex.wrong }), ex.correct ? el('span', { class: 'example-right', text: ex.correct }) : null] : '–')));
    });
    if (!rows.length) tbody.append(el('tr', {}, el('td', { colspan: '4', class: 'muted-empty', text: 'No grammar mistakes recorded in these sessions.' })));
  }

  // ── Strengths, goals, timeline ──────────────────────────────────────────
  function renderStrengths(r) {
    const list = $('strengths-list');
    list.replaceChildren();
    (r.strengths || []).forEach((s) => list.append(el('li', { class: 'quote' },
      el('q', { text: s.quote }),
      el('p', { text: [s.note, s.date ? fmt.date(s.date, { day: 'numeric', month: 'short' }) : ''].filter(Boolean).join(' · ') }))));
    if (!list.children.length) list.append(el('li', { class: 'muted-empty', text: 'Highlights from your sessions will appear here.' }));
  }

  function renderGoals(r) {
    const list = $('goals-list');
    list.replaceChildren();
    (r.learner_goals || []).forEach((g) => {
      const idx = GOAL_INDEX[g.status] ?? 0;
      list.append(el('li', {},
        el('p', { class: 'goal-desc', text: g.description }),
        el('div', { class: 'goal-steps', 'aria-label': `Status: ${GOAL_STEPS[idx]}` },
          GOAL_STEPS.map((s, i) => el('span', { class: `goal-step${i <= idx ? ' on' : ''}`, text: s })))));
    });
    if (!list.children.length) {
      list.append(el('li', { class: 'muted-empty', text: 'Tell your tutor what you need English for — interviews, client calls, presentations — and it will be tracked here.' }));
    }
  }

  function renderTimeline(r) {
    const list = $('timeline');
    list.replaceChildren();
    (r.sessions || []).forEach((s) => {
      const mins = s.is_valid
        ? `${s.learner_minutes != null ? `${Math.round(s.learner_minutes)} min speaking · ` : ''}${Math.round(s.duration_min)} min session`
        : 'Skipped — no clear speech from you';
      list.append(el('li', {},
        el('div', {}, el('p', { class: 'tl-date', text: fmt.date(s.date) }), el('p', { class: 'tl-mins', text: mins })),
        el('div', {},
          el('p', { class: 'tl-summary', text: s.summary || (s.is_valid ? 'No summary available.' : '') }),
          (s.topics || []).length ? el('div', { class: 'tl-topics' }, s.topics.map((t) => el('span', { class: 'badge', text: t }))) : null)));
    });
    if (!list.children.length) list.append(el('li', { class: 'muted-empty', text: 'No sessions yet.' }));
  }

  // ── Tutor view: knowledge graph (vis-network, loaded on demand) ─────────
  let visPromise = null;
  let network = null;

  function loadVis() {
    if (window.vis) return Promise.resolve();
    if (!visPromise) {
      visPromise = new Promise((resolve, reject) => {
        const s = el('script', { src: 'https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js' });
        s.onload = resolve;
        s.onerror = () => { visPromise = null; reject(new Error('Could not load the graph library.')); };
        document.head.append(s);
      });
    }
    return visPromise;
  }

  async function loadGraph() {
    if (!learnerId) return;
    const box = $('graph');
    box.replaceChildren(el('div', { class: 'empty' }, el('div', { class: 'spinner' })));
    try {
      const [graph] = await Promise.all([
        api(`/kg/learners/${encodeURIComponent(learnerId)}/graph?n=${selectWindow.value}`),
        loadVis(),
      ]);
      box.replaceChildren();
      if (network) network.destroy();
      const colors = {
        learner: '#6d28d9', session: '#a78bfa', grammar: '#b45309', word: '#0f766e', goal: '#1d4ed8',
      };
      network = new window.vis.Network(box, {
        nodes: graph.nodes.map((nd) => ({
          id: nd.id, label: nd.label.length > 28 ? `${nd.label.slice(0, 27)}…` : nd.label,
          title: nd.title || nd.label, color: colors[nd.group] || '#6b6681',
          shape: nd.group === 'learner' ? 'dot' : 'dot', size: nd.group === 'learner' ? 22 : nd.group === 'session' ? 14 : 9,
          font: { face: 'Inter', size: 12, color: '#1c1830' },
        })),
        edges: graph.edges.map((e) => ({
          from: e.source, to: e.target, color: { color: '#d4cfe4' }, width: 1,
          title: e.label || '', smooth: false,
        })),
      }, { physics: { stabilization: { iterations: 150 } }, interaction: { hover: true } });
    } catch (err) {
      box.replaceChildren(el('div', { class: 'empty' }, el('p', { text: err.message })));
    }
  }
  $('tutor-tools').addEventListener('toggle', (e) => { if (e.target.open) loadGraph(); });

  // ── Print: show every vocabulary group and the level guide ──────────────
  window.addEventListener('beforeprint', () => {
    if (!report) return;
    renderVocab(report, 'all');
    document.querySelector('.level-guide').open = true;
  });
  window.addEventListener('afterprint', () => { if (report) renderVocab(report); });

  loadLearners();
})();
