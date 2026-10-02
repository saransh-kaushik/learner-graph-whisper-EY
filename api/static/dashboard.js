document.addEventListener('DOMContentLoaded', () => {
  const selectLearner = document.getElementById('learner-select');
  const btnRefresh = document.getElementById('btn-refresh');
  const loadingOverlay = document.getElementById('loading-overlay');
  const dashboardContent = document.getElementById('dashboard-content');
  
  let neoVisInstance = null;

  // Init
  loadLearners();

  btnRefresh.addEventListener('click', () => {
    const learnerId = selectLearner.value;
    if (learnerId) fetchDashboardData(learnerId);
  });

  selectLearner.addEventListener('change', (e) => {
    fetchDashboardData(e.target.value);
  });

  async function loadLearners() {
    try {
      const res = await fetch('/kg/learners');
      const learners = await res.json();
      
      selectLearner.innerHTML = '<option value="" disabled selected>Select a learner...</option>';
      learners.forEach(l => {
        const opt = document.createElement('option');
        opt.value = l.id;
        opt.textContent = `${l.name || l.id} (${l.session_count} sessions)`;
        selectLearner.appendChild(opt);
      });
    } catch (e) {
      console.error("Failed to load learners", e);
      selectLearner.innerHTML = '<option value="" disabled selected>Error loading learners</option>';
    }
  }

  async function fetchDashboardData(learnerId) {
    loadingOverlay.classList.remove('hidden');
    dashboardContent.classList.add('hidden');

    try {
      // 1. Fetch Report
      const res = await fetch(`/kg/learners/${learnerId}/report`);
      const report = await res.json();

      // Update Header
      document.getElementById('current-learner-name').textContent = report.learner_name || learnerId;
      document.getElementById('current-learner-level').textContent = `CEFR Level: ${report.cefr_level || 'Unknown'}`;

      // Update AI Summary
      document.getElementById('ai-summary-text').textContent = report.llm_summary;
      document.getElementById('ai-practice-next').textContent = report.practice_next;

      // Update Metrics
      updateMetric('ttr', report.metrics.talk_time_ratio, '%', (v) => (v * 100).toFixed(1));
      updateMetric('wpm', report.metrics.speaking_rate_wpm, '');
      updateMetric('wpt', report.metrics.words_per_turn, '');
      updateMetric('csr', report.metrics.code_switch_ratio, '%', (v) => (v * 100).toFixed(1));

      // Update Vocab
      updateVocabTable(report.vocabulary.new_words, report.vocabulary.mastery_upgrades);

      // Update Goals
      updateGoalsList(report.goals);

      // Update Grammar
      updateGrammarLists(report.grammar.fading_errors, report.grammar.persistent_errors);

      // Render Graph
      renderGraph(learnerId);

      // Show UI
      loadingOverlay.classList.add('hidden');
      dashboardContent.classList.remove('hidden');

    } catch (e) {
      console.error("Error fetching report:", e);
      alert("Failed to load dashboard data.");
      loadingOverlay.classList.add('hidden');
    }
  }

  function updateMetric(idSuffix, metricData, unit = '', formatter = (v) => v) {
    const valEl = document.getElementById(`metric-${idSuffix}`);
    const trendEl = document.getElementById(`trend-${idSuffix}`);
    
    if (!metricData) {
      valEl.textContent = '--';
      trendEl.textContent = '';
      trendEl.className = 'trend-badge';
      return;
    }

    valEl.textContent = `${formatter(metricData.current)}${unit}`;
    
    trendEl.textContent = metricData.trend.toUpperCase();
    trendEl.className = `trend-badge trend-${metricData.trend}`;
  }

  function updateVocabTable(newWords, upgrades) {
    const tbody = document.getElementById('vocab-table-body');
    tbody.innerHTML = '';
    
    // Mix them up for display
    const items = [];
    newWords.forEach(w => items.push({ word: w, mastery: 'introduced', cefr: 'B1' }));
    upgrades.forEach(u => items.push({ word: u.word, mastery: u.to, cefr: 'B1' })); // simplified

    if (items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="3" style="text-align:center; color:gray;">No new vocabulary recorded.</td></tr>';
      return;
    }

    items.slice(0, 10).forEach(item => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${item.word}</strong></td>
        <td class="mastery-${item.mastery}">${item.mastery}</td>
        <td><span class="cefr-badge">${item.cefr}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  function updateGoalsList(goals) {
    const container = document.getElementById('goals-list');
    container.innerHTML = '';

    if (!goals || goals.length === 0) {
      container.innerHTML = '<p style="color:gray;">No goals tracked yet.</p>';
      return;
    }

    const statuses = { 'mentioned': 33, 'practiced': 66, 'demonstrated': 100 };

    goals.forEach(g => {
      const pct = statuses[g.status] || 0;
      const html = `
        <div class="goal-item">
          <div class="goal-desc">${g.description}</div>
          <div class="progress-track">
            <div class="progress-fill" style="width: ${pct}%"></div>
          </div>
          <span class="status-text">${g.status.toUpperCase()}</span>
        </div>
      `;
      container.innerHTML += html;
    });
  }

  function updateGrammarLists(fading, persistent) {
    const fadingUl = document.getElementById('fading-errors-list');
    const persistentUl = document.getElementById('persistent-errors-list');
    
    fadingUl.innerHTML = fading.map(e => `<li>${e.pattern.replace(/_/g, ' ')} <span class="error-count">↓</span></li>`).join('') || '<li style="color:gray">None detected</li>';
    persistentUl.innerHTML = persistent.map(e => `<li>${e.pattern.replace(/_/g, ' ')} <span class="error-count">!</span></li>`).join('') || '<li style="color:gray">None detected</li>';
  }

  function renderGraph(learnerId) {
    if (neoVisInstance) {
      neoVisInstance.clearNetwork();
    }
    
    // We assume neo4j is on default port on localhost for this demo visualization
    // In production, the backend should serve the graph data or proxy the neo4j connection.
    const config = {
      container_id: "viz",
      server_url: "bolt://localhost:7687", // Adjust if needed
      server_user: "neo4j",
      server_password: "password", // Demo password
      labels: {
        "Learner": { caption: "name", size: "pagerank" },
        "Session": { caption: "date" },
        "GrammarPattern": { caption: "pattern_name" },
        "Word": { caption: "lemma" },
        "Goal": { caption: "description" }
      },
      relationships: {
        "ATTENDED": { thickness: "weight", caption: false },
        "MADE_ERROR": { caption: false },
        "USED_WORD": { caption: false },
        "HAS_GOAL": { caption: false }
      },
      initial_cypher: `MATCH (l:Learner {id: "${learnerId}"})-[r*1..2]-(m) RETURN l, r, m LIMIT 100`
    };

    try {
      neoVisInstance = new NeoVis.default(config);
      neoVisInstance.render();
    } catch(e) {
      document.getElementById('viz').innerHTML = `<p style="color:var(--danger); padding:1rem;">Could not connect to Neo4j directly from browser for visualization. Ensure WebSockets/Bolt is exposed.</p>`;
    }
  }
});
