/* ─────────────────────────────────────────────────────────────────────────
   app.js — Agentic Travel Planner · Phase 1
   Responsibilities:
     1. Health check → update header status dot
     2. Interest chips & style selector toggle
     3. Example tag quick-fill
     4. Form submit → POST /chat/plan → render markdown itinerary
     5. Copy button
     6. Replan button
───────────────────────────────────────────────────────────────────────── */

const API_BASE = '';   // same origin — no CORS needed

// ── DOM refs ────────────────────────────────────────────────────────────────
const statusDot     = document.getElementById('statusDot');
const statusText    = document.getElementById('statusText');
const tripForm      = document.getElementById('tripForm');
const planBtn       = document.getElementById('planBtn');
const btnText       = planBtn.querySelector('.btn-text');
const btnLoader     = document.getElementById('btnLoader');
const emptyState    = document.getElementById('emptyState');
const resultContent = document.getElementById('resultContent');
const errorState    = document.getElementById('errorState');
const errorMsg      = document.getElementById('errorMsg');

// ── Health check ────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      const configured = data.llm_configured;
      statusDot.className = 'status-dot' + (configured ? '' : ' offline');
      statusText.textContent = configured
        ? `${data.llm_provider.toUpperCase()} · ${data.llm_model}`
        : 'API key not configured';
    } else {
      throw new Error('Server error');
    }
  } catch {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Server unreachable';
  }
}
checkHealth();

// ── Interest chips ──────────────────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    chip.classList.toggle('active');
    const activeChips = [...document.querySelectorAll('.chip.active')]
      .map(c => c.dataset.value)
      .join(', ');
    document.getElementById('interests').value = activeChips;
  });
});

// ── Travel style selector ───────────────────────────────────────────────────
document.querySelectorAll('.style-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('travelStyle').value = btn.dataset.value;
  });
});

// ── Example tags quick-fill ─────────────────────────────────────────────────
const examples = [
  { dest: 'Tokyo, Japan',   days: 7, month: 'October',  budget: '$2000', interests: 'Food & Culture, History & Museums' },
  { dest: 'Goa, India',     days: 5, month: 'December', budget: '₹60,000', interests: 'Beaches & Relaxation' },
  { dest: 'Paris, France',  days: 3, month: 'April',    budget: '€1200', interests: 'History & Museums, Shopping & Nightlife' },
];
['eg1','eg2','eg3'].forEach((id, i) => {
  document.getElementById(id)?.addEventListener('click', () => {
    const ex = examples[i];
    document.getElementById('destination').value = ex.dest;
    document.getElementById('days').value        = ex.days;
    document.getElementById('month').value       = ex.month;
    document.getElementById('budget').value      = ex.budget;
    document.getElementById('interests').value   = ex.interests;
    // Sync chips
    document.querySelectorAll('.chip').forEach(c => {
      c.classList.toggle('active', ex.interests.includes(c.dataset.value));
    });
  });
});

// ── State helpers ────────────────────────────────────────────────────────────
function showEmpty()  { emptyState.hidden=false; resultContent.hidden=true; errorState.hidden=true; }
function showResult() { emptyState.hidden=true;  resultContent.hidden=false; errorState.hidden=true; }
function showError(msg) { emptyState.hidden=true; resultContent.hidden=true; errorState.hidden=false; errorMsg.textContent=msg; }

function setLoading(on) {
  planBtn.disabled = on;
  btnText.hidden   = on;
  btnLoader.hidden = !on;
  if (on) showSkeleton();
}

function showSkeleton() {
  emptyState.hidden = true;
  resultContent.hidden = false;
  errorState.hidden = true;
  document.getElementById('resultDestination').textContent = '✈️  Planning your trip...';
  document.getElementById('resultTags').innerHTML = '';
  document.getElementById('itineraryBody').innerHTML = `
    <div class="skeleton-wrap">
      <div class="sk sk-title"></div>
      <div class="sk sk-long"></div>
      <div class="sk sk-mid"></div>
      <div class="sk sk-short"></div>
      <div class="sk sk-long" style="margin-top:16px"></div>
      <div class="sk sk-long"></div>
      <div class="sk sk-mid"></div>
      <div class="sk sk-short"></div>
      <div class="sk sk-long" style="margin-top:16px"></div>
      <div class="sk sk-long"></div>
      <div class="sk sk-mid"></div>
    </div>`;
  document.getElementById('providerBadge').textContent = '';
}

// ── Form submit ──────────────────────────────────────────────────────────────
tripForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const destination  = document.getElementById('destination').value.trim();
  const days         = parseInt(document.getElementById('days').value, 10);
  const month        = document.getElementById('month').value;
  const budget       = document.getElementById('budget').value.trim() || '$1000';
  const interests    = document.getElementById('interests').value.trim() || null;
  const travelStyle  = document.getElementById('travelStyle').value || null;

  if (!destination) {
    document.getElementById('destination').focus();
    return;
  }

  setLoading(true);

  try {
    const res = await fetch(`${API_BASE}/chat/plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ destination, days, month, budget, interests, travel_style: travelStyle }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      // Give a friendlier message for service-unavailable errors
      if (res.status === 503) {
        throw new Error(
          `🔄 ${err.detail || 'The AI service is currently busy.'}\n\nThis is temporary — please wait a moment and try again.`
        );
      }
      throw new Error(err.detail || `Server error ${res.status}`);
    }


    const data = await res.json();
    renderResult(data);

  } catch (err) {
    showError(`Failed to generate itinerary:\n${err.message}`);
  } finally {
    setLoading(false);
  }
});

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data) {
  document.getElementById('resultDestination').textContent = `✈️  ${data.destination}`;

  // Tags
  const tags = document.getElementById('resultTags');
  tags.innerHTML = [
    `📅 ${data.days} days`,
    `🌤️ ${data.month}`,
  ].map(t => `<span class="result-tag">${t}</span>`).join('');

  // Markdown → HTML
  const html = marked.parse(data.itinerary || '_No itinerary returned._');
  document.getElementById('itineraryBody').innerHTML = html;

  // Provider badge
  document.getElementById('providerBadge').textContent =
    `⚡ Powered by ${data.llm_provider.toUpperCase()} · Phase 1 Chain`;

  showResult();
}

// ── Copy button ───────────────────────────────────────────────────────────────
document.getElementById('copyBtn')?.addEventListener('click', async () => {
  const itineraryEl = document.getElementById('itineraryBody');
  const text = itineraryEl.innerText;
  try {
    await navigator.clipboard.writeText(text);
    const btn = document.getElementById('copyBtn');
    btn.textContent = '✅ Copied!';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = '📋 Copy'; btn.classList.remove('copied'); }, 2000);
  } catch {
    // fallback silently
  }
});

// ── Replan button ─────────────────────────────────────────────────────────────
document.getElementById('replanBtn')?.addEventListener('click', () => {
  showEmpty();
  document.getElementById('destination').focus();
});

// ── Retry button ──────────────────────────────────────────────────────────────
document.getElementById('retryBtn')?.addEventListener('click', () => {
  tripForm.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
});
