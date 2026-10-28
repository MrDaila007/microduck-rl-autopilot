const CHARTS = [
  ["Train/mean_reward", "награда"],
  ["Train/mean_episode_length", "длина эпизода"],
  ["Episode_Termination/fell_over", "падения"],
  ["Loss/entropy", "энтропия"],
  ["Perf/total_fps", "fps"],
];

const RESULT_RU = {
  launched: "запущено",
  stopped: "остановлено",
  saved: "настройки записаны",
  applied: "применено: smoke прошёл, прогон перезапущен",
  smoke_failed: "smoke не прошёл, живой прогон не тронут",
  rejected: "отклонено",
  already_running: "обучение уже идёт",
  no_process: "процесс обучения не найден",
  no_runs: "нет прогона",
  reset: "сброшено",
  ready_to_apply: "готово к применению",
};

const state = {
  overview: null,
  run: null,
  series: null,
  queue: null,
  recs: null,
  config: null,
  settingsDirty: false,
  queueNames: null,
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function fmtEta(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return "—";
  const s = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `${h} ч ${m} мин`;
  if (m > 0) return `${m} мин`;
  return `${s} с`;
}

function fmtNum(value, digits = 3) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  const n = Number(value);
  if (Math.abs(n) >= 100) return String(Math.round(n));
  return n.toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
}

function notice(text, bad = false) {
  const el = $("notice");
  el.hidden = !text;
  el.textContent = text || "";
  el.className = bad ? "notice bad" : "notice";
}

async function getJSON(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path} ${response.status}`);
  return response.json();
}

function busy() {
  const overview = state.overview;
  return Boolean(overview && (overview.busy || overview.intervening));
}

async function postAction(body, confirmText) {
  if (confirmText && !window.confirm(confirmText)) return;
  const response = await fetch("/api/actions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (response.status === 409) {
    notice("Автопилот занят: идёт smoke или другое действие.", true);
    return;
  }
  if (!response.ok) {
    notice((data.errors || [data.error || "ошибка"]).join("; "), true);
    return;
  }
  notice("действие поставлено в очередь…");
  await pollAction(data.id);
}

async function pollAction(id) {
  for (let i = 0; i < 4000; i += 1) {
    const action = await getJSON(`/api/actions/${id}`);
    if (action.status === "done" || action.status === "failed") {
      const result = action.result || {};
      const label = RESULT_RU[result.status] || result.status || action.error || action.status;
      const extra = (result.errors || []).join("; ");
      notice(extra ? `${label}: ${extra}` : label, action.status === "failed" || result.status === "rejected" || result.status === "smoke_failed");
      state.settingsDirty = false;
      state.queueNames = null;
      await refresh(true);
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

function stat(label, value) {
  return `<div class="stat"><span>${esc(label)}</span><strong>${esc(value)}</strong></div>`;
}

function renderOverview() {
  const overview = state.overview || {};
  const training = overview.training ? `pid ${overview.training.pid}` : "нет";
  const ollama = overview.ollama && overview.ollama.ok ? "на связи" : "нет";
  const auto = overview.auto && overview.auto.enabled ? "вкл" : "выкл";
  $("conn").textContent = overview.time ? "жив" : "ждём цикл";
  $("stats").innerHTML = [
    stat("фаза", overview.phase || "—"),
    stat("вердикт", overview.verdict || "—"),
    stat("итерация", overview.iters ?? (state.run && state.run.iters) ?? "—"),
    stat("с/итер", fmtNum(overview.sec_per_iter, 2)),
    stat("ETA", fmtEta(overview.eta_s)),
    stat("обучение", training),
    stat("ollama", ollama),
    stat("auto", auto),
    stat("бюджет", overview.budget ? `${overview.budget.used}/${overview.budget.max}` : "—"),
  ].join("");
  const bits = [];
  if (overview.intervening) {
    bits.push(`<div class="banner bad">Идёт smoke или перезапуск. Живой прогон не трогается, пока проверка не закончится.</div>`);
  }
  if (overview.needs_human) {
    bits.push(`<div class="banner bad">Нужен человек: ${esc(overview.needs_human_reason || "needs_human")}</div>`);
  }
  if (overview.diagnosis) {
    bits.push(`<div class="banner">${esc(overview.diagnosis)}</div>`);
  }
  $("banner").innerHTML = bits.join("");
}

function chartSVG(points) {
  if (!points || points.length < 2) {
    return `<p class="hint">нет точек</p>`;
  }
  const width = 360;
  const height = 92;
  const pad = 6;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const d = points.map((p, i) => {
    const x = pad + ((p[0] - minX) / spanX) * (width - pad * 2);
    const y = height - pad - ((p[1] - minY) / spanY) * (height - pad * 2);
    return `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
  const last = points[points.length - 1][1];
  return `<svg viewBox="0 0 ${width} ${height}" role="img"><path d="${d}"></path></svg>
    <p class="hint">${esc(fmtNum(minY))} … ${esc(fmtNum(maxY))} · сейчас ${esc(fmtNum(last))}</p>`;
}

function renderCharts() {
  const series = (state.series && state.series.series) || {};
  $("charts").innerHTML = CHARTS.map(([tag, title]) => `
    <article class="chart">
      <h2>${esc(title)}</h2>
      <p class="hint"><code>${esc(tag)}</code></p>
      ${chartSVG(series[tag])}
    </article>
  `).join("");
}

function renderHealth() {
  const run = state.run;
  if (!run || run.status === "no_runs") {
    $("health").innerHTML = `<article class="panel"><h2>Прогон</h2><p class="hint">Прогонов пока нет.</p></article>`;
    return;
  }
  const health = run.health || {};
  const progress = run.progress || {};
  const plateau = run.plateau || {};
  const rows = Object.entries(health)
    .filter(([key]) => key !== "window")
    .map(([key, value]) => `<div class="row"><span>${esc(key)}</span><strong>${esc(fmtNum(value))}</strong></div>`)
    .join("");
  const pacing = ((run.curriculum || {}).pacing_warnings || [])
    .map((item) => `<li>${esc(item.metric)} ${esc(item.step)} на итерации ${esc(item.at_iter)}</li>`)
    .join("");
  $("health").innerHTML = `
    <article class="panel">
      <h2>${esc(run.experiment)} / ${esc(run.name)}</h2>
      <p class="hint">${run.active ? "активен" : "завершён"} · итерация ${esc(run.iters)} · чекпойнт ${esc(progress.last_checkpoint_iter ?? "—")} · до цели ${esc(progress.remaining_to_target ?? "—")}</p>
      <div class="row"><span>плато</span><strong>${plateau.flat ? "да" : "нет"} · gain ${esc(plateau.rel_gain ?? "—")}</strong></div>
      ${rows || `<p class="hint">Метрик ещё нет.</p>`}
      ${pacing ? `<h3>curriculum</h3><ul>${pacing}</ul>` : ""}
    </article>`;
}

function renderSide() {
  const run = state.run || {};
  const findings = run.findings || [];
  const audit = run.penalty_audit || [];
  const mass = (run.reward_mass || {}).top || [];
  const findingHTML = findings.length
    ? `<ul>${findings.map((item) => `<li><span class="tag ${item.severity === "critical" || item.severity === "high" ? "bad" : "info"}">${esc(item.code)}</span> ${esc(item.detail)}</li>`).join("")}</ul>`
    : `<p class="hint">Правила молчат.</p>`;
  const auditHTML = audit.length
    ? audit.map((row) => `<div class="row"><span>${esc(row.term)}</span><strong class="${row.status === "VIOLATION" ? "tag bad" : ""}">${esc(row.status)} ${esc(fmtNum(row.max))}</strong></div>`).join("")
    : `<p class="hint">Штрафных членов нет.</p>`;
  const massHTML = mass.length
    ? mass.map((row) => `<div class="row"><span>${esc(row.term)}</span><strong>${esc(fmtNum((row.share || 0) * 100, 1))}%</strong></div>`).join("")
    : `<p class="hint">Положительной массы нет.</p>`;
  $("side").innerHTML = `
    <article class="panel"><h2>Findings</h2>${findingHTML}</article>
    <article class="panel" style="margin-top:0.7rem"><h2>Штрафы</h2>${auditHTML}</article>
    <article class="panel" style="margin-top:0.7rem"><h2>Масса наград</h2>${massHTML}</article>`;
}

function moveQueue(index, delta) {
  const names = state.queueNames.slice();
  const next = index + delta;
  if (next < 0 || next >= names.length) return;
  const [item] = names.splice(index, 1);
  names.splice(next, 0, item);
  state.queueNames = names;
  renderQueue();
}

function renderQueue() {
  const queue = state.queue || { queue: [], scenarios: [] };
  if (!state.queueNames) state.queueNames = queue.queue.map((item) => item.name);
  const byStatus = Object.fromEntries(queue.queue.map((item) => [item.name, item.status]));
  const items = state.queueNames.map((name, index) => `
    <div class="queue-item">
      <span class="tag ${byStatus[name] || "pending"}">${esc(byStatus[name] || "new")}</span>
      <div><strong>${esc(name)}</strong></div>
      <div class="actions">
        <button type="button" class="ghost" data-up="${index}">вверх</button>
        <button type="button" class="ghost" data-down="${index}">вниз</button>
        <button type="button" class="ghost" data-drop="${index}">убрать</button>
      </div>
    </div>`).join("");
  const options = (queue.scenarios || []).map((item) => `<option value="${esc(item.name)}">${esc(item.name)}</option>`).join("");
  const catalog = (queue.scenarios || []).map((item) => `
    <div class="row">
      <span>${esc(item.name)} <span class="muted">${esc(item.task || "")}</span></span>
      <button type="button" data-launch="${esc(item.name)}" ${busy() ? "disabled" : ""}>Запустить</button>
    </div>`).join("");
  $("queue").innerHTML = `
    <article class="panel">
      <h2>Очередь</h2>
      <p class="hint">Дальше: ${esc(queue.next || "—")} · по умолчанию ${esc(queue.default_scenario || "—")}. Каталог task только для чтения.</p>
      ${items || `<p class="hint">Очередь пустая.</p>`}
      <div class="actions" style="margin-top:0.8rem">
        <select id="add-scenario">${options}</select>
        <button type="button" class="ghost" id="add-queue">В очередь</button>
        <button type="button" id="save-queue" ${busy() ? "disabled" : ""}>Сохранить очередь</button>
        <button type="button" class="danger" id="stop-run" ${busy() ? "disabled" : ""}>Стоп</button>
        <label class="check"><input type="checkbox" id="park"> поставить сценарий на паузу</label>
      </div>
      <h2 style="margin-top:1rem">Каталог</h2>
      ${catalog}
    </article>`;
  $("queue").querySelectorAll("[data-up]").forEach((btn) => btn.addEventListener("click", () => moveQueue(Number(btn.dataset.up), -1)));
  $("queue").querySelectorAll("[data-down]").forEach((btn) => btn.addEventListener("click", () => moveQueue(Number(btn.dataset.down), 1)));
  $("queue").querySelectorAll("[data-drop]").forEach((btn) => btn.addEventListener("click", () => {
    state.queueNames = state.queueNames.filter((_, i) => i !== Number(btn.dataset.drop));
    renderQueue();
  }));
  $("queue").querySelectorAll("[data-launch]").forEach((btn) => btn.addEventListener("click", () => {
    postAction(
      { type: "launch", scenario: btn.dataset.launch },
      `Запустить сценарий ${btn.dataset.launch}?`,
    );
  }));
  $("add-queue").addEventListener("click", () => {
    const name = $("add-scenario").value;
    if (!name || state.queueNames.includes(name)) return;
    state.queueNames = state.queueNames.concat(name);
    renderQueue();
  });
  $("save-queue").addEventListener("click", () => {
    postAction({ type: "set_queue", queue: state.queueNames });
  });
  $("stop-run").addEventListener("click", () => {
    const park = $("park").checked;
    postAction(
      { type: "stop", park },
      "Остановить обучение? SIGINT, до save_interval итераций может не сохраниться.",
    );
  });
}

function renderRecs() {
  const recs = (state.recs && state.recs.recs) || [];
  const cards = recs.slice(0, 12).map((rec) => {
    const changes = (rec.changes || []).map((change) => `${change.key}=${change.value}`).join(", ");
    const apply = rec.applyable
      ? `<button type="button" data-apply="${esc(rec.file)}" ${busy() ? "disabled" : ""}>Применить</button>`
      : "";
    return `<article class="rec">
      <div class="row"><strong>${esc(rec.verdict || "—")}</strong><span class="tag">${esc(rec.status || "")}</span></div>
      <p>${esc(rec.diagnosis || "")}</p>
      <p class="hint">${esc(rec.file)}${changes ? ` · ${esc(changes)}` : ""}</p>
      ${rec.errors && rec.errors.length ? `<p class="hint">${esc(rec.errors.join("; "))}</p>` : ""}
      ${apply}
    </article>`;
  }).join("");
  $("recs").innerHTML = `
    <article class="panel">
      <div class="actions">
        <h2>Рекомендации</h2>
        <button type="button" class="ghost" id="ask-model" ${busy() ? "disabled" : ""}>Спросить модель</button>
        <button type="button" class="ghost" id="plan-run" ${busy() ? "disabled" : ""}>План прогона</button>
      </div>
      <p class="hint">«Применить» проходит тот же путь: проверка → smoke 64×5 → стоп → resume. При выключенном auto демон сам это не делает.</p>
      ${cards || `<p class="hint">Рекомендаций нет.</p>`}
    </article>`;
  const ask = $("ask-model");
  const plan = $("plan-run");
  if (ask) ask.addEventListener("click", () => postAction({ type: "suggest" }, "Спросить модель по активному прогону?"));
  if (plan) plan.addEventListener("click", () => postAction({ type: "plan" }, "Построить план последнего прогона?"));
  $("recs").querySelectorAll("[data-apply]").forEach((btn) => btn.addEventListener("click", () => {
    postAction({ type: "apply_rec", file: btn.dataset.apply }, "Применить рекомендацию? Сначала smoke, живой прогон остановится только после него.");
  }));
}

function renderManual() {
  const run = state.run || {};
  const knobs = run.knobs || [];
  const maxChanges = run.max_changes || (state.config && state.config.safety.max_changes) || 3;
  const fields = knobs.map((knob, index) => `
    <div class="knob">
      <label class="check"><input type="checkbox" data-use="${index}"> <span>${esc(knob.label)} <code>${esc(knob.key)}</code></span></label>
      <input type="number" step="any" data-value="${index}" value="${knob.value ?? ""}" placeholder="${esc((knob.bounds || []).join(" … "))}">
    </div>`).join("");
  $("manual").innerHTML = `
    <article class="panel">
      <h2>Ручная правка</h2>
      <p class="hint">Не больше ${esc(maxChanges)} ключей из allowlist. Живой прогон не останавливается, пока smoke (64 env, 5 iter) не пройдёт. Бюджет автопилота это не тратит.</p>
      ${fields || `<p class="hint">Нет прогона — не из чего прочитать текущие веса.</p>`}
      <div class="actions" style="margin-top:0.8rem">
        <button type="button" id="apply-manual" ${busy() || !fields ? "disabled" : ""}>Проверить и применить</button>
      </div>
    </article>`;
  const button = $("apply-manual");
  if (!button) return;
  button.addEventListener("click", () => {
    const changes = [];
    knobs.forEach((knob, index) => {
      const use = $("manual").querySelector(`[data-use="${index}"]`);
      const input = $("manual").querySelector(`[data-value="${index}"]`);
      if (use && use.checked && input && input.value !== "") {
        changes.push({ key: knob.key, value: Number(input.value) });
      }
    });
    if (!changes.length) {
      notice("Отметьте хотя бы один параметр.", true);
      return;
    }
    if (changes.length > maxChanges) {
      notice(`Слишком много правок: ${changes.length} > ${maxChanges}.`, true);
      return;
    }
    postAction(
      { type: "manual_adjust", changes },
      "Запустить smoke и, если он пройдёт, перезапустить с чекпойнта?",
    );
  });
}

function monitorFields(config) {
  const monitor = config.monitor || {};
  return Object.keys(monitor).map((key) => `
    <label class="setting"><span>${esc(key)}</span>
      <input data-monitor="${esc(key)}" type="number" step="any" value="${esc(monitor[key])}">
    </label>`).join("");
}

function focusedIn(id) {
  const root = $(id);
  return Boolean(root && root.contains(document.activeElement));
}

function renderSettings() {
  if (state.settingsDirty || focusedIn("settings")) return;
  const config = state.config;
  if (!config) {
    $("settings").innerHTML = "";
    return;
  }
  const ranges = config.safety.ranges || {};
  const rangeFields = Object.entries(ranges).map(([key, bounds]) => `
    <label class="setting"><span><code>${esc(key)}</code></span>
      <span class="inline">
        <input data-range="${esc(key)}" data-end="0" type="number" step="any" value="${esc(bounds[0])}">
        <input data-range="${esc(key)}" data-end="1" type="number" step="any" value="${esc(bounds[1])}">
      </span>
    </label>`).join("");
  const scenarios = (config.scenarios || []).map((name) => `<option value="${esc(name)}" ${name === config.train.default_scenario ? "selected" : ""}>${esc(name)}</option>`).join("");
  const queue = (config.train.queue || []).join(", ");
  $("settings").innerHTML = `
    <article class="panel">
      <h2>Настройки процесса</h2>
      <p class="hint">Пишется в autopilot.toml точечно: комментарии и каталог сценариев остаются. Allow/deny и smoke 64/5 отсюда не меняются. Очередь — в блоке выше (${esc(queue || "пустая")}).</p>
      <label class="check"><input id="cfg-auto" type="checkbox" ${config.auto.enabled ? "checked" : ""}> auto.enabled — демон сам применяет правки</label>
      <label class="check"><input id="cfg-idle" type="checkbox" ${config.auto.start_if_idle ? "checked" : ""}> start_if_idle — запускать очередь, когда никто не учится</label>
      <label class="setting"><span>сценарий по умолчанию</span>
        <select id="cfg-default"><option value="">—</option>${scenarios}</select>
      </label>
      <h2 style="margin-top:1rem">Монитор</h2>
      ${monitorFields(config)}
      <label class="setting"><span>safety.max_changes</span>
        <input id="cfg-max" type="number" step="1" value="${esc(config.safety.max_changes)}">
      </label>
      <h2 style="margin-top:1rem">Диапазоны</h2>
      ${rangeFields}
      <h2 style="margin-top:1rem">Модель</h2>
      <label class="setting"><span>model</span><input id="cfg-model" type="text" value="${esc(config.llm.model)}"></label>
      <label class="setting"><span>temperature</span><input id="cfg-temp" type="number" step="any" value="${esc(config.llm.temperature)}"></label>
      <label class="setting"><span>timeout_s</span><input id="cfg-timeout" type="number" step="1" value="${esc(config.llm.timeout_s)}"></label>
      <div class="actions" style="margin-top:0.8rem">
        <button type="button" id="save-config" ${busy() ? "disabled" : ""}>Сохранить настройки</button>
        <button type="button" class="ghost" id="reset-budget" ${busy() ? "disabled" : ""}>Сбросить бюджет</button>
        <button type="button" class="ghost" id="reset-queue" ${busy() ? "disabled" : ""}>Сбросить прогресс очереди</button>
      </div>
    </article>`;
  $("settings").querySelectorAll("input, select").forEach((el) => {
    el.addEventListener("input", () => { state.settingsDirty = true; });
    el.addEventListener("change", () => { state.settingsDirty = true; });
  });
  $("save-config").addEventListener("click", saveConfig);
  $("reset-budget").addEventListener("click", () => postAction({ type: "reset" }, "Сбросить бюджет и отметки завершённых прогонов?"));
  $("reset-queue").addEventListener("click", () => postAction({ type: "reset", queue: true }, "Сбросить прогресс очереди и бюджет?"));
}

function numberValue(id) {
  return Number($(id).value);
}

function saveConfig() {
  const config = state.config;
  const monitor = {};
  $("settings").querySelectorAll("[data-monitor]").forEach((input) => {
    monitor[`monitor.${input.dataset.monitor}`] = Number(input.value);
  });
  const ranges = {};
  Object.keys(config.safety.ranges || {}).forEach((key) => {
    const inputs = $("settings").querySelectorAll(`[data-range="${CSS.escape(key)}"]`);
    ranges[key] = [Number(inputs[0].value), Number(inputs[1].value)];
  });
  const values = {
    "auto.enabled": $("cfg-auto").checked,
    "auto.start_if_idle": $("cfg-idle").checked,
    "train.default_scenario": $("cfg-default").value,
    "safety.max_changes": numberValue("cfg-max"),
    "safety.ranges": ranges,
    "llm.model": $("cfg-model").value,
    "llm.temperature": numberValue("cfg-temp"),
    "llm.timeout_s": numberValue("cfg-timeout"),
    ...monitor,
  };
  postAction({ type: "set_config", values });
}

function renderLogs(autoText, trainText) {
  $("logs").innerHTML = `
    <article class="panel">
      <h2>Журналы</h2>
      <div class="split">
        <div><p class="hint">autopilot.log</p><pre>${esc(autoText || "пусто")}</pre></div>
        <div><p class="hint">последний train-*.log</p><pre>${esc(trainText || "пусто")}</pre></div>
      </div>
    </article>`;
}

async function refresh(full) {
  try {
    state.overview = await getJSON("/api/overview");
    renderOverview();
    if (full) {
      const [run, series, queue, recs, config, autoLog, trainLog] = await Promise.all([
        getJSON("/api/run"),
        getJSON("/api/series"),
        getJSON("/api/queue"),
        getJSON("/api/recs"),
        getJSON("/api/config"),
        getJSON("/api/log?which=autopilot&tail=80"),
        getJSON("/api/log?which=train&tail=80"),
      ]);
      state.run = run;
      state.series = series;
      const previous = (state.queue && state.queue.queue || []).map((item) => item.name).join("|");
      const next = (queue.queue || []).map((item) => item.name).join("|");
      state.queue = queue;
      if (previous !== next) state.queueNames = null;
      state.recs = recs;
      state.config = config;
      renderCharts();
      renderHealth();
      renderSide();
      if (!focusedIn("queue")) renderQueue();
      renderRecs();
      if (!focusedIn("manual")) renderManual();
      renderSettings();
      renderLogs(autoLog.text, trainLog.text);
    }
  } catch (error) {
    $("conn").textContent = "нет связи";
    notice(String(error.message || error), true);
  }
}

$("refresh").addEventListener("click", () => {
  state.settingsDirty = false;
  state.queueNames = null;
  refresh(true);
});
refresh(true);
setInterval(() => refresh(false), 2000);
setInterval(() => refresh(true), 10000);
