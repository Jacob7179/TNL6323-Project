/* MakanSense front-end: render dashboard from injected data + live classifier.
   Until a dataset is imported (A.n_reviews present), the dashboard shows an
   empty state and the classifier is disabled. */
(function () {
  "use strict";
  const A = JSON.parse(document.getElementById("analytics-data").textContent) || {};
  const M = JSON.parse(document.getElementById("metrics-data").textContent) || {};
  const initialized = !!(A && A.n_reviews);
  const $ = (s) => document.querySelector(s);
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  const pct = (n) => Math.round(n) + "%";
  const cap = (s) => s ? s[0].toUpperCase() + s.slice(1) : s;

  /* ---------- health badge ---------- */
  fetch("api/health").then(r => r.json()).then(d => {
    if (!d.initialized) {
      $("#htext").textContent = "no model yet · import data in step 01";
      $("#hdot").classList.add("off");
      return;
    }
    $("#htext").textContent = d.transformer === "ready"
      ? "classical + transformer ready"
      : (d.transformer === "unavailable"
        ? "classical ready · transformer offline"
        : "classical ready · transformer on demand");
    if (d.transformer !== "ready") $("#hdot").classList.add("off");
  }).catch(() => { $("#htext").textContent = "ready"; });

  /* ---------- sample review chips (classifier) ---------- */
  const SAMPLES = [
    "Food was delicious and staff super friendly 😍 definitely coming back!",
    "Service damn slow, waited 40 minutes and the waiter was rude.",
    "Ambience is nice and cozy but the food a bit pricey lah.",
    "Average only, nothing special. Quite normal taste.",
  ];
  const sc = $("#samples");
  SAMPLES.forEach(s => {
    const c = el("span", "chip", s.length > 46 ? s.slice(0, 44) + "…" : s);
    c.title = s;
    c.onclick = () => { if (!$("#review").disabled) $("#review").value = s; };
    sc.appendChild(c);
  });

  /* ===================== DASHBOARD (only when data exists) ===================== */
  if (initialized) renderDashboard();
  else renderEmpty();

  function renderEmpty() {
    const box = (h) => `<div class="emptybox">📥 ${h}</div>`;
    $("#overviewLine").textContent = "Import a dataset in step 01 to see this.";
    $("#overviewGrid").innerHTML = box("Import a dataset in <b>step 01</b> to see the overall sentiment.");
    $("#reviewQs").innerHTML = box("These insights appear after you import data and train.");
    $("#aspectGrid").innerHTML = box("Food / service / ambience breakdown appears after training.");
    $("#ownerCard").innerHTML = box("Your owner action plan appears here once the model is trained.");
  }

  function renderDashboard() {
    /* ---------- overall ---------- */
    const ov = A.overall;
    $("#overviewLine").textContent = A.questions.overall_summary.line;
    $("#stN").textContent = A.n_reviews;
    $("#stStars").innerHTML = (ov.avg_stars != null ? ov.avg_stars : "–") + " <small>/ 5 ★</small>";
    drawDonut([
      ["pos", ov.pos_pct, getC("--pos")],
      ["neu", ov.neu_pct, getC("--neu")],
      ["neg", ov.neg_pct, getC("--neg")],
    ], ov.pos_pct);

    const sb = $("#starBars");
    const dist = A.star_distribution || {};
    const maxStar = Math.max(1, ...Object.values(dist));
    [5, 4, 3, 2, 1].forEach(s => {
      const n = dist[String(s)] || 0;
      const row = el("div", "starbar");
      row.innerHTML = `<span>${s}★</span><div class="t"><div class="f" style="width:${100 * n / maxStar}%"></div></div><span class="mono">${n}</span>`;
      sb.appendChild(row);
    });
    if (!Object.keys(dist).length) sb.innerHTML = `<div class="note" style="color:var(--muted)">No star ratings in this dataset.</div>`;

    /* ---------- review-based questions ---------- */
    const RQ = A.questions;
    const rq = $("#reviewQs");
    rq.appendChild(qcard("👍", RQ.like_most.title,
      kwHTML(RQ.like_most.keywords, "pos") + noteHTML(RQ.like_most.summary)));
    rq.appendChild(qcard("⚠️", RQ.complaints.title,
      kwHTML(RQ.complaints.keywords, "neg") + noteHTML(RQ.complaints.summary)));
    rq.appendChild(qcard("🔎", RQ.negative_themes.title,
      kwHTML(RQ.negative_themes.keywords, "neg") + quotesHTML(RQ.negative_themes.examples)));
    const amb = RQ.ambience.aspect || {};
    rq.appendChild(qcard("🪑", RQ.ambience.title,
      `<div class="note">Across <b>${amb.mentions || 0}</b> ambience mentions, sentiment is
        <b>${amb.pos_pct || 0}% positive</b> / ${amb.neg_pct || 0}% negative
        (avg ${amb.avg_stars || "–"}★).</div>` +
      kwHTML((amb.top_words || []).map(w => [w, null]), "") +
      (amb.example_positive ? quotesHTML([amb.example_positive]) : "")));
    rq.appendChild(qcard("📊", "Summary of overall sentiment",
      `<div class="note">${RQ.overall_summary.line}</div>`, true));

    /* ---------- aspect-based sentiment ---------- */
    const ag = $("#aspectGrid");
    ["food", "service", "ambiance", "price"].forEach(name => {
      const a = A.aspects[name];
      if (!a) return;
      const tot = a.positive + a.neutral + a.negative || 1;
      const card = el("div", "acard");
      card.innerHTML = `
        <div class="atop">
          <div class="aname">${name}</div>
          <div class="amentions">${a.mentions} mentions · ${a.avg_stars || "–"}★</div>
        </div>
        <div class="sentbar">
          <i class="p" style="width:${100 * a.positive / tot}%"></i>
          <i class="u" style="width:${100 * a.neutral / tot}%"></i>
          <i class="g" style="width:${100 * a.negative / tot}%"></i>
        </div>
        <div class="akw"><b>${a.pos_pct}%</b> positive · ${a.neg_pct}% negative</div>
        <div class="akw" style="margin-top:6px">Top terms: <b>${(a.top_words || []).slice(0, 6).join(", ")}</b></div>`;
      ag.appendChild(card);
    });
    if (!ag.children.length) ag.innerHTML = `<div class="note" style="color:var(--muted)">No aspect mentions detected in this dataset.</div>`;

    /* ---------- owner action plan ---------- */
    renderOwner(A.owner_summary);
  }

  function renderOwner(o) {
    const host = $("#ownerCard");
    if (!o) { host.innerHTML = ""; return; }
    const wellItems = (o.doing_well || []).map(w =>
      `<li>Customers love your <b>${w.aspect}</b> — <b>${w.pos_pct}%</b> of ${w.aspect} mentions are positive.</li>`
    ).join("") || `<li>Not enough data yet to highlight a clear strength.</li>`;
    const impItems = (o.improve || []).map(m =>
      `<li><b>${cap(m.aspect)}</b> — ${m.neg_pct}% of mentions are negative.
         <div class="adv">${m.advice}</div>
         ${m.keywords && m.keywords.length ? `<div class="kws">mentioned: ${m.keywords.join(", ")}</div>` : ""}</li>`
    ).join("") || `<li>No major problem areas stand out — keep it up.</li>`;
    host.innerHTML = `
      <div class="ownercard">
        <div class="ownerhead">
          <div class="ownericon">📋</div>
          <div>
            <div class="ownerheadline">${o.headline || ""}</div>
            <div class="ownertone">${o.tone || ""}</div>
          </div>
        </div>
        <div class="ownergrid">
          <div class="ownerblock good"><h4>What's working</h4><ul>${wellItems}</ul></div>
          <div class="ownerblock improve"><h4>Where to improve</h4><ol>${impItems}</ol></div>
        </div>
        ${o.priority ? `<div class="ownerpriority">${o.priority}</div>` : ""}
        <div class="ownernote">Based on ${A.n_reviews} reviews. Import more data above to refresh this.</div>
      </div>`;
  }

  /* ===================== IMPORT DATA & TRAIN ===================== */
  let TRAIN_MODE = "append";
  document.querySelectorAll("#modeToggle button").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll("#modeToggle button").forEach(x => x.classList.remove("active"));
      b.classList.add("active"); TRAIN_MODE = b.dataset.mode;
    };
  });
  const fileInput = $("#trainFile");
  fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    $("#fileName").textContent = f ? f.name : "Choose a CSV or Excel file…";
    $("#fileLabel").classList.toggle("hasfile", !!f);
  });

  function runTrain(opts, workingMsg) {
    const st = $("#trainStatus");
    const btn = $("#trainBtn"), sbtn = $("#sampleBtn");
    btn.disabled = true; sbtn.disabled = true; btn.textContent = "Training…";
    st.style.display = "block"; st.className = "trainstatus working"; st.textContent = workingMsg;
    const url = opts.sample ? "api/train_sample" : "api/train";
    const init = opts.sample ? { method: "POST" } : { method: "POST", body: opts.body };
    fetch(url, init)
      .then(r => r.json().then(j => ({ ok: r.ok, j })))
      .then(({ ok, j }) => {
        if (!ok || j.error) {
          st.className = "trainstatus err"; st.textContent = "⚠ " + (j.error || "Training failed.");
        } else {
          st.className = "trainstatus ok";
          st.innerHTML = `✓ Trained on ${j.n_total} reviews. Best model: ${j.best_model}, `
            + `accuracy ${Math.round(j.accuracy * 100)}%, macro-F1 ${j.macro_f1}. Loading dashboard…`;
          setTimeout(() => location.reload(), 1500);
        }
      })
      .catch(() => { st.className = "trainstatus err"; st.textContent = "⚠ Something went wrong. Please try again."; })
      .finally(() => { btn.disabled = false; sbtn.disabled = false; btn.textContent = "Train model"; });
  }

  $("#trainBtn").onclick = () => {
    const f = fileInput.files[0];
    const st = $("#trainStatus");
    if (!f) { st.style.display = "block"; st.className = "trainstatus err"; st.textContent = "Please choose a CSV or Excel file first."; return; }
    const fd = new FormData();
    fd.append("file", f); fd.append("mode", TRAIN_MODE);
    runTrain({ body: fd }, "Uploading data and training the model… this takes a few seconds.");
  };

  $("#sampleBtn").onclick = () => {
    runTrain({ sample: true }, "Training on the sample dataset… this takes a few seconds.");
  };

  /* ===================== LIVE CLASSIFIER ===================== */
  let MODEL = "classical";
  document.querySelectorAll("#modeltoggle button").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll("#modeltoggle button").forEach(x => x.classList.remove("active"));
      b.classList.add("active"); MODEL = b.dataset.model;
    };
  });

  if (!initialized) {
    $("#review").disabled = true;
    $("#review").placeholder = "Import a dataset in step 01 first — the classifier turns on once the model is trained.";
    $("#analyzeBtn").disabled = true;
    $("#placeholder").textContent = "The classifier activates once you import data and train the model (step 01 above).";
  } else {
    $("#analyzeBtn").onclick = analyze;
    $("#review").addEventListener("keydown", e => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
    });
  }

  function analyze() {
    const text = $("#review").value.trim();
    if (!text) { $("#review").focus(); return; }
    const btn = $("#analyzeBtn");
    btn.disabled = true; btn.textContent = MODEL === "transformer" ? "Loading model…" : "Analyzing…";
    fetch("api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model: MODEL }),
    })
      .then(r => r.json())
      .then(renderResult)
      .catch(() => alert("Something went wrong. Please try again."))
      .finally(() => { btn.disabled = false; btn.textContent = "Analyze sentiment"; });
  }

  function renderResult(d) {
    if (d.error) { alert(d.error); return; }
    $("#placeholder").style.display = "none";
    $("#resultBody").style.display = "block";
    const s = d.sentiment;
    const undef = (s === "undefined");
    const stamp = $("#stamp");
    stamp.className = "stamp " + (undef ? "und" : s.slice(0, 3));
    stamp.textContent = undef ? "N/A" : s;
    stamp.classList.remove("show"); void stamp.offsetWidth; stamp.classList.add("show");
    $("#vlabel").textContent = undef ? "Undefined" : s;
    $("#vconf").textContent = undef ? "no restaurant aspect to judge" : ("confidence " + pct(d.confidence * 100));
    $("#vmodel").textContent = "model: " + d.model;

    const pb = $("#probBars"); pb.innerHTML = "";
    if (!undef) {
      ["positive", "neutral", "negative"].forEach(lab => {
        const p = (d.probabilities && d.probabilities[lab]) || 0;
        const row = el("div", "barrow");
        row.innerHTML = `<span>${lab}</span>
          <div class="bartrack"><div class="barfill ${lab.slice(0, 3)}"></div></div>
          <span class="mono">${pct(p * 100)}</span>`;
        pb.appendChild(row);
        requestAnimationFrame(() => { row.querySelector(".barfill").style.width = (p * 100) + "%"; });
      });
    }

    const mr = $("#metaRow"); mr.innerHTML = "";
    const aspects = d.aspects || {};
    const names = Object.keys(aspects);
    if (names.length) {
      names.forEach(a => {
        const as = aspects[a].sentiment;
        mr.appendChild(el("span", "tag " + as.slice(0, 3), a + ": " + as));
      });
    } else {
      mr.appendChild(el("span", "tag hint",
        "No food, service or ambience mentioned — try a review about the food, service, or ambience."));
    }
    (d.emojis || []).forEach(e => mr.appendChild(el("span", "tag emoji", e + " emoji")));
  }

  /* ---------- helpers ---------- */
  function qcard(icon, q, bodyHTML, span2) {
    const c = el("div", "qcard" + (span2 ? " span2" : ""));
    c.innerHTML = `<div class="qq"><span class="ic">${icon}</span><span>${q}</span></div>${bodyHTML}`;
    return c;
  }
  function kwHTML(pairs, cls) {
    if (!pairs || !pairs.length) return "";
    const items = pairs.map(p => {
      const w = Array.isArray(p) ? p[0] : p;
      const n = Array.isArray(p) ? p[1] : null;
      return `<span class="kw ${cls}">${w}${n != null ? `<span class="n">${n}</span>` : ""}</span>`;
    }).join("");
    return `<div class="kwrap">${items}</div>`;
  }
  function quotesHTML(arr) {
    if (!arr || !arr.length) return "";
    return arr.filter(Boolean).map(q => `<div class="quote">“${q}”</div>`).join("");
  }
  function noteHTML(t) { return t ? `<div class="note" style="color:var(--muted)">${t}</div>` : ""; }
  function getC(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

  function drawDonut(parts, centerPct) {
    const svg = $("#donut");
    const r = 15.9155, cx = 21, cy = 21;
    let offset = 25;
    const ns = "http://www.w3.org/2000/svg";
    parts.forEach(([, val, color]) => {
      const circ = document.createElementNS(ns, "circle");
      circ.setAttribute("cx", cx); circ.setAttribute("cy", cy); circ.setAttribute("r", r);
      circ.setAttribute("fill", "transparent");
      circ.setAttribute("stroke", color); circ.setAttribute("stroke-width", "5");
      circ.setAttribute("stroke-dasharray", `${val} ${100 - val}`);
      circ.setAttribute("stroke-dashoffset", offset);
      svg.appendChild(circ);
      offset = (offset - val + 100) % 100;
    });
    const big = document.createElementNS(ns, "text");
    big.setAttribute("x", cx); big.setAttribute("y", 20);
    big.setAttribute("text-anchor", "middle"); big.setAttribute("dominant-baseline", "central");
    big.setAttribute("class", "big"); big.textContent = Math.round(centerPct) + "%";
    svg.appendChild(big);
    const small = document.createElementNS(ns, "text");
    small.setAttribute("x", cx); small.setAttribute("y", 26.5);
    small.setAttribute("text-anchor", "middle"); small.setAttribute("dominant-baseline", "central");
    small.setAttribute("class", "small"); small.textContent = "POSITIVE";
    svg.appendChild(small);
  }
})();
