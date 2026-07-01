/* MakanSense front-end: render dashboard from injected data + live classifier */
(function () {
  "use strict";
  const A = JSON.parse(document.getElementById("analytics-data").textContent);
  const M = JSON.parse(document.getElementById("metrics-data").textContent);
  const $ = (s) => document.querySelector(s);
  const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h != null) e.innerHTML = h; return e; };
  const pct = (n) => Math.round(n) + "%";

  /* ---------- health ---------- */
  fetch("api/health").then(r => r.json()).then(d => {
    const t = d.transformer === "ready"
      ? "classical + transformer ready"
      : (d.transformer === "unavailable"
        ? "classical ready · transformer offline"
        : "classical ready · transformer on demand");
    $("#htext").textContent = t;
    if (d.transformer !== "ready") $("#hdot").classList.add("off");
  }).catch(() => { $("#htext").textContent = "classical ready"; });

  /* ---------- sample reviews ---------- */
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
    c.onclick = () => { $("#review").value = s; };
    sc.appendChild(c);
  });

  /* ---------- overall ---------- */
  const ov = A.overall;
  $("#overviewLine").textContent = A.questions.overall_summary.line;
  $("#stN").textContent = A.n_reviews;
  $("#stStars").innerHTML = ov.avg_stars + " <small>/ 5 ★</small>";

  // donut
  drawDonut([
    ["pos", ov.pos_pct, getC("--pos")],
    ["neu", ov.neu_pct, getC("--neu")],
    ["neg", ov.neg_pct, getC("--neg")],
  ], ov.pos_pct);

  // star bars
  const sb = $("#starBars");
  const maxStar = Math.max(...Object.values(A.star_distribution));
  [5, 4, 3, 2, 1].forEach(s => {
    const n = A.star_distribution[String(s)] || 0;
    const row = el("div", "starbar");
    row.innerHTML = `<span>${s}★</span><div class="t"><div class="f" style="width:${maxStar ? (100 * n / maxStar) : 0}%"></div></div><span class="mono">${n}</span>`;
    sb.appendChild(row);
  });

  /* ---------- review-based questions ---------- */
  const RQ = A.questions;
  const rq = $("#reviewQs");
  rq.appendChild(qcard("👍", RQ.like_most.title,
    kwHTML(RQ.like_most.keywords, "pos") + noteHTML(RQ.like_most.summary)));
  rq.appendChild(qcard("⚠️", RQ.complaints.title,
    kwHTML(RQ.complaints.keywords, "neg") + noteHTML(RQ.complaints.summary)));
  rq.appendChild(qcard("🔎", RQ.negative_themes.title,
    kwHTML(RQ.negative_themes.keywords.map(x => x), "neg") +
    quotesHTML(RQ.negative_themes.examples)));
  // ambience
  const amb = RQ.ambience.aspect || {};
  rq.appendChild(qcard("🪑", RQ.ambience.title,
    `<div class="note">Across <b>${amb.mentions || 0}</b> ambience mentions, sentiment is
      <b>${amb.pos_pct || 0}% positive</b> / ${amb.neg_pct || 0}% negative
      (avg ${amb.avg_stars || "–"}★).</div>` +
    kwHTML((amb.top_words || []).map(w => [w, null]), "") +
    (amb.example_positive ? quotesHTML([amb.example_positive]) : "")));
  // overall summary span2
  rq.appendChild(qcard("📊", "Summary of overall sentiment",
    `<div class="note">${RQ.overall_summary.line}</div>` +
    `<div class="note" style="color:var(--muted)">Positive reviews dominate, but roughly
      1 in 5 reviews is negative — concentrated in the aspects shown below.</div>`, true));

  /* ---------- aspect-based sentiment ---------- */
  const ag = $("#aspectGrid");
  const order = ["food", "service", "ambiance", "price"];
  order.forEach(name => {
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

  /* ===================== LIVE CLASSIFIER ===================== */
  let MODEL = "classical";
  document.querySelectorAll("#modeltoggle button").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll("#modeltoggle button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      MODEL = b.dataset.model;
    };
  });

  $("#analyzeBtn").onclick = analyze;
  $("#review").addEventListener("keydown", e => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
  });

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
    $("#vconf").textContent = undef ? "no restaurant aspect to judge"
                                    : ("confidence " + pct(d.confidence * 100));
    $("#vmodel").textContent = "model: " + d.model;

    // probability bars (hidden when undefined)
    const pb = $("#probBars"); pb.innerHTML = "";
    if (!undef) {
      ["positive", "neutral", "negative"].forEach(lab => {
        const p = (d.probabilities && d.probabilities[lab]) || 0;
        const row = el("div", "barrow");
        row.innerHTML = `<span>${lab}</span>
          <div class="bartrack"><div class="barfill ${lab.slice(0,3)}"></div></div>
          <span class="mono">${pct(p * 100)}</span>`;
        pb.appendChild(row);
        requestAnimationFrame(() => { row.querySelector(".barfill").style.width = (p * 100) + "%"; });
      });
    }

    // aspect-based sentiment + emojis
    const mr = $("#metaRow"); mr.innerHTML = "";
    const aspects = d.aspects || {};
    const names = Object.keys(aspects);
    if (names.length) {
      names.forEach(a => {
        const as = aspects[a].sentiment;              // positive | neutral | negative
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
  function cmpHTML(rows) {
    const max = 5;
    return `<div class="cmp">` + rows.map(([k, v]) => {
      const w = v != null ? (100 * v / max) : 0;
      return `<div class="cmprow"><span>${k}</span>
        <div class="cmptrack"><div class="cmpfill" style="width:${w}%"></div></div>
        <span class="mono">${v != null ? v + "★" : "–"}</span></div>`;
    }).join("") + `</div>`;
  }
  function getC(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

  function drawDonut(parts, centerPct) {
    const svg = $("#donut");
    const r = 15.9155, cx = 21, cy = 21;
    let offset = 25; // start at top
    const ns = "http://www.w3.org/2000/svg";
    parts.forEach(([, val, color]) => {
      const circ = document.createElementNS(ns, "circle");
      circ.setAttribute("cx", cx); circ.setAttribute("cy", cy); circ.setAttribute("r", r);
      circ.setAttribute("fill", "transparent");
      circ.setAttribute("stroke", color);
      circ.setAttribute("stroke-width", "5");
      circ.setAttribute("stroke-dasharray", `${val} ${100 - val}`);
      circ.setAttribute("stroke-dashoffset", offset);
      svg.appendChild(circ);
      offset = (offset - val + 100) % 100;
    });
    const big = document.createElementNS(ns, "text");
    big.setAttribute("x", cx); big.setAttribute("y", 20);
    big.setAttribute("text-anchor", "middle");
    big.setAttribute("dominant-baseline", "central");
    big.setAttribute("class", "big");
    big.textContent = Math.round(centerPct) + "%";
    svg.appendChild(big);
    const small = document.createElementNS(ns, "text");
    small.setAttribute("x", cx); small.setAttribute("y", 26.5);
    small.setAttribute("text-anchor", "middle");
    small.setAttribute("dominant-baseline", "central");
    small.setAttribute("class", "small");
    small.textContent = "POSITIVE";
    svg.appendChild(small);
  }
})();
