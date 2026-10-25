/* pprof_py documentation: interactive validation figures (precision ladder; per-provider explorer).
   Plain JavaScript and inline SVG, no dependencies. Inlined into _static/validation/<name>.html by
   validation/make_figures.py; the data is the JSON record in <script id="pprof-data"> ({kind, title, question,
   caption, report, benchmark}). Hover (and keyboard focus in the table) shows a tooltip; the table is always available. */
(function () {
  'use strict';
  var D = JSON.parse(document.getElementById('pprof-data').textContent);
  var root = document.getElementById('root');
  var reduced = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  var SVGNS = 'http://www.w3.org/2000/svg';

  /* ---------- theme: ?theme= in the URL, else the embedding page (same origin), else the OS setting (CSS) ---------- */
  function parentMode() {
    try {
      if (window.parent && window.parent !== window) {
        var el = window.parent.document.documentElement;
        var mode = el.getAttribute('data-color-mode') || el.getAttribute('data-theme');
        if (mode === 'dark' || mode === 'light') return mode;
      }
    } catch (e) { /* cross-origin frame: prefers-color-scheme decides */ }
    return null;
  }
  function applyTheme() {
    var q = /[?&]theme=(dark|light)\b/.exec(window.location.search);
    var mode = q ? q[1] : parentMode();
    if (mode) document.documentElement.setAttribute('data-theme', mode); else document.documentElement.removeAttribute('data-theme');
  }
  applyTheme();
  try { if (window.parent && window.parent !== window && window.MutationObserver) new MutationObserver(applyTheme).observe(window.parent.document.documentElement, { attributes: true, attributeFilter: ['data-color-mode', 'data-theme'] }); } catch (e) { /* cross-origin */ }

  /* ---------- helpers ---------- */
  function h(tag, attrs, children) {
    var el = tag.indexOf('svg:') === 0 ? document.createElementNS(SVGNS, tag.slice(4)) : document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { if (attrs[k] === null || attrs[k] === undefined) return; if (k === 'text') el.textContent = attrs[k]; else if (k === 'html') el.innerHTML = attrs[k]; else el.setAttribute(k, attrs[k]); });
    (children || []).forEach(function (c) { if (c) el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return el;
  }
  function sci(v) { if (v === null || v === undefined || isNaN(v)) return '–'; if (v === 0) return '0'; var e = Math.floor(Math.log10(Math.abs(v))); var m = v / Math.pow(10, e); return m.toFixed(2) + 'e' + (e < 0 ? '−' : '+') + Math.abs(e); }
  function fix(v, d) { return (v === null || v === undefined || isNaN(v)) ? '–' : Number(v).toFixed(d).replace(/^-/, '−'); }
  function fmtAuto(v) { var a = Math.abs(v); return (a !== 0 && (a < 1e-3 || a >= 1e5)) ? sci(v) : fix(v, a >= 100 ? 2 : 5); }
  function fmtTick(v) { var a = Math.abs(v); if (a !== 0 && (a < 1e-3 || a >= 1e5)) return sci(v); return Number(v.toFixed(4)).toString().replace(/^-/, '−'); }
  function niceTicks(lo, hi, n) { var span = hi - lo || 1, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step; step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1; var t = [], v = Math.ceil(lo / step) * step; for (; v <= hi + 1e-12; v += step) t.push(+v.toFixed(10)); return t; }
  function tooltip(wrap) { var tip = h('div', { class: 'tip', role: 'status', 'aria-live': 'polite' }); wrap.appendChild(tip); return { show: function (x, y, html) { tip.innerHTML = html; tip.style.left = (x + 14) + 'px'; tip.style.top = (y - 10) + 'px'; tip.classList.add('show'); }, at: function (evt, html) { var r = wrap.getBoundingClientRect(); this.show(evt.clientX - r.left, evt.clientY - r.top, html); }, hide: function () { tip.classList.remove('show'); } }; }
  var BANDS = [{ to: -10, label: 'floating-point limit', cls: 'b0' }, { to: -5, label: 'library / order of operations', cls: 'b1' }, { to: -3, label: 'tolerance / rounding', cls: 'b2' }, { to: 2, label: 'investigate', cls: 'b3' }];

  /* ---------- precision ladder ---------- */
  function ladder(R, B) {
    var items = [], groups = [];
    var live = 'Example data|' + fmtInt(R.data.rows) + ' rows · ' + R.data.providers + ' providers';   /* '|' splits the two label lines */
    R.summary.forEach(function (s) { if (s.value && s.tolerance && s.tolerance !== '=1' && s.quantity.indexOf('degenerate') < 0) { var q = s.quantity.replace(/ \(.*\)/, '').replace('test ', '').replace(': p-value max|diff|', ' p').replace('poibin_exact', 'exact'); items.push({ group: live, comp: q, v: s.value, fam: 'py', src: 'live comparison, pprof::ExampleDataBinary' }); } });
    (B || []).forEach(function (b) {
      var g = 'Benchmark|' + fmtInt(b.rows) + ' rows · ' + fmtInt(b.providers) + ' providers';
      [['beta_maxdiff', 'β'], ['gamma_maxdiff', 'γ']].forEach(function (k) { if (b.agreement && b.agreement[k[0]] > 0) items.push({ group: g, comp: k[1], v: b.agreement[k[0]], fam: 'three', src: 'scaling benchmark, R at its default tol = 1e−5' }); });
      var t8 = b['agreement_r_tol_1e-8'];
      if (t8) [['beta_maxdiff', 'β (R tol 1e−8)'], ['gamma_maxdiff', 'γ (R tol 1e−8)']].forEach(function (k) { if (t8[k[0]] > 0) items.push({ group: g, comp: k[1], v: t8[k[0]], fam: 'r', src: 'scaling benchmark, R rerun at tol = 1e−8' }); });
    });
    items.forEach(function (it) { if (groups.indexOf(it.group) < 0) groups.push(it.group); });
    var W = 840, rowH = 52, m = { t: 34, r: 24, b: 40, l: 168 }, H = m.t + groups.length * rowH + m.b, lo = -17, hi = 0;
    function sx(v) { return m.l + (v - lo) / (hi - lo) * (W - m.l - m.r); }
    var fig = h('figure', { class: 'fig' }), wrap = h('div', { class: 'fig-wrap' }), tip = tooltip(wrap);
    var svg = h('svg:svg', { class: 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Largest absolute difference between R and Python per comparison and component, log scale' });
    var prev = lo; BANDS.forEach(function (b) { var x0 = sx(prev), x1 = sx(Math.min(b.to, hi)); svg.appendChild(h('svg:rect', { class: 'band ' + b.cls, x: x0, y: m.t - 6, width: Math.max(0, x1 - x0), height: groups.length * rowH + 12 })); svg.appendChild(h('svg:text', { class: 'band-label', x: (x0 + x1) / 2, y: m.t - 12, 'text-anchor': 'middle', text: b.label })); prev = b.to; });
    var axis = h('svg:g', { class: 'axis' });
    for (var t = lo + 1; t <= hi; t += 2) { axis.appendChild(h('svg:line', { class: 'tick', x1: sx(t), x2: sx(t), y1: m.t - 6, y2: H - m.b + 4 })); axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 18, 'text-anchor': 'middle', text: '1e' + t })); }
    svg.appendChild(axis);
    svg.appendChild(h('svg:text', { class: 'label', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', text: 'max |R − Py| (log scale; smaller is better)' }));
    groups.forEach(function (g, gi) {
      var y = m.t + gi * rowH + rowH / 2;
      svg.appendChild(h('svg:line', { class: 'row', x1: m.l, x2: W - m.r, y1: y, y2: y }));
      var gl = g.split('|');
      svg.appendChild(h('svg:text', { class: 'rowlab', x: m.l - 10, y: y - 3, 'text-anchor': 'end', text: gl[0] }));
      if (gl[1]) svg.appendChild(h('svg:text', { class: 'rowsub', x: m.l - 10, y: y + 11, 'text-anchor': 'end', text: gl[1] }));
      var rowItems = items.filter(function (it) { return it.group === g; }).sort(function (a, b) { return a.v - b.v; }), lastLabX = -1e9, flip = false;
      rowItems.forEach(function (it, i) {
        var x = sx(Math.max(lo, Math.log10(it.v)));
        var c = h('svg:circle', { class: 'pt ' + it.fam + (reduced ? '' : ' enter'), cx: x, cy: y, style: 'animation-delay:' + (gi * 60 + i * 30) + 'ms' });
        var close = x - lastLabX < 48, lab = null;
        if (x - lastLabX > 24) { flip = close ? !flip : false; lab = h('svg:text', { class: 'ptlab', x: x, y: flip ? y + 17 : y - 9, 'text-anchor': 'middle', text: it.comp }); lastLabX = x; }
        var hit = h('svg:circle', { class: 'hit', cx: x, cy: y });
        hit.addEventListener('mousemove', function (e) { tip.at(e, '<b>' + it.group.replace('|', ', ') + ' · ' + it.comp + '</b><div class="row"><span>max |R − Py|</span><span>' + sci(it.v) + '</span></div><div class="row"><span>source</span><span>' + it.src + '</span></div>'); });
        hit.addEventListener('mouseleave', tip.hide);
        svg.appendChild(c); if (lab) svg.appendChild(lab); svg.appendChild(hit);
      });
    });
    fig.appendChild(h('div', { class: 'fig-head' }, [h('span', { class: 't', text: 'Precision ladder · largest difference per comparison and component' }), h('span', { class: 'legend' }, [h('span', { text: 'live comparison (example data)' }), h('span', { class: 'three', text: 'scaling benchmark (R at its default tol = 1e−5)' }), h('span', { class: 'r', text: 'benchmark, R rerun at tol = 1e−8' })])]));
    wrap.appendChild(svg); fig.appendChild(wrap); root.appendChild(fig);
    // table view
    var tbl = h('table', { class: 'data' }, [h('caption', { text: 'The same values as a table: comparison, component, source and the largest absolute difference.' })]);
    tbl.appendChild(h('thead', {}, [h('tr', {}, [h('th', { text: 'Comparison' }), h('th', { text: 'Component' }), h('th', { text: 'Source' }), h('th', { class: 'num', text: 'max |R − Py|' })])]));
    var tb = h('tbody'); items.forEach(function (it) { tb.appendChild(h('tr', {}, [h('td', { text: it.group.replace('|', ', ') }), h('td', { text: it.comp }), h('td', { text: it.src }), h('td', { class: 'num mono', text: sci(it.v) })])); });
    tbl.appendChild(tb); root.appendChild(h('div', { class: 'table-scroll' }, [tbl]));
  }

  /* ---------- per-provider explorer ---------- */
  var Q = [{ key: 'gamma', label: 'γ provider effect', tol: 1e-5 }, { key: 'indirect_ratio', label: 'Indirect standardized ratio', tol: 1e-5 }, { key: 'indirect_rate', label: 'Indirect standardized rate (%)', tol: 1e-4 }, { key: 'direct_ratio', label: 'Direct standardized ratio', tol: 1e-5 }, { key: 'direct_rate', label: 'Direct standardized rate (%)', tol: 1e-4 }, { key: 'wald_p', label: 'Wald test p-value', tol: 1e-3 }, { key: 'score_p', label: 'Score test p-value', tol: 1e-3 }, { key: 'poibin_exact_p', label: 'Exact (Poisson-binomial) test p-value', tol: 1e-3 }];
  function diffOf(p, key) { return Math.abs(p[key + '_r'] - p[key + '_py']); }
  function explorer(R) {
    var P = R.providers, rV = R.provenance.pprof_r_version, pV = R.provenance.pprof_py_version;
    var state = { q: 'gamma', deg: true, search: '', sort: 'diff', dir: 'desc', limit: 25 };
    var controls = h('div', { class: 'controls' }), sel = h('select', { 'aria-label': 'Quantity' }); Q.forEach(function (q) { sel.appendChild(h('option', { value: q.key, text: q.label })); });
    var chk = h('input', { type: 'checkbox', checked: 'checked' }), search = h('input', { type: 'search', placeholder: 'provider id…', 'aria-label': 'Find provider' }), nInfo = h('span', { class: 'n' });
    controls.appendChild(h('label', {}, ['Quantity ', sel])); controls.appendChild(h('label', {}, [chk, ' include zero-event providers'])); controls.appendChild(h('label', {}, ['Find ', search])); controls.appendChild(nInfo);
    var figS = h('figure', { class: 'fig' }), figD = h('figure', { class: 'fig' }), figB = h('figure', { class: 'fig' }), tableWrap = h('div', { class: 'table-scroll' });
    var wrap = h('div', { class: 'fig-wrap' }, [controls, figS, figD, figB, tableWrap]); root.appendChild(wrap);
    var tip = tooltip(wrap);
    function rows() { return P.filter(function (p) { return (state.deg || !p.degenerate) && (!state.search || String(p.provider).indexOf(state.search) === 0); }); }
    function qdef() { return Q.filter(function (q) { return q.key === state.q; })[0]; }
    function tipHtml(p) { return '<b>Provider ' + p.provider + '</b>' + (p.degenerate ? ' · zero events' : '') + '<div class="row"><span>n / events</span><span>' + p.n + ' / ' + p.events + '</span></div><div class="row"><span>R</span><span>' + fmtAuto(p[state.q + '_r']) + '</span></div><div class="row"><span>Python</span><span>' + fmtAuto(p[state.q + '_py']) + '</span></div><div class="row"><span>|R − Py|</span><span>' + sci(diffOf(p, state.q)) + '</span></div>'; }
    function highlight(id) { wrap.querySelectorAll('.pt').forEach(function (c) { c.classList.toggle('hl', c.dataset.id === id); c.classList.toggle('dim', !!id && c.dataset.id !== id); }); wrap.querySelectorAll('tbody tr').forEach(function (tr) { tr.classList.toggle('hl', tr.dataset.id === id); }); }
    function bind(hit, p) { hit.addEventListener('mousemove', function (e) { tip.at(e, tipHtml(p)); highlight(String(p.provider)); }); hit.addEventListener('mouseleave', function () { tip.hide(); highlight(null); }); }
    function chart(fig, head, legend, W, H, m, lo, hi, ylo, yhi, caption, draw) {
      fig.innerHTML = '';
      var svg = h('svg:svg', { class: 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': head });
      var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
      function sx(v) { return m.l + (v - lo) / (hi - lo) * (W - m.l - m.r); } function sy(v) { return H - m.b - (v - ylo) / (yhi - ylo) * (H - m.t - m.b); }
      niceTicks(lo, hi, 6).forEach(function (t) { grid.appendChild(h('svg:line', { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b })); axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 16, 'text-anchor': 'middle', text: fmtTick(t) })); });
      niceTicks(ylo, yhi, 5).forEach(function (t) { grid.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) })); axis.appendChild(h('svg:text', { x: m.l - 8, y: sy(t) + 4, 'text-anchor': 'end', text: fmtTick(t) })); });
      svg.appendChild(grid); svg.appendChild(axis);
      draw(svg, sx, sy);
      fig.appendChild(h('div', { class: 'fig-head' }, [h('span', { class: 't', text: head }), legend]));
      fig.appendChild(svg); fig.appendChild(h('p', { class: 'caption', html: caption }));
    }
    function drawScatter() {
      var q = qdef(), d = rows(); var xs = d.map(function (p) { return p[q.key + '_r']; }), ys = d.map(function (p) { return p[q.key + '_py']; });
      var lo = Math.min.apply(null, xs.concat(ys)), hi = Math.max.apply(null, xs.concat(ys)); var pad = (hi - lo || 1) * .05; lo -= pad; hi += pad;
      chart(figS, 'Figure 1 · Per provider: R result (x) versus Python result (y)', h('span', { class: 'legend' }, [h('span', { text: 'provider with events' }), h('span', { class: 'deg', text: 'zero-event provider' }), h('span', { class: 'line', text: 'identity line y = x' })]), 640, 400, { t: 14, r: 18, b: 44, l: 64 }, lo, hi, lo, hi,
        '<strong>' + q.label + '</strong>. One dot per provider; a dot on the dashed line means both implementations give the same number. Zero-event providers have no finite γ in either implementation and leave the identity line.',
        function (svg, sx, sy) {
          svg.appendChild(h('svg:line', { class: 'identity', x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi) }));
          svg.appendChild(h('svg:text', { class: 'label', x: (64 + 640 - 18) / 2, y: 394, 'text-anchor': 'middle', text: 'R · pprof ' + rV })); svg.appendChild(h('svg:text', { class: 'label', transform: 'translate(14 185) rotate(-90)', 'text-anchor': 'middle', text: 'Python · pprof_py ' + pV }));
          d.forEach(function (p, i) { var c = h('svg:circle', { class: 'pt' + (p.degenerate ? ' deg' : '') + (reduced ? '' : ' enter'), cx: sx(p[q.key + '_r']), cy: sy(p[q.key + '_py']), 'data-id': String(p.provider), style: 'animation-delay:' + Math.min(600, i * 6) + 'ms' }); var hit = h('svg:circle', { class: 'hit', cx: sx(p[q.key + '_r']), cy: sy(p[q.key + '_py']) }); bind(hit, p); svg.appendChild(c); svg.appendChild(hit); });
        });
    }
    function drawStrip() {
      var q = qdef(), d = rows(), W = 640, H = 150, m = { t: 26, r: 18, b: 36, l: 64 };
      var ds = d.map(function (p) { var dd = diffOf(p, q.key); return dd > 0 ? Math.log10(dd) : -16; });
      var lo = Math.floor(Math.min.apply(null, ds.concat([Math.log10(q.tol) - 1]))), hi = Math.ceil(Math.max.apply(null, ds.concat([Math.log10(q.tol) + 1])));
      figD.innerHTML = '';
      function sx(v) { return m.l + (v - lo) / (hi - lo) * (W - m.l - m.r); }
      var svg = h('svg:svg', { class: 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Distribution of the differences against the tolerance' });
      var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
      for (var t = lo; t <= hi; t++) { grid.appendChild(h('svg:line', { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b })); axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 16, 'text-anchor': 'middle', text: '1e' + t })); }
      svg.appendChild(grid); svg.appendChild(axis);
      var tx = sx(Math.log10(q.tol)); svg.appendChild(h('svg:line', { class: 'refline', x1: tx, x2: tx, y1: m.t - 6, y2: H - m.b })); svg.appendChild(h('svg:text', { class: 'refline-text', x: tx + 4, y: m.t - 8, text: 'tolerance ' + sci(q.tol) }));
      svg.appendChild(h('svg:text', { class: 'label', x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', text: '|R − Py| (log scale)' }));
      d.forEach(function (p, i) { var y = m.t + 12 + ((i * 7) % 5) * 16 + ((i * 13) % 3) * 3; var c = h('svg:circle', { class: 'pt' + (p.degenerate ? ' deg' : '') + (reduced ? '' : ' enter'), cx: sx(ds[i]), cy: y, 'data-id': String(p.provider), style: 'animation-delay:' + Math.min(600, i * 6) + 'ms' }); var hit = h('svg:circle', { class: 'hit', cx: sx(ds[i]), cy: y }); bind(hit, p); svg.appendChild(c); svg.appendChild(hit); });
      figD.appendChild(h('div', { class: 'fig-head' }, [h('span', { class: 't', text: 'Figure 2 · Per-provider difference |R − Py| against the tolerance line' })])); figD.appendChild(svg);
      figD.appendChild(h('p', { class: 'caption', text: 'Log scale: 1e−6 means the two differ by one part in a million. Dots left of the orange tolerance line count as agreement; exact zeros are drawn at the far left.' }));
    }
    function drawBA() {
      var q = qdef(), d = rows().filter(function (p) { return !p.degenerate; });
      var xs = d.map(function (p) { return p[q.key + '_r']; }), ds = d.map(function (p) { return p[q.key + '_py'] - p[q.key + '_r']; });
      var lo = Math.min.apply(null, xs), hi = Math.max.apply(null, xs), pad = (hi - lo || 1) * .05; lo -= pad; hi += pad;
      var dmax = Math.max(q.tol, Math.max.apply(null, ds.map(Math.abs))) * 1.15;
      chart(figB, 'Figure 3 · Bland–Altman view: Python − R against the R value', h('span', { class: 'legend' }, [h('span', { text: 'provider with events' }), h('span', { class: 'line', text: 'tolerance ±' + sci(q.tol) })]), 640, 300, { t: 16, r: 18, b: 44, l: 70 }, lo, hi, -dmax, dmax,
        '<strong>' + q.label + '</strong>, providers with events only. A difference that grows with the value would indicate a scale or definition mismatch; a flat band inside the tolerance lines indicates convergence noise only.',
        function (svg, sx, sy) {
          svg.appendChild(h('svg:line', { class: 'identity', x1: sx(lo), x2: sx(hi), y1: sy(0), y2: sy(0) }));
          [q.tol, -q.tol].forEach(function (t) { svg.appendChild(h('svg:line', { class: 'refline', x1: sx(lo), x2: sx(hi), y1: sy(t), y2: sy(t) })); });
          svg.appendChild(h('svg:text', { class: 'label', x: (70 + 640 - 18) / 2, y: 294, 'text-anchor': 'middle', text: 'R value' })); svg.appendChild(h('svg:text', { class: 'label', transform: 'translate(14 136) rotate(-90)', 'text-anchor': 'middle', text: 'Python − R' }));
          d.forEach(function (p, i) { var c = h('svg:circle', { class: 'pt' + (reduced ? '' : ' enter'), cx: sx(xs[i]), cy: sy(ds[i]), 'data-id': String(p.provider), style: 'animation-delay:' + Math.min(600, i * 6) + 'ms' }); var hit = h('svg:circle', { class: 'hit', cx: sx(xs[i]), cy: sy(ds[i]) }); bind(hit, p); svg.appendChild(c); svg.appendChild(hit); });
        });
    }
    function drawTable() {
      var q = qdef(), d = rows().slice(), key = state.sort, dir = state.dir === 'asc' ? 1 : -1;
      function val(p) { return key === 'diff' ? diffOf(p, q.key) : key === 'provider' ? +p.provider : key === 'r' ? p[q.key + '_r'] : key === 'py' ? p[q.key + '_py'] : p[key]; }
      d.sort(function (a, b) { var va = val(a), vb = val(b); return (va > vb ? 1 : va < vb ? -1 : 0) * dir; });
      var maxd = Math.max.apply(null, d.map(function (p) { return diffOf(p, q.key); })) || 1;
      tableWrap.innerHTML = '';
      var tbl = h('table', { class: 'data' }, [h('caption', { text: 'Providers shown in the charts: hover or focus a row to find it in the charts; sort by a column.' })]), trh = h('tr');
      [['provider', 'Provider'], ['n', 'n'], ['events', 'Events'], ['r', 'R'], ['py', 'Python'], ['diff', '|R − Py|']].forEach(function (c) { var th = h('th', { scope: 'col', class: 'sortable ' + (c[0] === 'provider' ? '' : 'num ') + (state.sort === c[0] ? state.dir : ''), text: c[1], 'aria-sort': state.sort === c[0] ? (state.dir === 'asc' ? 'ascending' : 'descending') : 'none' }); th.addEventListener('click', function () { if (state.sort === c[0]) state.dir = state.dir === 'asc' ? 'desc' : 'asc'; else { state.sort = c[0]; state.dir = c[0] === 'provider' ? 'asc' : 'desc'; } drawTable(); }); trh.appendChild(th); });
      tbl.appendChild(h('thead', {}, [trh])); var tb = h('tbody');
      d.slice(0, state.limit).forEach(function (p) { var dd = diffOf(p, q.key); var tr = h('tr', { class: p.degenerate ? 'deg' : '', 'data-id': String(p.provider), tabindex: '0' }, [h('td', { text: p.provider }), h('td', { class: 'num', text: p.n }), h('td', { class: 'num', text: p.events }), h('td', { class: 'num mono', text: fmtAuto(p[q.key + '_r']) }), h('td', { class: 'num mono', text: fmtAuto(p[q.key + '_py']) }), h('td', { class: 'num mono' }, [h('span', { class: 'bar', style: 'width:' + Math.max(2, Math.round(60 * (dd > 0 ? (Math.log10(dd) + 16) / (Math.log10(maxd) + 16 || 1) : 0))) + 'px' }), sci(dd)])]); tr.addEventListener('mouseenter', function () { highlight(String(p.provider)); }); tr.addEventListener('mouseleave', function () { highlight(null); }); tr.addEventListener('focus', function () { highlight(String(p.provider)); var c = wrap.querySelector('.pt[data-id="' + p.provider + '"]'); if (c) { var r = c.getBoundingClientRect(), w = wrap.getBoundingClientRect(); tip.show(r.left - w.left + 6, r.top - w.top + 6, tipHtml(p)); } }); tr.addEventListener('blur', function () { highlight(null); tip.hide(); }); tb.appendChild(tr); });
      tbl.appendChild(tb); tableWrap.appendChild(tbl);
      if (d.length > state.limit) { var more = h('button', { type: 'button', class: 'more-btn', text: 'Show all ' + d.length + ' providers' }); more.addEventListener('click', function () { state.limit = Infinity; drawTable(); }); tableWrap.appendChild(more); }
      else if (state.limit === Infinity && d.length > 25) { var less = h('button', { type: 'button', class: 'more-btn', text: 'Show the first 25 only' }); less.addEventListener('click', function () { state.limit = 25; drawTable(); }); tableWrap.appendChild(less); }
      nInfo.textContent = Math.min(d.length, state.limit) + ' of ' + P.length + ' providers';
    }
    function redraw() { drawScatter(); drawStrip(); drawBA(); drawTable(); }
    sel.addEventListener('change', function () { state.q = sel.value; redraw(); }); chk.addEventListener('change', function () { state.deg = chk.checked; redraw(); }); search.addEventListener('input', function () { state.search = search.value.trim(); redraw(); });
    redraw();
  }


  /* ---------- benchmark: runtime against size (log-log), speed-up bars, table ---------- */
  function fmtInt(v) { return Math.round(v).toLocaleString('en-US'); }
  function benchmark(B) {
    var figT = h('figure', { class: 'fig' }), figS = h('figure', { class: 'fig' }), wrap = h('div', { class: 'fig-wrap' }, [figT, figS]); root.appendChild(wrap); var tip = tooltip(wrap);
    // runtime chart
    (function () {
      var W = 760, H = 380, m = { t: 16, r: 18, b: 46, l: 60 };
      var series = [{ side: 'r', cls: 'ln r', label: 'Fit · R (C++)', get: function (r) { return r.r.fit; } }, { side: 'py', cls: 'ln py', label: 'Fit · Python (NumPy)', get: function (r) { return r.py.fit; } }, { side: 'r', cls: 'ln r dash', label: 'Exact test · R', get: function (r) { return r.r['test_exact.poisbinom']; } }, { side: 'py', cls: 'ln py dash', label: 'Exact test · Python', get: function (r) { return r.py.test_poibin_exact; } }];
      var xs = B.map(function (r) { return Math.log10(r.rows); }), ys = []; series.forEach(function (sr) { B.forEach(function (r) { ys.push(Math.log10(Math.max(sr.get(r), 1e-3))); }); });
      var xlo = Math.floor(Math.min.apply(null, xs) - .2), xhi = Math.ceil(Math.max.apply(null, xs) + .2), ylo = Math.floor(Math.min.apply(null, ys)), yhi = Math.ceil(Math.max.apply(null, ys));
      function sx(v) { return m.l + (v - xlo) / (xhi - xlo) * (W - m.l - m.r); } function sy(v) { return H - m.b - (v - ylo) / (yhi - ylo) * (H - m.t - m.b); }
      var svg = h('svg:svg', { class: 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Runtime against data size, R and Python, both axes logarithmic' });
      var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
      for (var t = xlo; t <= xhi; t++) { grid.appendChild(h('svg:line', { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b })); axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 16, 'text-anchor': 'middle', text: t === 6 ? '1M' : t === 5 ? '100k' : t === 4 ? '10k' : '1e' + t })); }
      for (var u = ylo; u <= yhi; u++) { grid.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: sy(u), y2: sy(u) })); axis.appendChild(h('svg:text', { x: m.l - 8, y: sy(u) + 4, 'text-anchor': 'end', text: u >= 0 ? Math.pow(10, u) + ' s' : (Math.pow(10, u) * 1000) + ' ms' })); }
      svg.appendChild(grid); svg.appendChild(axis);
      svg.appendChild(h('svg:text', { class: 'label', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', text: 'rows (log scale)' })); svg.appendChild(h('svg:text', { class: 'label', transform: 'translate(14 ' + (m.t + H - m.b) / 2 + ') rotate(-90)', 'text-anchor': 'middle', text: 'seconds (log scale)' }));
      series.forEach(function (sr) {
        var pts = B.map(function (r) { return [sx(Math.log10(r.rows)), sy(Math.log10(Math.max(sr.get(r), 1e-3)))]; });
        svg.appendChild(h('svg:path', { class: sr.cls + (reduced ? '' : ' draw'), d: pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ') }));
        pts.forEach(function (p, i) { var c = h('svg:circle', { class: 'pt ' + sr.side, cx: p[0], cy: p[1] }); var hit = h('svg:circle', { class: 'hit', cx: p[0], cy: p[1] }); hit.addEventListener('mousemove', function (e) { tip.at(e, '<b>' + sr.label + '</b><div class="row"><span>rows / providers</span><span>' + fmtInt(B[i].rows) + ' / ' + fmtInt(B[i].providers) + '</span></div><div class="row"><span>time</span><span>' + fix(sr.get(B[i]), 2) + ' s</span></div>'); }); hit.addEventListener('mouseleave', tip.hide); svg.appendChild(c); svg.appendChild(hit); });
      });
      figT.appendChild(h('div', { class: 'fig-head' }, [h('span', { class: 't', text: 'Runtime against data size (both axes logarithmic)' }), h('span', { class: 'legend' }, [h('span', { class: 'r', text: 'R (C++)' }), h('span', { text: 'Python (NumPy)' }), h('span', { class: 'line', text: 'solid = fit, dashed = exact test' })])]));
      figT.appendChild(svg); figT.appendChild(h('p', { class: 'caption', text: 'Hover a point for the exact timing. The fit scales linearly on both sides with Python about twice as fast; the exact Poisson-binomial provider test is the one step where Python is slower, by a factor of 4–7.' }));
    })();
    // speed-up bars
    (function () {
      var steps = [{ key: 'fit', label: 'fit', r: function (x) { return x.r.fit; }, p: function (x) { return x.py.fit; } }, { key: 'exact', label: 'exact test', r: function (x) { return x.r['test_exact.poisbinom']; }, p: function (x) { return x.py.test_poibin_exact; } }];
      var W = 760, H = 300, m = { t: 20, r: 24, b: 56, l: 60 }, ratios = []; steps.forEach(function (st) { B.forEach(function (x) { ratios.push(st.r(x) / st.p(x)); }); });
      var lo = Math.floor(Math.log10(Math.min.apply(null, ratios)) * 2) / 2 - .2, hi = Math.ceil(Math.log10(Math.max.apply(null, ratios)) * 2) / 2 + .2, n = B.length, gw = (W - m.l - m.r) / n, bw = Math.min(42, gw / 3);
      function sy(v) { return H - m.b - (Math.log10(v) - lo) / (hi - lo) * (H - m.t - m.b); }
      var svg = h('svg:svg', { class: 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Speed-up of Python over R by step and data size' });
      var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
      [0.1, 0.2, 0.5, 1, 2, 5, 10].forEach(function (v) { if (Math.log10(v) < lo || Math.log10(v) > hi) return; grid.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v) })); axis.appendChild(h('svg:text', { x: m.l - 8, y: sy(v) + 4, 'text-anchor': 'end', text: v + '×' })); });
      svg.appendChild(grid); svg.appendChild(axis);
      svg.appendChild(h('svg:line', { class: 'identity', x1: m.l, x2: W - m.r, y1: sy(1), y2: sy(1) })); svg.appendChild(h('svg:text', { class: 'label', x: W - m.r, y: sy(1) - 6, 'text-anchor': 'end', text: 'equal speed' }));
      B.forEach(function (x, i) {
        var cx = m.l + gw * i + gw / 2;
        axis.appendChild(h('svg:text', { x: cx, y: H - m.b + 18, 'text-anchor': 'middle', text: fmtInt(x.rows) + ' rows' })); axis.appendChild(h('svg:text', { x: cx, y: H - m.b + 32, 'text-anchor': 'middle', text: fmtInt(x.providers) + ' providers' }));
        steps.forEach(function (st, j) {
          var ratio = st.r(x) / st.p(x), bx = cx + (j - .5) * (bw + 6) - bw / 2, y0 = sy(1), y1 = sy(ratio);
          var rect = h('svg:rect', { class: 'bar ' + st.key, x: bx, y: Math.min(y0, y1), width: bw, height: Math.abs(y1 - y0), rx: 3 });
          rect.addEventListener('mousemove', function (e) { tip.at(e, '<b>' + st.label + ' · ' + fmtInt(x.rows) + ' rows</b><div class="row"><span>R</span><span>' + fix(st.r(x), 2) + ' s</span></div><div class="row"><span>Python</span><span>' + fix(st.p(x), 2) + ' s</span></div><div class="row"><span>R ÷ Python</span><span>' + ratio.toFixed(2) + '×</span></div>'); }); rect.addEventListener('mouseleave', tip.hide);
          svg.appendChild(rect); svg.appendChild(h('svg:text', { class: 'barlab', x: bx + bw / 2, y: ratio >= 1 ? y1 - 5 : y1 + 13, 'text-anchor': 'middle', text: ratio.toFixed(1) + '×' }));
        });
      });
      figS.appendChild(h('div', { class: 'fig-head' }, [h('span', { class: 't', text: 'Speed-up of Python over R (R time ÷ Python time, log scale)' }), h('span', { class: 'legend' }, [h('span', { class: 'sq', text: 'fit' }), h('span', { class: 'sq r', text: 'exact test' })])]));
      figS.appendChild(svg); figS.appendChild(h('p', { class: 'caption', text: 'Bars above the line mean Python is faster.' }));
    })();
    // table
    var tbl = h('table', { class: 'data' }, [h('caption', { text: 'Timings per step and size, with the agreement of the estimates between R and Python.' })]);
    tbl.appendChild(h('thead', {}, [h('tr', {}, ['Rows', 'Providers', 'Events', 'Fit R', 'Fit Python', 'Exact test R', 'Exact test Python', 'Wald + score (both)', 'β max diff', 'γ max diff'].map(function (t) { return h('th', { class: 'num', scope: 'col', text: t }); }))]));
    var tb = h('tbody'); B.forEach(function (r) { tb.appendChild(h('tr', {}, [h('td', { class: 'num', text: fmtInt(r.rows) }), h('td', { class: 'num', text: fmtInt(r.providers) }), h('td', { class: 'num', text: fmtInt(r.events) }), h('td', { class: 'num mono', text: fix(r.r.fit, 2) + ' s' }), h('td', { class: 'num mono', text: fix(r.py.fit, 2) + ' s' }), h('td', { class: 'num mono', text: fix(r.r['test_exact.poisbinom'], 2) + ' s' }), h('td', { class: 'num mono', text: fix(r.py.test_poibin_exact, 2) + ' s' }), h('td', { class: 'num mono', text: '≤ ' + fix(Math.max(r.r.test_wald, r.r.test_score, r.py.test_wald, r.py.test_score), 2) + ' s' }), h('td', { class: 'num mono', text: sci(r.agreement.beta_maxdiff) }), h('td', { class: 'num mono', text: sci(r.agreement.gamma_maxdiff) })])); });
    tbl.appendChild(tb); root.appendChild(h('div', { class: 'table-scroll' }, [tbl]));
  }

  /* ---------- reference reproduction: one dot per regenerated reference file ---------- */
  function repro(P) {
    var groups = ['CoxPH (survival)', 'phase 4', 'selection', 'penalized (glmnet)', 'pprofR clustered tests'];
    P.files.forEach(function (f) { if (groups.indexOf(f.group) < 0) groups.push(f.group); });
    var FLOOR = -17, lo = FLOOR, hi = 1, W = 840, rowH = 44, m = { t: 34, r: 24, b: 40, l: 168 }, H = m.t + groups.length * rowH + m.b;
    function xOf(f) { return f.identical || f.max_abs <= 0 ? FLOOR : Math.max(FLOOR, Math.log10(f.max_abs)); }
    function sx(v) { return m.l + (v - lo) / (hi - lo) * (W - m.l - m.r); }
    function cls(f) { return f.identical ? 'three' : f.max_rel > 1e-6 && f.max_abs > 1e-6 ? 'r' : 'py'; }
    var ident = P.files.filter(function (f) { return f.identical; }).length, mat = P.files.filter(function (f) { return cls(f) === 'r'; }).length;
    var fig = h('figure', { class: 'fig' }), wrap = h('div', { class: 'fig-wrap' }), tip = tooltip(wrap);
    var svg = h('svg:svg', { class: 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Largest absolute difference between committed and regenerated reference values, per file, log scale' });
    var prev = lo; BANDS.forEach(function (b) { var x0 = sx(prev), x1 = sx(Math.min(b.to, hi)); svg.appendChild(h('svg:rect', { class: 'band ' + b.cls, x: x0, y: m.t - 6, width: Math.max(0, x1 - x0), height: groups.length * rowH + 12 })); svg.appendChild(h('svg:text', { class: 'band-label', x: (x0 + x1) / 2, y: m.t - 12, 'text-anchor': 'middle', text: b.label })); prev = b.to; });
    var axis = h('svg:g', { class: 'axis' });
    for (var t = -16; t <= 0; t += 2) { axis.appendChild(h('svg:line', { class: 'tick', x1: sx(t), x2: sx(t), y1: m.t - 6, y2: H - m.b + 4 })); axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 18, 'text-anchor': 'middle', text: '1e' + t })); }
    axis.appendChild(h('svg:text', { x: sx(FLOOR), y: H - m.b + 18, 'text-anchor': 'middle', text: '0' }));
    svg.appendChild(axis);
    svg.appendChild(h('svg:text', { class: 'label', x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', text: 'largest |committed − regenerated| in the file (log scale; 0 = identical)' }));
    groups.forEach(function (g, gi) {
      var y = m.t + gi * rowH + rowH / 2, fs = P.files.filter(function (f) { return f.group === g; });
      svg.appendChild(h('svg:line', { class: 'row', x1: m.l, x2: W - m.r, y1: y, y2: y }));
      svg.appendChild(h('svg:text', { class: 'rowlab', x: m.l - 10, y: y - 3, 'text-anchor': 'end', text: g }));
      svg.appendChild(h('svg:text', { class: 'rowsub', x: m.l - 10, y: y + 11, 'text-anchor': 'end', text: fs.length + ' files · ' + fs.filter(function (f) { return f.identical; }).length + ' identical' }));
      fs.forEach(function (f, i) {
        var x = sx(xOf(f)), jit = ((i % 5) - 2) * 3;
        var c = h('svg:circle', { class: 'pt ' + cls(f) + (reduced ? '' : ' enter'), cx: x, cy: y + jit, style: 'animation-delay:' + (gi * 60 + i * 8) + 'ms' });
        var hit = h('svg:circle', { class: 'hit', cx: x, cy: y + jit });
        hit.addEventListener('mousemove', function (e) { tip.at(e, '<b>' + f.file + '</b><div class="row"><span>group</span><span>' + f.group + '</span></div><div class="row"><span>rows</span><span>' + fmtInt(f.rows) + '</span></div><div class="row"><span>max |diff|</span><span>' + (f.identical ? 'identical' : sci(f.max_abs)) + '</span></div><div class="row"><span>max relative</span><span>' + (f.identical ? '0' : sci(f.max_rel)) + '</span></div>' + (f.worst_column ? '<div class="row"><span>column</span><span>' + f.worst_column + '</span></div>' : '')); });
        hit.addEventListener('mouseleave', tip.hide);
        svg.appendChild(c); svg.appendChild(hit);
      });
    });
    fig.appendChild(h('div', { class: 'fig-head' }, [h('span', { class: 't', text: 'Committed against regenerated · ' + P.files.length + ' reference files' }), h('span', { class: 'legend' }, [h('span', { class: 'three', text: 'identical (' + ident + ')' }), h('span', { text: 'last digits only' }), h('span', { class: 'r', text: 'differs materially (' + mat + ')' })])]));
    wrap.appendChild(svg); fig.appendChild(wrap); root.appendChild(fig);
    var env = h('table', { class: 'data' }, [h('caption', { text: 'The two environments. Inputs: ' + P.inputs + '.' })]);
    env.appendChild(h('thead', {}, [h('tr', {}, [h('th', { scope: 'col', text: '' }), h('th', { scope: 'col', text: 'Committed' }), h('th', { scope: 'col', text: 'Regenerated' })])]));
    var eb = h('tbody'); [['R', 'R'], ['os', 'Platform'], ['blas', 'BLAS'], ['survival', 'survival'], ['glmnet', 'glmnet'], ['pprofR', 'pprofR']].forEach(function (k) { eb.appendChild(h('tr', {}, [h('td', { text: k[1] }), h('td', { text: P.committed[k[0]] || '' }), h('td', { text: P.regenerated[k[0]] || '' })])); });
    env.appendChild(eb); root.appendChild(h('div', { class: 'table-scroll' }, [env]));
    var T = P.tests_against_regenerated, G = P.glmnet_418_rerun;
    root.appendChild(h('p', { class: 'caption', html: '<b>pprof_py against the regenerated references:</b> ' + T.passed + ' of ' + (T.passed + T.failed) + ' tests pass; the failure is ' + T.failing + '. <b>glmnet 4.1-8 rerun:</b> ' + G.conclusion + ' (cvm within ' + sci(G['penalized_wide_cv.csv'].cvm) + ', lambda_1se exact).' }));
    var tbl = h('table', { class: 'data' }, [h('caption', { text: 'Every file, largest difference first.' })]);
    tbl.appendChild(h('thead', {}, [h('tr', {}, [h('th', { scope: 'col', text: 'File' }), h('th', { scope: 'col', text: 'Group' }), h('th', { scope: 'col', class: 'num', text: 'Rows' }), h('th', { scope: 'col', class: 'num', text: 'max |diff|' }), h('th', { scope: 'col', class: 'num', text: 'max relative' }), h('th', { scope: 'col', text: 'Column' })])]));
    var tb = h('tbody'); P.files.slice().sort(function (a, b) { return b.max_abs - a.max_abs; }).forEach(function (f) { tb.appendChild(h('tr', { class: cls(f) === 'r' ? 'deg' : '' }, [h('td', { class: 'mono', text: f.file }), h('td', { text: f.group }), h('td', { class: 'num', text: fmtInt(f.rows) }), h('td', { class: 'num mono', text: f.identical ? '0' : sci(f.max_abs) }), h('td', { class: 'num mono', text: f.identical ? '0' : sci(f.max_rel) }), h('td', { text: f.worst_column || '' })])); });
    tbl.appendChild(tb); root.appendChild(h('div', { class: 'table-scroll' }, [tbl]));
  }

  root.appendChild(h('h1', { text: D.title })); if (D.question) root.appendChild(h('p', { class: 'question', text: D.question }));
  if (D.kind === 'ladder') ladder(D.report, D.benchmark); else if (D.kind === 'explorer') explorer(D.report); else if (D.kind === 'benchmark') benchmark(D.benchmark); else if (D.kind === 'repro') repro(D.repro);
  if (D.caption) root.appendChild(h('p', { class: 'caption', html: D.caption }));

  /* ---------- height handshake: tell the embedding page (same origin) how tall this document is ---------- */
  var heightPending = false;
  function sendHeight() {
    heightPending = false;
    try {
      if (!window.parent || window.parent === window) return;
      var height = Math.ceil(document.body.getBoundingClientRect().height);   // the body, so the frame can shrink again
      window.parent.postMessage({ pprof: 'height', display: D.kind, height: height }, '/');   // '/' = this origin only
    } catch (e) { /* no parent or a cross-origin one: the frame keeps its fixed height */ }
  }
  function postHeight() { if (heightPending) return; heightPending = true; if (window.requestAnimationFrame) window.requestAnimationFrame(sendHeight); else setTimeout(sendHeight, 16); }
  postHeight();
  window.addEventListener('load', postHeight);
  window.addEventListener('resize', postHeight);
  if (window.ResizeObserver) new ResizeObserver(postHeight).observe(document.body);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(postHeight, function () {});
})();
