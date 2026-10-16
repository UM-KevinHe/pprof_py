/* pprof_py documentation: interactive figures (every display the gallery renders: funnel plot, interval plot,
   observed versus expected, coefficient forest, data quality, null calibration, reliability, between-provider
   variation, shrinkage, flag stability, several measures and measure agreement).
   Plain JavaScript and inline SVG, no dependencies. Inlined into _static/interactive/<name>.html by
   _ext/pprof_interactive.py; the data is the JSON record in <script id="pprof-data">. Each display draws an SVG
   chart with a hover layer (tooltip), a legend that names every category (shape and colour, never colour alone),
   and a sortable table linked to the chart by hover and keyboard focus (one row per provider, or per term or bin
   where the display has no provider marks), with an optional second table of the series behind the figure. */
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
        var mode = el.getAttribute('data-color-mode') || el.getAttribute('data-theme');   // Shibuya, or a stamp
        if (mode === 'dark' || mode === 'light') return mode;
      }
    } catch (e) { /* a cross-origin frame: prefers-color-scheme decides */ }
    return null;
  }
  function applyTheme() {
    var q = /[?&]theme=(dark|light)\b/.exec(window.location.search);
    var mode = q ? q[1] : parentMode();
    if (mode) document.documentElement.setAttribute('data-theme', mode);
    else document.documentElement.removeAttribute('data-theme');
  }
  applyTheme();
  try {
    if (window.parent && window.parent !== window && window.MutationObserver) {
      new MutationObserver(applyTheme).observe(window.parent.document.documentElement,
        { attributes: true, attributeFilter: ['data-color-mode', 'data-theme'] });
    }
  } catch (e) { /* cross-origin: nothing to follow */ }

  /* ---------- helpers ---------- */
  function h(tag, attrs, children) {
    var el = tag.indexOf('svg:') === 0 ? document.createElementNS(SVGNS, tag.slice(4)) : document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (attrs[k] === null || attrs[k] === undefined) return;
      if (k === 'text') el.textContent = attrs[k]; else el.setAttribute(k, attrs[k]);
    });
    (children || []).forEach(function (c) { if (c) el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c); });
    return el;
  }
  function minus(s) { return String(s).replace(/-/g, '−'); }
  function fmt(v, col) {
    if (v === null || v === undefined || (typeof v === 'number' && isNaN(v))) return col && col.missing ? col.missing : '–';
    if (col && col.p) return v < 0.0005 ? '< 0.001' : Number(v).toFixed(3);
    if (col && col.percent) return (Number(v) * 100).toFixed(1) + '%';
    var d = col && col.digits !== undefined ? col.digits : 2;
    var s = Number(v).toFixed(d);
    if (d === 0) s = Number(s).toLocaleString('en-US');
    return minus(s);
  }
  function fmtTick(v) { var a = Math.abs(v); if (a !== 0 && (a < 1e-3 || a >= 1e6)) return minus(v.toExponential(0)); return minus(Number(v.toFixed(6)).toLocaleString('en-US', { maximumFractionDigits: 6 })); }
  function niceTicks(lo, hi, n) {
    var span = hi - lo || 1, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step;
    step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    var t = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + 1e-12; v += step) t.push(+v.toFixed(10));
    return t;
  }
  function logTicks(lo, hi) {       // decades with 2 and 5, as the package's log_ticks
    var out = [], major = [];
    for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      [1, 2, 5].forEach(function (m) { var v = m * Math.pow(10, e); if (v >= lo && v <= hi) { out.push(v); if (m === 1) major.push(v); } });
    }
    return { all: out, major: major.length >= 2 ? major : out };
  }
  function nice(v) { var e = Math.pow(10, Math.floor(Math.log10(v))), m = v / e; return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * e; }
  function sqrtTicks(top) {          // the package's sqrt_ticks: even on the root scale, rounded to nice values
    var raw = [], ticks = {};
    for (var i = 1; i <= 5; i++) raw.push(Math.pow(Math.sqrt(top) * i / 5, 2));
    raw.forEach(function (v) { if (v > 0) ticks[nice(v)] = true; });
    return [0].concat(Object.keys(ticks).map(Number).filter(function (v) { return v <= top && (v >= 1 || top < 3); }).sort(function (a, b) { return a - b; }));
  }
  function extent(values) { var lo = Infinity, hi = -Infinity; values.forEach(function (v) { if (v !== null && v !== undefined && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }); return [lo, hi]; }
  function median(values) { var v = values.slice().sort(function (a, b) { return a - b; }); return v.length ? v[Math.floor((v.length - 1) / 2)] : NaN; }
  function spread(values, gap, lo, hi) {   // push labels apart so none overlap, inside [lo, hi]
    var idx = values.map(function (v, i) { return i; }).sort(function (a, b) { return values[a] - values[b]; });
    var pos = values.slice();
    for (var k = 1; k < idx.length; k++) pos[idx[k]] = Math.max(pos[idx[k]], pos[idx[k - 1]] + gap);
    var over = pos[idx[idx.length - 1]] - hi; if (over > 0) idx.forEach(function (i) { pos[i] -= over; });
    for (var j = 1; j < idx.length; j++) pos[idx[j]] = Math.max(pos[idx[j]], pos[idx[j - 1]] + gap);
    return pos.map(function (p) { return Math.min(hi, Math.max(lo, p)); });
  }
  function pdf(x, m, s) { var z = (x - m) / s; return Math.exp(-0.5 * z * z) / (s * Math.sqrt(2 * Math.PI)); }
  function rng(seed) {               // mulberry32: deterministic jitter, the same on every load
    var a = seed >>> 0;
    return function () { a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function wrap(text, width) {       // whole words, at most `width` characters per line
    var lines = [], line = '';
    String(text).split(/\s+/).forEach(function (w) { if (line && (line + ' ' + w).length > width) { lines.push(line); line = w; } else line = line ? line + ' ' + w : w; });
    if (line) lines.push(line);
    return lines;
  }
  function linear(lo, hi, a, b) { return function (v) { return a + (v - lo) / ((hi - lo) || 1) * (b - a); }; }
  function logScale(lo, hi, a, b) { var l = Math.log10(lo), r = Math.log10(hi); return function (v) { return a + (Math.log10(v) - l) / ((r - l) || 1) * (b - a); }; }

  /* status marks: shape and colour (CSS variable --c per status), filled with a surface-coloured halo or open */
  var SHAPES = {
    triangle_up: function (x, y, s) { return h('svg:path', { d: 'M' + x + ',' + (y - s * 1.15) + 'L' + (x + s) + ',' + (y + s * 0.65) + 'L' + (x - s) + ',' + (y + s * 0.65) + 'Z', class: 'filled' }); },
    triangle_down: function (x, y, s) { return h('svg:path', { d: 'M' + x + ',' + (y + s * 1.15) + 'L' + (x + s) + ',' + (y - s * 0.65) + 'L' + (x - s) + ',' + (y - s * 0.65) + 'Z', class: 'filled' }); },
    triangle_left: function (x, y, s) { return h('svg:path', { d: 'M' + (x - s * 1.15) + ',' + y + 'L' + (x + s * 0.65) + ',' + (y - s) + 'L' + (x + s * 0.65) + ',' + (y + s) + 'Z', class: 'filled' }); },
    triangle_right: function (x, y, s) { return h('svg:path', { d: 'M' + (x + s * 1.15) + ',' + y + 'L' + (x - s * 0.65) + ',' + (y - s) + 'L' + (x - s * 0.65) + ',' + (y + s) + 'Z', class: 'filled' }); },
    circle: function (x, y, s) { return h('svg:circle', { cx: x, cy: y, r: s * 0.8, class: 'filled' }); },
    circle_open: function (x, y, s) { return h('svg:circle', { cx: x, cy: y, r: s * 0.9, class: 'open' }); },
    square_open: function (x, y, s) { return h('svg:rect', { x: x - s * 0.85, y: y - s * 0.85, width: s * 1.7, height: s * 1.7, class: 'open' }); },
    square: function (x, y, s) { return h('svg:rect', { x: x - s * 0.8, y: y - s * 0.8, width: s * 1.6, height: s * 1.6, class: 'filled' }); },
    diamond: function (x, y, s) { return h('svg:path', { d: 'M' + x + ',' + (y - s * 1.1) + 'L' + (x + s * 1.1) + ',' + y + 'L' + x + ',' + (y + s * 1.1) + 'L' + (x - s * 1.1) + ',' + y + 'Z', class: 'filled' }); }
  };
  function glyph(name, x, y, s, cls) { var el = SHAPES[name](x, y, s); el.setAttribute('class', cls); return el; }
  function markOf(p) { return p.edge ? (p.edge === 'lower' ? 'triangle_left' : 'triangle_right') : p.zero_events ? 'square_open' : D.statuses[p.status].marker; }
  function statusKey(p) { return p.edge ? 'no_finite_estimate' : p.zero_events ? 'zero_events' : p.status; }
  function mark(p, x, y, size) {
    var el = SHAPES[markOf(p)](x, y, size);
    el.setAttribute('class', 'mark ' + el.getAttribute('class') + ' st-' + statusKey(p));
    el.setAttribute('data-id', p.id);
    return el;
  }
  function statusMark(status, id, x, y, size) {   // a status mark for one value of a provider (scenario, measure)
    var el = SHAPES[D.statuses[status].marker](x, y, size);
    el.setAttribute('class', 'mark ' + el.getAttribute('class') + ' st-' + status);
    el.setAttribute('data-id', id);
    return el;
  }
  function markSize(p, dense) { var s = p.status === 'above' || p.status === 'below' ? 5 : p.status === 'not_different' ? 3.4 : 4.2; return dense && p.status === 'not_different' ? s * 0.8 : s; }
  function statusSize(status, dense) { return markSize({ status: status }, dense); }

  /* ---------- tooltip and linking between chart and table ---------- */
  var wrap_, tip, marks = {}, rows = {}, current = [];
  function tooltip(container) {
    var el = h('div', { class: 'tip', 'aria-hidden': 'true' }); container.appendChild(el);   // the table carries the same values for assistive technology
    return {
      show: function (x, y, html) {
        el.innerHTML = html; el.classList.add('show');
        var w = container.clientWidth, tw = el.offsetWidth, th = el.offsetHeight;
        el.style.left = Math.max(0, Math.min(w - tw, x + 14)) + 'px';
        el.style.top = (y - th - 10 < 0 ? y + 16 : y - th - 10) + 'px';
      },
      hide: function () { el.classList.remove('show'); }
    };
  }
  function colType(c) { return c.type || (c.key === 'id' ? 'id' : c.key === 'status' ? 'status' : 'num'); }
  function cell(p, c) {
    var t = colType(c);
    if (t === 'id') return p.id;
    if (t === 'status') { var k = c.key === 'status' ? statusKey(p) : p[c.key]; return k && D.statuses[k] ? D.statuses[k].symbol + ' ' + D.statuses[k].label : (c.missing || '–'); }
    if (t === 'text') { var v = p[c.key]; return v === null || v === undefined || v === '' ? (c.missing || '–') : String(v); }
    return fmt(p[c.key], c);
  }
  function tipHtml(p, columns, title) {
    var s = '<b>' + (title || ((D.id_label === 'term' ? 'Term ' : 'Provider ') + p.id)) + '</b>';
    (columns || D.columns).forEach(function (c) { if (c.key !== 'id') s += '<div class="row"><span>' + c.label + '</span><span>' + cell(p, c) + '</span></div>'; });
    if (p.edge) s += '<div class="row"><span>Estimate</span><span>no finite estimate; shown at the axis edge</span></div>';
    return s;
  }
  function highlight(ids) {          // one id, a list of ids, or null
    ids = ids === null || ids === undefined ? [] : Array.isArray(ids) ? ids : [ids];
    current.forEach(function (id) { (marks[id] || []).forEach(function (m) { m.classList.remove('hl'); }); if (rows[id]) rows[id].classList.remove('hl'); });
    current = ids;
    root.classList.toggle('has-hl', ids.length > 0);
    ids.forEach(function (id) { (marks[id] || []).forEach(function (m) { m.classList.add('hl'); }); if (rows[id]) rows[id].classList.add('hl'); });
  }
  function bindHit(hit, p, html, ids) {
    hit.addEventListener('mouseenter', function () { highlight(ids || p.id); });
    hit.addEventListener('mousemove', function (e) { var r = wrap_.getBoundingClientRect(); tip.show(e.clientX - r.left, e.clientY - r.top, html ? (typeof html === 'function' ? html() : html) : tipHtml(p)); });
    hit.addEventListener('mouseleave', function () { tip.hide(); highlight(null); });
  }
  function showAtMark(p) {           // keyboard: the tooltip beside the provider's mark
    var m = marks[p.id] && marks[p.id][0]; if (!m) return;
    var r = m.getBoundingClientRect(), w = wrap_.getBoundingClientRect();
    tip.show(r.left - w.left + r.width / 2, r.top - w.top, tipHtml(p));
  }
  function register(p, el) { (marks[p.id] = marks[p.id] || []).push(el); }

  /* ---------- legend ---------- */
  var LINE_CLASS = { own: 'curve own', other: 'curve other', reference: 'reference', limit: 'limit', guide: 'guide', interval: 'interval' };
  function legend() {
    var ul = h('ul', { class: 'legend', 'aria-label': 'Legend' });
    D.legend.forEach(function (e) {
      var sw = h('svg:svg', { class: 'swatch', viewBox: '0 0 26 14', 'aria-hidden': 'true' }), el;
      if (e.kind === 'mark') { el = SHAPES[D.statuses[e.status].marker](13, 7, 5); el.setAttribute('class', 'mark ' + el.getAttribute('class') + ' st-' + e.status); }
      else if (e.kind === 'shape') el = glyph(e.shape, 13, 7, e.area ? Math.sqrt(e.size) * 0.9 : (e.size || 5), e.cls);
      else if (e.kind === 'band') el = h('svg:rect', { x: 1, y: 2, width: 24, height: 10, class: e.cls || 'band', style: e.cls ? null : 'stroke: var(--ink-2); stroke-width: 1' });
      else if (e.kind === 'bar') el = h('svg:rect', { x: 1, y: 4, width: 24, height: 6, class: 'volume' });
      else el = h('svg:line', { x1: 1, x2: 25, y1: 7, y2: 7, class: e.cls || LINE_CLASS[e.style] || 'curve own' });
      sw.appendChild(el);
      ul.appendChild(h('li', {}, [sw, e.label]));
    });
    return ul;
  }

  /* ---------- the tables: a linked table of the marks (providers or terms), optionally a table of the series ---------- */
  var STATUS_ORDER = ['above', 'below', 'not_different', 'not_tested', 'suppressed', 'no_finite_estimate', 'zero_events', 'excluded'];
  function controls(container) {
    var label = D.id_label || 'provider';
    var find = h('input', { type: 'search', placeholder: label + ' id', 'aria-label': 'Find a ' + label + ' by id' });
    var only = h('button', { type: 'button', 'aria-pressed': 'false', text: 'Table only' });
    var count = h('span', { class: 'n' });
    container.appendChild(h('div', { class: 'controls' }, [h('label', {}, ['Find ', find]), only, count]));
    only.addEventListener('click', function () { var on = only.getAttribute('aria-pressed') !== 'true'; only.setAttribute('aria-pressed', on ? 'true' : 'false'); only.textContent = on ? 'Chart and table' : 'Table only'; root.classList.toggle('table-only', on); });
    return { find: find, count: count };
  }
  function table(container, spec, ctl) {
    var cols = spec.columns, list0 = spec.rows, linked = !!spec.linked;
    var state = { sort: spec.ordered ? null : cols[0].key, dir: 'asc', q: '' };   // ordered: keep the given order until a header is clicked
    var scroll = h('div', { class: 'table-scroll' + (linked ? '' : ' series') });
    var tbl = h('table'), thead = h('thead'), tbody = h('tbody'), tr = h('tr');
    tbl.appendChild(h('caption', { text: spec.caption }));
    cols.forEach(function (c) {
      var t = colType(c);
      var th = h('th', { scope: 'col', class: t === 'num' ? 'num' : '' });
      var b = h('button', { type: 'button', text: c.label });
      b.addEventListener('click', function () {
        if (state.sort === c.key) state.dir = state.dir === 'asc' ? 'desc' : 'asc';
        else { state.sort = c.key; state.dir = t === 'num' ? 'desc' : 'asc'; }
        body();
      });
      th.appendChild(b); tr.appendChild(th);
    });
    thead.appendChild(tr); tbl.appendChild(thead); tbl.appendChild(tbody); scroll.appendChild(tbl);
    function value(p) {
      var c = cols.filter(function (c) { return c.key === state.sort; })[0], t = colType(c);
      if (t === 'id') return p.id;
      if (t === 'status') { var k = c.key === 'status' ? statusKey(p) : p[c.key]; var i = STATUS_ORDER.indexOf(k); return i < 0 ? 99 : i + (p.edge || p.zero_events ? 0.5 : 0); }
      if (t === 'text') return String(p[c.key] === null || p[c.key] === undefined ? '' : p[c.key]);
      var v = p[c.key]; return v === null || v === undefined ? (state.dir === 'asc' ? Infinity : -Infinity) : v;
    }
    function body() {
      Array.prototype.forEach.call(thead.querySelectorAll('th'), function (th, i) { if (cols[i].key === state.sort) th.setAttribute('aria-sort', state.dir === 'asc' ? 'ascending' : 'descending'); else th.removeAttribute('aria-sort'); });
      var dir = state.dir === 'asc' ? 1 : -1, q = state.q.toLowerCase();
      var list = list0.filter(function (p) { return !q || String(p.id).toLowerCase().indexOf(q) >= 0; });
      if (state.sort !== null) list.sort(function (a, b) { var va = value(a), vb = value(b); return (va > vb ? 1 : va < vb ? -1 : 0) * dir; });
      tbody.innerHTML = ''; if (linked) rows = {};
      list.forEach(function (p) {
        var r = h('tr', { tabindex: linked ? '0' : null, 'data-id': linked ? p.id : null, class: p.status === 'above' || p.status === 'below' ? 'flagged' : '' });
        cols.forEach(function (c) { var t = colType(c); r.appendChild(h('td', { class: t === 'id' ? '' : t === 'status' ? 'status' : t === 'text' ? 'text' + (c.wrap ? ' wrap' : '') : 'num', text: cell(p, c) })); });
        if (linked) {
          r.addEventListener('mouseenter', function () { highlight(p.id); });
          r.addEventListener('mouseleave', function () { if (document.activeElement !== r) highlight(null); });
          r.addEventListener('focus', function () { highlight(p.id); showAtMark(p); });
          r.addEventListener('blur', function () { highlight(null); tip.hide(); });
          rows[p.id] = r;
        }
        tbody.appendChild(r);
      });
      if (ctl) ctl.count.textContent = list.length + ' of ' + list0.length + ' ' + (D.id_label || 'provider') + 's';
    }
    if (ctl) ctl.find.addEventListener('input', function () { state.q = ctl.find.value.trim(); body(); });
    body();
    container.appendChild(scroll);
  }

  /* ---------- the figure scaffold ---------- */
  function figure(W, H, alt) {
    var fig = h('figure', { class: 'fig' });
    wrap_ = h('div', { class: 'fig-wrap' }); fig.appendChild(wrap_); tip = tooltip(wrap_);
    var svg = h('svg:svg', { class: 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-labelledby': 'chart-title chart-desc' });
    svg.appendChild(h('svg:title', { id: 'chart-title', text: alt }));
    svg.appendChild(h('svg:desc', { id: 'chart-desc', text: D.caption }));
    wrap_.appendChild(svg);
    return { fig: fig, svg: svg };
  }
  function axisText(svg, x, y, text, anchor, rotate) {
    svg.appendChild(h('svg:text', { class: 'axis-label', x: x, y: y, 'text-anchor': anchor || 'middle', transform: rotate ? 'rotate(-90 ' + x + ' ' + y + ')' : null, text: text }));
  }
  function xTicks(grid, axis, ticks, sx, top, bottom, labels) {
    ticks.forEach(function (t) { grid.appendChild(h('svg:line', { x1: sx(t), x2: sx(t), y1: top, y2: bottom })); if (labels !== false) axis.appendChild(h('svg:text', { x: sx(t), y: bottom + 16, 'text-anchor': 'middle', text: fmtTick(t) })); });
  }
  function yTicks(grid, axis, ticks, sy, left, right) {
    ticks.forEach(function (t) { grid.appendChild(h('svg:line', { x1: left, x2: right, y1: sy(t), y2: sy(t) })); axis.appendChild(h('svg:text', { x: left - 8, y: sy(t) + 4, 'text-anchor': 'end', text: fmtTick(t) })); });
  }
  function clip(svg, id, x, y, w, hgt) { svg.appendChild(h('svg:defs', {}, [h('svg:clipPath', { id: id }, [h('svg:rect', { x: x, y: y, width: w, height: hgt })])])); }

  /* ---------- funnel plot ---------- */
  function funnel(container) {
    var P = D.providers, W = 780, H = 470, m = { t: 14, r: 66, b: 46, l: 58 };
    var xe = extent(P.map(function (p) { return p.precision; })), span = Math.log10(xe[1] / xe[0]) || 1;
    var xlo = xe[0] / Math.pow(10, 0.06 * span), xhi = xe[1] * Math.pow(10, 0.06 * span);
    var med = median(P.map(function (p) { return p.precision; }));
    var ys = P.map(function (p) { return p.ratio; });
    D.curves.forEach(function (c) { c.precision.forEach(function (x, i) { if (x >= med) { ys.push(c.lower[i]); ys.push(c.upper[i]); } }); });
    if (D.reference.value !== null) ys.push(D.reference.value);
    var ye = extent(ys), pad = 0.07 * ((ye[1] - ye[0]) || 1), ylo = Math.min(0, ye[0]) - pad, yhi = ye[1] + pad;
    var sx = logScale(xlo, xhi, m.l, W - m.r), sy = linear(ylo, yhi, H - m.b, m.t);
    var f = figure(W, H, D.alt), svg = f.svg;
    clip(svg, 'plot', m.l, m.t, W - m.l - m.r, H - m.t - m.b);
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
    var lt = logTicks(xlo, xhi);
    lt.all.forEach(function (t) { grid.appendChild(h('svg:line', { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b })); });
    lt.major.forEach(function (t) { axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 16, 'text-anchor': 'middle', text: fmtTick(t) })); });
    yTicks(grid, axis, niceTicks(ylo, yhi, 6), sy, m.l, W - m.r);
    axis.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, (m.l + W - m.r) / 2, H - 6, D.axes.x.label);
    axisText(svg, 14, (m.t + H - m.b) / 2, D.axes.y.label, 'middle', true);
    var plot = h('svg:g', { 'clip-path': 'url(#plot)' }); svg.appendChild(plot);
    var labels = [];
    D.curves.forEach(function (c) {                     // the own-level corridor, then every curve
      var pts = function (arr) { return c.precision.map(function (x, i) { return arr[i] === null ? null : sx(x) + ',' + sy(arr[i]); }).filter(Boolean); };
      if (c.own && D.legend.some(function (e) { return e.kind === 'band'; })) {
        plot.appendChild(h('svg:polygon', { class: 'band', points: pts(c.upper).concat(pts(c.lower).reverse()).join(' ') }));
      }
      ['upper', 'lower'].forEach(function (side) { plot.appendChild(h('svg:polyline', { class: 'curve ' + (c.own ? 'own' : 'other'), points: pts(c[side]).join(' ') })); });
      var last = c.upper[c.upper.length - 1]; if (last !== null) labels.push({ y: last, text: (c.level * 100).toLocaleString('en-US') + '%' });
    });
    if (D.reference.value !== null) {
      plot.appendChild(h('svg:line', { class: 'reference', x1: m.l, x2: W - m.r, y1: sy(D.reference.value), y2: sy(D.reference.value) }));
      labels.push({ y: D.reference.value, text: fmt(D.reference.value, { digits: 2 }) });
    }
    var ly = spread(labels.map(function (l) { return sy(l.y); }), 13, m.t + 6, H - m.b - 4);
    labels.forEach(function (l, i) { svg.appendChild(h('svg:text', { class: 'curve-label', x: W - m.r + 6, y: ly[i] + 4, text: l.text })); });
    var layer = h('svg:g'), hits = h('svg:g');
    ['not_different', 'not_tested', 'below', 'above'].forEach(function (key) {
      P.forEach(function (p) { if (p.status !== key) return; var el = mark(p, sx(p.precision), sy(p.ratio), markSize(p)); layer.appendChild(el); register(p, el); });
    });
    P.forEach(function (p) { var hit = h('svg:circle', { class: 'hit', cx: sx(p.precision), cy: sy(p.ratio), r: 8 }); bindHit(hit, p); hits.appendChild(hit); });
    svg.appendChild(layer); svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- interval plot ("caterpillar") with a volume panel ---------- */
  function caterpillar(container) {
    var P = D.providers, n = P.length, labelled = n <= 60, rowH = labelled ? 14 : 3.4, dense = !labelled;
    var W = 780, m = { t: 12, r: 16, b: 46, l: labelled ? 70 : 34 }, gap = 28, volW = 150;
    var H = m.t + n * rowH + m.b, mainR = W - m.r - volW - gap;
    var xs = [];
    P.forEach(function (p) { if (p.estimate !== null) xs.push(p.estimate); if (p.lower !== null) xs.push(p.lower); if (p.upper !== null) xs.push(p.upper); });
    if (D.reference.value !== null) xs.push(D.reference.value);
    var xe = extent(xs), pad = 0.04 * ((xe[1] - xe[0]) || 1), xlo = xe[0] - pad, xhi = xe[1] + pad;
    var vmax = extent(P.map(function (p) { return p.volume; }))[1] || 1;
    var sx = linear(xlo, xhi, m.l, mainR);
    function sy(i) { return m.t + (i + 0.5) * rowH; }
    function sv(v) { return mainR + gap + v / (vmax * 1.05) * volW; }
    var f = figure(W, H, D.alt), svg = f.svg;
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
    xTicks(grid, axis, niceTicks(xlo, xhi, 6), sx, m.t, H - m.b);
    xTicks(grid, axis, niceTicks(0, vmax, 3), sv, m.t, H - m.b);
    axis.appendChild(h('svg:line', { x1: m.l, x2: mainR, y1: H - m.b, y2: H - m.b }));
    axis.appendChild(h('svg:line', { x1: mainR + gap, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, (m.l + mainR) / 2, H - 6, D.axes.x.label + (D.reference.value === null ? '' : '; reference ' + fmt(D.reference.value, { digits: 2 })));
    axisText(svg, mainR + gap + volW / 2, H - 6, D.axes.volume.label);
    axisText(svg, 12, (m.t + H - m.b) / 2, D.axes.y.label, 'middle', true);
    if (D.reference.value !== null) svg.appendChild(h('svg:line', { class: 'reference', x1: sx(D.reference.value), x2: sx(D.reference.value), y1: m.t, y2: H - m.b }));
    var bars = h('svg:g'), layer = h('svg:g'), vols = h('svg:g'), hits = h('svg:g');
    P.forEach(function (p, i) {
      var y = sy(i), lo = p.lower === null ? xlo : Math.max(xlo, p.lower), hi = p.upper === null ? xhi : Math.min(xhi, p.upper);
      if (p.has_interval && (!p.edge || (p.lower !== null) !== (p.upper !== null))) {    // one-sided: from the edge to the finite bound
        var seg = h('svg:line', { class: 'interval st-' + p.status, 'data-id': p.id, x1: sx(lo), x2: sx(hi), y1: y, y2: y, style: dense ? 'stroke-width: 1.6' : null });
        bars.appendChild(seg); register(p, seg);
      }
      var el = mark(p, p.edge ? (p.edge === 'lower' ? m.l : mainR) : sx(p.estimate), y, markSize(p, dense));
      layer.appendChild(el); register(p, el);
      if (p.volume !== null) { var v = h('svg:rect', { class: 'volume', 'data-id': p.id, x: sv(0), y: y - rowH * (labelled ? 0.3 : 0.4), width: sv(p.volume) - sv(0), height: rowH * (labelled ? 0.6 : 0.8) }); vols.appendChild(v); register(p, v); }
      if (labelled) svg.appendChild(h('svg:text', { class: 'row-label', x: m.l - 6, y: y + 4, 'text-anchor': 'end', text: p.id }));
      var hit = h('svg:rect', { class: 'hit', x: m.l, y: y - rowH / 2, width: W - m.r - m.l, height: rowH }); bindHit(hit, p); hits.appendChild(hit);
    });
    svg.appendChild(bars); svg.appendChild(layer); svg.appendChild(vols); svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- observed versus expected ---------- */
  function observed(container) {
    var P = D.providers, W = 600, H = 560, m = { t: 14, r: 60, b: 46, l: 58 };
    var top = Math.max(extent(P.map(function (p) { return p.observed; }))[1], extent(P.map(function (p) { return p.expected; }))[1]) * 1.06;
    function sx(v) { return m.l + Math.sqrt(Math.max(v, 0) / top) * (W - m.l - m.r); }
    function sy(v) { return H - m.b - Math.sqrt(Math.max(v, 0) / top) * (H - m.t - m.b); }
    var f = figure(W, H, D.alt), svg = f.svg;
    clip(svg, 'plot', m.l, m.t, W - m.l - m.r, H - m.t - m.b);
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
    sqrtTicks(top).forEach(function (t) {
      grid.appendChild(h('svg:line', { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b })); grid.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) }));
      axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 16, 'text-anchor': 'middle', text: fmtTick(t) })); axis.appendChild(h('svg:text', { x: m.l - 8, y: sy(t) + 4, 'text-anchor': 'end', text: fmtTick(t) }));
    });
    axis.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b })); axis.appendChild(h('svg:line', { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b }));
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, (m.l + W - m.r) / 2, H - 6, D.axes.x.label);
    axisText(svg, 14, (m.t + H - m.b) / 2, D.axes.y.label, 'middle', true);
    var plot = h('svg:g', { 'clip-path': 'url(#plot)' }); svg.appendChild(plot);
    function curve(ratio, cls) { var pts = []; for (var i = 0; i <= 60; i++) { var e = top * Math.pow(i / 60, 2); pts.push(sx(e) + ',' + sy(ratio * e)); } plot.appendChild(h('svg:polyline', { class: cls, points: pts.join(' ') })); }
    curve(0.5, 'guide'); curve(2, 'guide'); curve(1, 'reference');
    [[1, 'O = E', 'reference'], [0.5, 'O/E = 0.5', 'guide']].forEach(function (l) { svg.appendChild(h('svg:text', { class: 'curve-label', x: W - m.r + 5, y: sy(l[0] * top * 0.97) + 4, text: l[1] })); });
    svg.appendChild(h('svg:text', { class: 'curve-label', x: sx(top * 0.5 * 0.97) - 2, y: m.t + 12, 'text-anchor': 'end', text: 'O/E = 2' }));
    var limits = h('svg:g'), layer = h('svg:g'), hits = h('svg:g');
    P.forEach(function (p) { ['lower', 'upper'].forEach(function (s) { if (p[s] === null) return; var l = h('svg:line', { class: 'limit', 'data-id': p.id, x1: sx(p.expected) - 4, x2: sx(p.expected) + 4, y1: sy(p[s]), y2: sy(p[s]) }); limits.appendChild(l); }); });
    ['not_different', 'not_tested', 'below', 'above'].forEach(function (key) {
      P.forEach(function (p) { if (p.status !== key) return; var el = mark(p, sx(p.expected), sy(p.observed), markSize(p)); layer.appendChild(el); register(p, el); });
    });
    P.forEach(function (p) { var hit = h('svg:circle', { class: 'hit', cx: sx(p.expected), cy: sy(p.observed), r: 8 }); bindHit(hit, p); hits.appendChild(hit); });
    svg.appendChild(limits); svg.appendChild(layer); svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- coefficient forest ---------- */
  function forest(container) {
    var P = D.terms, k = P.length, W = 780, rowH = 36, m = { t: 26, r: 200, b: 46, l: 90 };
    var H = m.t + k * rowH + m.b, log = D.axes.x.scale === 'log', ref = D.reference.value;
    var xs = [ref];
    P.forEach(function (p) { xs.push(p.estimate); xs.push(p.lower); xs.push(p.upper); });
    var xe = extent(xs), xlo, xhi, sx, ticks;
    if (log) { var span = Math.log10(xe[1] / xe[0]) || 1; xlo = xe[0] / Math.pow(10, 0.06 * span); xhi = xe[1] * Math.pow(10, 0.06 * span); sx = logScale(xlo, xhi, m.l, W - m.r); ticks = xhi / xlo < 10 ? niceTicks(xlo, xhi, 5).filter(function (t) { return t > 0; }) : logTicks(xlo, xhi).major; }
    else { var pad = 0.06 * ((xe[1] - xe[0]) || 1); xlo = xe[0] - pad; xhi = xe[1] + pad; sx = linear(xlo, xhi, m.l, W - m.r); ticks = niceTicks(xlo, xhi, 6); }
    function sy(i) { return m.t + (i + 0.5) * rowH; }
    var f = figure(W, H, D.alt), svg = f.svg;
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
    xTicks(grid, axis, ticks, sx, m.t, H - m.b);
    axis.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, (m.l + W - m.r) / 2, H - 6, D.axes.x.label);
    if (ref !== null) svg.appendChild(h('svg:line', { class: 'reference', x1: sx(ref), x2: sx(ref), y1: m.t, y2: H - m.b }));
    svg.appendChild(h('svg:text', { class: 'curve-label', x: W - 12, y: m.t - 8, 'text-anchor': 'end', text: D.column_head }));
    var layer = h('svg:g'), hits = h('svg:g');
    P.forEach(function (p, i) {
      var y = sy(i), lo = Math.max(xlo, p.lower === null ? xlo : p.lower), hi = Math.min(xhi, p.upper === null ? xhi : p.upper);
      var seg = h('svg:line', { class: 'interval ink', 'data-id': p.id, x1: sx(lo), x2: sx(hi), y1: y, y2: y }); layer.appendChild(seg); register(p, seg);
      var el = glyph('circle', sx(p.estimate), y, 5, 'mark filled dot'); el.setAttribute('data-id', p.id); layer.appendChild(el); register(p, el);
      svg.appendChild(h('svg:text', { class: 'row-label big', x: m.l - 8, y: y + 4, 'text-anchor': 'end', text: p.id }));
      svg.appendChild(h('svg:text', { class: 'row-label big', x: W - 12, y: y + 4, 'text-anchor': 'end', text: p.text }));
      var hit = h('svg:rect', { class: 'hit', x: 0, y: y - rowH / 2, width: W, height: rowH }); bindHit(hit, p); hits.appendChild(hit);
    });
    svg.appendChild(layer); svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- data quality: provider accounting and volume by group ---------- */
  function quality(container) {
    var A = D.accounting, G = D.groups, P = D.providers, hasVol = D.has_volume && G.length;
    var W = 780, m = { t: 14, b: 46 }, rowH = 24, aL = 190, aR = hasVol ? 395 : W - 30, vL = 520, vR = W - 20;
    var gh = 60, H = m.t + Math.max(A.length * rowH, hasVol ? G.length * gh : 0) + m.b;
    var top = Math.max.apply(null, A.map(function (a) { return a.count || 0; }).concat([1]));
    var sx = linear(0, top * 1.22, aL, aR);
    var f = figure(W, H, D.alt), svg = f.svg;
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
    xTicks(grid, axis, niceTicks(0, top * 1.22, 5).filter(function (t) { return t <= top * 1.22; }), sx, m.t, H - m.b);
    axis.appendChild(h('svg:line', { x1: aL, x2: aR, y1: H - m.b, y2: H - m.b }));
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, (aL + aR) / 2, H - 6, D.axes.x.label);
    var bars = h('svg:g'), hits = h('svg:g'), firstAttr = -1;
    A.forEach(function (a, i) {
      var y = m.t + (i + 0.5) * rowH;
      if (a.key === 'no_finite_estimate' && firstAttr < 0) { firstAttr = i; svg.appendChild(h('svg:line', { class: 'separator', x1: aL, x2: aR, y1: y - rowH / 2, y2: y - rowH / 2 })); }
      svg.appendChild(h('svg:text', { class: 'row-label', x: aL - 8, y: y + 4, 'text-anchor': 'end', text: a.label }));
      if (a.count === null) svg.appendChild(h('svg:text', { class: 'annot muted', x: aL + 4, y: y + 4, text: 'not recorded' }));
      else {
        var cls = ['above', 'below', 'not_different', 'not_tested'].indexOf(a.key) >= 0 ? 'abar st-' + a.key : ['total', 'analysed'].indexOf(a.key) >= 0 ? 'abar ink' : a.key === 'excluded' ? 'abar excl' : 'abar attr';
        bars.appendChild(h('svg:rect', { class: cls, x: sx(0), y: y - rowH * 0.31, width: Math.max(sx(a.count) - sx(0), a.count > 0 ? 1.5 : 0), height: rowH * 0.62 }));
        svg.appendChild(h('svg:text', { class: 'annot', x: sx(a.count) + 5, y: y + 4, text: fmt(a.count, { digits: 0 }) }));
      }
      var html = '<b>' + a.label + '</b><div class="row"><span>Providers</span><span>' + fmt(a.count, { digits: 0, missing: 'not recorded' }) + '</span></div>' +
        '<div class="row"><span>Share</span><span>' + fmt(a.share, { percent: true }) + '</span></div><div class="row"><span>Definition</span><span>' + a.definition + '</span></div>';
      var hit = h('svg:rect', { class: 'hit', x: 0, y: y - rowH / 2, width: aR + 10, height: rowH }); bindHit(hit, null, html, a.ids); hits.appendChild(hit);
    });
    svg.appendChild(bars);
    if (hasVol) {
      var vols = P.map(function (p) { return p.volume; }).filter(function (v) { return v !== null && v > 0; });
      var ve = extent(vols), span = Math.log10(ve[1] / ve[0]) || 1, vlo = ve[0] / Math.pow(10, 0.08 * span), vhi = ve[1] * Math.pow(10, 0.08 * span);
      var sv = logScale(vlo, vhi, vL, vR), lt = logTicks(vlo, vhi), g2 = h('svg:g', { class: 'grid' }), a2 = h('svg:g', { class: 'axis' });
      lt.all.forEach(function (t) { g2.appendChild(h('svg:line', { x1: sv(t), x2: sv(t), y1: m.t, y2: H - m.b })); });
      (lt.all.length <= 8 ? lt.all : lt.major).forEach(function (t) { a2.appendChild(h('svg:text', { x: sv(t), y: H - m.b + 16, 'text-anchor': 'middle', text: fmtTick(t) })); });
      a2.appendChild(h('svg:line', { x1: vL, x2: vR, y1: H - m.b, y2: H - m.b }));
      svg.appendChild(g2); svg.appendChild(a2);
      axisText(svg, (vL + vR) / 2, H - 6, D.axes.volume.label);
      if (D.min_volume !== null) svg.appendChild(h('svg:line', { class: 'reference dashed', x1: sv(D.min_volume), x2: sv(D.min_volume), y1: m.t, y2: H - m.b }));
      var random = rng(20261001), layer = h('svg:g'), vhits = h('svg:g');
      G.forEach(function (g, gi) {
        var y0 = m.t + (gi + 0.5) * gh;
        svg.appendChild(h('svg:text', { class: 'row-label', x: vL - 8, y: y0 + 4, 'text-anchor': 'end', text: g }));
        P.forEach(function (p) {
          if (p.group !== g || p.volume === null || p.volume <= 0) return;
          var y = y0 + (random() * 0.56 - 0.28) * gh * 0.9;
          var el = glyph('circle', sv(p.volume), y, 3.6, 'mark filled ' + (g === 'Analysed' ? 'dot' : 'muted-dot')); el.setAttribute('data-id', p.id);
          layer.appendChild(el); register(p, el);
          var hit = h('svg:circle', { class: 'hit', cx: sv(p.volume), cy: y, r: 6 }); bindHit(hit, p); vhits.appendChild(hit);
        });
      });
      svg.appendChild(layer); svg.appendChild(vhits);
    }
    svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- null calibration: one histogram per null group ---------- */
  function calibration(container) {
    var G = D.groups, E = D.edges, P = D.providers, g = G.length, ncol = Math.min(g, 3), nrow = Math.ceil(g / ncol);
    var W = 780, pw = (W - 16) / ncol, ph = 250, H = 8 + nrow * ph, lo = E[0], hi = E[E.length - 1], width = E[1] - E[0];
    var byBin = {};
    P.forEach(function (p) { if (p.bin === null) return; var key = p.group + '/' + p.bin; (byBin[key] = byBin[key] || []).push(p.id); });
    var f = figure(W, H, D.alt), svg = f.svg;
    G.forEach(function (grp, i) {
      var ox = 16 + (i % ncol) * pw, oy = 8 + Math.floor(i / ncol) * ph, m = { t: 30, r: 10, b: 46, l: 44 };
      var x0 = ox + m.l, x1 = ox + pw - m.r, y0 = oy + m.t, y1 = oy + ph - m.b;
      var scale = grp.providers * width, cmax = Math.max.apply(null, grp.counts.concat([0]));
      var peak = Math.max(cmax, scale * pdf(0, 0, 1), grp.null_sd !== null ? scale * pdf(0, 0, grp.null_sd) : 0);
      var ymax = peak > 0 ? 1.45 * peak : 1, sx = linear(lo, hi, x0, x1), sy = linear(0, ymax, y1, y0);   /* headroom for the text */
      var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
      xTicks(grid, axis, niceTicks(lo, hi, 5), sx, y0, y1);
      yTicks(grid, axis, niceTicks(0, ymax, 4).filter(function (t) { return t === Math.round(t); }), sy, x0, x1);
      axis.appendChild(h('svg:line', { x1: x0, x2: x1, y1: y1, y2: y1 }));
      svg.appendChild(grid); svg.appendChild(axis);
      svg.appendChild(h('svg:text', { class: 'panel-title', x: x0, y: oy + 16, text: grp.label + ' (' + fmt(grp.providers, { digits: 0 }) + ')' }));
      if (Math.floor(i / ncol) === nrow - 1 || i + ncol >= g) axisText(svg, (x0 + x1) / 2, oy + ph - 6, D.axes.x.label);
      if (i % ncol === 0) axisText(svg, ox + 12, (y0 + y1) / 2, D.axes.y.label, 'middle', true);
      var bins = h('svg:g'), hits = h('svg:g');
      grp.counts.forEach(function (c, b) {
        var ids = byBin[grp.key + '/' + b] || [];
        var bar = h('svg:rect', { class: 'bin', x: sx(E[b]), y: sy(c), width: Math.max(sx(E[b + 1]) - sx(E[b]) - 0.5, 0.5), height: sy(0) - sy(c) });
        bins.appendChild(bar);
        ids.forEach(function (id) { register({ id: id }, bar); });
        var html = '<b>' + grp.label + ': z from ' + minus(fmtTick(E[b])) + ' to ' + minus(fmtTick(E[b + 1])) + '</b><div class="row"><span>Providers</span><span>' + c + '</span></div>' + (ids.length ? '<div class="row"><span>Ids</span><span>' + ids.slice(0, 8).join(', ') + (ids.length > 8 ? ', …' : '') + '</span></div>' : '');
        var hit = h('svg:rect', { class: 'hit', x: sx(E[b]), y: y0, width: sx(E[b + 1]) - sx(E[b]), height: y1 - y0 }); bindHit(hit, null, html, ids); hits.appendChild(hit);
      });
      svg.appendChild(bins);
      function curve(mean, sd, cls) { var pts = []; for (var j = 0; j <= 200; j++) { var x = lo + (hi - lo) * j / 200; pts.push(sx(x) + ',' + sy(scale * pdf(x, mean, sd))); } svg.appendChild(h('svg:polyline', { class: cls, points: pts.join(' ') })); }
      curve(0, 1, 'reference dashed');
      if (D.fitted_differs && grp.null_mean !== null && grp.null_sd !== null) curve(grp.null_mean, grp.null_sd, 'curve fitted');
      if (grp.beyond_left) svg.appendChild(h('svg:text', { class: 'annot muted', x: x0 + 3, y: y1 - 5, text: '← ' + grp.beyond_left }));
      if (grp.beyond_right) svg.appendChild(h('svg:text', { class: 'annot muted', x: x1 - 3, y: y1 - 5, 'text-anchor': 'end', text: grp.beyond_right + ' →' }));
      var fitted = grp.above_fitted + grp.below_fitted;
      var l1 = D.fitted_differs ? 'mean ' + fmt(grp.null_mean, { digits: 2 }) + ', SD ' + fmt(grp.null_sd, { digits: 2 }) : 'theoretical null';
      var l2 = D.compared ? 'flagged ' + (grp.above_theoretical + grp.below_theoretical) + ' → ' + fitted : 'flagged ' + fitted;
      svg.appendChild(h('svg:text', { class: 'annot', x: x0 + 6, y: y0 + 14, text: l1 }));
      svg.appendChild(h('svg:text', { class: 'annot', x: x0 + 6, y: y0 + 30, text: l2 }));
      svg.appendChild(hits);
    });
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- reliability by provider size ---------- */
  function reliability(container) {
    var P = D.providers, W = 780, H = 400, m = { t: 14, r: 16, b: 46, l: 58 };
    var se = extent(P.map(function (p) { return p.size; })), span = Math.log10(se[1] / se[0]) || 1;
    var xlo = se[0] / Math.pow(10, 0.05 * span), xhi = se[1] * Math.pow(10, 0.05 * span);
    var sx = logScale(xlo, xhi, m.l, W - m.r), sy = linear(0, 1, H - m.b, m.t);
    var f = figure(W, H, D.alt), svg = f.svg;
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' }), lt = logTicks(xlo, xhi);
    lt.all.forEach(function (t) { grid.appendChild(h('svg:line', { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b })); });
    (lt.all.length <= 8 ? lt.all : lt.major).forEach(function (t) { axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 16, 'text-anchor': 'middle', text: fmtTick(t) })); });
    yTicks(grid, axis, niceTicks(0, 1, 5), sy, m.l, W - m.r);
    axis.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, (m.l + W - m.r) / 2, H - 6, D.axes.x.label);
    axisText(svg, 14, (m.t + H - m.b) / 2, D.axes.y.label, 'middle', true);
    svg.appendChild(h('svg:line', { class: 'guide dotted', x1: sx(D.n_prime), x2: sx(D.n_prime), y1: m.t, y2: H - m.b }));
    svg.appendChild(h('svg:text', { class: 'annot muted', x: sx(D.n_prime) + 4, y: H - m.b - 6, text: 'n′ = ' + fmt(D.n_prime, { digits: 0 }) }));
    svg.appendChild(h('svg:polyline', { class: 'curve fitted', points: P.map(function (p) { return sx(p.size) + ',' + sy(p.reliability); }).join(' ') }));
    svg.appendChild(h('svg:line', { class: 'reference dashed', x1: m.l, x2: W - m.r, y1: sy(D.overall), y2: sy(D.overall) }));
    svg.appendChild(h('svg:text', { class: 'annot', x: m.l + 6, y: sy(D.overall) - 5, text: 'overall IUR ' + fmt(D.overall, { digits: 2 }) }));
    var layer = h('svg:g'), hits = h('svg:g');
    P.forEach(function (p) {
      var el = glyph('circle', sx(p.size), sy(p.reliability), 3.6, 'mark filled dot'); el.setAttribute('data-id', p.id); layer.appendChild(el); register(p, el);
      var hit = h('svg:circle', { class: 'hit', cx: sx(p.size), cy: sy(p.reliability), r: 7 }); bindHit(hit, p); hits.appendChild(hit);
    });
    svg.appendChild(layer); svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- between-provider variation: BLUP histogram and the fitted distribution ---------- */
  function variation(container) {
    var P = D.providers, E = D.edges, C = D.counts, W = 780, H = 430, m = { t: 14, r: 16, b: 46, l: 58 };
    var n = P.length, width = E[1] - E[0], scale = n * width, half = D.half;
    var drawn = [D.sigma].concat(D.lower !== null ? [D.lower, D.upper] : []).filter(function (s) { return s !== null && s >= 1e-6; });
    var peak = Math.max.apply(null, [Math.max.apply(null, C.concat([0]))].concat(drawn.map(function (s) { return scale * pdf(0, 0, s); })));
    var ymax = 1.85 * peak || 1, sx = linear(-half, half, m.l, W - m.r), sy = linear(0, ymax, H - m.b, m.t);
    var byBin = {};
    P.forEach(function (p) { if (p.bin !== null) (byBin[p.bin] = byBin[p.bin] || []).push(p.id); });
    var f = figure(W, H, D.alt), svg = f.svg;
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
    xTicks(grid, axis, niceTicks(-half, half, 6), sx, m.t, H - m.b);
    yTicks(grid, axis, niceTicks(0, ymax, 6).filter(function (t) { return t === Math.round(t); }), sy, m.l, W - m.r);
    axis.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, (m.l + W - m.r) / 2, H - 6, D.axes.x.label);
    axisText(svg, 14, (m.t + H - m.b) / 2, D.axes.y.label, 'middle', true);
    var bins = h('svg:g'), hits = h('svg:g');
    C.forEach(function (c, b) {
      var ids = byBin[b] || [];
      var bar = h('svg:rect', { class: 'bin', x: sx(E[b]), y: sy(c), width: Math.max(sx(E[b + 1]) - sx(E[b]) - 0.5, 0.5), height: sy(0) - sy(c) });
      bins.appendChild(bar); ids.forEach(function (id) { register({ id: id }, bar); });
      var html = '<b>BLUPs from ' + minus(E[b].toFixed(2)) + ' to ' + minus(E[b + 1].toFixed(2)) + '</b><div class="row"><span>Providers</span><span>' + c + '</span></div>' + (ids.length ? '<div class="row"><span>Ids</span><span>' + ids.slice(0, 8).join(', ') + (ids.length > 8 ? ', …' : '') + '</span></div>' : '');
      var hit = h('svg:rect', { class: 'hit', x: sx(E[b]), y: m.t, width: sx(E[b + 1]) - sx(E[b]), height: H - m.t - m.b }); bindHit(hit, null, html, ids); hits.appendChild(hit);
    });
    svg.appendChild(bins);
    function curve(sd, cls) { var pts = []; for (var j = 0; j <= 200; j++) { var x = -half + 2 * half * j / 200; pts.push(sx(x) + ',' + sy(scale * pdf(x, 0, sd))); } svg.appendChild(h('svg:polyline', { class: cls, points: pts.join(' ') })); }
    if (!D.degenerate) curve(D.sigma, 'curve fitted');
    if (D.lower !== null) [D.lower, D.upper].forEach(function (s) { if (s >= 1e-6) curve(s, 'curve dashed'); });
    svg.appendChild(h('svg:line', { class: 'reference', x1: sx(0), x2: sx(0), y1: sy(1.06 * peak), y2: sy(0) }));
    if (D.degenerate) svg.appendChild(h('svg:text', { class: 'annot', x: (m.l + W - m.r) / 2, y: sy(0.62 * ymax), 'text-anchor': 'middle', text: 'σ is estimated at 0: no between-provider variation is detected' }));
    else {
      var yb = sy(1.18 * peak), t = sy(1.21 * peak) - sy(1.15 * peak);
      svg.appendChild(h('svg:line', { class: 'bracket', x1: sx(D.range[0]), x2: sx(D.range[1]), y1: yb, y2: yb }));
      [D.range[0], D.range[1]].forEach(function (x) { svg.appendChild(h('svg:line', { class: 'bracket', x1: sx(x), x2: sx(x), y1: yb + t / 2, y2: yb - t / 2 })); });
      svg.appendChild(h('svg:text', { class: 'annot', x: sx(0), y: yb - 22, 'text-anchor': 'middle', text: (D.level * 100).toLocaleString('en-US') + '% of true effects if normal:' }));
      svg.appendChild(h('svg:text', { class: 'annot', x: sx(0), y: yb - 8, 'text-anchor': 'middle', text: minus(D.range_text) }));
    }
    svg.appendChild(h('svg:text', { class: 'annot', x: m.l + 6, y: m.t + 14, text: minus(D.sigma_text) }));
    svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- shrinkage: fixed-effect estimate against BLUP, sized by volume ---------- */
  function shrinkage(container) {
    var P = D.providers, W = 600, H = 580, m = { t: 14, r: 16, b: 46, l: 58 }, lo = D.limits[0], hi = D.limits[1], span = hi - lo;
    var sx = linear(lo, hi, m.l, W - m.r), sy = linear(lo, hi, H - m.b, m.t);
    function radius(v) { return Math.sqrt(v === null || v <= 0 ? 8 : 4 + 40 * v / D.top) * 0.9; }
    var f = figure(W, H, D.alt), svg = f.svg;
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' }), ticks = niceTicks(lo, hi, 7);
    xTicks(grid, axis, ticks, sx, m.t, H - m.b); yTicks(grid, axis, ticks, sy, m.l, W - m.r);
    axis.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, (m.l + W - m.r) / 2, H - 6, D.axes.x.label);
    axisText(svg, 14, (m.t + H - m.b) / 2, D.axes.y.label, 'middle', true);
    svg.appendChild(h('svg:line', { class: 'reference', x1: sx(lo), x2: sx(hi), y1: sy(lo), y2: sy(hi) }));
    svg.appendChild(h('svg:line', { class: 'guide dashed', x1: m.l, x2: W - m.r, y1: sy(0), y2: sy(0) }));
    var fin = P.filter(function (p) { return p.fixed !== null; });
    function crowd(cx, cy) { return fin.filter(function (p) { return Math.abs(p.fixed - cx) < 0.12 * span && Math.abs(p.random - cy) < 0.05 * span; }).length; }
    var diag = [0.62, 0.85, 0.35, 0.15].map(function (t) { return [lo + t * span, lo + t * span + 0.07 * span]; });
    var best = diag.reduce(function (a, c) { return crowd(c[0] - 0.06 * span, c[1] + 0.02 * span) < crowd(a[0] - 0.06 * span, a[1] + 0.02 * span) ? c : a; });
    svg.appendChild(h('svg:text', { class: 'annot', x: sx(best[0]), y: sy(best[1]) - 2, 'text-anchor': 'end', text: 'no shrinkage' }));
    var pool = [[lo + 0.02 * span, 0.015 * span, 'start', -3], [hi - 0.02 * span, 0.015 * span, 'end', -3], [lo + 0.02 * span, -0.015 * span, 'start', 12], [hi - 0.02 * span, -0.015 * span, 'end', 12]];
    var bp = pool.reduce(function (a, c) { var f1 = function (c) { return crowd(c[0] + (c[2] === 'start' ? 0.08 : -0.08) * span, c[1] + (c[3] < 0 ? 0.02 : -0.02) * span); }; return f1(c) < f1(a) ? c : a; });
    svg.appendChild(h('svg:text', { class: 'annot muted', x: sx(bp[0]), y: sy(bp[1]) + bp[3], 'text-anchor': bp[2], text: 'complete pooling' }));
    var layer = h('svg:g'), hits = h('svg:g');
    P.forEach(function (p) {
      var el, x = p.edge ? (p.edge === 'lower' ? m.l : W - m.r) : sx(p.fixed), y = sy(p.random);
      if (p.edge) el = glyph(p.edge === 'lower' ? 'triangle_left' : 'triangle_right', x, y, 4.5, 'mark filled muted-dot');
      else el = h('svg:circle', { cx: x, cy: y, r: radius(p.volume), class: 'mark open sdot' });
      el.setAttribute('data-id', p.id); layer.appendChild(el); register(p, el);
      var hit = h('svg:circle', { class: 'hit', cx: x, cy: y, r: Math.max(7, radius(p.volume) + 2) }); bindHit(hit, p); hits.appendChild(hit);
    });
    svg.appendChild(layer); svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- flag stability: providers by scenario ---------- */
  function stability(container) {
    var S = D.scenarios, P = D.providers, k = S.length, n = P.length, labelled = n <= 60, rowH = labelled ? 15 : 3.4, dense = !labelled;
    var colW = Math.min(130, (780 - 120) / (k + 1)), W = 780, heads = S.map(function (s) { return wrap(s.label, 14); });
    var maxLines = Math.max.apply(null, heads.map(function (l) { return l.length; })), m = { t: 14 * (maxLines + 1) + 22, r: 16, b: 24, l: labelled ? 70 : 34 };
    var H = m.t + Math.max(n, 1) * rowH + m.b, cx = function (j) { return m.l + 30 + (j + 0.5) * colW; }, chX = m.l + 30 + k * colW + 20;
    var f = figure(W, H, D.alt), svg = f.svg;
    S.forEach(function (s, j) {
      var lines = heads[j], y = 14 + (maxLines - lines.length) * 14;      // bottom-aligned, as the static headers
      lines.forEach(function (l) { svg.appendChild(h('svg:text', { class: 'col-head', x: cx(j), y: y, 'text-anchor': 'middle', text: l })); y += 14; });
      svg.appendChild(h('svg:text', { class: 'col-head', x: cx(j), y: y + 2, 'text-anchor': 'middle', text: D.statuses.above.symbol + ' ' + s.above + '  ' + D.statuses.below.symbol + ' ' + s.below }));
    });
    svg.appendChild(h('svg:text', { class: 'col-head', x: chX, y: 14 * maxLines + 2, 'text-anchor': 'middle', text: 'Changes' }));
    if (dense) axisText(svg, 12, (m.t + H - m.b) / 2, D.axes.y.label, 'middle', true);
    var grid = h('svg:g', { class: 'grid' }), layer = h('svg:g'), hits = h('svg:g');
    P.forEach(function (p, i) {
      var y = m.t + (i + 0.5) * rowH;
      grid.appendChild(h('svg:line', { x1: m.l, x2: chX + 12, y1: y, y2: y }));
      if (labelled) svg.appendChild(h('svg:text', { class: 'row-label', x: m.l - 6, y: y + 4, 'text-anchor': 'end', text: p.id }));
      S.forEach(function (s, j) { var el = statusMark(p['s' + j], p.id, cx(j), y, statusSize(p['s' + j], dense) * (dense ? 0.8 : 1)); layer.appendChild(el); register(p, el); });
      if (p.changed) { var d = glyph('diamond', chX, y, dense ? 2.4 : 3.4, 'mark filled dot'); d.setAttribute('data-id', p.id); layer.appendChild(d); register(p, d); }
      var hit = h('svg:rect', { class: 'hit', x: 0, y: y - rowH / 2, width: W, height: rowH }); bindHit(hit, p); hits.appendChild(hit);
    });
    svg.appendChild(grid); svg.appendChild(layer); svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- several measures: one interval panel per measure, common rows ---------- */
  function multi(container) {
    var M = D.measures, P = D.providers, k = M.length, n = P.length, labelled = n <= 60, rowH = labelled ? 14 : 3.4, dense = !labelled;
    var W = 780, m = { t: 30, r: 16, b: 46, l: labelled ? 70 : 34 }, gap = 34, pw = (W - m.l - m.r - gap * (k - 1)) / k;
    var H = m.t + n * rowH + m.b;
    function sy(i) { return m.t + (i + 0.5) * rowH; }
    var f = figure(W, H, D.alt), svg = f.svg;
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' }), bars = h('svg:g'), layer = h('svg:g'), hits = h('svg:g'), texts = h('svg:g');
    var sxs = M.map(function (mm, j) {
      var x0 = m.l + j * (pw + gap), x1 = x0 + pw, sx = linear(mm.limits[0], mm.limits[1], x0, x1);
      xTicks(grid, axis, niceTicks(mm.limits[0], mm.limits[1], 5), sx, m.t, H - m.b);
      axis.appendChild(h('svg:line', { x1: x0, x2: x1, y1: H - m.b, y2: H - m.b }));
      svg.appendChild(h('svg:text', { class: 'panel-title', x: x0, y: m.t - 12, text: mm.label }));
      axisText(svg, (x0 + x1) / 2, H - 6, D.axes.x.label);
      svg.appendChild(h('svg:line', { class: 'reference', x1: sx(mm.null), x2: sx(mm.null), y1: m.t, y2: H - m.b }));
      return sx;
    });
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, 12, (m.t + H - m.b) / 2, D.axes.y.label, 'middle', true);
    P.forEach(function (p, i) {
      var y = sy(i);
      if (labelled) svg.appendChild(h('svg:text', { class: 'row-label', x: m.l - 6, y: y + 4, 'text-anchor': 'end', text: p.id }));
      M.forEach(function (mm, j) {
        var sx = sxs[j], kind = p['k' + j], lo = mm.limits[0], hi = mm.limits[1];
        if (kind !== 'finite') { if (!dense) texts.appendChild(h('svg:text', { class: 'annot muted', x: sx(mm.null), y: y + 3.5, 'text-anchor': 'middle', text: kind === 'na' ? 'n/a' : 'NE' })); return; }
        var l = p['l' + j], u = p['u' + j], st = p['s' + j];
        if (l !== null || u !== null) {
          var seg = h('svg:line', { class: 'interval st-' + st, 'data-id': p.id, x1: sx(Math.max(lo, l === null ? lo : l)), x2: sx(Math.min(hi, u === null ? hi : u)), y1: y, y2: y, style: dense ? 'stroke-width: 1.6' : null });
          bars.appendChild(seg); register(p, seg);
        }
        var el = statusMark(st, p.id, sx(p['e' + j]), y, statusSize(st, dense)); layer.appendChild(el); register(p, el);
      });
      var hit = h('svg:rect', { class: 'hit', x: 0, y: y - rowH / 2, width: W, height: rowH }); bindHit(hit, p); hits.appendChild(hit);
    });
    svg.appendChild(bars); svg.appendChild(layer); svg.appendChild(texts); svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  /* ---------- agreement of two measures ---------- */
  function agreement(container) {
    var P = D.providers, W = 600, H = 580, m = { t: 14, r: 16, b: 46, l: 58 };
    var sx = linear(D.x.limits[0], D.x.limits[1], m.l, W - m.r), sy = linear(D.y.limits[0], D.y.limits[1], H - m.b, m.t);
    var f = figure(W, H, D.alt), svg = f.svg;
    clip(svg, 'plot', m.l, m.t, W - m.l - m.r, H - m.t - m.b);
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
    xTicks(grid, axis, niceTicks(D.x.limits[0], D.x.limits[1], 6), sx, m.t, H - m.b);
    yTicks(grid, axis, niceTicks(D.y.limits[0], D.y.limits[1], 6), sy, m.l, W - m.r);
    axis.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    svg.appendChild(grid); svg.appendChild(axis);
    axisText(svg, (m.l + W - m.r) / 2, H - 6, D.x.label);
    axisText(svg, 14, (m.t + H - m.b) / 2, D.y.label, 'middle', true);
    var plot = h('svg:g', { 'clip-path': 'url(#plot)' }); svg.appendChild(plot);
    plot.appendChild(h('svg:line', { class: 'reference', x1: sx(D.x.null), x2: sx(D.x.null), y1: m.t, y2: H - m.b }));
    plot.appendChild(h('svg:line', { class: 'reference', x1: m.l, x2: W - m.r, y1: sy(D.y.null), y2: sy(D.y.null) }));
    var crosses = h('svg:g'), layer = h('svg:g'), hits = h('svg:g');
    P.forEach(function (p) {
      if (p.lox !== null && p.hix !== null) { var cx_ = h('svg:line', { class: 'limit thin', 'data-id': p.id, x1: sx(p.lox), x2: sx(p.hix), y1: sy(p.ey), y2: sy(p.ey) }); crosses.appendChild(cx_); register(p, cx_); }
      if (p.loy !== null && p.hiy !== null) { var cy_ = h('svg:line', { class: 'limit thin', 'data-id': p.id, x1: sx(p.ex), x2: sx(p.ex), y1: sy(p.loy), y2: sy(p.hiy) }); crosses.appendChild(cy_); register(p, cy_); }
    });
    ['neither', 'untested', 'one', 'opposite', 'same'].forEach(function (key) {
      P.forEach(function (p) {
        if (p.joint !== key) return;
        var j = D.joint[key], el = glyph(j.filled ? j.shape : (j.shape === 'circle' ? 'circle_open' : j.shape), sx(p.ex), sy(p.ey), key === 'neither' ? 3 : 4.2, 'mark ' + (j.filled ? 'filled' : 'open') + ' jt-' + key);
        el.setAttribute('data-id', p.id); layer.appendChild(el); register(p, el);
      });
    });
    P.forEach(function (p) { var hit = h('svg:circle', { class: 'hit', cx: sx(p.ex), cy: sy(p.ey), r: 7 }); bindHit(hit, p); hits.appendChild(hit); });
    plot.appendChild(crosses); plot.appendChild(layer); svg.appendChild(hits);
    container.appendChild(legend()); container.appendChild(f.fig);
  }

  var container = document.getElementById('chart');
  ({ funnel: funnel, caterpillar: caterpillar, observed_expected: observed, forest: forest, data_quality: quality,
     null_calibration: calibration, reliability: reliability, provider_variation: variation, shrinkage: shrinkage,
     flag_stability: stability, multi_measure: multi, measure_agreement: agreement })[D.display](container);
  var ctl = controls(container);
  if (D.series) table(container, { rows: D.series.rows, columns: D.series.columns, caption: D.series.caption, linked: false, ordered: true });
  table(container, { rows: D.providers || D.terms, columns: D.columns, linked: true,
                     caption: D.table_caption || 'Providers shown in the chart: hover or focus a row to find it in the chart; sort by a column.' }, ctl);
  if (reduced) root.classList.add('reduced-motion');

  /* ---------- height: tell the embedding page how tall the document is, so its iframe can fit (same origin only) ---------- */
  var heightPending = false;
  function sendHeight() {
    heightPending = false;
    try {
      if (!window.parent || window.parent === window) return;
      /* the body's height, not documentElement.scrollHeight: that never falls below the frame's own viewport, so the
         frame could grow but never shrink again (after "Table only" or a filter) */
      var height = Math.ceil(document.body.getBoundingClientRect().height);
      window.parent.postMessage({ pprof: 'height', display: D.display, height: height }, '/');   // '/' = this origin
    } catch (e) { /* no parent, or a cross-origin one: the frame keeps its fixed height */ }
  }
  function postHeight() {
    if (heightPending) return;
    heightPending = true;
    if (window.requestAnimationFrame) window.requestAnimationFrame(sendHeight); else setTimeout(sendHeight, 16);
  }
  postHeight();
  window.addEventListener('load', postHeight);
  window.addEventListener('resize', postHeight);
  if (window.ResizeObserver) new ResizeObserver(postHeight).observe(document.body);   // table only, find, sort
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(postHeight, function () {});
})();
