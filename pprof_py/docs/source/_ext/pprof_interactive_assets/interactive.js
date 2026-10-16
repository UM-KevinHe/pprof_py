/* pprof_py documentation: interactive figures (funnel plot, interval plot, observed versus expected).
   Plain JavaScript and inline SVG, no dependencies. Inlined into _static/interactive/<name>.html by
   _ext/pprof_interactive.py; the data is the JSON record in <script id="pprof-data">. Each display draws an SVG
   chart with a hover layer (tooltip), a legend that names every category (shape and colour, never colour alone),
   and a sortable table of the providers linked to the chart by hover and keyboard focus. */
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
    if (v === null || v === undefined || isNaN(v)) return '–';
    if (col && col.p) return v < 0.0005 ? '< 0.001' : Number(v).toFixed(3);
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

  /* status marks: shape and colour (CSS variable --c per status), filled with a surface-coloured halo or open */
  var SHAPES = {
    triangle_up: function (x, y, s) { return h('svg:path', { d: 'M' + x + ',' + (y - s * 1.15) + 'L' + (x + s) + ',' + (y + s * 0.65) + 'L' + (x - s) + ',' + (y + s * 0.65) + 'Z', class: 'filled' }); },
    triangle_down: function (x, y, s) { return h('svg:path', { d: 'M' + x + ',' + (y + s * 1.15) + 'L' + (x + s) + ',' + (y - s * 0.65) + 'L' + (x - s) + ',' + (y - s * 0.65) + 'Z', class: 'filled' }); },
    triangle_left: function (x, y, s) { return h('svg:path', { d: 'M' + (x - s * 1.15) + ',' + y + 'L' + (x + s * 0.65) + ',' + (y - s) + 'L' + (x + s * 0.65) + ',' + (y + s) + 'Z', class: 'filled' }); },
    triangle_right: function (x, y, s) { return h('svg:path', { d: 'M' + (x + s * 1.15) + ',' + y + 'L' + (x - s * 0.65) + ',' + (y - s) + 'L' + (x - s * 0.65) + ',' + (y + s) + 'Z', class: 'filled' }); },
    circle: function (x, y, s) { return h('svg:circle', { cx: x, cy: y, r: s * 0.8, class: 'filled' }); },
    circle_open: function (x, y, s) { return h('svg:circle', { cx: x, cy: y, r: s * 0.9, class: 'open' }); },
    square_open: function (x, y, s) { return h('svg:rect', { x: x - s * 0.85, y: y - s * 0.85, width: s * 1.7, height: s * 1.7, class: 'open' }); }
  };
  function markOf(p) { return p.edge ? (p.edge === 'lower' ? 'triangle_left' : 'triangle_right') : p.zero_events ? 'square_open' : D.statuses[p.status].marker; }
  function statusKey(p) { return p.edge ? 'no_finite_estimate' : p.zero_events ? 'zero_events' : p.status; }
  function mark(p, x, y, size) {
    var el = SHAPES[markOf(p)](x, y, size);
    el.setAttribute('class', 'mark ' + el.getAttribute('class') + ' st-' + statusKey(p));
    el.setAttribute('data-id', p.id);
    return el;
  }
  function markSize(p, dense) { var s = p.status === 'above' || p.status === 'below' ? 5 : p.status === 'not_different' ? 3.4 : 4.2; return dense && p.status === 'not_different' ? s * 0.8 : s; }

  /* ---------- tooltip and linking between chart and table ---------- */
  var wrap, tip, marks = {}, rows = {}, current = null;
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
  function cell(p, col) { if (col.key === 'id') return p.id; if (col.key === 'status') return D.statuses[statusKey(p)].symbol + ' ' + D.statuses[statusKey(p)].label; return fmt(p[col.key], col); }
  function tipHtml(p) {
    var s = '<b>Provider ' + p.id + '</b>';
    D.columns.forEach(function (c) { if (c.key !== 'id') s += '<div class="row"><span>' + c.label + '</span><span>' + cell(p, c) + '</span></div>'; });
    if (p.edge) s += '<div class="row"><span>Estimate</span><span>no finite estimate; shown at the axis edge</span></div>';
    return s;
  }
  function highlight(id) {
    if (current && marks[current]) marks[current].forEach(function (m) { m.classList.remove('hl'); });
    if (current && rows[current]) rows[current].classList.remove('hl');
    current = id;
    root.classList.toggle('has-hl', !!id);
    if (id && marks[id]) marks[id].forEach(function (m) { m.classList.add('hl'); });
    if (id && rows[id]) rows[id].classList.add('hl');
  }
  function bindHit(hit, p) {
    hit.addEventListener('mouseenter', function () { highlight(p.id); });
    hit.addEventListener('mousemove', function (e) { var r = wrap.getBoundingClientRect(); tip.show(e.clientX - r.left, e.clientY - r.top, tipHtml(p)); });
    hit.addEventListener('mouseleave', function () { tip.hide(); highlight(null); });
  }
  function showAtMark(p) {           // keyboard: the tooltip beside the provider's mark
    var m = marks[p.id] && marks[p.id][0]; if (!m) return;
    var r = m.getBoundingClientRect(), w = wrap.getBoundingClientRect();
    tip.show(r.left - w.left + r.width / 2, r.top - w.top, tipHtml(p));
  }
  function register(p, el) { (marks[p.id] = marks[p.id] || []).push(el); }

  /* ---------- legend ---------- */
  function legend() {
    var ul = h('ul', { class: 'legend', 'aria-label': 'Legend' });
    D.legend.forEach(function (e) {
      var sw = h('svg:svg', { class: 'swatch', viewBox: '0 0 26 14', 'aria-hidden': 'true' }), el;
      if (e.kind === 'mark') { el = SHAPES[D.statuses[e.status].marker](13, 7, 5); el.setAttribute('class', 'mark ' + el.getAttribute('class') + ' st-' + e.status); }
      else if (e.kind === 'band') el = h('svg:rect', { x: 1, y: 2, width: 24, height: 10, class: 'band', style: 'stroke: var(--ink-2); stroke-width: 1' });
      else if (e.kind === 'bar') el = h('svg:rect', { x: 1, y: 4, width: 24, height: 6, class: 'volume' });
      else el = h('svg:line', { x1: 1, x2: 25, y1: 7, y2: 7, class: { own: 'curve own', other: 'curve other', reference: 'reference', limit: 'limit', guide: 'guide', interval: 'interval' }[e.style] || 'curve own' });
      sw.appendChild(el);
      ul.appendChild(h('li', {}, [sw, e.label]));
    });
    return ul;
  }

  /* ---------- the table ---------- */
  function table(container) {
    var state = { sort: 'id', dir: 'asc', q: '' };
    var find = h('input', { type: 'search', placeholder: 'provider id', 'aria-label': 'Find a provider by id' });
    var only = h('button', { type: 'button', 'aria-pressed': 'false', text: 'Table only' });
    var count = h('span', { class: 'n' });
    var controls = h('div', { class: 'controls' }, [h('label', {}, ['Find ', find]), only, count]);
    var scroll = h('div', { class: 'table-scroll' });
    var tbl = h('table'), thead = h('thead'), tbody = h('tbody'), tr = h('tr');
    tbl.appendChild(h('caption', { text: 'Providers shown in the chart: hover or focus a row to find it in the chart; sort by a column.' }));
    D.columns.forEach(function (c) {
      var th = h('th', { scope: 'col', class: c.key === 'id' || c.key === 'status' ? '' : 'num' });
      var b = h('button', { type: 'button', text: c.label });
      b.addEventListener('click', function () {
        if (state.sort === c.key) state.dir = state.dir === 'asc' ? 'desc' : 'asc';
        else { state.sort = c.key; state.dir = c.key === 'id' || c.key === 'status' ? 'asc' : 'desc'; }
        body();
      });
      th.appendChild(b); tr.appendChild(th);
    });
    thead.appendChild(tr); tbl.appendChild(thead); tbl.appendChild(tbody); scroll.appendChild(tbl);
    var ORDER = { above: 0, below: 1, not_different: 2, not_tested: 3 };
    function value(p) { var k = state.sort; if (k === 'id') return p.id; if (k === 'status') return ORDER[p.status] + (p.edge || p.zero_events ? 0.5 : 0); var v = p[k]; return v === null || v === undefined ? (state.dir === 'asc' ? Infinity : -Infinity) : v; }
    function body() {
      Array.prototype.forEach.call(thead.querySelectorAll('th'), function (th, i) { if (D.columns[i].key === state.sort) th.setAttribute('aria-sort', state.dir === 'asc' ? 'ascending' : 'descending'); else th.removeAttribute('aria-sort'); });
      var dir = state.dir === 'asc' ? 1 : -1, q = state.q.toLowerCase();
      var list = D.providers.filter(function (p) { return !q || p.id.toLowerCase().indexOf(q) >= 0; });
      list.sort(function (a, b) { var va = value(a), vb = value(b); return (va > vb ? 1 : va < vb ? -1 : 0) * dir; });
      tbody.innerHTML = ''; rows = {};
      list.forEach(function (p) {
        var r = h('tr', { tabindex: '0', 'data-id': p.id, class: p.status === 'above' || p.status === 'below' ? 'flagged' : '' });
        D.columns.forEach(function (c) { r.appendChild(h('td', { class: c.key === 'id' ? '' : c.key === 'status' ? 'status' : 'num', text: cell(p, c) })); });
        r.addEventListener('mouseenter', function () { highlight(p.id); });
        r.addEventListener('mouseleave', function () { if (document.activeElement !== r) highlight(null); });
        r.addEventListener('focus', function () { highlight(p.id); showAtMark(p); });
        r.addEventListener('blur', function () { highlight(null); tip.hide(); });
        rows[p.id] = r; tbody.appendChild(r);
      });
      count.textContent = list.length + ' of ' + D.providers.length + ' providers';
    }
    find.addEventListener('input', function () { state.q = find.value.trim(); body(); });
    only.addEventListener('click', function () { var on = only.getAttribute('aria-pressed') !== 'true'; only.setAttribute('aria-pressed', on ? 'true' : 'false'); only.textContent = on ? 'Chart and table' : 'Table only'; root.classList.toggle('table-only', on); });
    body();
    container.appendChild(controls); container.appendChild(scroll);
  }

  /* ---------- the figure scaffold ---------- */
  function figure(W, H, alt) {
    var fig = h('figure', { class: 'fig' });
    wrap = h('div', { class: 'fig-wrap' }); fig.appendChild(wrap); tip = tooltip(wrap);
    var svg = h('svg:svg', { class: 'chart', viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-labelledby': 'chart-title chart-desc' });
    svg.appendChild(h('svg:title', { id: 'chart-title', text: alt }));
    svg.appendChild(h('svg:desc', { id: 'chart-desc', text: D.caption }));
    wrap.appendChild(svg);
    return { fig: fig, svg: svg };
  }
  function axisText(svg, x, y, text, anchor, rotate) {
    svg.appendChild(h('svg:text', { class: 'axis-label', x: x, y: y, 'text-anchor': anchor || 'middle', transform: rotate ? 'rotate(-90 ' + x + ' ' + y + ')' : null, text: text }));
  }

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
    function sx(v) { return m.l + (Math.log10(v) - Math.log10(xlo)) / (Math.log10(xhi) - Math.log10(xlo)) * (W - m.l - m.r); }
    function sy(v) { return H - m.b - (v - ylo) / (yhi - ylo) * (H - m.t - m.b); }
    var f = figure(W, H, D.alt), svg = f.svg;
    svg.appendChild(h('svg:defs', {}, [h('svg:clipPath', { id: 'plot' }, [h('svg:rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b })])]));
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
    var lt = logTicks(xlo, xhi);
    lt.all.forEach(function (t) { grid.appendChild(h('svg:line', { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b })); });
    lt.major.forEach(function (t) { axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 16, 'text-anchor': 'middle', text: fmtTick(t) })); });
    niceTicks(ylo, yhi, 6).forEach(function (t) { grid.appendChild(h('svg:line', { x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) })); axis.appendChild(h('svg:text', { x: m.l - 8, y: sy(t) + 4, 'text-anchor': 'end', text: fmtTick(t) })); });
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
    function sx(v) { return m.l + (v - xlo) / (xhi - xlo) * (mainR - m.l); }
    function sy(i) { return m.t + (i + 0.5) * rowH; }
    function sv(v) { return mainR + gap + v / (vmax * 1.05) * volW; }
    var f = figure(W, H, D.alt), svg = f.svg;
    var grid = h('svg:g', { class: 'grid' }), axis = h('svg:g', { class: 'axis' });
    niceTicks(xlo, xhi, 6).forEach(function (t) { grid.appendChild(h('svg:line', { x1: sx(t), x2: sx(t), y1: m.t, y2: H - m.b })); axis.appendChild(h('svg:text', { x: sx(t), y: H - m.b + 16, 'text-anchor': 'middle', text: fmtTick(t) })); });
    niceTicks(0, vmax, 3).forEach(function (t) { grid.appendChild(h('svg:line', { x1: sv(t), x2: sv(t), y1: m.t, y2: H - m.b })); axis.appendChild(h('svg:text', { x: sv(t), y: H - m.b + 16, 'text-anchor': 'middle', text: fmtTick(t) })); });
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
    svg.appendChild(h('svg:defs', {}, [h('svg:clipPath', { id: 'plot' }, [h('svg:rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b })])]));
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

  var container = document.getElementById('chart');
  ({ funnel: funnel, caterpillar: caterpillar, observed_expected: observed })[D.display](container);
  table(container);
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
