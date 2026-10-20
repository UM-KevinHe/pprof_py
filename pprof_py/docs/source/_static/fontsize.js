/* pprof_py documentation: text-size control for the Shibuya header.
 *
 * Adds an "A-" / "A+" pair next to the colour-mode toggle. The chosen step
 * (small / default / large) is written to the data-font-size attribute on
 * <html>; custom.css turns it into a scale on the article text. The choice is
 * remembered in localStorage when that is available. Without JavaScript the
 * control is simply absent and the page uses the default size.
 */
(function () {
  'use strict';

  var KEY = 'pprof_font_size';
  var STEPS = ['small', 'default', 'large'];
  var root = document.documentElement;

  function readStored() {
    try {
      var value = window.localStorage.getItem(KEY);
      return STEPS.indexOf(value) >= 0 ? value : 'default';
    } catch (e) {
      return 'default';
    }
  }

  function store(value) {
    try {
      if (value === 'default') {
        window.localStorage.removeItem(KEY);
      } else {
        window.localStorage.setItem(KEY, value);
      }
    } catch (e) {
      /* Private mode or storage disabled: the choice lasts for this page only. */
    }
  }

  function apply(value) {
    if (value === 'default') {
      root.removeAttribute('data-font-size');
    } else {
      root.setAttribute('data-font-size', value);
    }
  }

  var current = readStored();
  apply(current);

  function makeButton(text, label, delta) {
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'pp-fontsize-btn';
    button.textContent = text;
    button.setAttribute('aria-label', label);
    button.title = label;
    button.addEventListener('click', function () {
      if (button.getAttribute('aria-disabled') === 'true') {
        return;
      }
      var index = STEPS.indexOf(current) + delta;
      if (index < 0 || index >= STEPS.length) {
        return;
      }
      current = STEPS[index];
      apply(current);
      store(current);
      update();
      status.textContent = 'Text size: ' + current;
    });
    return button;
  }

  var group = document.createElement('div');
  group.className = 'pp-fontsize';
  group.setAttribute('role', 'group');
  group.setAttribute('aria-label', 'Text size');

  var decrease = makeButton('A−', 'Decrease text size', -1);
  var increase = makeButton('A+', 'Increase text size', 1);
  var status = document.createElement('span');
  status.className = 'pp-fontsize-status';
  status.setAttribute('aria-live', 'polite');

  group.appendChild(decrease);
  group.appendChild(increase);
  group.appendChild(status);

  /* aria-disabled rather than disabled, so a button that reaches the end of
     the range keeps keyboard focus instead of dropping it. */
  function update() {
    decrease.setAttribute('aria-disabled', current === STEPS[0] ? 'true' : 'false');
    increase.setAttribute('aria-disabled', current === STEPS[STEPS.length - 1] ? 'true' : 'false');
  }
  update();

  function mount() {
    var actions = document.querySelector('.sy-head-actions');
    if (actions) {
      /* Next to the colour-mode toggle, before it. */
      actions.insertBefore(group, actions.querySelector('.js-theme') || actions.firstChild);
      return true;
    }
    var links = document.querySelector('.sy-head-links > ul');
    if (links) {
      var item = document.createElement('li');
      item.className = 'link';
      item.appendChild(group);
      links.appendChild(item);
      return true;
    }
    return false;
  }

  /* The theme loads scripts at the end of <body>, so the header is already
     parsed; the DOMContentLoaded path covers a script placed in <head>. */
  if (!mount() && document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount);
  }
})();
