/* Print the static figure kept in a collapsed <details class="pprof-static-figure"> under an interactive view:
   open those blocks before printing and close them again afterwards. custom.css hides the interactive frame in print. */
(function () {
  var opened = [];
  window.addEventListener('beforeprint', function () {
    var blocks = document.querySelectorAll('details.pprof-static-figure');
    for (var i = 0; i < blocks.length; i++) {
      if (!blocks[i].open) { blocks[i].open = true; opened.push(blocks[i]); }
    }
  });
  window.addEventListener('afterprint', function () {
    for (var i = 0; i < opened.length; i++) opened[i].open = false;
    opened = [];
  });
})();
