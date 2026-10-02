IBM Plex Sans, bundled with pprof_py
====================================

These four font files are IBM Plex Sans, version 3.005, as published by IBM at
https://github.com/IBM/plex (packages/plex-sans/fonts/complete/ttf). They are
unmodified: not subset, renamed or otherwise changed.

  IBMPlexSans-Regular.ttf   200500 bytes  sha256 975dcda37d80f038dcd143c22e33ca2d97a0cc5a929aace1c749153b0fe1afa5
  IBMPlexSans-Italic.ttf   207920 bytes  sha256 a9c6ef9942c49e49d11e11a6dacc0b3a087978757e9b22a06b8ac22a6400fb15
  IBMPlexSans-Medium.ttf   202460 bytes  sha256 331c8639d7598b2cde62a911a71db195e30cb655cd6bdf2e324a7e984955f907
  IBMPlexSans-SemiBold.ttf   202632 bytes  sha256 a20caf8286023a6a7a85e40b1d2a4ae9fc3e3b1f9eda8f4c542dd4986af67bb1

Copyright 2017 IBM Corp. with Reserved Font Name "Plex". The fonts are licensed
under the SIL Open Font License, Version 1.1; the copyright notice and the full
licence are in OFL.txt in this directory and must accompany the font files.
The licence covers the fonts only; pprof_py itself is MIT-licensed.

pprof_py's figures load these files per text element, through Matplotlib's
FontProperties(fname=...). They are never registered with Matplotlib's font
manager and change no rcParams. If the files are missing or unreadable, figures
fall back to DejaVu Sans, which ships with Matplotlib.
