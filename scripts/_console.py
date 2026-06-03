"""Console-safe text for Windows output.

The user runs scripts in a PowerShell console whose codepage is NOT UTF-8.
Printing model/corpus text (smart quotes, em-dashes, accents) raw either crashes
(`UnicodeEncodeError: 'charmap' codec ...`) or, if stdout is reconfigured to UTF-8,
shows as mojibake when piped/redirected (`| Tee-Object`, `> file`). The robust fix
is to transliterate to plain ASCII before printing -- which renders correctly on
ANY console / pipe / file, with no codepage assumptions.

Single source of truth: every demo/script that prints text it does not fully
control imports `to_ascii` from here rather than re-implementing it.
"""

from __future__ import annotations

import unicodedata

# Common "smart" punctuation -> ASCII equivalents. (NFKD handles accents; this map
# handles punctuation NFKD leaves as non-ASCII.)
_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "‑": "-", "−": "-",
    "…": "...", " ": " ",
}


def to_ascii(s: str) -> str:
    """Transliterate `s` to plain ASCII: smart quotes -> ' / ", dashes -> -,
    ellipsis -> ..., accented letters -> base letter (cafe). Anything still
    non-ASCII after that is dropped. Guarantees `to_ascii(s).encode("ascii")`
    never raises, so the result is safe on any Windows console/pipe/file."""
    for bad, good in _PUNCT.items():
        s = s.replace(bad, good)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
