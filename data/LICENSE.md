# Wikipedia text attribution and license

Copyright in Wikipedia prose belongs to its contributors. The cleaned excerpts
in `eval_sv.jsonl` and `eval_en.jsonl` are distributed under **Creative Commons
Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**:

https://creativecommons.org/licenses/by-sa/4.0/

License legal text: https://creativecommons.org/licenses/by-sa/4.0/legalcode

Each JSONL record credits Wikipedia contributors and contains the article URL,
the exact revision URL and the history URL listing contributors. These provide
article-specific attribution; retain them when redistributing the excerpts.
Additional imported-text credit notices detected in the rendered article are
retained in `attribution_notices`, with their links. No images or other media
are redistributed. No endorsement by Wikipedia or Wikimedia is implied.

Changes: the collector removes HTML/markup, citations and nonprose elements,
normalizes whitespace/Unicode and selects an opening prose excerpt. It does not
translate or generate replacement text. These adapted excerpts remain under
CC BY-SA 4.0; preserve the license link, attribution and change notices, and
license further adaptations in accordance with the ShareAlike terms.

The source policy is Wikimedia's Terms of Use, section 7:
https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content

