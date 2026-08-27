# DESIGN.md — HarrisonGPT Visual System

**Direction: "Clinical Print."** The interface is the modern binding around a
printed page, not a chatbot that happens to cite one.

Status: proposed, v1.0.0. Stack-independent — these are CSS custom properties,
so they survive whichever frontend option gets picked.

- Tokens: `design-tokens.json`
- Preview: `design-preview.html` (self-contained, open it in a browser)

---

## 1. The constraint that decides everything

`visual_context` returns `thumbnail_url` / `full_url` pointing at real scans of
printed Harrison's pages ([main.py:215](../../backend/api/main.py#L215)). The UI
is never seen on its own — it is always seen *next to a photograph of paper*.

Three consequences, and they are not stylistic preferences:

| Default choice | Why it fails here | What we do instead |
|---|---|---|
| `#FFFFFF` background | Scanned paper is ~`#F4F1EA`. On pure white every scan reads as grey and dirty. | Ground is `--paper #FAF8F4` |
| `#000000` text | Pure black is harder-edged than the scan's own ink; the UI shouts over the source. | Warm ink ramp topping out at `#1A1714` |
| Rounded image corners | A book page is a rectangle. Rounding the scan is a small lie about the artefact. | `radius: 0` on scans, always |

## 2. Colour

Warm-neutral ink ramp on warm paper, one oxblood accent (Harrison's own print
identity). Every value below was computed, not eyeballed:

| Token | Hex | On paper | Role |
|---|---|---|---|
| `ink-900` | `#1A1714` | 16.83:1 | headings |
| `ink-700` | `#3D3630` | 11.19:1 | answer prose |
| `ink-500` | `#6B6259` | 5.63:1 | secondary UI |
| `ink-300` | `#786E62` | 4.71:1 | page labels, timings |
| `ink-100` | `#E5DFD6` | 1.25:1 | borders — never text |
| `accent-600` | `#8B2635` | 8.16:1 | citations, links, primary action |

The first draft of `ink-300` was `#A69C91` and measured **2.54:1** — it would
have failed WCAG AA on exactly the elements that carry provenance (page labels).
Darkened to `#786E62`. This is why the palette ships with a contrast script
rather than adjectives.

## 3. The four states that carry clinical weight

This is the part of the system that is not decoration. `QueryResponse` has a
`confidence` field that is never hardcoded and comes from `calculate_confidence()`,
plus a verbatim refusal path. The UI must render all four unambiguously.

| State | fg / bg | Glyph | Reasoning |
|---|---|---|---|
| High | `#2D6A4F` / `#E6F0EA` | `●●●` | 5.48:1 |
| Medium | `#8A5A00` / `#F7EDDA` | `●●○` | 5.10:1 |
| Low | `#55606B` / `#EDEEF0` | `●○○` | 5.53:1 |
| Refusal | `#3D3630` / `#F0EBE3` | — | 10.01:1, heavy left rule |

Two deliberate decisions:

**Low confidence is slate, not red.** Red would collide with the oxblood accent,
and more importantly it over-signals: low confidence is *uncertainty*, not an
*error*. The system already refuses outright when it has nothing; "Low" means
"this is thin," and the visual weight should match.

**Never colour alone.** Each state carries a glyph and the literal word. This is
an accessibility requirement and a safety one — a colourblind resident reading a
dosage must not be one hue away from misreading the system's own hedge.

**Refusal is a state, not a styled error.** `REFUSAL_STR` renders verbatim,
full-width, ink on `ink-050`, left rule in `ink-500`. No warning triangle, no
sad emoji, no "try rephrasing!" No softening of any kind — `CODING_RULES.md` §3.2
makes that a correctness rule, and the visual layer is the last place it could
quietly be undone.

## 4. Typography

System stacks only. No webfont request, no FOUT, no CDN dependency.

- **Answer body — serif, 17px/28px, 68ch max.** It is textbook prose set beside
  a scanned textbook page; a serif harmonises with the scan and a sans fights it.
  17px because this is read, not skimmed. `68ch` because a 1200px-wide paragraph
  is unreadable regardless of how good the font is.
- **UI chrome — system sans.** Buttons, labels, mode toggle.
- **Page labels and timings — mono.** `p.3074` and `6.14s` are data. Tabular
  figures stop the pipeline strip from jittering as numbers change.

## 5. Rules that keep it from drifting

**Radius is meaningful.** `0` scans · `3px` chips/buttons/inputs · `6px` cards ·
`999px` the confidence dot and nothing else.

**Exactly one shadow exists** — the lightbox overlay. Everything else gets a 1px
`ink-100` border. Paper does not float. This single rule prevents most of the
drift that makes an interface look generic.

**Citations hoist past sentence punctuation.** The chip carries 5px of internal
padding, so `[p:3254].` renders with a visually orphaned period. The renderer
moves a trailing citation after the full stop -- `...subsequent rate.` `p.3254` --
the standard reference convention. One regex in the markdown post-pass.

**Two marker shapes reach the renderer.** The prompt asks the model for
`[p:NNN]` ([llm.py:761](../../backend/llm/llm.py#L761)) but shows it context
already marked `[p:NNN|c:NNNN]` ([llm.py:659](../../backend/llm/llm.py#L659)),
and the model copies what it sees. In practice the chunk-id form dominates, so
the renderer accepts both -- `\[p:(\d+)(?:\|c:\d+)?\]`. Matching only the
documented form silently drops every real citation; that is exactly what
happened, and only a live `/ask` surfaced it.

**A cited page is not always a retrievable one.** `sources` is built from the
retrieved chunks, not from the answer text, so the model can cite a page with no
`visual_context` entry. The frontend must not construct the URL itself --
`resolve_page_urls()` is the only permitted constructor
([ARCHITECTURE.md](../../ARCHITECTURE.md)) -- so those chips render `.inert` and
`disabled` rather than as a click that does nothing.

**Motion is functional.** 120ms on hover/focus, 180ms on the lightbox, and one
looping animation: the wait skeleton. Nothing animates on scroll. Everything
collapses under `prefers-reduced-motion`.

## 6. The wait problem

`/ask` is a blocking POST with a 90s budget
([config.py:54](../../backend/config.py#L54)) and no token stream. Real waits are
10–40s. A spinner for 40 seconds reads as a hang.

v1 uses a staged skeleton labelled with the pipeline's real stage names —
retrieval → rerank → draft → verify — advancing on median durations.

> `ponytail:` this is an estimate, not telemetry. `timings` only arrives *with*
> the answer, so the bar cannot be truthful mid-flight. Known ceiling: on a slow
> request it will sit at "verifying" longer than it claims. The real fix is an
> SSE endpoint emitting stage transitions, which is a backend change and should
> not gate the frontend.

## 7. Deliberately not built

- **Dark mode.** Every scan is white paper; in a dark shell they glow and take
  over the layout. Half-done dark mode is worse than none. All values route
  through tokens, so it is later one `[data-theme]` block — not a refactor.
- **A component library.** Nine components, one page. A design system this size
  is a stylesheet.
- **An icon set.** Four glyphs are needed. Four inline SVGs.

## 8. Anti-slop contract

Explicitly banned, because these are the defaults that would otherwise arrive:

purple→blue gradients · glassmorphism · gradient text · full-round everything ·
centred hero over a gradient · scroll-reveal animation · shadow on every card ·
neon "AI" accents · generic geometric sans with no rationale · a chat bubble UI
(this is a reference tool — the answer is a document, not a message).

A medical reference that looks like a landing page for a Series A undermines its
own credibility before a word is read.
