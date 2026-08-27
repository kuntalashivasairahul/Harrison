"""Scrollytelling copy for the helix hero, as data rather than markup.

Adding, reordering or rewording a step must never require touching
``partials/hero.html`` or any JavaScript.  ``len(HERO_STEPS)`` is the single
source of truth for the section's scroll height — it is passed to the template
as a CSS custom property and nothing hardcodes a viewport count.

Import-time purity: frozen dataclasses and literals only.  No I/O, no network,
no imports from anywhere else under ``backend/`` (ARCHITECTURE, "import-time
purity").
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeroStep:
    """One scroll step: the copy on the left, the card along the bottom."""

    index: str        # "01" — displayed, so it is a string, not an int
    eyebrow: str
    heading: str
    lede: str
    rail_key: str     # must match an entry in RAIL
    card_meta: str    # the small line above the card title
    card_title: str
    card_body: str
    cta_label: str = ""
    cta_href: str = ""


# The rail down the left edge.  Order defines display order; the keys are
# matched against HeroStep.rail_key to move aria-current as you scroll.
RAIL: tuple[tuple[str, str], ...] = (
    ("title", "Start"),
    ("intro", "Intro"),
    ("retrieve", "Retrieve"),
    ("rerank", "Rerank"),
    ("draft", "Draft"),
    ("verify", "Verify"),
)

# Every number and mechanism named below is real and matches the running
# system: RRF k=60 and the -3.0 rerank threshold are the frozen constants in
# backend/config.py, BGE-M3 is the encoder, and the refusal string is verbatim.
# This is a medical tool — the marketing copy does not get to round up.
HERO_STEPS: tuple[HeroStep, ...] = (
    HeroStep(
        index="01",
        eyebrow="Reference assistant",
        heading="HarrisonGPT",
        lede="A grounded reference assistant over Harrison's Principles of "
             "Internal Medicine. Ask a clinical question and get an answer "
             "built from the retrieved text — every claim linked back to the "
             "scanned page it came from.",
        rail_key="title",
        card_meta="Harrison's Principles of Internal Medicine",
        card_title="Built to be checked, not trusted blindly",
        card_body="Five stages sit between your question and the answer — "
                  "retrieval, reranking, drafting, verification. Scroll to "
                  "see how each one works.",
        cta_label="Try it now",
        cta_href="/chat",
    ),
    HeroStep(
        index="02",
        eyebrow="What this is",
        heading="Every answer points back to the page it came from.",
        lede="A reference assistant that reads Harrison's Principles of Internal "
             "Medicine and cites it — not from memory, but from the retrieved "
             "text, with the scanned page one click away.",
        rail_key="intro",
        card_meta="The whole idea",
        card_title="Grounded, not remembered",
        card_body="The model never answers from training. It answers from "
                  "passages pulled out of the book a moment earlier, and every "
                  "claim carries the page it came from.",
        cta_label="Try it now",
        cta_href="/chat",
    ),
    HeroStep(
        index="03",
        eyebrow="Stage one",
        heading="Retrieval happens before generation.",
        lede="Your question is expanded — acronyms resolved, clinical framing "
             "added — and anything non-medical stops right here. What survives "
             "runs through dense vector search and BM25 in parallel.",
        rail_key="retrieve",
        card_meta="Hybrid search · RRF k=60",
        card_title="Two searches, one ranking",
        card_body="BGE-M3 embeddings catch meaning, BM25 catches exact terms "
                  "and drug names. Reciprocal rank fusion merges the two lists "
                  "so neither method's blind spot decides the answer.",
    ),
    HeroStep(
        index="04",
        eyebrow="Stage two",
        heading="Weak matches are dropped, not ranked lower.",
        lede="A cross-encoder reads your question and each candidate passage "
             "together and scores the pair. Anything below the threshold is "
             "discarded outright rather than pushed down the list.",
        rail_key="rerank",
        card_meta="Cross-encoder · threshold −3.0",
        card_title="A hard filter, not a soft sort",
        card_body="This is what makes refusal possible. If nothing clears the "
                  "bar there is no weak context left to write a plausible "
                  "answer from, so the system says so instead of guessing.",
    ),
    HeroStep(
        index="05",
        eyebrow="Stage three",
        heading="The draft can only use what survived.",
        lede="The answer is written from the passages that cleared the filter "
             "and from nothing else, citing the page behind each claim as it "
             "goes.",
        rail_key="draft",
        card_meta="Grounded generation",
        card_title="Citations are load-bearing",
        card_body="A page marker is not a footnote added at the end — it is the "
                  "record of which retrieved passage a sentence came from, and "
                  "it is checked in the next stage.",
    ),
    HeroStep(
        index="06",
        eyebrow="Stage four",
        heading="A second pass checks it before you see it.",
        lede="An independent verification pass re-reads the draft against the "
             "same passages. Claims the context does not support come out. "
             "What reaches you carries a confidence grade derived from "
             "retrieval quality, never asserted by the model about itself.",
        rail_key="verify",
        card_meta="Independent verifier",
        card_title="Insufficient information in the provided context.",
        card_body="That sentence is returned verbatim and unsoftened when the "
                  "book does not answer your question. Knowing where a "
                  "reference tool stops is the useful part.",
        cta_label="Ask it something",
        cta_href="/chat",
    ),
)
