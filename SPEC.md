# Noise — Phase 0 Project Spec

## Problem Statement

Finding reliable answers online means opening multiple tabs, scanning noisy pages, and mentally stitching together fragments from different sources. **Noise** collapses that into a single step: ask a question, get one concise answer backed by cited sources.

## Input

A natural-language question typed into the app's search bar.

## Output

A single summarized answer that:
- Directly addresses the question
- Cites the sources it drew from (linked)
- Is readable in under 30 seconds

## First Supported Sources

1. **Web search results** (via a search API — e.g. SearXNG, Brave Search, or Serper)
2. **Web page content** (extracted/scraped from top results)

_Future phases may add: PDFs, YouTube transcripts, docs sites, code repos._

## What Success Means

- User types a question and gets a useful, cited answer within ~5 seconds.
- The answer is factually grounded in the retrieved sources (no hallucinated claims).
- At least 2–3 source links are provided so the user can verify.
- Works on a local iOS/Android emulator via Expo.

## What Is NOT in Phase 0

- Login / user accounts
- Persistent memory or conversation history
- Multiple agent personas
- Real-time streaming UI
- Eval / benchmarking platform
- Fancy UI beyond the search bar + result card
