"""Phase 6 — Answer aggregation pipeline.

Modules:
    source_reader      – normalise filtered evidence into a uniform shape
    claim_extractor    – pull individual factual claims from each source
    evidence_ranker    – score source quality and rank evidence
    consensus_builder  – detect agreement/disagreement, merge duplicates
    final_writer       – compose the aggregated judgment answer
"""
