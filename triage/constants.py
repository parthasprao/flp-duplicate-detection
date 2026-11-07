"""Taxonomy, constants, and CourtListener integration mappings."""

# The 8 canonical labels in the hackathon schema
LABEL_PROBABLE_DUPLICATE = "probable_duplicate"
LABEL_PROBABLE_VERSION = "probable_version"
LABEL_PARALLEL_SPLIT = "parallel_split_cluster"
LABEL_RELATED_DECISION = "related_decision"
LABEL_SHARED_PAGE = "shared_page_ambiguity"
LABEL_CITATION_ERROR = "citation_data_error"
LABEL_UNRELATED = "unrelated"
LABEL_INSUFFICIENT = "insufficient_evidence"

VALID_LABELS = {
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    LABEL_PARALLEL_SPLIT,
    LABEL_RELATED_DECISION,
    LABEL_SHARED_PAGE,
    LABEL_CITATION_ERROR,
    LABEL_UNRELATED,
    LABEL_INSUFFICIENT,
}

# Labels proposing that records describe the same decision and should be consolidated
MERGE_LABELS = {
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    LABEL_PARALLEL_SPLIT,
}

# Labels designating that records represent distinct judicial acts and must stay separate
KEEP_SEPARATE_LABELS = {
    LABEL_RELATED_DECISION,
    LABEL_SHARED_PAGE,
    LABEL_CITATION_ERROR,
    LABEL_UNRELATED,
}

# Mapping from merge labels to CourtListener ClusterRedirection reasons
# In CourtListener: 1 = VERSION, 2 = DUPLICATE, 3 = CONSOLIDATION, 4 = SEALED
CLUSTER_REDIRECTION_MAP = {
    LABEL_PROBABLE_VERSION: {
        "reason_code": 1,
        "reason_str": "VERSION",
        "doc": "Opinion.main_version link or revised/slip consolidation",
    },
    LABEL_PROBABLE_DUPLICATE: {
        "reason_code": 2,
        "reason_str": "DUPLICATE",
        "doc": "Identical document or scraping re-ingestion",
    },
    LABEL_PARALLEL_SPLIT: {
        "reason_code": 3,
        "reason_str": "CONSOLIDATION",
        "doc": "Parallel reporter split or majority/dissent split cluster",
    },
}

# Review queue priorities
PRIORITY_CRITICAL = "CRITICAL"    # Transitive contradiction or dangerous false-merge risk
PRIORITY_HIGH = "HIGH"            # Proposed merge awaiting maintainer confirmation
PRIORITY_MEDIUM = "MEDIUM"        # Ambiguous / AI disagreement / needs adjudication
PRIORITY_LOW = "LOW"              # Confident keep-separate / low-risk background audit
