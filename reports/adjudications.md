# Adjudication Report & Annotation Guide v2 Proposal

**Track:** Case-Identity Adjudication (Track 3)  
**Dataset:** `adjudicate_me.jsonl` (12 hard disagreement pairs without ground truth)  
**Target:** Data maintainers and annotation leads at Free Law Project / CourtListener

---

## Part 1: Adjudications of the 12 Ambiguous Pairs

The 12 pairs in `adjudicate_me.jsonl` represent cases where two state-of-the-art LLMs (Claude Opus and GPT) reading full opinion texts reached conflicting conclusions. Below are our expert adjudications, the decisive factual evidence, and the CourtListener consolidation recommendation.

---

### Pair 1: `adj-001` (Clusters `7717118` & `8433379`) — *Trim v. McPherson*
- **Jurisdiction:** Supreme Court of Tennessee (1869-09-15)
- **Model Conflict:** Claude = `parallel_split_cluster` (0.80) vs. GPT = `insufficient_evidence` (0.60)
- **Our Adjudication:** **`insufficient_evidence`** (Confidence: 0.70)
- **Decisive Fact:** Cluster `8433379` is a Shannon's Cases cross-reference stub containing no extracted opinion text or docket metadata. Under CourtListener's safety rule, a merge cannot be substantiated without corroborating text evidence.
- **Maintainer Action:** Do not consolidate. Keep stub separate until fuller 19th-century reporter text is ingested.

---

### Pair 2: `adj-002` (Clusters `3775915` & `3775936`) — *Shakarzahi v. Malooly*
- **Jurisdiction:** Texas Court of Appeals, Third District (2016-06-30), Docket `03-15-00229-CV`
- **Model Conflict:** Claude = `parallel_split_cluster` (0.62) vs. GPT = `related_decision` (0.88)
- **Our Adjudication:** **`related_decision`** (Sub-type: *Separate Judgment vs. Opinion*) (Confidence: 0.85)
- **Decisive Fact:** The Texas appellate court entered a formal one-page judgment order dismissing the appeal and a separate multi-page memorandum opinion explaining the rationale. They represent distinct judicial instruments entered on the same docket.
- **Maintainer Action:** Keep both clusters distinct. Exposes Gap #1 in Annotation Guide v1 (no label for companion judgment/opinion).

---

### Pair 3: `adj-003` (Clusters `9472736` & `9472737`) — *Brockman v. State*
- **Jurisdiction:** Texas Court of Appeals, Sixth District (2024-02-02), Docket `06-23-00067-CR`
- **Model Conflict:** Claude = `parallel_split_cluster` (0.60) vs. GPT = `related_decision` (0.88)
- **Our Adjudication:** **`related_decision`** (Sub-type: *Separate Judgment vs. Opinion*) (Confidence: 0.88)
- **Decisive Fact:** Identical pattern to `adj-002`. Cluster `9472736` is the formal appellate judgment; `9472737` is the memorandum opinion. Under Texas practice, judgments and opinions carry separate legal significance and should not be collapsed into a single opinion document.
- **Maintainer Action:** Keep both clusters distinct.

---

### Pair 4: `adj-004` (Clusters `1119670` & `4907530`) — *Key Design Inc. v. Moser*
- **Jurisdiction:** Supreme Court of Washington (1999-09-09), Citations `983 P.2d 653` and `138 Wash. 2d 875`
- **Model Conflict:** Claude = `parallel_split_cluster` (0.80) vs. GPT = `probable_version` (0.98)
- **Our Adjudication:** **`parallel_split_cluster`** (Confidence: 0.90)
- **Decisive Fact:** This is a single Washington Supreme Court decision whose parallel citations (West P.2d vs. Official Washington Reports Wash. 2d) were ingested into separate clusters via Lawbox and CAP. Both clusters contain the same merits opinion text.
- **Maintainer Action:** Consolidate under `ClusterRedirection(reason=CONSOLIDATION, code=3)`.

---

### Pair 5: `adj-005` (Clusters `9472734` & `9472735`) — *Brockman v. State* (Companion)
- **Jurisdiction:** Texas Court of Appeals, Sixth District (2024-02-02), Docket `06-23-00066-CR`
- **Model Conflict:** Claude = `parallel_split_cluster` (0.60) vs. GPT = `related_decision` (0.90)
- **Our Adjudication:** **`related_decision`** (Confidence: 0.88)
- **Decisive Fact:** Companion criminal appeal on docket `06-23-00066-CR`. Separate judgment entry and memorandum opinion.
- **Maintainer Action:** Keep both clusters distinct.

---

### Pair 6: `adj-006` (Clusters `2825532` & `2832402`) — *JC Fab, Inc. v. OESC*
- **Jurisdiction:** Oklahoma Court of Civil Appeals (2015-04-27), Citation `2015 OK CIV APP 65`
- **Model Conflict:** Claude = `probable_duplicate` (0.85) vs. GPT = `probable_version` (0.99)
- **Our Adjudication:** **`probable_duplicate`** (Confidence: 0.92)
- **Decisive Fact:** Identical Oklahoma State Courts Network (OSCN) slip opinion document scraped twice into CourtListener. No textual modifications, corrections, or reporter pagination differences exist.
- **Maintainer Action:** Merge under `ClusterRedirection(reason=DUPLICATE, code=2)`.

---

### Pair 7: `adj-007` (Clusters `566430` & `9006481`) — *El Rescate Legal Services v. EOIR*
- **Jurisdiction:** U.S. Court of Appeals for the Ninth Circuit (1991-08-12), Docket `90-55292`
- **Model Conflict:** Claude = `related_decision` (0.60) vs. GPT = `probable_version` (0.99)
- **Our Adjudication:** **`probable_version`** (Confidence: 0.85)
- **Decisive Fact:** Following a petition for rehearing, the Ninth Circuit panel issued an amended merits opinion superseding the slip opinion on the exact same docket and filing date. In legal publishing, amended opinions are canonical versions of the decision, not independent cases.
- **Maintainer Action:** Consolidate under `ClusterRedirection(reason=VERSION, code=1)` and set `Opinion.main_version`.

---

### Pair 8: `adj-008` (Clusters `7717090` & `8433382`) — *Carson v. Prater*
- **Jurisdiction:** Supreme Court of Tennessee (1869-09-15)
- **Model Conflict:** Claude = `parallel_split_cluster` (0.82) vs. GPT = `insufficient_evidence` (0.60)
- **Our Adjudication:** **`insufficient_evidence`** (Confidence: 0.75)
- **Decisive Fact:** Identical 19th-century Shannon's Cases stub defect as `adj-001`. The secondary cluster lacks substantive content.
- **Maintainer Action:** Abstain / Defer until complete texts are available.

---

### Pair 9: `adj-009` (Clusters `5246728` & `5247138`) — *Brown v. State*
- **Jurisdiction:** Missouri Court of Appeals, Western District (1988-08-30 vs. 1988-11-15)
- **Model Conflict:** Claude = `unrelated` (0.90) vs. GPT = `shared_page_ambiguity` (0.99)
- **Our Adjudication:** **`unrelated`** (Confidence: 0.95)
- **Decisive Fact:** Docket numbers conflict explicitly (`WD 40231` vs. `WD 40703`), and decisions were filed 77 days apart. Both records share the common name "Brown v. State" and contain identical Missouri Rule 84.16(b) stock post-conviction summary order boilerplate. They represent two completely different criminal defendants.
- **Maintainer Action:** Keep records strictly separate.

---

### Pair 10: `adj-010` (Clusters `1113466` & `6976979`) — *State v. JS*
- **Jurisdiction:** Washington Court of Appeals (1993-06-21), Citations `70 Wash. App. 1040` vs. `70 Wash. App. 659`
- **Model Conflict:** Claude = `parallel_split_cluster` (0.75) vs. GPT = `probable_version` (0.99)
- **Our Adjudication:** **`probable_version`** (Confidence: 0.90)
- **Decisive Fact:** Washington appellate practice originally posted an unpublished table disposition (`70 Wash. App. 1040`), which was subsequently ordered published as a full citable opinion (`70 Wash. App. 659`). The published opinion supersedes the table entry.
- **Maintainer Action:** Consolidate under `ClusterRedirection(reason=VERSION, code=1)`.

---

### Pair 11: `adj-011` (Clusters `2329805` & `4951558`) — *Cantu v. Dept. of Labor & Industries*
- **Jurisdiction:** Washington Court of Appeals (2012-03-27), Citations `277 P.3d 685` and `168 Wash. App. 14`
- **Model Conflict:** Claude = `parallel_split_cluster` (0.80) vs. GPT = `probable_version` (0.98)
- **Our Adjudication:** **`parallel_split_cluster`** (Confidence: 0.92)
- **Decisive Fact:** One Washington Court of Appeals decision split across parallel reporter citations (P.3d in Lawbox scraper vs. Wash. App. in CAP). The text and citations are identical; this is an artifact of multiple bulk ingestions.
- **Maintainer Action:** Consolidate under `ClusterRedirection(reason=CONSOLIDATION, code=3)`.

---

### Pair 12: `adj-012` (Clusters `2825575` & `2832442`) — *Eagle Energy Production v. Corporation Commission*
- **Jurisdiction:** Oklahoma Court of Civil Appeals (2014-12-05), Citation `2015 OK CIV APP 18`
- **Model Conflict:** Claude = `probable_duplicate` (0.85) vs. GPT = `probable_version` (0.99)
- **Our Adjudication:** **`probable_duplicate`** (Confidence: 0.92)
- **Decisive Fact:** Re-ingestion of the identical OSCN document under two separate cluster IDs with no text modifications.
- **Maintainer Action:** Merge under `ClusterRedirection(reason=DUPLICATE, code=2)`.

---

## Part 2: Proposed Annotation Guide v2

### Gap Analysis of Current Guide (v1)

1. **No Label for Judgment vs. Opinion:**  
   In states like Texas, courts frequently issue a 1-page formal decree ("Judgment") and a 15-page "Memorandum Opinion" on the same docket and date. Annotators in v1 are forced to choose between `parallel_split_cluster` (inaccurate, because they are not parallel citations of the same document) and `related_decision` (misleading, because they are the same proceeding).
2. **Missing Multi-Tagging for Compound Errors:**  
   Records frequently exhibit two simultaneous phenomena: e.g., two records represent the exact same decision, but one cluster has a mis-scraped citation string. In v1, annotators must choose either `probable_duplicate` or `citation_data_error`, losing half the truth.
3. **Ambiguity Between Shared Page and Citation Error:**  
   When two unrelated decisions cite the same reporter volume and page, annotators cannot tell whether the citation was mistyped or whether multiple short orders appeared on that page.

### Proposed Additions in Guide v2

#### 1. New Canonical Label: `companion_instrument`
- **Definition:** Distinct legal instruments entered by the court in the same proceeding on the same date (e.g., formal judgment order, explanatory memorandum, findings of fact, or sentencing entry).
- **Consolidation Policy:** **Keep distinct**, but link via a parent-child or companion document relationship in CourtListener rather than merging text.

#### 2. Secondary Tagging Flags
Allow primary classification with optional modifier tags:
- `+citation_error`: Records should be consolidated, but citation metadata requires scrubbing.
- `+rehearing_revision`: The document is a substantive post-rehearing version of an earlier opinion.
- `+stub_cluster`: One of the records is a citation-only stub lacking substantive text.

#### 3. Formal Decision Tree for Reporter Pages
When two records share a citation:
- If `page_cluster_span >= 2` AND `min_hamming > 20`: Label as `shared_page_ambiguity`.
- If `page_cluster_span <= 1` AND case names completely disagree: Label as `citation_data_error`.
- If `min_hamming <= 15` AND parties match: Label as `probable_version` or `probable_duplicate`.
