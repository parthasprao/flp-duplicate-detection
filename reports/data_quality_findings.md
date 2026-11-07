# Data-Quality Findings & Systematic Corpus Defects for CourtListener

**Target Audience:** Data maintainers, pipeline developers, and database administrators at Free Law Project (CourtListener).  
**Track:** Track 4 (Data-Quality Findings)  
**Investigation Source:** CourtListener bulk snapshot, `candidates.jsonl`, `dev_silver.jsonl`, and `flp_cluster_metadata.jsonl`.

---

## 1. SCOTUS Cert Denials Systematically Misassigned to CAFC (`cafc`)

### The Defect
Multiple Supreme Court cert denials from the October 2004 term (reported in **Volume 543 U.S.**) were ingested into CourtListener under two conflicting records:
1. One record filed correctly under court `scotus` (U.S. Supreme Court).
2. A duplicate/companion record filed erroneously under court **`cafc`** (U.S. Court of Appeals for the Federal Circuit).

Because the court assignment disagrees (`scotus` vs. `cafc`), standard duplicate-detection heuristics that require `same_court == 1` completely missed these duplicates, while the baseline rule classifier mistakenly called them `citation_data_error`.

### Concrete Examples

| Example | SCOTUS Cluster ID | CAFC Cluster ID | Case Name | Shared Citation | Live CourtListener URLs |
|---|---|---|---|---|---|
| **Pair 1** | `141222` | `7356301` | *Parsons v. Price* / *Wood v. United States* | `543 U.S. 916`, `125 S. Ct. 34` | [SCOTUS 141222](https://www.courtlistener.com/opinion/141222/x/) · [CAFC 7356301](https://www.courtlistener.com/opinion/7356301/x/) |
| **Pair 2** | `139285` | `7471861` | *Bolden v. Fischer* | `543 U.S. 916`, `125 S. Ct. 137` | [SCOTUS 139285](https://www.courtlistener.com/opinion/139285/x/) · [CAFC 7471861](https://www.courtlistener.com/opinion/7471861/x/) |
| **Pair 3** | `139582` | `7471863` | *Travis County v. Perry* | `543 U.S. 941`, `125 S. Ct. 352` | [SCOTUS 139582](https://www.courtlistener.com/opinion/139582/x/) · [CAFC 7471863](https://www.courtlistener.com/opinion/7471863/x/) |
| **Pair 4** | `138693` | `7471862` | *Rifenbury v. Dretke* | `543 U.S. 916`, `125 S. Ct. 138` | [SCOTUS 138693](https://www.courtlistener.com/opinion/138693/x/) · [CAFC 7471862](https://www.courtlistener.com/opinion/7471862/x/) |
| **Pair 5** | `139274` | `7471861` | *Hospitality Mgmt v. Shell Oil* | `543 U.S. 916`, `125 S. Ct. 136` | [SCOTUS 139274](https://www.courtlistener.com/opinion/139274/x/) · [CAFC 7471861](https://www.courtlistener.com/opinion/7471861/x/) |

### Root Cause in CourtListener Scraper/Parser
In October 2004, the Federal Circuit released order summaries referencing Supreme Court cert dispositions in patent/claims appeals. The Federal Circuit scraper (or PACER/GPO importer) parsed the order header citing *U.S. Reports* orders and erroneously assigned the originating court to `cafc` instead of `scotus`. Furthermore, multiple distinct cert dispositions printed on pages 916 and 941 of 543 U.S. were clustered together.

### Recommended Maintainer Action
1. Run a database audit query:
   ```sql
   SELECT oc.id, oc.case_name, c.citation_key
   FROM search_opinioncluster oc
   JOIN search_citation c ON c.cluster_id = oc.id
   WHERE oc.court_id = 'cafc' AND c.reporter = 'U.S.' AND c.volume = '543';
   ```
2. Reassign court from `cafc` to `scotus`, and create `ClusterRedirection` records pointing duplicate clusters to the canonical SCOTUS records.

---

## 2. 19th-Century "14 N.Y." Citations that are Sickels Nominative Volumes

### The Defect
In 19th-century New York case law (circa 1874–1876), decisions reported by State Reporter Hiram E. Sickels were frequently cited by volume number within his tenure (e.g., "14 Sickels"). In early CourtListener ingestions, these were stripped or keyed simply as `14|NY|<page>`.

However, Volume 14 of the official *New York Reports* (`14 N.Y.`) covers decisions from **1856** (reported by George F. Comstock), NOT the 1870s Sickels decisions. As a result:
- Dozens of 1870s Court of Appeals decisions (such as *Perrine v. Hotchkiss*, *National Life Ins. v. Jones*, *Wiles v. Bogert*, *Board of Supervisors v. Wandel*, *Thrasher v. Bentley*) collide on identical citation keys like `14 N.Y. 623` and `14 N.Y. 624`.
- Clusters `5632050`, `5632051`, `5632044`, `5632045`, `5632049` each carry overlapping citations that corrupt the citation graph.

### Recommended Maintainer Action
1. Run a targeted scrubber on New York Court of Appeals opinions filed between 1872 and 1880 that carry `14 N.Y.` citations.
2. Re-key `14 Sickels` to the true corresponding official New York Reports volume (Volume 59 N.Y. through 63 N.Y.).

---

## 3. Systematic Shared Reporter Page Collisions in NY Appellate Divisions

### The Defect
In the New York Appellate Division reports (both *A.D.3d* and *N.Y.S.3d*), short memorandum dispositions and motion orders are printed consecutively, often 4 to 8 per printed page.
In multiple instances, CourtListener ingested the entire page text into one cluster, and then ingested an individual case from that page into another cluster:
- **Pair `4383809-6234821`:** Cluster `4383809` (*Buchwald v. Silverman Shin & Byrne PLLC*) and Cluster `6234821` (*People v. Melgurrero*) share page 416 of 149 A.D.3d.
- **Pair `4378273-6234619`:** Cluster `4378273` (*Matter of Brady J.*) and Cluster `6234619` (*People v. Hawkins*) share page 312 of 64 N.Y.S.3d.

The rule classifier flagged these as duplicates or citation errors because of `shared_citations == 1`. In reality, they are two completely distinct appeals that share physical real estate in the paper reporter.

---

## 4. Analysis of Historical `ClusterRedirection` Table (`flp_cluster_metadata.jsonl`)

From the 2,664 clusters analyzed in the FLP metadata snapshot:
- **2,655 clusters (99.7%)** carry historical redirection entries in `cluster_redirections`.
- Exactly **3,813 historical redirections** were recorded across these clusters.
- All 3,813 historical entries carry `reason = 1` (`VERSION`), confirming that CourtListener's historical consolidation efforts were focused heavily on opinion versioning (consolidating slip opinions with official releases via issue [#6074](https://github.com/freelawproject/courtlistener/pull/6074) and [#4819](https://github.com/freelawproject/courtlistener/issues/4819)).
- There are virtually zero automated historical entries for `reason = 2` (`DUPLICATE`) or `reason = 3` (`CONSOLIDATION`), demonstrating that cross-cluster duplicate detection at ingestion has been an open gap in production.

Our triage tool directly closes this gap by outputting automated, calibrated `ClusterRedirection` proposals for reasons 1, 2, and 3.
