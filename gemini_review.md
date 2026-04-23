# Analysis Review: Gaps & Observations

This review highlights gaps in the current econometric argument and identifies areas for further development to strengthen the case that economic dislocations from the Dissolution were the primary drivers of the Pilgrimage of Grace.

## 1. The "Tenants vs. Employees" Mechanism (Off-site Land)
*   **Gap:** While the "OwnOther" (On-site vs. Off-site) specification exists in the code (`jn_10_AIPW.ipynb`), the results are not yet clearly presented in the LaTeX tables. 
*   **Observation:** The current text argues that off-site land (more likely to be tenanted) is a better predictor than on-site land (local employment). 
*   **Recommendation:** Generate a combined IPW table showing coefficients for `lown_arak` (on-site) and `loth_arak` (off-site) side-by-side to empirically anchor the "tenurial fear" claim.

## 2. Elite Motivations: The "Family Snub" Gap
*   **Gap:** The qualitative argument suggests elites were driven by personal slights (`fsnub`). However, `elite_vs_commons_primary.tex` and `elite_vs_commons_seats.tex` show the "Snubbed Family" variables are statistically insignificant.
*   **Observation:** Instead, for the `seats` outcome (rebel gentlemen), `Court Officer IDW Score` (0.296***) and `ln(Lay Subsidy per Capita)` (0.353***) are powerful predictors.
*   **Recommendation:** Re-evaluate the "snub" hypothesis. The data suggests elites may have been driven more by direct proximity to royal administrative centers (Court Officers) and fiscal extraction (Lay Subsidy) than by specific family slubs.

## 3. The "Large vs. Small" House Paradox
*   **Observation:** Large monastery land consistently outperforms small monastery land in significance across models (e.g., `full_monastic.tex`, `old_vs_new_primary.tex`).
*   **Argument Support:** This supports your "expectations" argument—that the 1536 Dissolution of small houses created an immediate credible threat to large houses.
*   **Recommendation:** Explicitly integrate the data you mentioned regarding **royal interference in elections** and **pre-Dissolution opposition** as interaction terms or additional controls to explain *why* specific large houses were perceived as "next in line."

## 4. Taxation: Level vs. Change
*   **Confirmation:** `old_vs_new_primary.tex` shows that both the absolute level of the Lay Subsidy and the *change* since 1332 are insignificant predictors of primary musters.
*   **Implication:** This robustly supports the argument that fiscal grievances were secondary to monastic ones for the commons, contrasting with the elite results where taxation level *is* significant.

## 5. Summary of Significant Drivers (Current Data)
*   **For the Commons (Musters):** Large monastery land, total population (urbanisation), and weather shocks (1536).
*   **For the Elites (Seats):** Proximity to Court Officers, Lay Subsidy levels, weather shocks, and Large monastery land.

## 6. Technical Gaps
*   **Missing Tables:** Some advanced specifications (IPW "OwnOther", AIPW with opposition scores) currently only exist as `.png` plots or are buried in notebook outputs. These need to be formalised into `.tex` tables for the final paper.
*   **Clustering:** Ensure that standard error clustering (currently at the "hundred" level) is consistent across all survival and logit specifications to ensure comparability.
