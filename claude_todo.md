# Claude TODO — NRP Analysis Refinement

## Operating Instructions (session setup — 2026-04-23)

- **Execution order:** Work through P1 → P4 strictly in order. Do not parallelize across priority blocks.
- **Variable verification:** Before executing each task, verify required variables (`lown_arak`, `loth_arak`, `county`, `hundred`, Percy proximity, etc.) exist in the processed data. Flag if missing.
- **Percy vs. fsnub:** The Percy variable has largely been subsumed into `fsnub`. Run diagnostics including both to compare information content; do not blindly restore Percy.
- **Sign analysis (P1 item 3):** Run additional diagnostics only (partial regressions, residuals, VIFs, collinearity checks). Nicholas will write the narrative prose.
- **Inference:** Wild cluster bootstrap is fine; use a state-of-the-art econometric package (e.g. `wildboottest`).
- **Snub rework (P3 item 3):** Ignore the "Statute of Uses" suggestion. If the rework requires substantial conceptual work, flag it and leave it for Nicholas.
- **LaTeX tables (P4):** Create or modify the scripts that *generate* the tables; Nicholas will run them to produce the final `.tex` files.
- **Session cadence:** Iterate notebook-by-notebook. Expect multiple sessions. Pause for checkpoint after each sub-task.
- **Check off tasks here as you perform them**
## Priority 0: Repo Cleanup
- [ ] Ensure that each analysis script in R is converted to an R-language Jupyter notebook that is fully up-to-date and outputs its tables to Output/Tables/ properly. 
- [ ] Include each version of the analysis in each R notebook, using each notebook's preferred specification (logit, aipw, survival analysis, etc.) for muster, primary muster, and rebel gentlemen's seats
	- [ ] Monastic land
	- [ ] Large vs Small monastery land
	- [ ] On- vs off-site land
## Priority 1: Existential Mechanism Fixes (The "Tenant" Story)

- [ ] **Restore On-site vs. Off-site Decomposition:** The "tenants vs. employees" argument relies on showing that land away from the monastery (tenanted) predicts revolt while site land (wage labor) does not. 
    - *Task:* Re-run the `primary_split` / "OwnOther" specification.
    - *Output:* Generate a LaTeX table showing `lown_arak` and `loth_arak` side-by-side.
- [ ] **Solve the "Large vs. Small" Paradox:** Large monastery land consistently outperforms small monastery land. 
    - *Task:* Explicitly integrate `mo_` (monastic opposition and crown interference) variables as interaction terms or controls to test if large houses were perceived as "next in line."
- [ ] **Address Variable Signs:** Explain the negative coefficients for **Net Income** and **Tithes**.
    - *Task:* Refine the "extractive relationship" narrative for tithes and investigate why net income flips sign when land is controlled for.

## Priority 2: Robustness & Regional Controls

- [ ] **County Fixed Effects:** Rule out the "Why the North?" factors (border militarization, regional poverty) being absorbed by monastic variables.
    - *Task:* Add county-level fixed effects to the main logit and survival models.
- [ ] **Sample Robustness (The "Dropped Parish" Problem):** 20% of parishes (including the most rebellious) are dropped when Lay Subsidy is included.
    - *Task:* Run a "Full Sample" robustness table (n ≈ 1,755) excluding the Lay Subsidy variable to ensure monastic coefficients remain stable.
- [ ] **Standardize Inference:** 
    - *Task:* Unify standard error clustering at the "hundred" level across all models. Consider wild cluster bootstrap for rare primary muster outcomes.

## Priority 3: Elite vs. Commons Model Differentiation

- [ ] **The "Gentry vs. Musters" Split:** The data suggests two different stories. 
    - *Task:* Explicitly frame the `seats` model as a joint product of Court patronage (Court Officer IDW) and Taxation, whereas `musters` are driven by Dissolution-economics.
- [ ] **Restore the Percy Network:** The Percy variable is a historical cornerstone of Northern rebellion.
    - *Task:* Re-add the Percy country seat proximity variable to the "Elite" models or provide a statistical justification for its substitution by the "Snubbed Family" variables.
- [ ] **Address the "Snub" Failure:** The `fsnub` variables are currently insignificant or have the "wrong" sign.
    - *Task:* Re-evaluate the operationalization of elite grievances beyond simple family snubs.

## Priority 4: Technical Formalization & Housekeeping

- [ ] **Formalize LaTeX Tables:** Several advanced specs (AIPW, opposition scores, off-site land) exist only as plots or notebook output.
    - *Task:* Export all key specifications to `.tex` tables in `Output/Tables/`.
- [ ] **Variable Labeling & Formatting:**
    - *Task:* Clean up stargazer headers (remove raw R code like `paste(...)`).
    - *Task:* Ensure all tables use labels from `pretty_dict.json` rather than internal codes (e.g., `lbg_arak`).
- [ ] **AIPW Substantive Effects:** 
    - *Task:* Calculate marginal effects/probabilities for AIPW estimates to move beyond significance stars to real-world impact (e.g., "doubling the baseline rate").
