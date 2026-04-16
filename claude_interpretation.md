# Interpretation of Results in `Output/Tables/`

This is my best reading of the regression output currently in `Output/Tables/`, in light of the literature and framing already laid out in `paper.tex`. Quantities are coefficients (log-odds for logits, log-hazard for Cox, log-rate for Poisson) on standardized continuous variables, so each "one unit" change is a one-standard-deviation move in that covariate. The specifications now express monastic stocks per arable km² rather than as raw totals, which is an important shift: the regressions are asking whether *intensity* of monastic activity in a parish predicts revolt, not whether the parish happens to be large.

---

## 1. The DAG specification (`dag.tex`)

The minimally controlled model — the one the DAG says is sufficient to identify a controlled direct effect of monastic land — gives the cleanest pro-Bush result in the entire folder.

- `ln(Land Owned / Arable km²)` is positive across all three outcomes and significant for muster (β = 0.263, p < 0.1) and primary muster (β = 0.433, p < 0.05). It is positive but insignificant for gentlemen seats.
- Population is the strongest predictor everywhere (very tight, very large), exactly as expected: more people, more chances for muster.
- `ln(Lay Subsidy per Capita)` is *negative* and not significant in any column. Hoyle's "tax revolt" reading does not survive even this minimal model — the sign is wrong.

Read alone, this table is the strongest single piece of evidence in the paper for H₂ and against H₁: once you condition on what the DAG says is enough, monastic land predicts rebellion and per-capita tax does not. The richer specifications below reduce significance but not the sign or the rank ordering of which variables matter.

---

## 2. Muster, primary, and seats with full controls (`muster_all.tex`, `primary_all.tex`, `seat_all.tex`)

These are the workhorse logits/Poissons that the paper cites as Tables \ref{tab:muster_all}, \ref{tab:primary_all}, and \ref{tab:seat_all}. The pattern across the three outcomes is more interesting than any single column.

**Large vs. small monastery land.** With everything in the regression, large-monastery land is the variable that consistently does work:

- Muster: β = 0.24–0.28, marginally significant only in column (2).
- Primary: β = 0.48–0.50, p < 0.05–0.10 across columns (2)–(4). This is the strongest and most stable effect of any monastic variable on the most informative outcome.
- Seats: β = 0.17–0.23, marginal.

Small-monastery land per arable km² is essentially flat (and *negative* for primary musters). This is the key interpretive point: the parishes that produced primary musters were not those where the immediately-threatened small houses owned land — they were the ones soaked in the holdings of the large abbeys. The paper already builds a story about this in the Furness/Holm Cultram passage; the per-arable-km² rescaling sharpens it. Rebels were not just trying to save *their* landlord; the geography of revolt tracks the parishes whose tenancy fabric was densely large-monastic.

**House dummies.** The "small house" and "large house" dummies are large and significant in earlier columns of every table but mostly attenuate to insignificance once geographic controls and per-arable-km² monastic stocks both enter. The exception is the seats Poisson, where "Small House Dummy" stays huge and significant (β ≈ 2.5, p < 0.01) all the way through. Combined with the positive small-land coefficient on seats, that suggests gentleman participation tracks proximity to small houses themselves more than the broader monastic-land economy. That is consistent with the reading in §Rebel Gentlemen: the threat to *their* networks was the immediate one.

**Tithe.** Negative in every muster and primary specification (β ≈ −0.22 to −0.36, occasionally significant). This is striking and worth flagging in the paper more explicitly than it currently is. The current text describes tithe as the more "extractive" form of monastic income, with farther-flung counterparties and weaker personal ties; the negative coefficient is consistent with that reading. Parishes that paid tithes to monasteries were if anything *less* likely to rebel, supporting the view that it was the warm tenant-relationship channel — not the cold tithe-extraction channel — that mobilized the commons. Harrison's Lake Counties evidence (rebels demanding *reductions* in monastic tithes) fits.

**Net income.** Mostly negative, sometimes significant for seats and primary. This is the variable most exposed to the "monasteries as bullies" reading — wealthy houses that dominated and oppressed neighboring secular communities. Once you have the land, tithe, and dummy variables in, net income is mopping up the residual "size" effect, and that residual works *against* rebellion. This too needs a sentence in the paper: the parishes living under the largest monastic budgets were not the parishes that rose.

**Alms.** Positive but small and only intermittently significant. Given the well-known limitation of the alms field in the *Valor* (perpetual benefactions only), this is about as much as can reasonably be asked of it. In the Cox/IPW results below it gains some significance.

**Tax.** `ln(Lay Subsidy)` is small and never significant for muster or primary in any of the three full tables. It is positive and consistently significant only for seats. That is consistent with H₁ being false at the level of commoner mobilization but having some bite as a measure of where wealthy gentleman seats happened to be.

**Weather.** `Wet 1536` is positive and significant for primary and seats and marginal for muster. The 1535 wet shock is null. The harvest immediately preceding the rising mattered; the older Cravendale-era memory does not show up here. That is consistent with the Davies/Hoyle account of a precipitating subsistence shock on top of structural grievances.

**Population.** Always large, always significant for muster and primary. Always essentially zero for seats — gentlemen seats are not where people lived, they are where the powerful did.

---

## 3. Single-variable monastic specifications (`muster_monastic.tex`, `primary_monastic.tex`, `seat_monastic.tex`)

Each of these tables runs eight horse-race columns with one monastic variable at a time plus the standard controls. Read this way:

- **Net income alone** is positive and significant for muster (β = 0.32, p < 0.01) and primary (β = 0.34, p < 0.05). This *contradicts* the negative net-income sign in the multivariate full-model tables. The reconciliation is collinearity: net income is mostly capturing "is there a big house and does it own the land around it." Once you include the large-house dummy and large-monastery land directly, net income flips. This is worth a footnote in the paper because the net income variable, on its own, looks like a strong pro-Bush result and only becomes "wrong-signed" after partialling.
- **Large monastery land alone** β = 0.30 (muster, p<0.1), 0.54 (primary, p<0.05), 0.28 (seats, p<0.05). This is the most consistently positive variable across all three outcomes when run alone.
- **Large house dummy alone** β = 1.92 (muster, p<0.01), 1.86 (primary, p<0.05). Big effect. Together with the large-land result, this is the simplest summary of the entire paper: parishes near large monasteries with extensive landholdings rebelled.
- **Small monastery land alone**: small and insignificant for muster and primary, marginally positive for seats.
- **Small house dummy alone**: significant only for primary and seats, not for muster. Suggests the "your own monastery is being dissolved" channel does some work, but a smaller share than the large-house tenancy channel.

**Distance to Scotland** is reliably negative — closer to the border, more rebellion, exactly as the paper's "Why the North?" section anticipates. It is the strongest single non-monastic structural predictor of seats.

---

## 4. Cox proportional hazards (`survival.tex`)

The survival framing — accounting for the time at which news of the rising "exposed" each parish — does not weaken the story. If anything it strengthens it for the most plausible primary-muster outcome:

- `ln(Large Monastery Land / Arable km²)`: hazard log-coefficient β = 0.46–0.52, p < 0.05–0.10, *stable across all three columns*, including the geography-controlled one. So even after timing, population, taxation, weather, and geography, a one-SD increase in large-monastery land density raises the hazard of becoming a primary muster site by roughly e^0.5 ≈ 1.65.
- Small-monastery land is negative throughout. Same story as the cross-sectional logits.
- Lay Subsidy is null. Again no support for H₁.
- Population, again, is the dominant control.
- Wet 1536 is positive and significant in (1)–(2) and falls to insignificance with full geography in (3). Some of the weather effect is location.

The Cox results are the cleanest econometric defense of the paper's central claim. The current paper text says the land coefficient "becomes statistically insignificant when more controls are added" — that is no longer accurate for *large*-monastery land per arable km². The text should be updated.

---

## 5. Inverse probability weighting (`IPW.tex`)

The IPW logit and IPW Cox both point the same way as the unweighted models. With CBPS-style weights and full controls:

- `lbg_arak` (large monastery land per arable km²): β ≈ 0.47–0.50, p < 0.10 in both logit and Cox.
- `lti_arak` (tithe): negative, marginal in the logit.
- Small-monastery land: negative, insignificant.
- Population, Y_COORD (latitude), and area dominate.

The fact that the central result survives reweighting toward observations that "shouldn't" have so much monastic land, given their other features, is meaningful. It is the closest the paper gets to an observational identification claim, and it reproduces the headline.

---

## 6. Full monastic robustness (`full_monastic.tex`)

This adds Distance to Scotland to the kitchen-sink monastic spec, across all three outcomes:

- Large monastery land stays positive and is significant for primary (p < 0.10) and seats (p < 0.05).
- Net income is significantly negative for seats; small house dummy is significantly positive for primary and seats. Same partial-vs.-marginal collinearity story.
- Tithe is negative and marginal for muster.
- Distance to Scotland is large, negative, and significant in all three columns. This robustness check should probably make it into the main text — it kills the worry that the monastic results are just picking up "north-ness."

---

## 7. Sensitivity to the gentleman buffer (`sensitivity_gent_buffer.tex`)

Across buffer thresholds 10 km–30 km, the coefficient on rebel-gentleman proximity is positive but only marginally significant at the smallest buffers (10 km, 15 km), and the loyalist-gentleman buffer is essentially noise. The 20 km buffer used in the paper is in the middle of this range and is one of the *less* favorable choices — at 10 km the coefficient is largest. None of this changes the population/large-monastery-land story that dominates the same regression, so the choice of buffer is not load-bearing for the main result, but the paper could honestly report that the gentleman channel is fragile.

---

## 8. Old vs. new grievances (`old_vs_new_*.tex`)

These tables horse-race the "old grievances" set (taxation level + change, Wet 1535, Wet 1536, 5-yr drought index, population) against the "new grievances" set (the monastic suite).

- **Old-only column**: lay subsidy and lay-subsidy *change* are both insignificant for every outcome. Wet 1536 is significant only for seats. Population dominates muster and primary; for seats, weather does. The old-grievances story has very little to say about who rebelled once population is conditioned on.
- **New-only column**: large-monastery land is significant for primary and seats, and the large-house dummy is huge for muster and primary. Even *without* geographic controls, the monastic story has bite.
- **Combined column**: the LR test that the new (monastic) variables add explanatory power *given* the old (taxation/weather/population) variables yields p = 0.023 for muster, p = 0.062 for primary, p = 0.42 for seats. So the data reject "the old grievances are sufficient" at conventional levels for muster and at marginal levels for primary participation. Seats are explained equally well by either set.

This is a clean nested-model way to summarize the paper's core empirical claim: the monastic variables add real explanatory power on top of the standard tax-and-weather story for the outcomes where commoner mobilization is most visible.

---

## 9. Elite vs. commons frameworks (`elite_vs_commons_*.tex`, `_aipw_*.tex`, `_shapley_owen.tex`)

These newer tables introduce two elite-side regressors — a "Snubbed Family IDW Score" and a "Court Officer IDW Score" — and run them against the commons (monastic) variables. They are the most ambitious causal-inference-style tables in the folder, with cluster-robust SEs at the hundred level, AIPW/DoubleML versions, and a Shapley–Owen decomposition.

**Logit / Cox results (`elite_vs_commons_*`, `_cox`):**

- For *muster*: snubbed-family score is *negative* (−0.44 to −0.50, marginal), large-monastery land is positive (≈ 0.36, p < 0.10), tithe is negative (≈ −0.36, p < 0.05), alms is *positive* and significant (≈ 0.24–0.26, p < 0.05). The commons set fits better than the elite set (McFadden 0.357 vs. 0.346) and the combined model is best.
- For *primary*: large-monastery land is the only significant monastic variable (≈ 0.57, p < 0.10), alms is marginal. Snubbed family and court officer scores are both null. McFadden 0.41 (commons) > 0.38 (elite).
- For *seats*: court-officer IDW score is positive and significant at 1% (β ≈ 0.37–0.41), lay subsidy is positive and significant at 1%, large monastery land is positive and significant at 5%. Wet 1536 is significant. Here the elite framework does have real bite — court officers and tax wealth predict where gentleman seats sat.
- For the Cox primary-survival model: large-monastery land β ≈ 0.56 (p<0.10), elite scores null. Same story as the cross-sectional primary logit.

**AIPW / DoubleML versions:**

- Muster: large-monastery land β = 0.012 (p<0.05 in combined), tithe negative (p<0.05), snubbed family marginally negative.
- Primary: small-monastery land *negative* and significant (β = −0.008, p<0.01 in combined), large-monastery land positive and marginal. The negative small-land coefficient is the strongest in any specification — the doubly-robust estimator is most insistent that small-monastery land does *not* drive primary mobilization. This is uncomfortable for the simplest "the rebels were defending their own monastery" reading, and it is the place where the paper's interpretation needs the most care.
- Seats: court officer (≈ 0.05, p < 0.01) and large-monastery land (≈ 0.011–0.014, p < 0.05) both significant.
- AIPW Cox: large-monastery land β = 0.53, p < 0.05; alms β = 0.32, p < 0.05–0.10. The alms variable becomes more credible in the doubly-robust framework, which is interesting — the coefficient is small in raw logits but stable here.

**Shapley–Owen decomposition.** This is the cleanest summary of the relative importance of each block:

- For muster: controls explain 17.1% of pseudo-R², commons block 2.3%, elite block 0.4%. So among the *non-control* variables, commons explains roughly five times what elite does.
- For primary: controls 20.6%, commons 2.9%, elite 1.2% — commons still wins by ~2.5×.
- For seats: controls 8.9%, commons 0.5%, elite 1.0% — elite wins for gentleman seats, by roughly 2×.

Within the commons block, the contributions are:
- Muster: alms (0.0115) > large land (0.0098) > tithe (0.0020).
- Primary: large land (0.0195) > alms (0.0080) > tithe (0.0012).
- Seats: large land (0.0038) > tithe (0.0011) > alms (0.0002).

So *alms* is doing more work than the paper text currently gives it credit for, especially for the muster outcome. The paper should probably loosen its skeptical framing of the alms variable: even though only the perpetual-bequest portion is recorded, that variable is contributing meaningful incremental explanation in the most flexible specifications.

The other take-away from Shapley–Owen is that controls dwarf both grievance frameworks — population, slope, and per-capita tax soak up most of the explanation. That is honest and worth saying. The question this paper actually answers is: *given* that you've conditioned on where people lived and the local geography, which grievance framework adds explanation? And the answer for the commoner outcomes (muster, primary) is "the monastic-economy framework adds about 2–3× as much as the elite-grievance framework."

---

## 10. Putting the results in dialogue with the existing paper

Reading these tables against the current `paper.tex`:

1. **The paper's main claim survives.** Across logit, Cox, IPW, AIPW/DoubleML, the Shapley decomposition, and the nested LR test, large-monastery-land density is the most consistently positive, most consistently significant, and largest-Shapley monastic predictor of the most informative outcome (primary musters). Bush's reading is supported.

2. **The paper's framing of "land becomes insignificant with controls" is now too pessimistic.** The new per-arable-km² scaling and the large/small split fix exactly that problem for *large* land. The Discussion section should be updated to reflect that the headline coefficient is robust across the strictest specifications including AIPW and Cox-with-geographic-controls.

3. **Small monastery land is doing the *opposite* of what a naïve "self-interest" hypothesis predicts.** In nearly every spec, small land is null or negative, while large land is positive. The paper's existing answer to this — large monasteries felt the heat from the dissolution of small ones, large monasteries had more tenant land, the Furness story — is the right one. The paper could lean harder on the AIPW result here, since it is the most causal-leaning specification and is the most insistent that small land is *not* the driver.

4. **Tithe is negative across every specification.** This is more than a footnote. It is a positive piece of evidence for the paper's "land = warm tenant ties; tithe = cold extractive relationship" framing. The Lake Counties episode that the paper currently treats as an interesting anomaly is actually the *median* parish in the data: tithe payments to a monastery, conditional on everything else, slightly *reduce* the probability of revolt. That deserves explicit treatment.

5. **Lay Subsidy per Capita is null for muster and primary in every specification.** H₁ in the form Hoyle states it ("more taxed parishes rebelled more") is not supported. The paper currently says the evidence on tax is "weaker and inconsistent"; it would be defensible to be sharper. Tax level is positive and significant only for seats — i.e., wealthy areas had more gentlemen, which is mechanical, not motivational. The "old vs. new" LR test backs this up: adding the monastic variables to the tax-and-weather model raises explanatory power significantly for muster and marginally for primary.

6. **The elite-grievance framework has real bite — but only for gentleman seats.** Court-officer IDW and lay subsidy are robust predictors of gentleman seats and nothing else. The cleanest reading is that elite networks explain elite participation, and monastic-tenant geography explains commoner participation. The paper currently treats seats as a noisy proxy for commoner sympathy; the elite-vs.-commons tables suggest seats may be more usefully understood as a *separate* outcome with a different generating process. That is worth making explicit.

7. **Wet 1536 is the only consistently positive non-control predictor besides the monastic block.** Subsistence shock matters; it is not a substitute for the monastic story; it is additive.

8. **The DAG result is doing the most work for the simplest reader.** If the paper's most skeptical readers want to see a single regression they can interpret without learning Cox or DoubleML, `dag.tex` is the table to put first. The minimal model that the DAG says identifies the controlled direct effect gives a positive, significant land coefficient and a null tax coefficient. Everything else is robustness.

## 11. Where I would push back on the current narrative

- **Net income.** The single-variable spec says net income is positively associated with rebellion; the multivariate spec says the opposite. The paper currently uses net income as a clean measure and treats it as supportive. That is misleading once the large-house dummy and large-monastery land variables are also in the regression. Either drop net income from the headline tables or clearly explain the partial-vs.-marginal flip.
- **Alms.** Currently dismissed in the paper text as too noisy to draw conclusions from. The Shapley decomposition disagrees: alms is the largest single commons-side contributor for the muster outcome. The narrative could be loosened.
- **Small monastery land.** The paper would be stronger if it stated up front that *the monastic land that predicts rebellion is the land of the houses that were not yet directly threatened*. That is the most counterintuitive and historically interesting fact in the data, and the AIPW results are the cleanest place to anchor it.
- **The Cox result.** The paper text says Cox land "becomes statistically insignificant when more controls are added." That is no longer true for large monastery land per arable km². Update the sentence.
- **Distance to Scotland.** It is the most reliable non-monastic, non-population predictor in the entire `full_monastic.tex` and `*_monastic.tex` tables, and it directly confirms the "Why the North?" section's mechanism. It belongs in the main results section, not just the controls.

## 12. Bottom line

The folder, taken as a whole, supports the following compact statement:

> Conditional on population, geography, tax burden, and weather, parishes with denser large-monastery landholdings per unit of arable land were significantly and robustly more likely to host primary rebel musters in the autumn of 1536. The same is not true for tax burden, for tithe income, or for the land of the small houses that were the immediate targets of the Dissolution. The pattern is most consistent with a "tenant-fabric" mechanism: the parishes whose everyday economic life ran through the largest monastic estates rebelled to defend a system whose collapse they could now see clearly, even where the legal language of the 1536 statute had not yet reached them. Elite grievances explain where rebel gentlemen lived, but they do not explain where the commons rose.

That is a defensible reading of the tables now in `Output/Tables/`.
