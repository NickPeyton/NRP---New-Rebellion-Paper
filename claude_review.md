# Claude Review — Dissolution → Rebellion Argument

Scope: updated tables in `Output/Tables/` only. Not a rewrite of `paper.tex`/`presentation.tex`, which are stale.
Ordering: descending threat to the core claim that economic dislocation from the Dissolution drove the rebellion.

---

## TIER 1 — Existential threats to the core claim

### 1. The "monastic tenants" mechanism is no longer supported by a table.
The old `paper.tex` rested its argument on the **on-site vs off-site land split** (`primary_split`), with off-site land (where tenants lived) predicting rebellion strongly and on-site (demesne/wage-labor) land predicting *less* rebellion. That was the key identification move distinguishing the "tenants" mechanism from an undifferentiated "monasteries in general" correlation. I do not see a primary_split or equivalent in the updated `Output/Tables/` folder. Without it, the final paragraph of §6 — the claim that "monastic tenants themselves are a key predictor of revolt" — is unsupported by the current results. This is the single biggest gap. Either restore the decomposition or retreat the mechanism claim.

### 2. Only *one* monastic economic variable is robustly signed in the direction the theory predicts, and only at 10%.
Across `primary_all` (col 4), `muster_all`, `full_monastic`, and the AIPW tables, **large-monastery land/arable km²** is the only Dissolution-economic variable with consistent positive sign at conventional significance. The rest of the monastic-economic bundle behaves badly for a referee reading the theory:

- **Net income** is *negative* and significant in `seat_all` (–0.669***), `full_monastic` for seats (–0.727***), and `primary_all` col 2 (–0.728**). If the mechanism is loss of monastic economic presence, bigger monasteries should have *larger* pro-rebellion effects, not smaller. Paper needs to explain why net income flips sign once land is controlled for — currently it does not.
- **Tithe** is *negative* and significant in `muster_all` col 3/4 (–0.336*, –0.338*), `full_monastic` muster (–0.358*), Commons spec (–0.355*), and AIPW muster (–0.0107**). The paper's own data description nods at this ("extractive relationship," Harrison's Lake Counties rebels rising *against* tithes), but the story for rebellion needs to be: "land-tenant bonds pro-rebellion, tithe-extraction bonds anti-rebellion, net effect positive." That is a subtler claim than the current text makes, and puts a lot of weight on an unmeasured distinction between "bond" and "extraction."
- **Alms** is significant in some specs (`muster_monastic` col 4: 0.215*, `primary_monastic` col 4: 0.283**, Commons 0.238**/0.267*) but you already flag in §5.4 that the alms variable in the *Valor* is bequest-only and "disallowed" totals can run 7× the allowed figures. You cannot both disclaim the alms variable as unreliable and lean on it for H₂. Pick one.

So H₂ as written — "Monastic landholding, alms, **or** net income will be positively and significantly associated with rebellion" — survives only through a disjunction that hides which specific channel is live. A referee will reduce this to: "large-monastery-land density in the parish, at 10%."

### 3. "Large House Dummy" is doing a lot of work and is not plausibly exogenous.
Across `primary_monastic` col 7 (1.859**), `muster_monastic` col 7 (1.919***), `primary_all` col 3 (2.687*), `full_monastic` primary (2.778), the large-house dummy is the single biggest monastic coefficient in the paper (log-odds of 1.5–3.4). But whether a large monastery was sited in a given parish is endogenous to medieval wealth, market access, political patronage, and demographic density — exactly the things that might independently produce rebellion. Geographic controls and `ln(population)` will not absorb this; population here is town-based Wallis/Udale/Buringh data, which undercounts rural settlement clustering that monasteries caused or tracked.

The referee-facing question is: **does the large-house dummy represent the Dissolution mechanism, or is it a fixed effect for "medieval urban/market center"?** A reasonable check is to restrict to parishes *without* a resident house and ask whether monastic-land-density still predicts rebellion. If it does, the mechanism survives; if the house dummy is doing all the work, the story is "revolt happened where the big abbeys were" — which is consistent with Bush but also consistent with a dozen competing stories.

### 4. The elite pathway is *mostly* closed off, but not completely, and the failure is in the expected place.
The Shapley–Owen decomposition is actually strong evidence for your case: Commons monastic variables contribute 0.0228–0.0273 of R² for muster/primary vs Elite variables at 0.0012–0.0055 — a 4–22× ratio. That is the headline to foreground.

However, two cracks a referee will widen:

- **Court Officer IDW is positive and significant for seats** across main regression (0.296***, 0.268**), AIPW (0.0429**), and Shapley–Owen (0.0079, larger than Commons 0.0049 for seats). The seats outcome *is* partly elite-network-driven, and you should say so explicitly rather than lumping all three outcomes together. The cleanest framing is: "commons-led musters and primary musters are driven by Dissolution-economic variables; gentry seats are jointly driven by taxation, court patronage, and monastic-land proximity." That is a more defensible position than "one story fits all three."
- **"Snubbed Family" IDW is negative and significant for muster** (–0.433*, –0.490**) and negative in AIPW (–0.0173***, –0.0110*). This is unexplained. It might be that "snubbed" families cluster far from the rebellion area, or that their estates were too loyal to participate — but leaving a significant elite variable with the "wrong" sign unexplained in a referee's mind opens the door to "your elite operationalization is underspecified." Two suggestions: (a) report *which* families are in the snubbed set and whether their geographic distribution is what drives the sign; (b) explicitly state that "snubbed" captures only one elite grievance channel, not the Statute of Uses, inheritance law, base-born ministers, or Percy-style clientage.

### 5. The Percy variable has disappeared.
The old paper, the presentation, and the historical literature all treat the Percy network as *the* elite-mobilization channel in the North. The updated `elite_vs_commons_*` tables replace Percy with Snubbed-Family and Court-Officer IDWs, neither of which is a Percy-network measure. This is a substantive change in the elite model without explanation. Either re-add Percy to the horse race or justify the substitution. A referee who has read Bush or Hoyle will notice immediately.

---

## TIER 2 — Major gaps a referee will flag

### 6. Old vs New Grievances LR test is weaker than the paper wants.
`old_vs_new_primary` reports LR p = 0.062 and `old_vs_new_seats` reports p = 0.42. Monastic-Dissolution variables add significant explanatory power over the "old" tax/weather/population story only for musters (p = 0.023) and marginally for primary musters (p = 0.062). For gentry seats, the null is not rejected at any conventional level — consistent with #4 above. You should (a) restrict the headline claim to musters and primary musters, and (b) explicitly note that gentry seats are a *different* model.

### 7. Missing sample robustness for the 364 dropped parishes.
The full-spec regressions drop from n = 1,755 to n = 1,391 when the lay subsidy variable is added — ~20% of parishes removed, concentrated in Cumberland, Westmorland, Northumberland, and Durham. These are exactly the counties of the Lake Counties risings and the *most* rebellious regions. §5.5 mentions this but does not run the regression without the lay-subsidy variable on the full sample to quantify how much the estimates shift. The current DAG table (n = 1,558) isn't a substitute because it drops the other monastic variables. Add a full-sample robustness table with and without the lay-subsidy restriction. Right now a referee can plausibly argue that the results survive *only* because the most rebellious areas were dropped.

### 8. Distance-to-Scotland is a major unexamined confounder.
In every `*_monastic` table, Distance to Scotland is negative and highly significant (e.g. seats: –0.899***, –0.982***, –0.926***). Parishes closer to Scotland rebelled more. This is larger than any monastic coefficient. Two questions the paper does not answer: (a) Are large Northern monasteries concentrated near the border? If so, "large monastery land" partially proxies for borderland militarization. (b) Distance to Scotland is a good instrument for *militia preparedness*, which is also the mechanism §3 identifies as making Northern rebellion possible. The paper mentions this in §3 but never ties it back to the regression interpretation. A referee will ask whether the monastic effect holds *within* border-distance strata.

### 9. No fixed effects despite strong regional confounders.
The paper clusters SEs at "hundred" (139 clusters for 1,391 obs — borderline but probably OK) but does not include county or hundred fixed effects. Given that §3 ("Why the North?") identifies county-level drivers (border, Northern poverty, distinct monastic landscape), not including county FEs leaves every monastic coefficient vulnerable to "you're picking up a county-level trait that happens to correlate with monastic density." Run a specification with county FEs; if large-house land still signs positive, the mechanism is within-county and much more defensible.

### 10. Mechanism interpretation of "large monastery land / arable km²" is not pinned.
The arable-km² denominator is a reasonable normalization choice, but the scaled variable now measures *intensity* of monastic land per unit of productive land, not total monastic presence. The paper needs one paragraph explaining why that normalization is the theoretically correct one. Candidate stories: it proxies for share of local agrarian economy tied to monasteries, which is what the "tenant bonds" mechanism implies. But it could also mean: marginal monastic estates in areas with little non-monastic arable — which gets you into "subsistence-edge, monastery-dependent" territory, a different mechanism from Bush's.

### 11. Monastic-opposition variables (`mo_*`) do not appear in any reported table.
Per my prior memory note, you built `mo_ci1`, `mo_ci05`, `mo_anyop` (and IDW variants) measuring proximity to religious houses that actively opposed the Reformation or saw pre-1536 Crown interference. This is the most direct operationalization of the "monks as agitators" channel mentioned in §2 (Furness, Holm Cultram). If these are in progress, flag that; if they were tried and dropped, say why. Referees will ask for this precisely because the paper tells the Furness story.

---

## TIER 3 — Defensive hardening a referee will ask for

### 12. H₁ taxation is not a clean null.
Lay Subsidy per Capita is positive and highly significant for seats in `seat_monastic` (all columns ~0.38***) and Commons/Combined specs (0.333***, 0.383***). The paper's framing ("taxation does not predict rebellion") is only true for the muster and primary outcomes. For gentry participation, the Hoyle taxation story is vindicated. A clean version: "H₁ predicts gentry participation but not commons participation; H₂ predicts commons participation." This is a stronger paper than one pretending Hoyle was wrong about everything.

### 13. Wet-1536 is carrying a lot of the story and is not from the Dissolution.
Wet 1536 is significant in most specs (e.g. primary_all col 3: 0.752**; seat_all col 4: 0.742***; survival col 1: 0.923***). This is a competing exogenous driver (subsistence crisis) that sits *outside* the Dissolution story. The paper should acknowledge: the rebellion's timing is overdetermined — it took a Dissolution *and* a bad harvest. Good news: if Wet 1536 and monastic variables both enter significantly in the same regression, they are not substitutes. But a referee will want a discussion of why the Dissolution + weather jointly mattered, rather than one eclipsing the other.

### 14. AIPW effect sizes are tiny.
DoubleML ATE estimates for Large Monastery Land / Arable km² are 0.0077 (primary), 0.0118 (muster), 0.0122 (seats). On the probability scale, these are small marginal effects — interpretable but not dramatic. The paper should give a substantive effect-size translation ("a 1-SD increase in large-monastery-land density raises the probability of a primary muster by ~0.8 percentage points, roughly doubling the baseline rate of primary-muster presence in the median parish") rather than relying on significance stars.

### 15. Inference at 139 clusters with rare outcomes.
Primary musters are rare (the log-likelihoods of ~85 on 1,391 obs suggest a very low base rate — likely <5% of parishes). Combined with 139 hundred-clusters, cluster-robust SEs may be under-covering. Consider wild cluster bootstrap p-values for the headline coefficients on large-monastery land.

### 16. The "Friary" variable is positive & significant only in `muster_all` col 2 (0.917**) and never thereafter.
Not a threat, but referees will ask what "friary" is doing if it is in the regression. If it is a nuisance control, say so; if it is theorized, it needs a paragraph in §5.

### 17. "Small House Dummy" is consistently positive & significant in seat and primary specs.
This is where the Dissolution statute bit hardest (houses under £200 dissolved in 1536). But you say in §6 that the land of *large* houses predicts rebellion more strongly than the land of small ones, and then argue that large houses saw the writing on the wall. The small-house *dummy* — which is what was actually dissolved — also strongly predicts rebellion. This is *more* consistent with a "direct dissolution → rebellion" story than the large-house-land result. The current paper doesn't quite braid these two findings together; the presentation foregrounds large houses, but the small-house dummy is a clearer match to the proximate cause. Reconcile.

---

## TIER 4 — Housekeeping

- `old_vs_new_*` stargazer formulas are printed as raw R code in the column headers (`paste(dep, "~", ...)`). Fix before circulating.
- Several tables still use the `lbg_arak`, `lti_arak`, etc. internal names (`IPW.tex`) rather than the pretty labels used elsewhere. Unify via `pretty_dict.json`.
- `elite_vs_commons_aipw_*` tables use Stata-style footnotes with `$^\dagger$` but other tables use stargazer defaults. Harmonize.
- `primary_sm.tex` and `primary_split.tex` are referenced in the current `paper.tex` but absent from `Output/Tables/`. Either restore them or update the paper.

---

## Headline recommendation

The strongest defensible version of the paper, given the current tables, is:

> "Among commons-led rebellion outcomes (musters and primary musters), parish-level density of land owned by large monasteries robustly predicts participation, controlling for population, wealth, weather, geography, and proximity to the border. Elite grievance channels predict gentry participation but not commons participation. Taxation predicts gentry participation but not commons participation. The Dissolution's economic footprint is therefore one — not the only — driver of the rebellion, and its channel is specifically tenant-adjacent monastic property rather than monastic institutions in the aggregate."

To get there from here: (1) restore the on-site / off-site decomposition, (2) add county FEs, (3) separate the seats story from the musters story explicitly, (4) bring back Percy or justify its removal, (5) run the `mo_*` variables in the main spec.
