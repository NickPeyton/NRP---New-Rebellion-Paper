# TO DO:


#### Examiners and CalTech:
---
##### Paper:
- [ ] Strengthen the conclusion
- [ ] Make every bit of the paper flow toward the conclusion
- [ ] Clearer Hoyle v Bush distinction
- [ ] Could threatened monastic land point to religious threat?
- [ ] What were the monasteries doing?
	- [ ] More critical orders?
	- [ ] Friaries?
- [ ] Emphasize entry fines as disruptive, monasteries as slow-moving and institutional
- [ ] Read Roper's "Summer of Fire and Blood"
- [ ] Check primary sources for rebel interrogations
- [ ] Examine treatment of tenants on Royal v monastic estates
- [ ] Distinguish "commons-driven" vs "elite-driven" views of rebellion
##### Analysis:
###### Claude:
- [x] Replace the monastic net income variable with two dummies, one indicating the presence of a large house (200 pounds or more in net income) and one indicating the presence of a small house (sub-200 pounds net income), should already have been created in the data creation scripts.
- [x] Modify monastic variables with "denominators" and use them for the analysis scripts
	- [x] "Per capita" version with VALUE / popC
	- [x] "Per square kilometer" version with VALUE / area
- [x] Replace the "Percy" variable with the "disgruntled gentlemen" variable for analysis
- [x] Add a "distance from Scottish border"
- [x] Alter main regressions to examine one monastic variable at a time while keeping geographic and other controls
- [x] Include "full model" with all monastic variables as robustness check
- [x] Create a new script (with new output table/graph names of course) with each monastic variable interacted with a measure of the inverse of its distance
###### Nick:
- [ ] 
- [ ] "Old" vs "New" Grievances analysis: which has higher R$^2$?
##### Data:
- [ ] Add variable for Royal-influenced abbatial elections?
- [ ] Full universe of top-level aristocrats and their seats/lands