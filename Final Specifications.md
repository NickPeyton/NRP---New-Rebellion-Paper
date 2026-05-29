## Variables
- Three rebellion outcomes:
	- Muster
	- Primary (Muster)
	- (Rebel Gentlemen) Seats
- Three main explanatory variables (logged):
	- Monastic land
	- Land of large monasteries
	- Off-site monastic land
Only use one of these explanatory variables at a time, using the complement of the last two (small monastery land and on-site land respectively)
- Include other standard monastic variables (logged) (tithes, alms, large and small house dummies)
- Three versions of monastic variables: log value, log value per km^2, log value per arable km^2. This will produce nine versions of each type of analysis.
- Controls: 
	- 20km binary buffers for fsnub and court officer gentlemen
	- Wet 1535 and 1536 variables
	- Lay Subsidy per capita
	- ln Population
	- Distance to Scotland
	- Mean Slope
	- Parish Area
- Standardize all variables obvi
## Standard Errors
- Where possible, use Conley Standard Errors with a **100km** distance cutoff. This distance was selected based on Moran's I tests showing that spatial autocorrelation in both outcome and explanatory variables decays significantly by this point.
## Outputs
- Tables should report muster, primary, seats coefficients in that order
- Create an accompanying regression plot for each table