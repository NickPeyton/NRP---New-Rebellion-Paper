#%% Imports
import os
import statsmodels.api as sm
import pandas as pd
import numpy as np
import geopandas as gp
from statsmodels.iolib.summary2 import summary_col

#%% Functions

def prettyReg(df, yVar, xList, nameList, function):
    pnd = {'popP': 'Population',
           'popDG': 'Population',
           'popC': 'Population',
           'shillH': 'Shillings H',
           'shillL': 'Shillings L',
           'shillM': 'Shillings M',
           'taxpayersH': 'Taxpayers H',
           'taxpayersL': 'Taxpayers L',
           'taxpayersM': 'Taxpayers M',
           'llandOutTo': 'ln(Land Outflow)',
           'ltitheOutT': 'ln(Tithe Outflow)',
           'lalmsInTot': 'ln(Alms Inflow)',
           'house': 'House',
           'lnetInc': 'ln(House Net Income)',
           'lnetFlow': 'ln(Net Flow)',
           'landPct': 'Percent Land',
           'landOwned': 'Land Owned',
           'llandOwned': 'ln(Land Owned)',
           'X_COORD': 'X Coordinate',
           'Y_COORD': 'Y Coordinate',
           'AREA': 'Area',
           'lpopP': 'ln(Population P)',
           'lpopDG': 'ln(Population B)',
           'lpopC': 'ln(Town Population)',
           'shillPerCapH': 'Shillings per Taxpayer 1525 H',
           'shillPerCapL': 'Shillings per Taxpayer 1525 L',
           'shillPerCapM': 'Shillings/Taxpayer M',
           'shillPerCa': 'Shillings per Taxpayer',
           'lidwLandOu': 'ln(IDW Land)',
           'lidwTitheO': 'ln(IDW Tithe)',
           'lidwAlmsIn': 'ln(IDW Alms)',
           'lidwNetInc': 'ln(IDW Net Income)',
           'lidwNetFlo': 'ln(IDW Net Flow)',
           'lidwLandOw': 'ln(IDW Land Owned)',
           'lidwPop': 'ln(IDW Population)',
           'lidwPopP': 'ln(IDW Population P)',
           'lidwPopDG': 'ln(IDW Population B)',
           'lidwPopC': 'ln(IDW Town Population)',
           'idwHouse': 'IDW House',
           'mean_slope': 'Mean Slope',
           'constant': 'Constant',
           'muster': 'Muster',
           'primary': 'Primary Muster',
           'seats': 'Seats',
           'day': 'Day',
           'reb': 'Rebelion',
           'dissLand': 'Land of Dissolved Houses',
           'ldissLand': 'ln(Land of Dissolved Houses)',
           'nonDissLand': 'Land of Non-Dissolved Houses',
           'lnonDissLand': 'ln(Land of Non-Dissolved Houses)',
           'idwHouseLand': 'IDW House*Land',
           'lidwHouseLAnd': 'ln(IDW House*Land)',
           'friary': 'Friary',
           'LS_pc_ch': 'Lay Subsidy Per Capita % Change',
           'lsmLand': 'ln(Land of Small Houses)',
           'lLStax_pc': 'ln(Lay Subsidy per Capita)',
           'idwLandOwn': 'IDW Land Owned',
           'idwTitheOu': 'IDW Tithe',
           'idwAlmsInT': 'IDW Alms',
           'idwNetInc': 'IDW Monastic Net Income',
           'idwPopC': 'IDW Town Population',
           'area': 'Area',
           }
    rdf = df.copy()
    rdf.rename(columns=pnd, inplace=True)
    yVar = pnd[yVar]
    xVars = []
    outputList = []
    for xVarList in xList:
        xVars = xVars + xVarList
        xVars = [pnd[x] if x in pnd else x for x in xVars]
        print(xVars)
        x = rdf[xVars]
        x = sm.add_constant(x)
        x.fillna(0, inplace=True)
        y = rdf[yVar]
        y.fillna(0, inplace=True)
        if str.lower(function) == 'ols':
            results = sm.OLS(y, x).fit(disp=0)
        elif str.lower(function) == 'nb':
            results = sm.NegativeBinomialP(y, x).fit_regularized(maxiter=10000, cov_type='HC3', disp=0)
        elif str.lower(function) == 'poisson':
            results = sm.Poisson(y, x).fit_regularized(maxiter=10000, cov_type='HC3', disp=0)
        elif str.lower(function) == 'logit':
            results = sm.Logit(y, x).fit_regularized(maxiter=10000, cov_type='HC3', disp=0)
        else:
            print('Pick a real function, idiot')
        outputList.append(results)
    output = summary_col(outputList,
                         stars=True,
                         float_format='%0.3f',
                         model_names=nameList,
                         info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                    'R2': lambda x: "{:.3f}".format(float(x.prsquared))},
                         regressor_order=xVars)
    return output, outputList
