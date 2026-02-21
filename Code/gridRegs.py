import pandas as pd
import numpy as np
import statsmodels.api as sm
import geopandas as gp
from statsmodels.iolib.summary2 import summary_col


#%% Grid shit

print('Loading 2km grid data')
g2df = gp.read_file(BNGPath + 'grid2.shp')
g2df['centroid'] = g2df.centroid
g2df['X_COORD'] = g2df['centroid'].x
g2df['Y_COORD'] = g2df['centroid'].y
print('Loading 5km grid data...')
g5df = gp.read_file(BNGPath + 'grid5.shp')
g5df['centroid'] = g5df.centroid
g5df['X_COORD'] = g5df['centroid'].x
g5df['Y_COORD'] = g5df['centroid'].y
print('Loading 10km grid data...')
g10df = gp.read_file(BNGPath + 'grid10.shp')
g10df['centroid'] = g10df.centroid
g10df['X_COORD'] = g10df['centroid'].x
g10df['Y_COORD'] = g10df['centroid'].y
print('Loading 20km grid data...')
g20df = gp.read_file(BNGPath + 'grid20.shp')
g20df['centroid'] = g20df.centroid
g20df['X_COORD'] = g20df['centroid'].x
g20df['Y_COORD'] = g20df['centroid'].y
#%%
outputList = []
tableNames = []
tableCaptions = []
tablePath = 'Tables\\'
terrainList = pdf.terrainTyp.unique().tolist()
terrainList.remove('Intermediate Lands')
parishVars = [['landOwnedS'],
              ['ltitheOutT', 'lalmsInTot'],
              ['lnetInc', 'friary'],
              ['lLStax_pc', 'LS_pc_ch', 'lpopC'],
              ['X_COORD', 'Y_COORD', 'area', 'mean_slope', 'total_cost'] + terrainList]
parList = []
for list in parishVars:
    parList = parList + list

gridVars = ['llandOwned',
            'lnetInc', 'friary',
            'lLStax_pc', 'lpopC',
            'landPct',
            'X_COORD', 'Y_COORD',
            'mean_slope'
            ]
gridList = [g2df, g5df, g10df, g20df]
kmList = ['2', '5', '10', '20']

musterResultList = []
primaryResultList = []
seatsResultList = []


for i, df in enumerate(gridList):
    df['lLStax_pc'] = np.log(df['LStax_pc'] + 1)
    rdf = df.copy()
    rdf = rdf.drop_duplicates(subset='geometry')
    x = rdf[gridVars]
    x = sm.add_constant(x)
    x.rename(columns=pnd, inplace=True)
    x.fillna(0, inplace=True)
    ym = rdf['muster']
    ym.fillna(0, inplace=True)
    #yp = rdf['primary']
    #yp.fillna(0, inplace=True)
    ys = rdf['seats']
    ys.fillna(0, inplace=True)
    musterResult = sm.Poisson(ym, x).fit_regularized(maxiter=10000, cov_type='HC3')
    #primaryResult = sm.Poisson(yp, x).fit_regularized(maxiter=10000, cov_type='HC3')
    seatsResult = sm.Poisson(ys, x).fit_regularized(maxiter=10000, cov_type='HC3')

    musterResultList.append(musterResult)
    #primaryResultList.append(primaryResult)
    seatsResultList.append(seatsResult)

musterTable = summary_col(musterResultList,
                          stars=True,
                          float_format='%0.3f',
                          model_names=kmList,
                          info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                     'R2': lambda x: "{:.3f}".format(float(x.prsquared))},
                          regressor_order=parList)
'''
primaryTable = summary_col(primaryResultList,
                          stars=True,
                         float_format='%0.3f',
                         model_names=kmList,
                         info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                    'R2': lambda x: "{:.3f}".format(float(x.prsquared))},
                         regressor_order=parList)
'''

seatsTable = summary_col(seatsResultList,
                          stars=True,
                         float_format='%0.3f',
                         model_names=kmList,
                         info_dict={'N': lambda x: "{0:d}".format(int(x.nobs)),
                                    'R2': lambda x: "{:.3f}".format(float(x.prsquared))},
                         regressor_order=parList)
