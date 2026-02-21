#%% TODO:
#%% Imports
import pandas as pd
import os
import geopandas as gp
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import shapely as sh
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
import sys
os.chdir('C:/PhD/DissolutionProgramming/REB---Rebellion-Paper/')
sys.path.insert(1, os.path.join(os.getcwd(), 'Code'))
from functions import prettyReg

#%% Loading Data
print('Loading Parish Data...')
pdf = gp.read_file('Data\\Processed\\northParishFlows.shp')
print('Loaded!')

#%% Regions
print('Creating regions...')
W = ['CUMBERLAND', 'LANCASHIRE', 'WESTMORLAND']
E = ['DURHAM', 'LINCOLNSHIRE', 'NORTHUMBERLAND', 'YORKSHIRE, EAST RIDING', 'YORKSHIRE, NORTH RIDING', 'YORKSHIRE, WEST RIDING']
ALL = W + E
POG = W + E
POG.remove('LINCOLNSHIRE')
LD = POG.copy()
LD.remove('NORTHUMBERLAND')
regions = [ALL, POG, LD]
regionNames = ['All Counties', 'PoG Only', 'Less-Damaged\nCounties in PoG']
regionNameDict = dict(zip(regionNames, regions))


pdf = pdf[pdf['county'].isin(POG)]
#%% Pretty Name Dictionary
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

#%% Regs

# List to add the outputs, table names, etc.
outputList = []
tableNames = []
tableCaptions = []
tablePath = 'Output\\Tables\\'
terrainList = pdf.terrainTyp.unique().tolist()
terrainList.remove('Intermediate Lands')
terrainList.remove('Other')
terrainList = [x.lower() for x in terrainList if x is not None]
parishVars = [['llandOwned'],
              ['ltitheOutT', 'lalmsInTot'],
              ['lnetInc', 'friary'],
              ['lLStax_pc', 'LS_pc_ch', 'lpopC'],
              ['X_COORD', 'Y_COORD', 'area', 'mean_slope'] + terrainList]
parList = []
for list in parishVars:
    parList = parList + list
parishModelNames = ['Land\nOwned', 'Tithe\nand\nAlms', 'Local\nMonasteries',
                    'Taxes\nand\nPopulation', 'Geography']

# Outputs for muster and seats of involved gentlemen
pogdf = pdf[pdf['county'].isin(POG)]
parishPrimary, parishPrimaryRaw= prettyReg(df=pogdf,
                                           yVar='primary',
                                           xList=parishVars,
                                           nameList=parishModelNames,
                                           function='logit')
outputList.append(parishPrimary)
tableNames.append('parishPrimary')
tableCaptions.append('Parish-Level Primary Muster Regression')

parishMuster, parishMusterRaw = prettyReg(pdf, 'muster', parishVars, parishModelNames, 'logit')
outputList.append(parishMuster)
tableNames.append('parishMuster')
tableCaptions.append('Parish-Level Muster Regression')

parishSeats, parishSeatsRaw = prettyReg(pdf, 'seats', parishVars, parishModelNames, 'poisson')
outputList.append(parishSeats)
tableNames.append('parishSeats')
tableCaptions.append('Parish-Level Seats Regression')

idwVars = [['idwLandOwn'],
           ['idwTitheOu', 'idwAlmsInT'],
           ['idwNetInc', 'friary'],
           ['lLStax_pc', 'LS_pc_ch', 'idwPopC'],
           ['X_COORD', 'Y_COORD', 'area', 'mean_slope', 'total_cost'] + terrainList]
idwModelNames = ['Land\nOwned', 'Tithe\nand\nAlms', 'Local\nMonasteries', 'Taxes\nand\nPopulation', 'Geography']

idwMuster, idwMusterRaw = prettyReg(pdf, 'primary', idwVars, idwModelNames, 'logit')
outputList.append(idwMuster)
tableNames.append('idwMuster')
tableCaptions.append('IDW Muster Regression')
try:
    idwSeats, idwSeatsRaw = prettyReg(pdf, 'seats', idwVars, idwModelNames, 'poisson')
    outputList.append(idwSeats)
    tableNames.append('idwSeats')
    tableCaptions.append('IDW Seats Regression')
except:
    print('\n\n\nIDW Seats Regression failed\n\n\n')
try:
    idwPrimary, idwPrimaryRaw = prettyReg(pdf, 'primary', idwVars, idwModelNames, 'poisson')
    outputList.append(idwPrimary)
    tableNames.append('idwPrimary')
    tableCaptions.append('IDW Primary Muster Regression')
except:
    print('\n\n\nIDW Primary Muster Regression failed\n\n\n')


#%% Outputting all tables to LaTeX
for i, table in enumerate(outputList):
    # Save each as a different file with different name
    tableName = tableNames[i]
    caption = tableCaptions[i]
    label = 'tab:' + tableName
    print(tableName + ' saved')
    with open(tablePath + tableName + '.tex', 'w') as tf:
        tf.write(table.as_latex(label=label).replace('caption{}', 'caption{' + caption + '}'))
        tf.close()
for i, table in enumerate(outputList):
    print('\n\n\n' + tableNames[i])
    print(table)