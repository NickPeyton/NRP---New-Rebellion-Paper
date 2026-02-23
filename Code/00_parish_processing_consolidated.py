# %% Imports and Setup

import pandas as pd
import numpy as np
import geopandas as gp
import shapely as sh
import re
import os
from tqdm import tqdm

tqdm.pandas()
os.chdir('C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper')
# Define Paths relative to project root
RAW = 'Data/Raw/'
PROCESSED = 'Data/Processed/'

# Target 'north' coverage and relevant counties
coverage = 'north'
north_list = ['NORTHUMBERLAND', 'CUMBERLAND', 'DURHAM', 'WESTMORLAND',
              'YORKSHIRE, NORTH RIDING', 'LANCASHIRE', 'YORKSHIRE, WEST RIDING',
              'YORKSHIRE, EAST RIDING', 'LINCOLNSHIRE']

# %% 1. Valor Dataset Creation (Logic from 00_Valor_dataset_creator.py)
print('--- Step 1: Loading and Processing Valor data ---')
line_items_df = pd.read_csv(RAW + 'CSV/ValorLineItems.csv')
line_items_df = line_items_df[line_items_df['multi'] != 1]
natl_archives_df = pd.read_csv(RAW + 'CSV/NationalArchivesData.csv')

# Load spatial buffers for urban/suburban classification
suburban_buffer_gdf = gp.read_file(RAW + 'GIS/BNG Projections/SuburbanBufferBNG.shp')
urban_buffer_gdf = gp.read_file(RAW + 'GIS/BNG Projections/UrbanBufferBNG1.shp')

# Merge and clean data
df = pd.merge(line_items_df, natl_archives_df, on='name', validate='m:1', how='right')
df = df[df['note'].str.contains('italic') != True]
df = df.drop(columns=['note', 'page', 'foundedSource'])
df['total'] = pd.to_numeric(df['total'], errors='coerce')
df = df.dropna(subset=['total'])

# Categorization Regex
catVars = ['land', 'tithe', 'mill', 'fee', 'alms', 'infra', 'transfer', 'court', 'annuity', 'education', 'synProx', 'unknown']
synProx = re.compile('(syn)|(prox)')
glebe = re.compile('(glebe)')
ownLand = re.compile('(own land)')

def get_categories(row):
    """Categorize items based on transfer type and party."""
    if (row['transfer'] == 1) and (re.search(synProx, row['counterParty']) is not None):
        row['synProx'] = 1
        row['transfer'] = 0
    else:
        row['synProx'] = 0
    if (row['land'] == 1) and (re.search(glebe, str(row['counterParty'])) is not None):
        row['tithe'] = 1
        row['land'] = 0
    row['incomplete'] = 1 if len(df[df['name'] == row['name']]) <= 3 else 0
    row['ladyHouse'] = 1 if row['members'] in (['Nuns', 'Canonesses']) else 0
    row['ownLand'] = 1 if row['land'] == 1 and re.search(ownLand, str(row['counterParty']).lower()) is not None else 0
    return row

df = df.progress_apply(get_categories, axis=1)
df[catVars + ['sum']] = df[catVars + ['sum']].fillna(0).apply(pd.to_numeric, errors='coerce')
df[['lat', 'long', 'latitude', 'longitude']] = df[['lat', 'long', 'latitude', 'longitude']].apply(pd.to_numeric, errors='coerce')

# Assign primary type
for var in catVars:
    df.loc[df[var] == 1, 'type'] = var
df['type'] = df['type'].fillna('sum')

# %% Create Lines and Flows
print('Creating geographic lines for flows...')
lines_gdf = df.dropna(subset=['lat', 'long']).copy()
income_df = lines_gdf[lines_gdf['total'] > 0].copy()
expenditure_df = lines_gdf[lines_gdf['total'] < 0].copy()

# Create LineString geometries based on flow direction
income_df['lineString'] = income_df.apply(lambda i: sh.geometry.LineString([(i['long'], i['lat']), (i['longitude'], i['latitude'])]), axis=1)
income_df['direction'] = 'in'
expenditure_df['lineString'] = expenditure_df.apply(lambda e: sh.geometry.LineString([(e['longitude'], e['latitude']), (e['long'], e['lat'])]), axis=1)
expenditure_df['direction'] = 'out'

lines_gdf = pd.concat([income_df, expenditure_df])
lines_gdf['total'] = abs(lines_gdf['total'])
lines_gdf = gp.GeoDataFrame(lines_gdf, geometry='lineString', crs='epsg:4326').to_crs('epsg:27700')
lines_gdf = lines_gdf.rename(columns={'lineString': 'geometry'}).set_geometry('geometry')

# Add smallHouse flag early for line processing
# First calculate total income per house
house_totals = df[df['total'] > 0].groupby('name')['total'].sum().rename('totalInc')
lines_gdf = lines_gdf.merge(house_totals, on='name', how='left')
lines_gdf['smallHouse'] = (lines_gdf['totalInc'] <= (200 * 240)).astype(int)

def get_urban_status(row):
    """Classify flow source/destination as Urban, Suburban, or Rural."""
    pt0 = sh.geometry.Point(row['geometry'].coords[0])
    pt1 = sh.geometry.Point(row['geometry'].coords[1])
    sub_src = suburban_buffer_gdf['geometry'].contains(pt0).any()
    sub_dest = suburban_buffer_gdf['geometry'].contains(pt1).any()
    urb_src = urban_buffer_gdf['geometry'].contains(pt0).any()
    urb_dest = urban_buffer_gdf['geometry'].contains(pt1).any()
    
    row['urbSrc'], row['subSrc'] = (1, 0) if urb_src else (0, 1 if sub_src else 0)
    row['urbDest'], row['subDest'] = (1, 0) if urb_dest else (0, 1 if sub_dest else 0)
    row['rurSrc'] = 1 if row['urbSrc'] == 0 and row['subSrc'] == 0 else 0
    row['rurDest'] = 1 if row['urbDest'] == 0 and row['subDest'] == 0 else 0
    
    if row['direction'] == 'out':
        row['urbHouse'], row['subHouse'], row['rurHouse'] = row['urbSrc'], row['subSrc'], row['rurSrc']
    else:
        row['urbHouse'], row['subHouse'], row['rurHouse'] = row['urbDest'], row['subDest'], row['rurDest']
    return row

print('Classifying urban status for lines...')
lines_gdf = lines_gdf.progress_apply(get_urban_status, axis=1)
lines_gdf['diss1536'] = (lines_gdf['year'] == 1536).astype(int)
north_lines_gdf = lines_gdf[lines_gdf['county'].isin(north_list)].copy()

# %% 2. Parish Shapefile Processing (Logic from 01_...)
print('--- Step 2: Processing Parish Shapefile ---')
parish_df = gp.read_file(RAW + 'GIS/BNG Projections/AncientParishesBNG.shp')
parish_df = parish_df[parish_df['GAZ_CNTY'].isin(north_list)]

# Load auxiliary datasets
lincs_rebel_df = pd.read_csv(RAW + 'CSV/LincsRebels.csv')
muster_df = gp.read_file(RAW + 'GIS/BNG Projections/rebPoints.shp')
muster_df['muster'] = 1
seat_df = gp.read_file(RAW + 'GIS/BNG Projections/gentlemenInvolved.shp')
seat_df['seats'] = 1
terrain_df = gp.read_file(RAW + 'GIS/BNG Projections/TerrainZones.shp')
population_df = gp.read_file(RAW + 'GIS/BNG Projections/CombinedPop.shp')

sheail_payers_df = gp.read_file(RAW + 'GIS/BNG Projections/SheailParishPops1525ND.shp')
sheail_payers_df = sheail_payers_df.drop(columns=['FID']).dissolve(by='taxpayersH').reset_index()
sheail_payers_df = gp.clip(sheail_payers_df, parish_df)

sheail_shillings_df = gp.read_file(RAW + 'GIS/BNG Projections/SheailParishShillings1525ND.shp')
sheail_shillings_df = sheail_shillings_df.drop(columns=['FID', 'layer', 'path']).dissolve(by='shilH').reset_index()
sheail_shillings_df = sheail_shillings_df.rename(columns={'shilH':'shillH', 'shilL':'shillL'})
sheail_shillings_df = gp.clip(sheail_shillings_df, parish_df)

print('Processing points for parish aggregation...')
# Small House categorization
net_df = north_lines_gdf[north_lines_gdf['counterParty']=='Net income'][['total', 'geometry']].rename(columns={'total':'rhNetInc'})
net_df['smHouse'] = (net_df['rhNetInc'] <= (200*240)).astype(int)

# Separate flows into source/destination points
lines_proc = north_lines_gdf[north_lines_gdf['sum']==0].copy()
in_parts, out_parts = [], []

for d in ['in', 'out']:
    temp = lines_proc.copy()
    if d == 'in':
        temp['geometry'] = temp['geometry'].map(lambda x: sh.geometry.Point(list(x.coords)[1]))
        temp['inTot'], temp['outTot'] = temp['total'], 0
        for v in catVars:
            temp[v + 'InTot'] = temp[v] * temp['inTot']
            temp[v + 'OutTot'] = 0
        temp['landOwned'] = temp['dissLand'] = temp['ownLandVal'] = temp['otherLandVal'] = temp['smLand'] = 0
        in_parts.append(temp)
    else:
        temp['geometry'] = temp['geometry'].map(lambda x: sh.geometry.Point(list(x.coords)[0]))
        temp['outTot'], temp['inTot'] = temp['total'], 0
        for v in catVars:
            temp[v + 'OutTot'] = temp[v] * temp['outTot']
            temp[v + 'InTot'] = 0
        temp['landOwned'] = temp.loc[temp['direction'] == 'in', 'landOutTot'].fillna(0)
        temp['dissLand'] = temp['diss1536'] * temp['landOwned']
        temp['ownLandVal'] = temp['landOwned'] * temp['ownLand']
        temp['otherLandVal'] = temp['landOwned'] - temp['ownLandVal']
        temp['smLand'] = temp['landOwned'] * temp['smallHouse']
        out_parts.append(temp)

in_out_df = gp.GeoDataFrame(pd.concat(in_parts + out_parts), geometry='geometry', crs='epsg:27700')

# %% Merge HRV Data
print('Merging HRV historical records...')
HRVPath = RAW + 'CSV/HRV/'
hrv_dfs = [pd.read_csv(HRVPath + f) for f in os.listdir(HRVPath) if f.endswith('.csv')]
big_df = pd.concat(hrv_dfs, axis=1).loc[:, ~pd.concat(hrv_dfs, axis=1).columns.duplicated()]
big_df = big_df.dropna(subset=['pla'])
if 'STARTcounty' in big_df.columns:
    big_df = big_df.rename(columns={'STARTcounty': 'county'})
big_df['hrv_land'] = big_df['lMincome'].apply(np.exp)

parish_df = parish_df.rename(columns={'PLA': 'pla', 'GAZ_CNTY': 'county', 'AREA': 'area'})
parish_df['area'] /= 1_000_000
parish_df = pd.merge(parish_df, big_df, how='left', on=['pla', 'county', 'area'])

# Aggregation Logic
paSum = ['area', 'NrGentry', 'mills', 'copyhold_count_1850', 'copyhold_count', 'NrPatents', 'NrGentry_1400', 'mills_1400', 'copyhold_count_1516', 'hrv_land']
paMean = ['X_COORD', 'Y_COORD', 'perc_catholics_1800', 'ind_share_1831', 'agr_share_1831', 'ind_share', 'agr_share', 'LS_pc_change', 'pc_change1525_1086', 'pc_change1525_1066', 'pc_change1332_1086', 'pc_change1332_1066', 'pc_change1086_1066', 'lLStax_pc', 'LStax_pc_1332', 'agr_share_1370', 'ind_share_1370', 'WheatYield', 'mean_elevation', 'mean_slope', 'wheatsuitability', 'distancetoriver', 'distancetomarkettown', 'distancetoborder', 'distancetolondon', 'distancetocoal', 'latitude', 'longitude']
paFirst = ['PAR1851_ID', 'PAR1851_', 'county', 'PAR', 'hundred']
parish_df['par_county'] = parish_df['PAR'] + '_' + parish_df['county']
aggDict = {v: 'sum' for v in paSum} | {v: 'mean' for v in paMean} | {v: 'first' for v in paFirst}
parish_df = parish_df.dissolve(by='par_county', aggfunc=aggDict)

# %% Spatial Aggregation of Flows, Musters, and Seats
print('Spatially joining flows and rebellion data...')
io_vars = ['outTot', 'inTot', 'landOwned', 'dissLand', 'smLand', 'ownLandVal', 'otherLandVal'] + [v + 'InTot' for v in catVars] + [v + 'OutTot' for v in catVars]
# Using intersects to ensure points on boundaries are captured
joined_io = gp.sjoin(in_out_df, parish_df[['geometry']], how='right', predicate='intersects').groupby('par_county').agg({v: 'sum' for v in io_vars})
parish_df = parish_df.join(joined_io)

muster_df[['muster', 'day', 'primary']] = muster_df[['muster', 'day', 'primary']].apply(pd.to_numeric, errors='coerce')
joined_muster = gp.sjoin(muster_df, parish_df[['geometry']], how='right', predicate='intersects').groupby('par_county').agg({'muster':'sum', 'day':'mean', 'primary':'max'})
joined_muster['muster'] = (joined_muster['muster'] >= 1).astype(int)
parish_df = parish_df.join(joined_muster)
parish_df['muster'] = parish_df['muster'].fillna(0).astype(int)
parish_df['primary'] = parish_df['primary'].fillna(0).astype(int)

joined_seats = gp.sjoin(seat_df, parish_df[['geometry']], how='right', predicate='intersects').groupby('par_county').agg({'seats':'sum'}).fillna(0)
parish_df = parish_df.join(joined_seats)

joined_net = gp.sjoin(net_df, parish_df[['geometry']], how='right', predicate='intersects').groupby('par_county').agg({'rhNetInc':'sum', 'smHouse':'sum'}).fillna(0)
parish_df = parish_df.join(joined_net)
parish_df['netFlow'] = parish_df['inTot'] - parish_df['outTot']

# %% Final Data Adding and Cleaning
print('Adding final variables...')


# Assign Rebellion dummy for Lincolnshire parishes based on LincsRebels.csv
parish_df['reb'] = 0
parNums = dict(zip(parish_df['PAR1851_ID'], parish_df.index))
for pid in lincs_rebel_df['PAR1851_ID']:
    if pid in parNums:
        parish_df.at[parNums[pid], 'reb'] = 1

# Join Friaries
friardf = gp.read_file(RAW + 'GIS/BNG Projections/friarPoints.shp').to_crs('epsg:27700')
friardf['friary'] = 1
joined_friar = gp.sjoin(parish_df[['geometry']], friardf[['geometry', 'friary']], how='left', predicate='contains').groupby('par_county').agg({'friary':'sum'}).fillna(0)
parish_df = parish_df.join(joined_friar)

# Join Terrain Type
print('Joining terrain data...')
# terrain_df was loaded earlier
joined_terrain = gp.sjoin(parish_df[['geometry']], terrain_df, how='left', predicate='intersects')
# Take the first terrain type if a parish overlaps multiple zones
joined_terrain = joined_terrain.groupby('par_county').first()

# Map T_TYPE to terrainTyp as found in the data
if 'T_TYPE' in joined_terrain.columns:
    joined_terrain = joined_terrain.rename(columns={'T_TYPE': 'terrainTyp'})

if 'terrainTyp' in joined_terrain.columns:
    parish_df = parish_df.join(joined_terrain[['terrainTyp']])
else:
    print(f"Warning: 'terrainTyp' not found in terrain_df columns: {joined_terrain.columns.tolist()}")
    # Create dummy column to prevent crash if absolutely missing
    parish_df['terrainTyp'] = 'Unknown'

# Add population data
print('Joining population data...')
# population_df was loaded earlier
population_df = population_df.rename(columns={'pop': 'popC'})
joined_pop = gp.sjoin(parish_df[['geometry']], population_df[['geometry', 'popC']], how='left', predicate='intersects').groupby('par_county').agg({'popC':'sum'}).fillna(0)
parish_df = parish_df.join(joined_pop)

# Truncate variable names for Shapefile compatibility
truncDict = {'copyhold_count_1850':'copys_1850', 'copyhold_count':'copys', 'NrGentry_1400':'gent_1400', 'copyhold_count_1516':'copys_1516', 'perc_catholics_1800':'cath_1800', 'ind_share_1831':'ind_1831', 'agr_share_1831':'agr_1831', 'LS_pc_change':'LS_pc_ch', 'pc_change1525_1086':'pc15251086', 'pc_change1525_1066':'pc15251066', 'pc_change1332_1086':'pc13321086', 'pc_change1332_1066':'pc13321066', 'pc_change1086_1066':'pc10861066', 'LStax_pc_1332':'lspc1332', 'agr_share_1370':'agsh1370', 'ind_share_1370':'indsh1370', 'mean_elevation':'mean_elev', 'wheatsuitability':'wheatsuit', 'distancetoriver':'distriver', 'distancetomarkettown':'distmkt', 'distancetoborder':'distborder', 'distancetolondon':'distlond', 'distancetocoal':'distcoal'}
parish_df = parish_df.rename(columns=truncDict)

# %% 3. Final Variable Processing (Logic from 02_processing.py)
print('--- Step 3: Final Variable Creation and Cleaning ---')
pdf = parish_df.copy()
pdf['muster'] = pdf['muster'].replace({2: 1, 3: 1})
pdf['primary'] = pdf['primary'].replace({2: 1, 3: 1})

# Generate terrain and county dummies
terrainDummies = pd.get_dummies(pdf['terrainTyp'], drop_first=True).astype(int)
terrainDummies.columns = [c.lower() for c in terrainDummies.columns]
pdf = pd.concat([pdf, terrainDummies], axis=1)

countyDummies = pd.get_dummies(pdf['county'], drop_first=True).astype(int)
pdf = pd.concat([pdf, countyDummies], axis=1)

# Logarithmic transformations and derived metrics
pdf['nonDissLand'] = pdf['landOwned'] - pdf['dissLand']
pdf['ldissLand'] = np.log(pdf['dissLand'] + 1)
pdf['lnonDissLand'] = np.log(pdf['nonDissLand'] + 1)
pdf['bigLand'] = pdf['landOwned'] - pdf['smLand']
pdf['lsmLand'] = np.log(pdf['smLand'] + 1)
pdf['lbigLand'] = np.log(pdf['bigLand'] + 1)
pdf['titheIncCalc'] = pdf['titheOutTot'] * 10
pdf['ltitheIncCalc'] = np.log(pdf['titheIncCalc'] + 1)
pdf['tithed'] = (pdf['titheIncCalc'] > 0).astype(int)
pdf['landOwnedShare'] = (pdf['landOwned'] / pdf['titheIncCalc']).replace([np.nan, np.inf], [0, 1000])
pdf.loc[pdf['landOwnedShare'] > 1, 'tithed'] = 0
pdf['smOwnLand'] = 0
pdf.loc[pdf['smHouse'] == 1, 'smOwnLand'] = pdf['ownLandVal']
pdf['smOtherLand'] = pdf['smLand'] - pdf['smOwnLand']
pdf['lpopC'] = np.log(pdf['popC'] + 1)
pdf['lnetInc'] = np.log(pdf['rhNetInc'] + 1)
pdf['llandOwned'] = np.log(pdf['landOwned'] + 1)
pdf['day'] = pdf['day'].fillna(40).astype(int)
pdf['lotherLand'] = np.log(pdf['otherLandVal'] + 1)
pdf['lownLand'] = np.log(pdf['ownLandVal'] + 1)


for monast_var in ['land', 'alms', 'tithe']:
    pdf[f'l{monast_var}InTot'] = np.log(pdf[f'{monast_var}InTot'] + 1)
    pdf[f'l{monast_var}OutTot'] = np.log(pdf[f'{monast_var}OutTot'] + 1)

# Final Output
# pdf.drop(columns=['par_county'], inplace=True)  # Drop ID for shapefile compatibility
print(f'Saving final output to {PROCESSED}northParishFlows.shp')
pdf.to_file(PROCESSED + 'northParishFlows.shp')
print('DONE!')