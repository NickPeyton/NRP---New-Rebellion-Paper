
# Script to extend the rebellion variables
# %% Loading
print("Loading packages and data...")
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint
from shapely.ops import unary_union
import os

os.chdir(r"C:/PhD/DissolutionProgramming/REB---Rebellion-Paper")
# Load the shapefiles
reb_points = gpd.read_file('./Data/Raw/GIS/BNG Projections/rebPoints.shp')
gentlemen = gpd.read_file('./Data/Raw/GIS/BNG Projections/gentlemenInvolved.shp')
parish_flows = gpd.read_file('./Data/Processed/northParishFlows.shp')

print(f"Loaded {len(reb_points)} rebel muster points")
print(f"Loaded {len(gentlemen)} gentleman points")
print(f"Loaded {len(parish_flows)} parishes")
print(f"\nUnique hosts in gentlemenInvolved: {sorted(gentlemen['host'].unique())}")

#%% Identifying parishes NEAR rebel musters
# Create a buffer of 10km (10000 meters in BNG projection) around all rebel muster points
reb_buffer = reb_points.geometry.buffer(10000).unary_union

# Check if each parish centroid is within 10km of any rebel muster point
parish_flows['near_reb_muster'] = parish_flows.geometry.centroid.within(reb_buffer).astype(int)

print(f"Parishes within 10km of rebel muster: {parish_flows['near_reb_muster'].sum()}")
print(f"Parishes not within 10km: {len(parish_flows) - parish_flows['near_reb_muster'].sum()}")

#%% Group gentleman points by host and create convex hull polygons
host_polygons = {}

for host_id in gentlemen['host'].unique():
    # Get all points for this host
    host_points = gentlemen[gentlemen['host'] == host_id]
    
    # If there's only one point, use a buffer to create a polygon
    if len(host_points) == 1:
        host_polygons[host_id] = host_points.geometry.iloc[0].buffer(100)  # Small buffer
    elif len(host_points) == 2:
        # For two points, create a line and buffer it
        multipoint = MultiPoint([p for p in host_points.geometry])
        host_polygons[host_id] = multipoint.convex_hull.buffer(100)
    else:
        # For 3+ points, create a convex hull
        multipoint = MultiPoint([p for p in host_points.geometry])
        host_polygons[host_id] = multipoint.convex_hull

print(f"Created polygons for {len(host_polygons)} hosts")
for host_id, polygon in host_polygons.items():
    num_points = len(gentlemen[gentlemen['host'] == host_id])
    print(f"  Host {host_id}: {num_points} points")

#%% Check which parishes intersect with any host polygon
# Initialize the host classification column
parish_flows['host'] = np.nan

# For each parish, determine which host(s) it's within 50km of
for idx, parish in parish_flows.iterrows():
    parish_centroid = parish.geometry.centroid
    
    # Find all hosts within 50km
    distances = {}
    for host_id, host_polygon in host_polygons.items():
        distance = parish_centroid.distance(host_polygon)
        if distance <= 50000:  # 50km in meters
            distances[host_id] = distance
    
    # Assign to closest host if any are within 50km
    if distances:
        closest_host = min(distances, key=distances.get)
        parish_flows.at[idx, 'host'] = closest_host

print(f"Parishes assigned to a host: {parish_flows['host'].notna().sum()}")
print(f"Parishes not assigned to any host: {parish_flows['host'].isna().sum()}")
print(f"\nParishes by host:")
for host_id in sorted(parish_flows['host'].dropna().unique()):
    count = (parish_flows['host'] == host_id).sum()
    print(f"  Host {int(host_id)}: {count} parishes")


#%% Target surnames of interest
target_surnames = ['Darcy', 'Percy', 'Latimer', 'Neville', 'Hussey']

# For each surname, find the set of host numbers that contain at least one
# gentleman with that surname. Match on surname exactly or as the part
# before a comma (e.g. 'Percy, Thomas' has surname 'Percy').
surname_to_hosts = {}
for surname in target_surnames:
    mask = gentlemen['gentleman'].fillna('').apply(
        lambda g, s=surname: g == s or g.startswith(s + ',')
    )
    hosts = set(gentlemen.loc[mask, 'host'].tolist())
    surname_to_hosts[surname] = hosts
    print(f"{surname}: found in host(s) {sorted(hosts)}")

#%% Disgruntled gentlemen proximity (within 20km of their points)
disg_mask = gentlemen['gentleman'].fillna('').apply(
    lambda g: any(g == s or g.startswith(s + ',') for s in target_surnames)
)
disg_points = gentlemen.loc[disg_mask]

if len(disg_points) == 0:
    parish_flows['disg_gnt'] = 0
    print("No disgruntled gentlemen points found; disg_gnt set to 0 for all parishes")
else:
    disg_buffer = disg_points.geometry.buffer(20000).unary_union
    parish_flows['disg_gnt'] = parish_flows.geometry.centroid.within(disg_buffer).astype(int)
    print(f"Parishes within 20km of disgruntled gentlemen points: {parish_flows['disg_gnt'].sum()}")

disg_gnt_count = int(parish_flows['disg_gnt'].sum())
print(f"Parishes with disg_gnt = 1: {disg_gnt_count}")

# Create per-surname proximity dummies (within 20km of each surname's points)
surname_dummy_cols = {
    'Darcy': 'dg_darcy',
    'Percy': 'dg_percy',
    'Latimer': 'dg_latim',
    'Neville': 'dg_nevil',
    'Hussey': 'dg_husse',
}

for surname, col in surname_dummy_cols.items():
    surname_mask = gentlemen['gentleman'].fillna('').apply(
        lambda g, s=surname: g == s or g.startswith(s + ',')
    )
    surname_points = gentlemen.loc[surname_mask]
    if len(surname_points) == 0:
        parish_flows[col] = 0
        print(f"{col} ({surname}): 0 parishes = 1 (no points)")
    else:
        surname_buffer = surname_points.geometry.buffer(20000).unary_union
        parish_flows[col] = parish_flows.geometry.centroid.within(surname_buffer).astype(int)
        print(f"{col} ({surname}): {parish_flows[col].sum()} parishes = 1")

# Create new dummy columns for each target surname based on host assignment
col_names = {
    'Darcy':   'darcy_host',
    'Percy':   'percy_host',
    'Latimer': 'latime_hst',
    'Neville': 'nevill_hst',
    'Hussey':  'hussey_hst',
}

# For each surname, create a dummy: 1 if the parish's host contains that
# surname, 0 otherwise. Parishes with no host assignment (NaN) get 0.
for surname, col in col_names.items():
    host_set = surname_to_hosts[surname]
    parish_flows[col] = parish_flows['host'].apply(
        lambda h: 1 if (pd.notna(h) and h in host_set) else 0
    )
    print(f"{col} ({surname}): {parish_flows[col].sum()} parishes = 1")

dummy_cols = list(col_names.values())
parish_flows['disg_gnt_h'] = (parish_flows[dummy_cols].max(axis=1)).astype(int)

surname_count_summary = {col: int(parish_flows[col].sum()) for col in surname_dummy_cols.values()}
print("Disgruntled surname dummy counts:")
for col, count in surname_count_summary.items():
    print(f"  {col}: {count}")

#%% Save the updated shapefile
output_path = './Data/Processed/northParishFlows.shp'
parish_flows.to_file(output_path)

print(f"Updated shapefile saved to {output_path}")
print(f"\nNew columns added:")
print(f"  - near_reb_muster: Binary indicator (0/1) for parishes within 10km of rebel muster")
print(f"    (Note: Saved as 'near_reb_m' in shapefile due to 10-char column name limit)")
print(f"  - disg_gnt: Binary indicator (0/1) for parishes within 20km of disgruntled gentlemen points")
print(f"  - host: Host classification (1-9 or NaN) for parishes within 50km of host polygons")
print(f"  - {dummy_cols}: Dummy variables for each target surname")
print(f"  - disg_gnt_h: 1 if any of the dummy variables is 1, 0 otherwise")