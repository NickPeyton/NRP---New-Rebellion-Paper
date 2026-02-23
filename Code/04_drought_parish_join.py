import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import os
os.chdir(r"C:/PhD/DissolutionProgramming/NRP---New-Rebellion-Paper")
# 1. Load the processed drought CSV
drought_csv = 'Data/Processed/drought_intensity_bng.csv'
df = pd.read_csv(drought_csv)

# 2. Create drought grid cells (polygons)
# OWDA grid is 0.5 x 0.5 degrees. Coordinates are centroids.
geometry = []
for _, row in df.iterrows():
    lon, lat = row['Longitude'], row['Latitude']
    geometry.append(box(lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25))

drought_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Reproject to BNG
drought_gdf = drought_gdf.to_crs("EPSG:27700")

# Rename columns early for consistency and shapefile limits
drought_gdf = drought_gdf.rename(columns={
    'pdsi_avg_1yr': 'dr_1yr',
    'pdsi_avg_2yr': 'dr_2yr',
    'pdsi_avg_3yr': 'dr_3yr',
    'pdsi_avg_5yr': 'dr_5yr',
    'pdsi_avg_10yr': 'dr_10yr',
    'pdsi_ext_1yr': 'exw_1yr',
    'pdsi_ext_2yr': 'exw_2yr',
    'pdsi_ext_3yr': 'exw_3yr',
    'pdsi_ext_5yr': 'exw_5yr',
    'pdsi_ext_10yr': 'exw_10yr',
    'pdsi_1535': 'wet_1535',
    'pdsi_1536': 'wet_1536',
    'pdsi_1535x1536': 'dwx_1536'
})

# Save drought cells shapefile
drought_shp_path = 'Data/Processed/drought_cells.shp'
drought_gdf.to_file(drought_shp_path)
print(f"Drought cells shapefile saved to {drought_shp_path}")

# 3. Load northParishFlows
parish_path = 'Data/Processed/northParishFlows.shp'
parishes = gpd.read_file(parish_path)

# Ensure parishes are in BNG
if parishes.crs is None or parishes.crs.to_epsg() != 27700:
    parishes = parishes.to_crs("EPSG:27700")

# 4. Sample drought cells using parish centroids
parish_centroids = parishes.copy()
parish_centroids.geometry = parishes.centroid

# Spatial join: centroids with drought polygons
# Keep only necessary drought columns
drought_data_cols = [
    'dr_1yr', 'dr_2yr', 'dr_3yr', 'dr_5yr', 'dr_10yr',
    'exw_1yr', 'exw_2yr', 'exw_3yr', 'exw_5yr', 'exw_10yr',
    'wet_1535', 'wet_1536', 'dwx_1536',
    'geometry'
]
joined = gpd.sjoin(parish_centroids, drought_gdf[drought_data_cols], how='left', predicate='within')

# 5. Attach the sampled variables back to the original parishes
# We use the index from parishes to ensure alignment
parishes['drought_1'] = joined['dr_1yr']
parishes['drought_2'] = joined['dr_2yr']
parishes['drought_3'] = joined['dr_3yr']
parishes['drought_5'] = joined['dr_5yr']
parishes['drought_10'] = joined['dr_10yr']
parishes['exw_1'] = joined['exw_1yr']
parishes['exw_2'] = joined['exw_2yr']
parishes['exw_3'] = joined['exw_3yr']
parishes['exw_5'] = joined['exw_5yr']
parishes['exw_10'] = joined['exw_10yr']
parishes['wet_1535'] = joined['wet_1535']
parishes['wet_1536'] = joined['wet_1536']
parishes['dwx_1536'] = joined['dwx_1536']

# 6. Save updated northParishFlows
parishes.to_file(parish_path)
print(f"Updated {parish_path} with drought variables.")
