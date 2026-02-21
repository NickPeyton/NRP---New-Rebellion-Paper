import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# Load the data
coords_path = 'Data/Raw/drought_coords.csv'
values_path = 'Data/Raw/drought_values.csv'

coords_df = pd.read_csv(coords_path)
values_df = pd.read_csv(values_path)

# Handle missing values
values_df = values_df.replace(-99.999, np.nan)

# Define the target year
target_year = 1536

# Windows: 1, 2, 3, 5, 10 years leading up to and including 1536
windows = {
    '1yr': [1536],
    '2yr': [1535, 1536],
    '3yr': [1534, 1535, 1536],
    '5yr': list(range(1532, 1537)),
    '10yr': list(range(1527, 1537))
}

results = {}

# Single-year drought (1535), flipped so higher = more severe drought
year_1535_df = values_df[values_df['year'] == 1535]
if not year_1535_df.empty:
    year_1535_values = year_1535_df.drop(columns=['year']).mean()
    results['pdsi_1535'] = -1 * year_1535_values

# Single-year wet weather (1536), unflipped so higher = wetter
year_1536_df = values_df[values_df['year'] == 1536]
if not year_1536_df.empty:
    year_1536_values = year_1536_df.drop(columns=['year']).mean()
    results['pdsi_1536'] = year_1536_values

if 'pdsi_1535' in results and 'pdsi_1536' in results:
    results['pdsi_1535x1536'] = results['pdsi_1535'] * results['pdsi_1536']

for name, years in windows.items():
    # Filter for the specific years
    window_df = values_df[values_df['year'].isin(years)]
    
    # Calculate mean for each column (grid cell), excluding 'year'
    means = window_df.drop(columns=['year']).mean()
    # Flip so higher values indicate more severe drought
    means = -1 * means

    # Extreme weather version: mean of absolute values
    abs_means = window_df.drop(columns=['year']).abs().mean()

    results[f'pdsi_avg_{name}'] = means
    results[f'pdsi_ext_{name}'] = abs_means

# Combine results into a single DataFrame
# Transpose means to have grid cells as index
results_df = pd.DataFrame(results)
results_df.index = results_df.index.astype(int)
results_df.index.name = 'Grid-cell'

# Merge with coordinates
merged_df = coords_df.merge(results_df, on='Grid-cell', how='inner')

# Convert to GeoDataFrame
# Assuming coords are in WGS84 (EPSG:4326)
geometry = [Point(xy) for xy in zip(merged_df['Longitude'], merged_df['Latitude'])]
gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")

# Reproject to British National Grid (EPSG:27700)
gdf_bng = gdf.to_crs("EPSG:27700")

# Extract BNG coordinates
gdf_bng['bng_x'] = gdf_bng.geometry.x
gdf_bng['bng_y'] = gdf_bng.geometry.y

# Drop the geometry column for CSV output
output_df = pd.DataFrame(gdf_bng.drop(columns='geometry'))

# Save to processed data folder
output_path = 'Data/Processed/drought_intensity_bng.csv'
output_df.to_csv(output_path, index=False)

print(f"Processed drought data saved to {output_path}")
