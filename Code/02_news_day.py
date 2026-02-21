# Script to combine transport networks to analyze news travel time from Louth Park Abbey
#%% Import packages and data
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from skimage.graph import MCP_Geometric
from shapely.geometry import Point
import os
import matplotlib.pyplot as plt

# Set paths
base_dir = r"C:\PhD\DissolutionProgramming\REB---Rebellion-Paper"
processed_dir = os.path.join(base_dir, "Data", "Processed")
raw_gis_dir = os.path.join(base_dir, "Data", "Raw", "GIS")
bng_dir = os.path.join(raw_gis_dir, "BNG Projections")

# Load Data
print("Loading Data...")
# Load Target File
north_parish_flows = gpd.read_file(os.path.join(processed_dir, "northParishFlows.shp"))

# Load England File
counties_base = gpd.read_file(r"C:\PhD\DissolutionProgramming\REB---Rebellion-Paper\Data\Raw\GIS\BNG Projections\countiesBNG.shp")

# Load Source Point
louth_park = gpd.read_file(os.path.join(bng_dir, "LouthParkAbbey.shp"))

# Load Cost Features
direct_evidence = gpd.read_file(os.path.join(bng_dir, "direct_evidence.shp"))
indirect_evidence = gpd.read_file(os.path.join(bng_dir, "indirect_evidence.shp"))
gough_routes = gpd.read_file(os.path.join(bng_dir, "gough_routes.shp"))
shipping = gpd.read_file(os.path.join(bng_dir, "shippingDissolved.shp"))

# Load Ancient Parishes for overlay
ancient_parishes = gpd.read_file(os.path.join(bng_dir, "AncientParishesBNG.shp"))

# Define northern counties
north_list = ['NORTHUMBERLAND', 'CUMBERLAND', 'DURHAM', 'WESTMORLAND',
              'YORKSHIRE, NORTH RIDING', 'LANCASHIRE', 'YORKSHIRE, WEST RIDING',
              'YORKSHIRE, EAST RIDING', 'LINCOLNSHIRE']

# Ensure all are in the same CRS (British National Grid)
target_crs = north_parish_flows.crs
print(f"Target CRS: {target_crs}")

for gdf in [louth_park, direct_evidence, indirect_evidence, gough_routes, shipping, ancient_parishes]:
    if gdf.crs != target_crs:
        print(f"Reprojecting {gdf}...")
        gdf.to_crs(target_crs, inplace=True)

#%% Define Grid
print("Defining Grid...")
# Define bounds with 20km buffer
minx, miny, maxx, maxy = north_parish_flows.total_bounds
buffer_dist = 20000
minx -= buffer_dist
miny -= buffer_dist
maxx += buffer_dist
maxy += buffer_dist

# Grid resolution (1km)
res = 1000
width = int((maxx - minx) / res)
height = int((maxy - miny) / res)

transform = from_origin(minx, maxy, res, res)
print(f"Grid Size: {width} x {height}")
print(f"Bounds: {minx}, {miny}, {maxx}, {maxy}")

#%% Create Cost Surface
print("Creating Cost Surface...")
# Initialize with high cost (background)
cost_grid = np.full((height, width), 0.2, dtype=np.float32)

# Function to rasterize and update minimum cost
def rasterize_and_update(gdf, cost_value, existing_grid, transform, shape):
    shapes = ((geom, cost_value) for geom in gdf.geometry)
    feature_grid = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=np.inf, # Use infinity for background to facilitate minimization
        dtype=np.float32,
        all_touched=True
    )
    return np.minimum(existing_grid, feature_grid)

# Rasterize layers
# 1. Land (General movement) - Cost 0.05
print("Rasterizing Land (0.05)...")
cost_grid = rasterize_and_update(counties_base, 0.05, cost_grid, transform, (height, width))

# 2. Roads and Evidence - Cost 0.02
print("Rasterizing Roads/Evidence (0.02)...")
for gdf in [direct_evidence, indirect_evidence, gough_routes]:
    cost_grid = rasterize_and_update(gdf, 0.02, cost_grid, transform, (height, width))

# 3. Shipping - Cost 0.01
print("Rasterizing Shipping (0.01)...")
cost_grid = rasterize_and_update(shipping, 0.01, cost_grid, transform, (height, width))

#%% Calculate Least Cost Path
print("Calculating Least Cost Paths...")
# Get Source Node (Louth Park Abbey)
source_geom = louth_park.geometry.iloc[0]
if isinstance(source_geom, Point):
    src_x, src_y = source_geom.x, source_geom.y
else:
    src_x, src_y = source_geom.centroid.x, source_geom.centroid.y

# Convert to matrix coordinates
# ~transform gives the inverse transform
inv_transform = ~transform
col, row = inv_transform * (src_x, src_y)
src_r, src_c = int(row), int(col)

# Initialize MCP with fully_connected=False for 4-connectivity (no diagonals)
mcp = MCP_Geometric(cost_grid, fully_connected=False)

# Calculate cumulative costs from source to all other points
cumulative_costs, traceback = mcp.find_costs([(src_r, src_c)])

#%% Extract Costs for Parishes
print("Extracting costs for parishes...")
# Function to get cost at centroid
def get_cost_at_centroid(geom):
    cx, cy = geom.centroid.x, geom.centroid.y
    c, r = inv_transform * (cx, cy)
    r, c = int(r), int(c)
    
    # Check bounds
    if 0 <= r < height and 0 <= c < width:
        return cumulative_costs[r, c]
    else:
        return np.nan

north_parish_flows['news_day'] = north_parish_flows.geometry.apply(get_cost_at_centroid)

#%% # 6. Save Result
output_path = os.path.join(processed_dir, "northParishFlows.shp")
north_parish_flows.to_file(output_path)
print(f"Saved to {output_path}")

#%% Plotting (Optional)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(cumulative_costs, cmap='viridis_r', origin='upper', vmax=np.percentile(cumulative_costs[cumulative_costs < 999], 95)) # Cap vmax for better viz
plt.colorbar(im, ax=ax, label='Cost (Days)')
ax.scatter([src_c], [src_r], c='red', marker='*', label='Louth Park Abbey', s=200, zorder=5)
ax.set_title('Least Cost Surface (Days from Louth Park)')
ax.legend()
plt.savefig(os.path.join('Output/Images/Maps/news_day_cost_surface.png'), dpi=300)
plt.show()

#%% Plot Least Cost Surface with Route Overlays
print("Creating route overlay plot...")

# Function to transform geometry coordinates to raster coordinates
def geom_to_raster_coords(gdf, inv_transform, height, width):
    """
    Convert geographic coordinates to raster row/col coordinates.
    Each LineString is kept separate to prevent artificial connections.
    Coordinates are clipped to raster bounds.
    """
    coords_list = []
    for geom in gdf.geometry:
        if geom.is_empty or geom is None:
            continue
            
        # Handle both LineString and MultiLineString
        lines = []
        if geom.geom_type == 'LineString':
            lines = [geom]
        elif geom.geom_type == 'MultiLineString':
            lines = list(geom.geoms)
        else:
            continue
        
        # Process each LineString separately to avoid artificial connections
        for line in lines:
            coords = list(line.coords)
            
            # Transform each coordinate and clip to bounds
            raster_coords = []
            for x, y in coords:
                col, row = inv_transform * (x, y)
                
                # Check if point is within raster bounds
                if 0 <= row < height and 0 <= col < width:
                    raster_coords.append((col, row))
                else:
                    # If we hit an out-of-bounds point, break the line
                    if raster_coords:
                        coords_list.append(raster_coords)
                        raster_coords = []
            
            # Add the final segment if it has points
            if len(raster_coords) >= 2:  # Need at least 2 points for a line
                coords_list.append(raster_coords)
    
    return coords_list

# Create the figure
fig, ax = plt.subplots(figsize=(14, 14))

# Plot the least cost surface
im = ax.imshow(cumulative_costs, cmap='viridis_r', origin='upper', 
               vmax=np.percentile(cumulative_costs[cumulative_costs < 999], 95))
plt.colorbar(im, ax=ax, label='Cost (Days)')

# Set axis limits to match raster dimensions exactly
ax.set_xlim(0, width)
ax.set_ylim(height, 0)  # Inverted because origin is upper

# Overlay ancient parishes
print("Plotting ancient parishes overlay...")

# Function to convert polygon coordinates to raster coordinates
def plot_polygon_overlay(gdf, inv_transform, ax, color, alpha, height, width, label=None):
    """Plot polygons as overlay on raster, clipped to raster bounds"""
    from matplotlib.patches import Polygon as MPLPolygon
    from matplotlib.collections import PatchCollection
    
    patches = []
    for geom in gdf.geometry:
        if geom.is_empty or geom is None:
            continue
        
        # Handle both Polygon and MultiPolygon
        polygons = []
        if geom.geom_type == 'Polygon':
            polygons = [geom]
        elif geom.geom_type == 'MultiPolygon':
            polygons = list(geom.geoms)
        else:
            continue
        
        for poly in polygons:
            # Get exterior coordinates
            coords = list(poly.exterior.coords)
            
            # Transform to raster coordinates and check bounds
            raster_coords = []
            all_in_bounds = True
            for x, y in coords:
                col, row = inv_transform * (x, y)
                raster_coords.append((col, row))
                
                # Check if any point is significantly out of bounds
                if not (0 <= row < height and 0 <= col < width):
                    all_in_bounds = False
            
            # Only add polygon if it has valid coordinates and intersects with raster
            if len(raster_coords) >= 3:
                # Check if at least some part might be visible
                cols, rows = zip(*raster_coords)
                min_col, max_col = min(cols), max(cols)
                min_row, max_row = min(rows), max(rows)
                
                # Check if polygon intersects with raster bounds
                if not (max_col < 0 or min_col >= width or max_row < 0 or min_row >= height):
                    # Clip coordinates to bounds for rendering
                    clipped_coords = []
                    for col, row in raster_coords:
                        clipped_col = max(0, min(width - 1, col))
                        clipped_row = max(0, min(height - 1, row))
                        clipped_coords.append((clipped_col, clipped_row))
                    
                    patches.append(MPLPolygon(clipped_coords, closed=True))
    
    if patches:
        collection = PatchCollection(patches, facecolor=color, edgecolor='none', 
                                    alpha=alpha, zorder=2)
        ax.add_collection(collection)

# Determine county field name
county_field = None
for field in ['GAZ_CNTY', 'county', 'County', 'COUNTY']:
    if field in ancient_parishes.columns:
        county_field = field
        break

if county_field:
    print(f"Using county field: {county_field}")
    # Split into northern and other parishes
    northern_parishes = ancient_parishes[ancient_parishes[county_field].isin(north_list)]
    other_parishes = ancient_parishes[~ancient_parishes[county_field].isin(north_list)]
    
    # Plot other parishes in light grey
    print(f"Plotting {len(other_parishes)} non-northern parishes...")
    plot_polygon_overlay(other_parishes, inv_transform, ax, color='#D3D3D3', alpha=0.3, 
                        height=height, width=width)
    
    # Plot northern parishes in dark grey
    print(f"Plotting {len(northern_parishes)} northern parishes...")
    plot_polygon_overlay(northern_parishes, inv_transform, ax, color='#505050', alpha=0.4,
                        height=height, width=width)
else:
    print("Warning: Could not find county field in ancient parishes shapefile")
    # Plot all parishes in light grey as fallback
    plot_polygon_overlay(ancient_parishes, inv_transform, ax, color='#D3D3D3', alpha=0.3,
                        height=height, width=width)

# Overlay routes with specified colors
# Medium blue for evidence routes
medium_blue = '#4682B4'
dark_blue = '#00008B'
red = '#FF0000'

# Plot direct evidence in medium blue
print("Plotting direct evidence routes...")
direct_coords = geom_to_raster_coords(direct_evidence, inv_transform, height, width)
for coords in direct_coords:
    if len(coords) >= 2:
        cols, rows = zip(*coords)
        ax.plot(cols, rows, color=medium_blue, linewidth=2, alpha=0.8, zorder=3)

# Plot indirect evidence in medium blue
print("Plotting indirect evidence routes...")
indirect_coords = geom_to_raster_coords(indirect_evidence, inv_transform, height, width)
for coords in indirect_coords:
    if len(coords) >= 2:
        cols, rows = zip(*coords)
        ax.plot(cols, rows, color=medium_blue, linewidth=2, alpha=0.8, zorder=3)

# Plot gough routes in red
print("Plotting Gough routes...")
gough_coords = geom_to_raster_coords(gough_routes, inv_transform, height, width)
for coords in gough_coords:
    if len(coords) >= 2:
        cols, rows = zip(*coords)
        ax.plot(cols, rows, color=red, linewidth=2.5, alpha=0.9, zorder=4)

# Plot shipping routes in dark blue
print("Plotting shipping routes...")
shipping_coords = geom_to_raster_coords(shipping, inv_transform, height, width)
for coords in shipping_coords:
    if len(coords) >= 2:
        cols, rows = zip(*coords)
        ax.plot(cols, rows, color=dark_blue, linewidth=2, alpha=0.8, zorder=3)

# Plot source point
ax.scatter([src_c], [src_r], c='yellow', marker='*', 
          edgecolors='black', linewidths=1, 
          label='Louth Park Abbey', s=300, zorder=6)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=red, linewidth=2.5, label='Gough Routes'),
    Line2D([0], [0], color=medium_blue, linewidth=2, label='River Shipping Routes'),
    Line2D([0], [0], color=dark_blue, linewidth=2, label='Shipping Routes'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', 
           markeredgecolor='black', markersize=15, label='Louth Park Abbey')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

ax.set_title('Least Cost Surface with Historical Route Networks', fontsize=14, fontweight='bold')

# Remove axis ticks and labels
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('')

# Save figure
output_path = os.path.join(base_dir, 'Output/Images/Maps/news_day_with_routes.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Route overlay saved to {output_path}")
plt.show()