import nbformat as nbf
nb = nbf.v4.new_notebook()

code = """# Modeling Rebellion Spread Animation

## Import packages and data
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Define Paths relative to project root
PROJECT_ROOT = Path.cwd().parent
RAW = PROJECT_ROOT / 'Data' / 'Raw'
PROCESSED = PROJECT_ROOT / 'Data' / 'Processed'
BNG = RAW / 'GIS' / 'BNG Projections'

# Load Data
print("Loading Data...")
north_parish_flows = gpd.read_file(PROCESSED / 'northParishFlows.shp')
reb_points = gpd.read_file(BNG / 'rebPoints.shp')
direct_evidence = gpd.read_file(BNG / 'direct_evidence.shp')
indirect_evidence = gpd.read_file(BNG / 'indirect_evidence.shp')
gough_routes = gpd.read_file(BNG / 'gough_routes.shp')

# Ensure all are in the same CRS (British National Grid)
target_crs = north_parish_flows.crs
print(f"Target CRS: {target_crs}")

for gdf in [reb_points, direct_evidence, indirect_evidence, gough_routes]:
    if gdf.crs != target_crs:
        print(f"Reprojecting...")
        gdf.to_crs(target_crs, inplace=True)

## Animation Setup
fig, ax = plt.subplots(figsize=(12, 12))

# Determine global bounds to keep camera still
minx, miny, maxx, maxy = north_parish_flows.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Function to draw each frame
def update(frame):
    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    
    # frame is day of October 1536 (0 = Oct 1)
    day_of_month = frame + 1
    month_name = "October"
    if day_of_month > 31:
        day_of_month -= 31
        month_name = "November"
    
    ax.set_title(f"Spread of News and Rebellion\\n{month_name} {day_of_month}, 1536", fontsize=20)
    
    # Base map: news_day + 1 <= day_of_october implies the parish has received news
    has_news = (north_parish_flows['news_day'] + 1) <= (frame + 1)
    
    # Plot parishes without news (light gray)
    if not north_parish_flows[~has_news].empty:
        north_parish_flows[~has_news].plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.2, zorder=1)
    
    # Plot parishes with news (orange)
    if not north_parish_flows[has_news].empty:
        north_parish_flows[has_news].plot(ax=ax, color='#ffb74d', edgecolor='white', linewidth=0.2, zorder=1)
    
    # Plot routes for context
    if not direct_evidence.empty:
        direct_evidence.plot(ax=ax, color='black', linewidth=0.8, alpha=0.6, zorder=2, label='Direct Evidence')
    if not indirect_evidence.empty:
        indirect_evidence.plot(ax=ax, color='gray', linewidth=0.6, alpha=0.6, linestyle='--', zorder=2)
    if not gough_routes.empty:
        gough_routes.plot(ax=ax, color='darkgray', linewidth=0.6, alpha=0.5, zorder=2)
    
    # Plot rebel points: day <= day_of_october implies the rebel muster has occurred
    rebel_now = reb_points[reb_points['day'] <= (frame + 1)]
    if not rebel_now.empty:
        rebel_now.plot(ax=ax, color='#d32f2f', markersize=50, zorder=3, edgecolors='black', linewidth=0.5)

print("Creating animation (this may take a minute)...")
ani = animation.FuncAnimation(fig, update, frames=range(0, 34), interval=1000/3)

## Save Result
output_path_mp4 = PROJECT_ROOT / 'Output' / 'Images' / 'Maps' / 'rebellion_spread.mp4'
output_path_gif = PROJECT_ROOT / 'Output' / 'Images' / 'Maps' / 'rebellion_spread.gif'

output_path_mp4.parent.mkdir(parents=True, exist_ok=True)

try:
    print(f"Saving MP4 to {output_path_mp4}...")
    ani.save(output_path_mp4, writer='ffmpeg', fps=3)
    print("MP4 saved successfully!")
except Exception as e:
    print(f"Could not save MP4 (possibly missing ffmpeg): {e}")

try:
    print(f"Saving GIF to {output_path_gif}...")
    ani.save(output_path_gif, writer='pillow', fps=3)
    print("GIF saved successfully!")
except Exception as e:
    print(f"Could not save GIF: {e}")
    
plt.close(fig) # Prevent static display of the last frame in the notebook
print("Done!")"""

nb['cells'] = [nbf.v4.new_code_cell(code)]
with open('Code/jn_03_rebellion_animation.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook Code/jn_03_rebellion_animation.ipynb created.")
