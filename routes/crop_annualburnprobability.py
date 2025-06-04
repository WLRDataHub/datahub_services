
import rasterio
import rasterio.mask
import geopandas as gpd
import requests
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sqlalchemy import create_engine
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import show
import matplotlib.patches as mpatches
import base64
import contextily as ctx  
from pyproj import Transformer
from sqlalchemy.orm import Session 
from sqlalchemy import text


def get_county_boundary(db: Session, county_name):
    """
    Retrieve the boundary for a specified county from the database
    
    Parameters:
    -----------
    db_connection_string : str
        Connection string for PostGIS database
    county_name : str
        Name of the county to retrieve
        
    Returns:
    --------
    county_data : GeoDataFrame
        County boundary in EPSG:3310 projection
    """
    # Connect to the database
    print(f"Connecting to database and retrieving {county_name} County boundary...")
     
    # Query to get the specified county geometry
    query = f"""
    SELECT name, geom 
    FROM boundary.ca_counties 
    WHERE name = '{county_name}'
    """
    
    # Execute query and get results
    with db.bind.connect() as conn:  
        county_data = gpd.read_postgis(text(query), conn, geom_col='geom')
    
    if county_data.empty:
        raise ValueError(f"County '{county_name}' not found in database")
    
    # Get county geometry and reproject to EPSG:3310 (California Albers)
    print("Reprojecting county geometry to EPSG:3310...")
    if county_data.crs is None:
        # If CRS is not defined, assume WGS84 (EPSG:4326)
        county_data.crs = "EPSG:4326"
    
    # Reproject to EPSG:3310
    county_data_3310 = county_data.to_crs("EPSG:3310")
    
    return county_data_3310

def get_wcs_coverage(geoserver_url, layer_name, minx, miny, maxx, maxy):
    """
    Retrieve a WCS coverage for the specified bounding box
    
    Parameters:
    -----------
    geoserver_url : str
        Base URL of the GeoServer
    layer_name : str
        Layer name (coverage ID) from the WCS service
    minx, miny, maxx, maxy : float
        Bounding box coordinates in EPSG:3310
        
    Returns:
    --------
    bytes
        Raw GeoTIFF data
    """
    # Build WCS request using EPSG:3310
    raster_url = f'{geoserver_url}/ows?service=WCS&version=2.0.0&request=GetCoverage&coverageId={layer_name}&format=image/geotiff'
    raster_url += f'&SUBSET=X({minx},{maxx})&SUBSET=Y({miny},{maxy})'
    raster_url += f'&SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/3310'
    
    print(f"WCS request URL: {raster_url}")
    
    # Download the coverage
    print(f"Downloading coverage...")
    response = requests.get(raster_url)
    
    if response.status_code != 200:
        print(f"Error status code: {response.status_code}")
        print(f"Response content: {response.text[:1000]}")  # Print first 1000 chars of response
        
        # Try alternate axes labels (some servers use different conventions)
        print("Trying alternate axis labels...")
        
        # Try with lowercase 'subset'
        raster_url_alt = f'{geoserver_url}/ows?service=WCS&version=2.0.0&request=GetCoverage&coverageId={layer_name}&format=image/geotiff'
        raster_url_alt += f'&subset=E({minx},{maxx})&subset=N({miny},{maxy})'
        raster_url_alt += f'&SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/3310'
        
        print(f"Alternative WCS request URL: {raster_url_alt}")
        response = requests.get(raster_url_alt)
        
        if response.status_code != 200:
            print(f"Alternative request also failed with status code: {response.status_code}")
            print(f"Response content: {response.text[:1000]}")
            raise Exception(f"Failed to download coverage")
    
    # Check if response is actually a GeoTIFF by looking at the first few bytes
    if response.content[:4] != b'II*\x00' and response.content[:4] != b'MM\x00*':
        print("Warning: Response does not appear to be a valid GeoTIFF")
        print(f"First 100 bytes: {response.content[:100]}")
    
    return response.content

def get_scatter_points(db_connection_string, county_code):
    """
    Retrieve points from the database for the specified county code
    
    Parameters:
    -----------
    db_connection_string : str
        Connection string for PostGIS database
    county_code : str
        County code for the query
        
    Returns:
    --------
    points_df : DataFrame
        DataFrame with x,y coordinates in EPSG:4326
    """
    # Create engine for database connection
    engine = create_engine(db_connection_string)
    
    # Get scatter point data with coordinates in EPSG:4326
    print(f"Retrieving point data for {county_code}...")
    query = f"""
        SELECT 
            county, 
            ST_X(ST_Transform(ST_Centroid(geom), 4326)) as x,
            ST_Y(ST_Transform(ST_Centroid(geom), 4326)) as y
        FROM its.activities_report_20241209
        WHERE county = '{county_code}'
    """
    
    points_df = pd.read_sql_query(query, engine)
    
    if points_df.empty:
        print(f"No points found for county code: {county_code}")
    else:
        print(f"Found {len(points_df)} points to plot")
        
    return points_df


def create_combined_map(db: Session, county_name, geoserver_url, layer_name, output_png, 
                       county_code=None, dpi=100, figsize=(8, 5), colormap='viridis', title=None,
                       point_color='blue', point_size=5, point_alpha=0.7):
    """
    Create a map with OpenStreetMap background, transparent burn probability raster, and treatment points
    Exactly matching the view window and appearance of the treatment map
    
    Parameters:
    -----------
    db: Session
        Connection string for PostGIS database
    county_name : str
        Name of the county to crop to
    geoserver_url : str
        Base URL of the GeoServer
    layer_name : str
        Layer name (coverage ID) from the WCS service
    output_png : str
        Path to save the output PNG file
    county_code : str, optional
        County code for scatter points query (if None, use county_name)
    dpi : int, optional
        Resolution of the output PNG (default: 100)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10.67, 8) - matches 800x600)
    colormap : str, optional
        Matplotlib colormap name (default: 'viridis')
    title : str, optional
        Map title. If None, a default title will be created.
    point_color : str, optional
        Color for scatter points (default: 'blue')
    point_size : int, optional
        Size for scatter points (default: 5 to match treatment map)
    point_alpha : float, optional
        Alpha (transparency) for scatter points (default: 0.7)
    """
    # Use the county_name as county_code if none provided
    if county_code is None:
        county_code = county_name
    
    # Get scatter points in EPSG:4326 (exactly matching the treatment map query)
    query = f"""
        SELECT 
            county, 
            ST_X(ST_Transform(ST_Centroid(geom), 4326)) as lon,
            ST_Y(ST_Transform(ST_Centroid(geom), 4326)) as lat,
            activity_quantity
        FROM its.activities_report_20241209
        WHERE county = '{county_code}'
    """
    with db.bind.connect() as conn:
        points_df = pd.read_sql_query(text(query), conn)
    
    if points_df.empty:
        print(f"No points found for county code: {county_code}")
        return None
    
    # Use the same column names as in treatment map for consistency
    points_df = points_df.rename(columns={'lon': 'x', 'lat': 'y'})
    
    # Calculate the center and bounds exactly as in the treatment map
    center_lat = points_df['y'].mean()
    center_lon = points_df['x'].mean()
    
    # Get county boundary for display and raster cropping
    county_data_4326 = get_county_boundary(db, county_name).to_crs("EPSG:4326")
    county_data_3310 = county_data_4326.to_crs("EPSG:3310")
    county_geom_3310 = county_data_3310.iloc[0].geom
    
    # Get the bounding box of the county in EPSG:3310 (for WCS request)
    minx, miny, maxx, maxy = county_geom_3310.bounds
    print(f"County bounds in EPSG:3310: {minx}, {miny}, {maxx}, {maxy}")
    
    # Get WCS coverage
    geotiff_data = get_wcs_coverage(geoserver_url, layer_name, minx, miny, maxx, maxy)
    
    # Find the extent that matches the Plotly map at zoom level 8
    # Plotly's zoom level 8 corresponds to roughly these scales
    lon_range = 2.4  # approximate degrees for zoom level 8
    lat_range = 1.5  # slightly smaller for lat due to projection
    
    min_lon = center_lon - lon_range/2
    max_lon = center_lon + lon_range/2
    min_lat = center_lat - lat_range/2
    max_lat = center_lat + lat_range/2
    
    # Process the raster in memory
    print("Processing raster in memory...")
    with MemoryFile(io.BytesIO(geotiff_data)) as memfile:
        with memfile.open() as src:
            print(f"Opened raster. Bounds: {src.bounds}, Shape: {src.shape}")
            
            # Crop to county boundary
            out_image, out_transform = rasterio.mask.mask(src, [county_geom_3310], crop=True, filled=False)
            out_meta = src.meta.copy()
            
            # Update metadata for the cropped image
            out_meta.update({
                'driver': 'GTiff',
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform
            })
            
            # Now reproject to EPSG:4326
            print("Reprojecting raster to EPSG:4326...")
            dst_crs = 'EPSG:4326'
            transform_4326, width_4326, height_4326 = calculate_default_transform(
                src.crs, dst_crs, out_image.shape[2], out_image.shape[1], 
                *rasterio.transform.array_bounds(out_image.shape[1], out_image.shape[2], out_transform)
            )
            
            # Create destination array for reprojected data
            dst_array = np.zeros((src.count, height_4326, width_4326), dtype=out_image.dtype)
            
            # Update metadata for reprojected image
            dst_meta = out_meta.copy()
            dst_meta.update({
                'crs': dst_crs,
                'transform': transform_4326,
                'width': width_4326,
                'height': height_4326
            })
            
            # Reproject the data
            for i in range(1, src.count + 1):
                reproject(
                    source=out_image[i-1],
                    destination=dst_array[i-1],
                    src_transform=out_transform,
                    src_crs=src.crs,
                    dst_transform=transform_4326,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
            
            # Create figure and axis with the same dimensions as treatment map
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get the first band
            image_data = dst_array[0]
            
            # Handle NoData values by creating a mask
            mask = None
            if src.nodata is not None:
                mask = image_data == src.nodata
            
            # Create a masked array
            if mask is not None:
                image_data = np.ma.masked_array(image_data, mask=mask)
            
            # Set up axis with exact bounds to match treatment map
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            
            # Add OpenStreetMap background first
            ctx.add_basemap(ax, crs=dst_crs, source=ctx.providers.OpenStreetMap.Mapnik)
            
            # Get original bounds for the raster data
            left = transform_4326[2]
            top = transform_4326[5]
            right = left + transform_4326[0] * image_data.shape[1]
            bottom = top + transform_4326[4] * image_data.shape[0]
            extent = [left, right, bottom, top]
            
            # Create a custom colormap with fully transparent background
            # This sets zero/low values to transparent
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_under('none')  # Set color for values below vmin to transparent
            
            # Plot the raster with custom vmin to make background transparent
            # Only show values above a certain threshold
            vmin = np.nanpercentile(image_data, 15)  # Start color at 5th percentile
            vmin = 0.01 
            vmax = 0.1

            im = ax.imshow(image_data, extent=extent, cmap=cmap, vmin=vmin,
                           interpolation='nearest', origin='upper', alpha=0.7)
            
            # Plot county boundary (thin line)
            county_data_4326.boundary.plot(ax=ax, color='black', linewidth=0.5)
            
            # Plot scatter points to exactly match treatment map's appearance
            ax.scatter(points_df['x'], points_df['y'], color=point_color, s=point_size, 
                       alpha=point_alpha, edgecolor='white', linewidth=0.2)
            
            # Add colorbar for burn probability
            cbar = plt.colorbar(
                im, 
                ax=ax,
                fraction=0.015,
                aspect=40,
                pad=0.01,
                location='right'
            )
            cbar.set_label('Burn Probability', size=10, labelpad=5)
            cbar.ax.tick_params(labelsize=8)

            # Remove all axis elements to match the treatment map's clean look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Set title if provided
            if title:
                ax.set_title(title)
            
            # Adjust layout to match the borderless appearance
            plt.tight_layout(pad=0)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Save to base64 with the same dimensions and resolution
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight', 
                        pad_inches=0, transparent=True)
            plt.close()
            img_buffer.seek(0)
            base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            return base64_image


def create_combined_map(db: Session, county_name, geoserver_url, layer_name, output_png, 
                       county_code=None, dpi=100, figsize=(8, 5), colormap='viridis', title=None,
                       point_color='blue', point_size=5, point_alpha=0.7):
    """
    Create a map with OpenStreetMap background, transparent burn probability raster, and treatment points
    Exactly matching the view window and appearance of the treatment map
    
    Parameters:
    -----------
    db: Session
        Connection string for PostGIS database
    county_name : str
        Name of the county to crop to
    geoserver_url : str
        Base URL of the GeoServer
    layer_name : str
        Layer name (coverage ID) from the WCS service
    output_png : str
        Path to save the output PNG file
    county_code : str, optional
        County code for scatter points query (if None, use county_name)
    dpi : int, optional
        Resolution of the output PNG (default: 100)
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (10.67, 8) - matches 800x600)
    colormap : str, optional
        Matplotlib colormap name (default: 'viridis')
    title : str, optional
        Map title. If None, a default title will be created.
    point_color : str, optional
        Color for scatter points (default: 'blue')
    point_size : int, optional
        Size for scatter points (default: 5 to match treatment map)
    point_alpha : float, optional
        Alpha (transparency) for scatter points (default: 0.7)
    """
    # Use the county_name as county_code if none provided
    if county_code is None:
        county_code = county_name
    
    # Get scatter points in EPSG:4326 (exactly matching the treatment map query)
    query = f"""
        SELECT 
            county, 
            ST_X(ST_Transform(ST_Centroid(geom), 4326)) as lon,
            ST_Y(ST_Transform(ST_Centroid(geom), 4326)) as lat
        FROM its.activities_report_20241209
        WHERE county = '{county_code}'
    """
    with db.bind.connect() as conn:
        points_df = pd.read_sql_query(text(query), conn)
    
    if points_df.empty:
        print(f"No points found for county code: {county_code}")
        return None
    
    # Use the same column names as in treatment map for consistency
    points_df = points_df.rename(columns={'lon': 'x', 'lat': 'y'})
    
    # Calculate the center and bounds exactly as in the treatment map
    center_lat = points_df['y'].mean()
    center_lon = points_df['x'].mean()
    
    # Get county boundary for display and raster cropping
    county_data_4326 = get_county_boundary(db, county_name).to_crs("EPSG:4326")
    county_data_3310 = county_data_4326.to_crs("EPSG:3310")
    county_geom_3310 = county_data_3310.iloc[0].geom
    
    # Get the bounding box of the county in EPSG:3310 (for WCS request)
    minx, miny, maxx, maxy = county_geom_3310.bounds
    print(f"County bounds in EPSG:3310: {minx}, {miny}, {maxx}, {maxy}")
    
    # Get WCS coverage
    geotiff_data = get_wcs_coverage(geoserver_url, layer_name, minx, miny, maxx, maxy)
    
    # Find the extent that matches the Plotly map at zoom level 8
    # Plotly's zoom level 8 corresponds to roughly these scales
    lon_range = 2.4  # approximate degrees for zoom level 8
    lat_range = 1.5  # slightly smaller for lat due to projection
    
    min_lon = center_lon - lon_range/2
    max_lon = center_lon + lon_range/2
    min_lat = center_lat - lat_range/2
    max_lat = center_lat + lat_range/2
    
    # Process the raster in memory
    print("Processing raster in memory...")
    with MemoryFile(io.BytesIO(geotiff_data)) as memfile:
        with memfile.open() as src:
            print(f"Opened raster. Bounds: {src.bounds}, Shape: {src.shape}")
            
            # Crop to county boundary
            out_image, out_transform = rasterio.mask.mask(src, [county_geom_3310], crop=True, filled=False)
            out_meta = src.meta.copy()
            
            # Update metadata for the cropped image
            out_meta.update({
                'driver': 'GTiff',
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform
            })
            
            # Now reproject to EPSG:4326
            print("Reprojecting raster to EPSG:4326...")
            dst_crs = 'EPSG:4326'
            transform_4326, width_4326, height_4326 = calculate_default_transform(
                src.crs, dst_crs, out_image.shape[2], out_image.shape[1], 
                *rasterio.transform.array_bounds(out_image.shape[1], out_image.shape[2], out_transform)
            )
            
            # Create destination array for reprojected data
            dst_array = np.zeros((src.count, height_4326, width_4326), dtype=out_image.dtype)
            
            # Update metadata for reprojected image
            dst_meta = out_meta.copy()
            dst_meta.update({
                'crs': dst_crs,
                'transform': transform_4326,
                'width': width_4326,
                'height': height_4326
            })
            
            # Reproject the data
            for i in range(1, src.count + 1):
                reproject(
                    source=out_image[i-1],
                    destination=dst_array[i-1],
                    src_transform=out_transform,
                    src_crs=src.crs,
                    dst_transform=transform_4326,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
            
            # Create figure and axis with the same dimensions as treatment map
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get the first band
            image_data = dst_array[0]
            
            # Handle NoData values by creating a mask
            mask = None
            if src.nodata is not None:
                mask = image_data == src.nodata
            
            # Create a masked array
            if mask is not None:
                image_data = np.ma.masked_array(image_data, mask=mask)
            
            # Set up axis with exact bounds to match treatment map
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            
            # Add OpenStreetMap background first
            ctx.add_basemap(ax, crs=dst_crs, source=ctx.providers.OpenStreetMap.Mapnik)
            
            # Get original bounds for the raster data
            left = transform_4326[2]
            top = transform_4326[5]
            right = left + transform_4326[0] * image_data.shape[1]
            bottom = top + transform_4326[4] * image_data.shape[0]
            extent = [left, right, bottom, top]
            
            # Create a custom colormap with fully transparent background
            # This sets zero/low values to transparent
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_under('none')  # Set color for values below vmin to transparent
            
            # FIXED COLORMAP RANGE: Use a fixed vmin (e.g., 0.01) and vmax (e.g., 0.1)
            vmin = 0.01
            vmax = 0.1  # Fixed upper bound for burn probability
            
            # Create a custom normalization
            norm = Normalize(vmin=vmin, vmax=vmax)
            
            # Plot the raster with fixed colormap range
            im = ax.imshow(image_data, extent=extent, cmap=cmap, norm=norm,
                           interpolation='nearest', origin='upper', alpha=0.7)
            
            # Plot county boundary (thin line)
            county_data_4326.boundary.plot(ax=ax, color='black', linewidth=0.5)
            
            # Plot scatter points to exactly match treatment map's appearance
            ax.scatter(points_df['x'], points_df['y'], color=point_color, s=point_size, 
                       alpha=point_alpha, edgecolor='white', linewidth=0.2)
            
            # Add colorbar for burn probability with fixed range
            cbar = plt.colorbar(
                im, 
                ax=ax,
                fraction=0.015,
                aspect=40,
                pad=0.01,
                location='right'
            )
            cbar.set_label('Annual Burn Probability', size=10, labelpad=5)
            cbar.ax.tick_params(labelsize=8)

            # Remove all axis elements to match the treatment map's clean look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Set title if provided
            if title:
                ax.set_title(title)
            
            # Adjust layout to match the borderless appearance
            plt.tight_layout(pad=0)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            # Save to base64 with the same dimensions and resolution
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight', 
                        pad_inches=0, transparent=True)
            plt.close()
            img_buffer.seek(0)
            base64_image = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            return base64_image



def create_map_plot_base64(db_connection_string, county_code):
    """
    Create an interactive map and return as base64 encoded PNG.
    
    Parameters:
    -----------
    db_connection_string : str
        Connection string for PostGIS database
    county_code : str
        County code to query
    
    Returns:
    --------
    str
        Base64 encoded PNG image
    """
    # Create engine for database connection
    engine = create_engine(db_connection_string)
    
    # Query precomputed centroids directly from the database
    query = f"""
        SELECT 
            county, 
            ST_X(ST_Transform(ST_Centroid(geom), 4326)) as lon,
            ST_Y(ST_Transform(ST_Centroid(geom), 4326)) as lat
        FROM its.activities_report_20241209
        WHERE county = '{county_code}'
    """
    
    df = pd.read_sql_query(query, engine)
    
    if df.empty:
        return None
    
    try:
        import plotly.graph_objects as go
        
        # Use WebGL-accelerated Scattermapbox
        fig = go.Figure(go.Scattermapbox(
            mode="markers",
            lon=df['lon'],
            lat=df['lat'],
            marker=dict(size=5, color='blue'),
            hoverinfo='none'
        ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=df['lat'].mean(), lon=df['lon'].mean()), zoom=8),
            margin=dict(l=5, r=5, t=5, b=5),
            height=400,
            width=600
        )

        # Export as PNG
        img_bytes = io.BytesIO()
        fig.write_image(img_bytes, format='png', engine='kaleido')  # Use Kaleido for static export
        img_bytes.seek(0)
        
        return base64.b64encode(img_bytes.read()).decode("utf-8")
    except ImportError:
        print("Plotly or kaleido not installed. Cannot create interactive map.")
        return None

