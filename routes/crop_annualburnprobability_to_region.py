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
import sys
from datetime import datetime


def debug_print(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    sys.stdout.write(f"[{timestamp}] {msg}\n")
    sys.stdout.flush()


def get_region_boundary(db: Session, table_name: str, column_name: str, region_name: str):
    """
    Retrieve the boundary for a specified region from the database
    
    Parameters:
    -----------
    db : Session
        Database session
    table_name : str
        Name of the PostGIS table containing region geometries
    column_name : str
        Name of the column containing region names
    region_name : str
        Name of the region to retrieve
        
    Returns:
    --------
    region_data : GeoDataFrame
        Region boundary in EPSG:3310 projection
    """
    debug_print(f"Retrieving {region_name} boundary from {table_name}.{column_name}...")
     
    # Query to get the specified region geometry
    query = f"""
    SELECT {column_name} as name, geom 
    FROM {table_name} 
    WHERE {column_name} = :region_name
    """
    
    # Execute query and get results
    with db.bind.connect() as conn:  
        region_data = gpd.read_postgis(text(query), conn, 
                                      geom_col='geom',
                                      params={"region_name": region_name})
    
    if region_data.empty:
        raise ValueError(f"Region '{region_name}' not found in {table_name}.{column_name}")
    
    # Reproject to EPSG:3310 (California Albers) if needed
    debug_print("Reprojecting region geometry to EPSG:3310...")
    if region_data.crs is None:
        region_data.crs = "EPSG:4326"
    
    region_data_3310 = region_data.to_crs("EPSG:3310")
    return region_data_3310

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
    
    debug_print(f"WCS request URL: {raster_url}")
    
    # Download the coverage
    debug_print(f"Downloading coverage...")
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

    debug_print(f"Downloaded coverage.")    
    return response.content


import concurrent.futures
from functools import partial
import requests
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
import io
import numpy as np
import warnings

def download_tile(geoserver_url, layer_name, tile_bbox):
    """Download a single WCS tile"""
    minx, miny, maxx, maxy = tile_bbox
    
    # Build WCS request using EPSG:3310
    raster_url = f'{geoserver_url}/ows?service=WCS&version=2.0.0&request=GetCoverage&coverageId={layer_name}&format=image/geotiff'
    raster_url += f'&SUBSET=X({minx},{maxx})&SUBSET=Y({miny},{maxy})'
    raster_url += f'&SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/3310'
    
    # Download the coverage
    response = requests.get(raster_url)
    
    if response.status_code != 200:
        # Try alternate axes labels (some servers use different conventions)
        raster_url_alt = f'{geoserver_url}/ows?service=WCS&version=2.0.0&request=GetCoverage&coverageId={layer_name}&format=image/geotiff'
        raster_url_alt += f'&subset=E({minx},{maxx})&subset=N({miny},{maxy})'
        raster_url_alt += f'&SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/3310'
        
        response = requests.get(raster_url_alt)
        
        if response.status_code != 200:
            raise Exception(f"Failed to download coverage for tile {tile_bbox}")
    
    # Check if response is actually a GeoTIFF
    if response.content[:4] != b'II*\x00' and response.content[:4] != b'MM\x00*':
        print(f"Warning: Response for tile {tile_bbox} does not appear to be a valid GeoTIFF")

    return response.content, tile_bbox


def safe_merge_tiles(tile_data):
    """
    Safely merge tiles using rasterio's merge function with careful memory management
    """
    if len(tile_data) == 1:
        return tile_data[0][0]

    memory_files = []
    datasets = []
    try:
        for data, _ in tile_data:
            memfile = MemoryFile(io.BytesIO(data))
            memory_files.append(memfile)
            datasets.append(memfile.open())
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            mosaic, out_transform = merge(datasets)
        
        # Preserve critical metadata from the first dataset
        profile = datasets[0].profile.copy()
        profile.update({
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_transform,
            'nodata': datasets[0].nodata  # Ensure nodata is preserved
        })
        
        # Copy colormap if exists
        colormap = datasets[0].colormap(1)
        
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(mosaic)
                if colormap:
                    dst.write_colormap(1, colormap)
            merged_bytes = memfile.read()
        
        return merged_bytes
    
    finally:
        for dataset in datasets:
            dataset.close()
        for memfile in memory_files:
            memfile.close()

def manual_merge_tiles(tile_data):
    if len(tile_data) == 1:
        return tile_data[0][0]
    
    debug_print("Using manual merge method...")
    
    tile_info = []
    for data, bbox in tile_data:
        try:
            with MemoryFile(io.BytesIO(data)) as memfile:
                with memfile.open() as src:
                    tile_info.append({
                        'data': src.read(),
                        'bounds': src.bounds,
                        'transform': src.transform,
                        'profile': src.profile,
                        'nodata': src.nodata
                    })
        except Exception as e:
            debug_print(f"Error reading tile {bbox}: {e}")
    
    if not tile_info:
        raise Exception("No valid tiles to merge")
    
    minx = min(info['bounds'].left for info in tile_info)
    miny = min(info['bounds'].bottom for info in tile_info)
    maxx = max(info['bounds'].right for info in tile_info)
    maxy = max(info['bounds'].top for info in tile_info)
    
    first_tile = tile_info[0]
    x_res = abs(first_tile['transform'].a)
    y_res = abs(first_tile['transform'].e)
    
    width = int((maxx - minx) / x_res) + 1
    height = int((maxy - miny) / y_res) + 1
    
    from rasterio.transform import from_origin
    new_transform = from_origin(minx, maxy, x_res, y_res)
    
    dtype = first_tile['data'].dtype
    count = first_tile['data'].shape[0]
    nodata = first_tile['nodata'] if first_tile['nodata'] is not None else 0
    
    merged_data = np.full((count, height, width), nodata, dtype=dtype)
    
    for info in tile_info:
        data = info['data']
        left = info['bounds'].left
        top = info['bounds'].top
        
        x_offset = int((left - minx) / x_res)
        y_offset = int((maxy - top) / y_res)
        
        tile_height, tile_width = data.shape[1], data.shape[2]
        
        # Calculate valid slices
        y_slice = slice(y_offset, y_offset + tile_height)
        x_slice = slice(x_offset, x_offset + tile_width)
        
        # Ensure slices are within bounds
        y_slice = slice(max(y_slice.start, 0), min(y_slice.stop, height))
        x_slice = slice(max(x_slice.start, 0), min(x_slice.stop, width))
        
        # Adjust tile data if necessary
        adj_y_start = max(0 - y_offset, 0)
        adj_x_start = max(0 - x_offset, 0)
        tile_data = data[
            :,
            adj_y_start:adj_y_start + (y_slice.stop - y_slice.start),
            adj_x_start:adj_x_start + (x_slice.stop - x_slice.start)
        ]
        
        merged_data[:, y_slice, x_slice] = tile_data
    
    profile = first_tile['profile'].copy()
    profile.update({
        'height': height,
        'width': width,
        'transform': new_transform,
        'nodata': nodata
    })
    
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(merged_data)
            if first_tile.get('colormap'):
                dst.write_colormap(1, first_tile['colormap'])
        merged_bytes = memfile.read()
    
    return merged_bytes


def get_wcs_coverage_concurrent(geoserver_url, layer_name, minx, miny, maxx, maxy, tile_size=None, max_workers=4):
    """
    Retrieve a WCS coverage for the specified bounding box using concurrent downloads
    
    Parameters:
    -----------
    geoserver_url : str
        Base URL of the GeoServer
    layer_name : str
        Layer name (coverage ID) from the WCS service
    minx, miny, maxx, maxy : float
        Bounding box coordinates in EPSG:3310
    tile_size : float, optional
        Size of each tile for parallel downloading. If None, uses original function.
    max_workers : int
        Maximum number of concurrent download workers
        
    Returns:
    --------
    bytes
        Raw GeoTIFF data (identical format to the original function)
    """
    # If no tile size is provided, use the original function
    if tile_size is None:
        return get_wcs_coverage(geoserver_url, layer_name, minx, miny, maxx, maxy)
    
    # Otherwise, use the concurrent approach
    debug_print(f"Downloading coverage using concurrent approach with {max_workers} workers...")
    
    # Calculate tile dimensions
    x_range = maxx - minx
    y_range = maxy - miny
    
    # Calculate number of tiles in x and y directions
    nx = max(1, int(x_range / tile_size))
    ny = max(1, int(y_range / tile_size))
    
    debug_print(f"Splitting request into {nx}x{ny} tiles...")
    
    # Create tile bounding boxes with a small overlap
    overlap = tile_size * 0.01  # 1% overlap
    tiles = []
    
    for i in range(nx):
        x_min = minx + i * (x_range / nx) - (0 if i == 0 else overlap)
        x_max = minx + (i + 1) * (x_range / nx) + (0 if i == nx-1 else overlap)
        
        for j in range(ny):
            y_min = miny + j * (y_range / ny) - (0 if j == 0 else overlap)
            y_max = miny + (j + 1) * (y_range / ny) + (0 if j == ny-1 else overlap)
            
            tiles.append((x_min, y_min, x_max, y_max))
    
    # Download tiles concurrently
    download_fn = partial(download_tile, geoserver_url, layer_name)
    tile_data = []
    
    debug_print(f"Starting concurrent downloads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {executor.submit(download_fn, tile): tile for tile in tiles}
        
        for future in concurrent.futures.as_completed(future_to_tile):
            tile = future_to_tile[future]
            try:
                data, bbox = future.result()
                tile_data.append((data, bbox))
                debug_print(f"Downloaded tile {tile}")
            except Exception as e:
                debug_print(f"Error downloading tile {tile}: {e}")
    
    debug_print(f"All tiles downloaded, merging results...")
    
    # Merge the tiles
    if not tile_data:
        raise Exception("Failed to download any tiles")
    
    # First try the safe merge method
    try:
        debug_print("Attempting standard merge method...")
        return safe_merge_tiles(tile_data)
    except Exception as e:
        debug_print(f"Standard merge failed: {e}")
        
        # Try the manual merge method as a fallback
        try:
            debug_print("Attempting manual merge method...")
            return manual_merge_tiles(tile_data)
        except Exception as e2:
            debug_print(f"Manual merge also failed: {e2}")
            
            # Last resort: return the largest tile
            debug_print("Returning largest tile as fallback...")
            largest_tile = max(tile_data, key=lambda x: len(x[0]))
            return largest_tile[0]



def get_wcs_coverage(geoserver_url, layer_name, minx, miny, maxx, maxy):
    """
    Original function - kept for reference and fallback
    
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
    
    debug_print(f"WCS request URL: {raster_url}")
    
    # Download the coverage
    debug_print(f"Downloading coverage...")
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

    debug_print(f"Downloaded coverage.")    
    return response.content






def get_scatter_points(db: Session, table_name, column_name, region_name):
    """
    Retrieve points from the database that intersect with the region geometry
    
    Parameters:
    -----------
    db : Session
        Database session
    region_geom : shapely.geometry
        Region geometry to intersect with
        
    Returns:
    --------
    points_df : DataFrame
        DataFrame with x,y coordinates in EPSG:4326
    """
    debug_print("Retrieving treatment points intersecting with region...")
    query = f"""
        SELECT ST_X(its.activities_report_20250110.geom) as lon,
               ST_Y(its.activities_report_20250110.geom) as lat,
               its.activities_report_20250110.activity_quantity                                                                                                   
          FROM its.activities_report_20250110,                                                                                                      
               { table_name }                                                                                                                       
         WHERE st_intersects(its.activities_report_20250110.geom, ST_Transform({ table_name }.geom, 4269))                                          
           AND { table_name }.{ column_name } = '{ region_name }'                                                                                   
           AND year_txt ~ '^[0-9]+$'                                                                                                                
           AND CAST(year_txt AS INTEGER) BETWEEN 2021 AND 2023     
    """

    query = f"""
           WITH region_geom AS (
                   SELECT ST_Transform(geom, 4269) AS geom
                   FROM {table_name}
                   WHERE {column_name} = '{region_name}'
                ),
                bbox AS (
                     SELECT ST_SetSRID(ST_Extent(geom), 4269)::geometry AS geom
                     FROM region_geom
                )
            SELECT
                ST_X(its.geom) AS lon,
                ST_Y(its.geom) AS lat,
                its.activity_quantity
            FROM its.activities_report_20250110 AS its, bbox, region_geom
            WHERE ST_Intersects(its.geom, bbox.geom)  -- Uses the bounding box for fast filtering
              AND st_contains(region_geom.geom, its.geom)
              AND year_txt ~ '^[0-9]+$'
              AND CAST(year_txt AS INTEGER) BETWEEN 2021 AND 2023;
    """
    debug_print(query)
    
    with db.bind.connect() as conn:
        points_df = pd.read_sql_query(text(query), conn)
    
    if points_df.empty:
        print("No treatment points found in this region")
    else:
        print(f"Found {len(points_df)} points to plot")

    debug_print(f"points_df: {points_df.shape}")     
        
    return points_df


def create_region_map(db: Session, table_name: str, column_name: str, region_name: str, 
                      geoserver_url: str, layer_name: str, output_png=None,
                      dpi=100, figsize=(12, 5), colormap='YlOrRd', title=None,  # Changed from (8, 5) to (12, 5)
                      point_color='blue', point_size=7, point_alpha=0.7, points_df=None,
                      color_by_year=True):
    """
    Create a map for a custom region with OpenStreetMap background, burn probability raster, and treatment points
    
    Parameters:
    -----------
    db : Session
        Database session
    table_name : str
        PostGIS table containing region geometries
    column_name : str
        Column containing region names
    region_name : str
        Name of the region to map
    geoserver_url : str
        GeoServer base URL
    layer_name : str
        WCS layer name for burn probability
    output_png : str, optional
        Output path for PNG file
    color_by_year : bool, optional
        If True, color points by year_txt column; if False, use point_color for all points
    """
    # Get region boundary
    region_data_3310 = get_region_boundary(db, table_name, column_name, region_name)
    region_geom_3310 = region_data_3310.iloc[0].geom
    region_data_4326 = region_data_3310.to_crs("EPSG:4326")

    # Get treatment points from the passed DataFrame or query
    if points_df is not None:
        # Use the provided points_df
        if points_df.empty:
            return None
        # Ensure columns are named x and y
        if 'x' not in points_df.columns or 'y' not in points_df.columns:
            if 'lon' in points_df.columns and 'lat' in points_df.columns:
                points_df = points_df.rename(columns={'lon': 'x', 'lat': 'y'})
            else:
                points_df = get_scatter_points(db, table_name, column_name, region_name)
    else:
        # Query the database if points_df not provided
        points_df = get_scatter_points(db, table_name, column_name, region_name)
    
    if points_df.empty:
        return None
    
    # Rename columns for consistency
    points_df = points_df.rename(columns={'lon': 'x', 'lat': 'y'})
    
    # Get bounding box in EPSG:3310
    minx, miny, maxx, maxy = region_geom_3310.bounds
    debug_print(f"Region bounds in EPSG:3310: {minx}, {miny}, {maxx}, {maxy}")
    
    width = maxx - minx
    height = maxy - miny
    tile_size = max(width, height) / 6
    geotiff_data = get_wcs_coverage_concurrent(geoserver_url, layer_name, minx, miny, maxx, maxy, tile_size, 8)
    
    # Calculate view bounds based on region geometry instead of points
    region_bounds_4326 = region_data_4326.total_bounds  # This gets [minx, miny, maxx, maxy]
    min_lon, min_lat, max_lon, max_lat = region_bounds_4326

    # Add a buffer (padding) around the region (10% of width/height)
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    buffer_factor = 0.1  # 10% buffer

    min_lon -= lon_range * buffer_factor
    max_lon += lon_range * buffer_factor
    min_lat -= lat_range * buffer_factor
    max_lat += lat_range * buffer_factor

    # Ensure aspect ratio is maintained according to the figure size
    fig_aspect = figsize[0] / figsize[1]
    map_aspect = lon_range / lat_range

    if map_aspect > fig_aspect:
        # Map is wider than figure - adjust latitude range
        center_lat = (min_lat + max_lat) / 2
        adjusted_lat_range = lon_range / fig_aspect
        min_lat = center_lat - adjusted_lat_range / 2
        max_lat = center_lat + adjusted_lat_range / 2
    else:
        # Map is taller than figure - adjust longitude range
        center_lon = (min_lon + max_lon) / 2
        adjusted_lon_range = lat_range * fig_aspect
        min_lon = center_lon - adjusted_lon_range / 2
        max_lon = center_lon + adjusted_lon_range / 2
    
    # Process raster
    with MemoryFile(io.BytesIO(geotiff_data)) as memfile:
        with memfile.open() as src:
            # Crop to region boundary
            out_image, out_transform = rasterio.mask.mask(src, [region_geom_3310], crop=True, filled=False)
            out_meta = src.meta.copy()
            out_meta.update({
                'driver': 'GTiff',
                'height': out_image.shape[1],
                'width': out_image.shape[2],
                'transform': out_transform
            })
            
            # Reproject to EPSG:4326
            dst_crs = 'EPSG:4326'

            transform_4326, width_4326, height_4326 = calculate_default_transform(
                src.crs, dst_crs, out_image.shape[2], out_image.shape[1], 
                *rasterio.transform.array_bounds(out_image.shape[1], out_image.shape[2], out_transform)
            )        
            
            dst_array = np.zeros((src.count, height_4326, width_4326), dtype=out_image.dtype)
            dst_meta = out_meta.copy()
            dst_meta.update({
                'crs': dst_crs,
                'transform': transform_4326,
                'width': width_4326,
                'height': height_4326
            })
            
            # Reproject data
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
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            image_data = dst_array[0]
            
            # Handle nodata
            mask = image_data == src.nodata if src.nodata else None
            if mask is not None:
                image_data = np.ma.masked_array(image_data, mask=mask)
            
            # Set map bounds
            ax.set_xlim(min_lon, max_lon)
            ax.set_ylim(min_lat, max_lat)
            
            # Add base map
            ctx.add_basemap(ax, crs=dst_crs, source=ctx.providers.OpenStreetMap.Mapnik)
            
            # Plot raster
            left = transform_4326[2]
            top = transform_4326[5]
            right = left + transform_4326[0] * image_data.shape[1]
            bottom = top + transform_4326[4] * image_data.shape[0]
            extent = [left, right, bottom, top]
            
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_under('none')
            
            # Fixed color range
            vmin = 0.01
            vmax = 0.1
            norm = Normalize(vmin=vmin, vmax=vmax)
            
            im = ax.imshow(image_data, extent=extent, cmap=cmap, norm=norm,
                           interpolation='nearest', origin='upper', alpha=0.7)
            
            # Plot boundary
            region_data_4326.boundary.plot(ax=ax, color='black', linewidth=0.5)
            
            # Plot points - either by year or with a single color
            if color_by_year and 'year_txt' in points_df.columns:
                # Define color mapping for years
                year_colors = {
                    '2021': '#008000',  # Green
                    '2022': '#8A2BE2',  # Purple
                    '2023': '#0000FF',  # Blue
                }
                
                # Define the desired plotting order (bottom to top)
                year_order = ['2021', '2022', '2023']  # Earlier years first, latest year last
                
                # Store scatter objects and their corresponding years
                scatter_objects = []
                
                # Plot each year in the specified order
                for year in year_order:
                    if year in points_df['year_txt'].values:
                        year_data = points_df[points_df['year_txt'] == year]
                        color = year_colors.get(year, 'gray')  # Default to gray if year not in mapping
                        
                        scatter = ax.scatter(
                            year_data['x'], year_data['y'], 
                            color=color, 
                            s=point_size, 
                            alpha=point_alpha, 
                            edgecolor='white', 
                            linewidth=0.2,
                            label=f'Activities {year}'
                        )
                        scatter_objects.append((year, scatter))
                
                # Create legend handles in reverse order (top to bottom in legend)
                # This makes the legend order match the visual prominence
                legend_handles = []
                for year, scatter in reversed(scatter_objects):
                    legend_handles.append(scatter)
                
                # Add legend
                if legend_handles:
                    ax.legend(handles=legend_handles, loc='upper right', 
                             fontsize=8, frameon=True, framealpha=0.9)
            else:
                # Plot all points with the same color
                ax.scatter(
                    points_df['x'], points_df['y'], 
                    color=point_color, 
                    s=point_size, 
                    alpha=point_alpha, 
                    edgecolor='white', 
                    linewidth=0.2
                )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.015, aspect=40, pad=0.01)
            cbar.set_label('Annual Burn Probability', size=10)
            cbar.ax.tick_params(labelsize=8)
            
            # Add title if provided
            if title:
                ax.set_title(title, fontsize=12)
            
            # Clean up axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            
            # Save to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=dpi, bbox_inches='tight', 
                        pad_inches=0, transparent=True)
            plt.close()
            img_buffer.seek(0)
            return base64.b64encode(img_buffer.getvalue()).decode('utf-8')



def create_region_map_base64(db: Session, table_name: str, column_name: str, region_name: str):
    """
    Create an interactive region map and return as base64 encoded PNG
    
    Parameters:
    -----------
    db : Session
        Database session
    table_name : str
        PostGIS table containing region geometries
    column_name : str
        Column containing region names
    region_name : str
        Name of the region to map
    
    Returns:
    --------
    str
        Base64 encoded PNG image
    """
    try:
        # Get region boundary in 4326 for visualization
        region_data = get_region_boundary(db, table_name, column_name, region_name).to_crs("EPSG:4326")
        region_geom = region_data.iloc[0].geom
        
        # Get points
        query = f"""
            SELECT 
                ST_X(ST_Transform(ST_Centroid(geom), 4326)) as lon,
                ST_Y(ST_Transform(ST_Centroid(geom), 4326)) as lat
            FROM its.activities_report_20250110
            WHERE ST_Intersects(geom, ST_Transform(ST_SetSRID(ST_GeomFromText(:region_geom), 4326), 4269))
        """
        with db.bind.connect() as conn:
            points_df = pd.read_sql_query(text(query), conn, params={"region_geom": region_geom.wkt})
        
        if points_df.empty:
            return None
        
        # Calculate bounds
        bounds = region_geom.bounds  # This gets (minx, miny, maxx, maxy)
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Calculate zoom level based on the bounding box size
        import math
        lat_range = abs(bounds[3] - bounds[1])
        lon_range = abs(bounds[2] - bounds[0])
        max_range = max(lat_range, lon_range)
        zoom = 12 - math.log2(max_range * 111)  # 111km is roughly 1 degree at the equator
        
        # Ensure zoom is within reasonable bounds
        zoom = max(5, min(15, zoom))
        
        # Create plot
        import plotly.graph_objects as go
        
        # Create a figure with the region boundary
        fig = go.Figure()
        
        # Add region boundary
        x, y = region_geom.exterior.xy
        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=list(x),
            lat=list(y),
            marker=dict(size=1, color='black'),
            line=dict(width=2, color='black'),
            hoverinfo='none',
            name='Region Boundary'
        ))
        
        # Add points
        fig.add_trace(go.Scattermapbox(
            mode="markers",
            lon=points_df['lon'],
            lat=points_df['lat'],
            marker=dict(size=5, color='blue'),
            hoverinfo='none',
            name='Treatment Points'
        ))

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=dict(lat=center_lat, lon=center_lon),
                zoom=zoom
            ),
            margin=dict(l=5, r=5, t=5, b=5),
            height=400,
            width=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.02,
                xanchor="right",
                x=0.95
            )
        )

        # Export to base64
        img_bytes = io.BytesIO()
        fig.write_image(img_bytes, format='png', engine='kaleido')
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")
    
    except Exception as e:
        print(f"Error creating region map: {str(e)}")
        return None