import rasterio
import rasterio.mask
import geopandas as gpd
import requests
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_rgba
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
import warnings
from rasterio.merge import merge

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
            'nodata': datasets[0].nodata
        })

        # Copy colormap if exists
        colormap = datasets[0].colormap(1) if datasets[0].colormaps else None

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
    """
    Manual merge of tiles as a fallback method
    """
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
        Raw GeoTIFF data
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
    from functools import partial
    import concurrent.futures
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
        print(f"Response content: {response.text[:1000]}")

        # Try alternate axes labels (some servers use different conventions)
        print("Trying alternate axis labels...")

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
    table_name : str
        Name of the PostGIS table
    column_name : str
        Name of the column containing region names
    region_name : str
        Name of the region

    Returns:
    --------
    points_df : DataFrame
        DataFrame with x,y coordinates in EPSG:4326
    """
    debug_print("Retrieving treatment points intersecting with region...")
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
                its.activity_quantity,
                its.year_txt
            FROM its.activities_report_20250110 AS its, bbox, region_geom
            WHERE ST_Intersects(its.geom, bbox.geom)
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
                     geoserver_url: str, layer_name: str, layer_title: str,
                     min_value: float, max_value: float, min_value_color: str, max_value_color: str,
                     output_png=None, dpi=100, figsize=(12, 5), point_color='blue',
                     point_size=7, point_alpha=0.7, points_df=None, color_by_year=True):
    """
    Create a map for a custom region with OpenStreetMap background, WCS raster, and treatment points

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
        WCS layer name
    layer_title : str
        Title for the layer in the map
    min_value : float
        Minimum value for the color scale
    max_value : float
        Maximum value for the color scale
    min_value_color : str
        Hex color code for the minimum value
    max_value_color : str
        Hex color code for the maximum value
    output_png : str, optional
        Output path for PNG file
    dpi : int, optional
        DPI for the output image
    figsize : tuple, optional
        Figure size (width, height)
    point_color : str, optional
        Color for points if not coloring by year
    point_size : float, optional
        Size of the scatter points
    point_alpha : float, optional
        Transparency of the scatter points
    points_df : DataFrame, optional
        DataFrame containing points to plot
    color_by_year : bool, optional
        If True, color points by year_txt column; if False, use point_color for all points

    Returns:
    --------
    str
        Base64 encoded PNG image
    """
    # Get region boundary
    region_data_3310 = get_region_boundary(db, table_name, column_name, region_name)
    region_geom_3310 = region_data_3310.iloc[0].geom
    region_data_4326 = region_data_3310.to_crs("EPSG:4326")

    # Get treatment points from the passed DataFrame or query
    if points_df is not None:
        if points_df.empty:
            return None
        if 'x' not in points_df.columns or 'y' not in points_df.columns:
            if 'lon' in points_df.columns and 'lat' in points_df.columns:
                points_df = points_df.rename(columns={'lon': 'x', 'lat': 'y'})
            else:
                points_df = get_scatter_points(db, table_name, column_name, region_name)
    else:
        points_df = get_scatter_points(db, table_name, column_name, region_name)

    if points_df.empty:
        return None

    points_df = points_df.rename(columns={'lon': 'x', 'lat': 'y'})

    # Get bounding box in EPSG:3310
    minx, miny, maxx, maxy = region_geom_3310.bounds
    debug_print(f"Region bounds in EPSG:3310: {minx}, {miny}, {maxx}, {maxy}")

    width = maxx - minx
    height = maxy - miny
    tile_size = max(width, height) / 6
    geotiff_data = get_wcs_coverage_concurrent(geoserver_url, layer_name, minx, miny, maxx, maxy, tile_size, 8)

    # Calculate view bounds based on region geometry
    region_bounds_4326 = region_data_4326.total_bounds
    min_lon, min_lat, max_lon, max_lat = region_bounds_4326

    # Add a larger buffer (padding) around the region to ensure basemap visibility
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    buffer_factor = 0.2

    min_lon -= lon_range * buffer_factor
    max_lon += lon_range * buffer_factor
    min_lat -= lat_range * buffer_factor
    max_lat += lat_range * buffer_factor

    # Ensure aspect ratio is maintained
    fig_aspect = figsize[0] / figsize[1]
    map_aspect = lon_range / lat_range

    if map_aspect > fig_aspect:
        center_lat = (min_lat + max_lat) / 2
        adjusted_lat_range = lon_range / fig_aspect
        min_lat = center_lat - adjusted_lat_range / 2
        max_lat = center_lat + adjusted_lat_range / 2
    else:
        center_lon = (min_lon + max_lon) / 2
        adjusted_lon_range = lat_range * fig_aspect
        min_lon = center_lon - adjusted_lon_range / 2
        max_lon = center_lon + adjusted_lon_range / 2

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Set axis limits *before* adding the basemap to ensure correct tile fetching
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)

    # Add basemap with error handling and explicit zoom level
    try:
        debug_print(f"Adding OpenStreetMap basemap with bounds: ({min_lon}, {min_lat}, {max_lon}, {max_lat})")
        ctx.add_basemap(
            ax,
            crs="EPSG:4326",
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom=10,
            attribution="(C) OpenStreetMap contributors"
        )
        debug_print("Basemap added successfully")
    except Exception as e:
        debug_print(f"Error adding OpenStreetMap basemap: {str(e)}")
        try:
            debug_print("Attempting fallback basemap provider (Stamen Terrain)...")
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.Stamen.Terrain,
                zoom=10,
                attribution="(C) Stamen Design, OpenStreetMap contributors"
            )
            debug_print("Fallback basemap added successfully")
        except Exception as e2:
            debug_print(f"Error adding fallback basemap: {str(e2)}")
            ax.set_facecolor('lightblue')
            debug_print("Using fallback light blue background")

    # Process the GeoTIFF data
    with MemoryFile(geotiff_data) as memfile:
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

            # Reproject to EPSG:4326 for display
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

            # Handle data
            image_data = dst_array[0]
            nodata = src.nodata if src.nodata is not None else -9999

            # Mask NoData and out-of-range values
            masked_data = np.ma.masked_where(
                (image_data == nodata) | (image_data < min_value) | (image_data > max_value),
                image_data
            )

            # Calculate extent for display
            left = transform_4326[2]
            right = left + width_4326 * transform_4326[0]
            top = transform_4326[5]
            bottom = top + height_4326 * transform_4326[4]

            # Create custom colormap
            colors = [to_rgba(min_value_color), to_rgba(max_value_color)]
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
            # Set NoData (masked) values to be fully transparent
            cmap.set_bad(alpha=0)

            # Normalize data to the specified range
            norm = Normalize(vmin=min_value, vmax=max_value)

            # Plot the raster
            im = ax.imshow(
                masked_data,
                cmap=cmap,
                norm=norm,
                extent=[left, right, bottom, top],
                alpha=0.6,
                zorder=2
            )

            # Add a clipping path to ensure only the region is displayed
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch

            # Get coordinates of the region boundary in EPSG:4326
            coords = []
            for geom in region_data_4326.geometry:
                if geom.geom_type == 'Polygon':
                    coords.extend(list(geom.exterior.coords))
                elif geom.geom_type == 'MultiPolygon':
                    for part in geom.geoms:
                        coords.extend(list(part.exterior.coords))

            # Create a Path from the coordinates
            path = Path(coords)

            # Create a PathPatch and use it as a clip path for the image
            patch = PathPatch(path, facecolor='none', edgecolor='none')
            ax.add_patch(patch)
            im.set_clip_path(patch)

    # Plot the region boundary
    region_data_4326.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=3)

    # Plot scatter points
    if color_by_year and 'year_txt' in points_df.columns:
        year_colors = {'2021': 'green', '2022': 'purple', '2023': 'blue'}
        unique_years = sorted([y for y in points_df['year_txt'].unique() if y in year_colors])

        for year in unique_years:
            year_points = points_df[points_df['year_txt'] == year]
            if not year_points.empty:
                ax.scatter(
                    year_points['x'],
                    year_points['y'],
                    color=year_colors[year],
                    s=point_size,
                    alpha=point_alpha,
                    label=f'{year} Activities',
                    marker='o',
                    edgecolor='white',
                    linewidth=0.2,
                    zorder=4
                )
    else:
        ax.scatter(
            points_df['x'],
            points_df['y'],
            color=point_color,
            s=point_size,
            alpha=point_alpha,
            label='Activities',
            marker='o',
            edgecolor='white',
            linewidth=0.2,
            zorder=4
        )

    # Add colorbar with full height
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, aspect=40, pad=0.01)
    cbar.set_label(f'{layer_title}', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Add legend
    ax.legend(loc='upper right', fontsize=8, frameon=True, edgecolor='black')

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Adjust layout
    plt.tight_layout()

    # Save to BytesIO and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    if output_png:
        with open(output_png, 'wb') as f:
            f.write(base64.b64decode(img_str))

    return img_str

