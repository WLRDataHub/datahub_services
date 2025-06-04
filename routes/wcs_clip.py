from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import Response
import requests
import zipfile
import io
import logging
from typing import Optional
from pydantic import BaseModel, Field
from geojson_pydantic import MultiPolygon, Feature, FeatureCollection
from geojson_pydantic.geometries import Polygon
import pyproj
from shapely.geometry import shape
from shapely.ops import transform
from functools import partial
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
import numpy as np
import concurrent.futures
import warnings
import uuid


router = APIRouter(tags=["Utility"], prefix='/Utility')

class WCSClipRequest(BaseModel):
    geoserver_url: str
    coverage_id: str
    geojson: dict  # Will accept any GeoJSON structure
    input_crs: str = Field(default="EPSG:4326", description="CRS of the input GeoJSON")
    output_format: Optional[str] = "image/tiff"
    wcs_crs: str = Field(default="EPSG:3310", description="CRS used by the WCS service")
    zip_filename: Optional[str] = Field(default=None, description="Custom name for the output ZIP file (must end with .zip)")

def debug_print(msg):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")

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
        # Try alternate axes labels
        raster_url_alt = f'{geoserver_url}/ows?service=WCS&version=2.0.0&request=GetCoverage&coverageId={layer_name}&format=image/geotiff'
        raster_url_alt += f'&subset=E({minx},{maxx})&subset=N({miny},{maxy})'
        raster_url_alt += f'&SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/3310'

        response = requests.get(raster_url_alt)

        if response.status_code != 200:
            raise Exception(f"Failed to download coverage for tile {tile_bbox}")

    # Check if response is a valid GeoTIFF
    if response.content[:4] not in [b'II*\x00', b'MM\x00*']:
        logging.warning(f"Response for tile {tile_bbox} does not appear to be a valid GeoTIFF")

    return response.content, tile_bbox

def safe_merge_tiles(tile_data):
    """Safely merge tiles using rasterio's merge function"""
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

        # Preserve metadata from the first dataset
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
    """Manual merge of tiles as a fallback method"""
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

        y_slice = slice(y_offset, y_offset + tile_height)
        x_slice = slice(x_offset, x_offset + tile_width)

        y_slice = slice(max(y_slice.start, 0), min(y_slice.stop, height))
        x_slice = slice(max(x_slice.start, 0), min(x_slice.stop, width))

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

def get_wcs_coverage_concurrent(geoserver_url, layer_name, minx, miny, maxx, maxy, tile_size=100000, max_workers=4):
    """Retrieve WCS coverage using concurrent tile downloads"""
    # Check if the area is small enough for a single request
    x_range = maxx - minx
    y_range = maxy - miny
    if x_range * y_range < tile_size * tile_size:
        debug_print("Area is small, using single WCS request...")
        raster_url = f'{geoserver_url}/ows?service=WCS&version=2.0.0&request=GetCoverage&coverageId={layer_name}&format=image/geotiff'
        raster_url += f'&SUBSET=X({minx},{maxx})&SUBSET=Y({miny},{maxy})'
        raster_url += f'&SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/3310'

        debug_print(f"WCS request URL: {raster_url}")
        response = requests.get(raster_url)

        if response.status_code != 200:
            raster_url_alt = f'{geoserver_url}/ows?service=WCS&version=2.0.0&request=GetCoverage&coverageId={layer_name}&format=image/geotiff'
            raster_url_alt += f'&subset=E({minx},{maxx})&subset=N({miny},{maxy})'
            raster_url_alt += f'&SubsettingCRS=http://www.opengis.net/def/crs/EPSG/0/3310'

            response = requests.get(raster_url_alt)

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=f"Failed to download coverage: {response.text[:500]}")

        return response.content

    # Concurrent tile-based approach
    debug_print(f"Downloading coverage using concurrent approach with {max_workers} workers...")
    nx = max(1, int(x_range / tile_size))
    ny = max(1, int(y_range / tile_size))

    debug_print(f"Splitting request into {nx}x{ny} tiles...")
    overlap = tile_size * 0.01  # 1% overlap
    tiles = []

    for i in range(nx):
        x_min = minx + i * (x_range / nx) - (0 if i == 0 else overlap)
        x_max = minx + (i + 1) * (x_range / nx) + (0 if i == nx-1 else overlap)
        for j in range(ny):
            y_min = miny + j * (y_range / ny) - (0 if j == 0 else overlap)
            y_max = miny + (j + 1) * (y_range / ny) + (0 if j == ny-1 else overlap)
            tiles.append((x_min, y_min, x_max, y_max))

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

    if not tile_data:
        raise HTTPException(status_code=500, detail="Failed to download any tiles")

    try:
        debug_print("Attempting standard merge method...")
        return safe_merge_tiles(tile_data)
    except Exception as e:
        debug_print(f"Standard merge failed: {e}")
        try:
            debug_print("Attempting manual merge method...")
            return manual_merge_tiles(tile_data)
        except Exception as e2:
            debug_print(f"Manual merge also failed: {e2}")
            largest_tile = max(tile_data, key=lambda x: len(x[0]))
            return largest_tile[0]

@router.post("/wcs/clip", response_class=Response)
async def clip_raster_with_polygon(request: WCSClipRequest):
    """
    Clip a WCS coverage with a GeoJSON polygon/multipolygon and return as zipped GeoTIFF.

    Args:
        request: Request containing GeoServer URL, coverage ID, GeoJSON, and optional ZIP filename

    Returns:
        Zipped GeoTIFF file
    """
    try:
        # Parse and validate GeoJSON
        geojson_data = request.geojson
        geometry = None
        geojson_type = geojson_data.get("type")

        if geojson_type in ["MultiPolygon", "Polygon"]:
            geometry = shape(geojson_data)
        elif geojson_type == "Feature":
            feature = Feature(**geojson_data)
            geometry = shape(feature.geometry.dict())
        elif geojson_type == "FeatureCollection":
            collection = FeatureCollection(**geojson_data)
            for feature in collection.features:
                geom = feature.geometry
                if geom.type in ["Polygon", "MultiPolygon"]:
                    geometry = shape(geom.dict())
                    break

        if not geometry:
            raise HTTPException(
                status_code=400,
                detail="Could not extract Polygon or MultiPolygon from provided GeoJSON"
            )

        # Set up coordinate transformation
        transformer = pyproj.Transformer.from_crs(
            request.input_crs,
            request.wcs_crs,
            always_xy=True
        )

        def project_func(x, y):
            return transformer.transform(x, y)

        project = partial(transform, project_func)
        transformed_geometry = project(geometry)

        # Get bounding box for WCS request
        minx, miny, maxx, maxy = transformed_geometry.bounds
        debug_print(f"Geometry bounds in {request.wcs_crs}: {minx}, {miny}, {maxx}, {maxy}")

        # Normalize GeoServer URL
        geoserver_url = request.geoserver_url.rstrip("/")

        # Download WCS coverage using concurrent approach
        geotiff_data = get_wcs_coverage_concurrent(
            geoserver_url,
            request.coverage_id,
            minx, miny, maxx, maxy
        )

        # Clip the raster with the geometry
        with MemoryFile(geotiff_data) as memfile:
            with memfile.open() as src:
                out_image, out_transform = rasterio.mask.mask(
                    src,
                    [transformed_geometry],
                    crop=True,
                    filled=False
                )
                out_meta = src.meta.copy()
                out_meta.update({
                    'driver': 'GTiff',
                    'height': out_image.shape[1],
                    'width': out_image.shape[2],
                    'transform': out_transform
                })

                # Write clipped raster to memory
                with MemoryFile() as clipped_memfile:
                    with clipped_memfile.open(**out_meta) as dst:
                        dst.write(out_image)
                    clipped_bytes = clipped_memfile.read()

        # Generate a unique identifier for filenames
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID for brevity

        # Create zip file in memory
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Use unique GeoTIFF filename
            tiff_filename = f"{request.coverage_id}_clip_{unique_id}.tif"
            zf.writestr(tiff_filename, clipped_bytes)

        memory_file.seek(0)

        # Determine ZIP filename
        zip_filename = request.zip_filename
        if zip_filename:
            # Ensure .zip extension
            if not zip_filename.lower().endswith('.zip'):
                zip_filename += '.zip'
            debug_print(f"Using user-specified ZIP filename: {zip_filename}")
        else:
            # Use unique ZIP filename
            zip_filename = f"{request.coverage_id}_clip_{unique_id}.zip"
            debug_print(f"Using default unique ZIP filename: {zip_filename}")

        return Response(
            content=memory_file.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{zip_filename}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Error processing WCS clip request")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

