from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import geopandas as gpd
import tempfile
import os
import shutil
import zipfile
import logging
from typing import List
import traceback


router = APIRouter(tags=["Utility"], prefix='/Utility')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_shapefile(directory):
    """
    Recursively search for .shp files in a directory and its subdirectories
    
    Args:
        directory: Directory to search in
        
    Returns:
        Path to the first .shp file found, or None if no .shp file is found
    """
    for root, dirs, files in os.walk(directory):
        shp_files = [os.path.join(root, f) for f in files if f.endswith('.shp')]
        if shp_files:
            return shp_files[0]
    return None


@router.post("/convert/shapefile_to_geojson", response_class=JSONResponse)
async def convert_shapefile_to_geojson(file: UploadFile = File(...)):
    """
    Convert a shapefile (in zip format) to GeoJSON
    
    - **file**: A zip file containing the required shapefile components (.shp, .shx, .dbf)
                The shapefile can be either directly in the zip root or within a subfolder
    
    Returns a GeoJSON representation of the data
    """
    # Check if file is a zip file
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Please upload a zip file containing shapefile components")
    
    # Create a temporary directory to extract the zip file
    temp_dir = tempfile.mkdtemp()
    temp_zip_path = os.path.join(temp_dir, "shapefile.zip")
    
    try:
        # Save the uploaded zip file
        with open(temp_zip_path, "wb") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
        
        # Extract the zip file
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the .shp file in the extracted contents (including subdirectories)
        shp_path = find_shapefile(temp_dir)
        
        if not shp_path:
            raise HTTPException(status_code=400, detail="No shapefile (.shp) found in the zip file")
        
        logger.info(f"Found shapefile at: {shp_path}")
        
        # Read the shapefile with geopandas
        gdf = gpd.read_file(shp_path)
        
        # Check if the shapefile has data
        if gdf.empty:
            raise HTTPException(status_code=400, detail="The shapefile does not contain any data")
        
        # Convert to GeoJSON (ensuring WGS84 projection for standard GeoJSON)
        gdf = gdf.to_crs(epsg=4326)
        geojson_data = gdf.__geo_interface__
        
        return geojson_data
        
    except Exception as e:
        logger.error(f"Error processing shapefile: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing shapefile: {str(e)}")
    
    finally:
        # Clean up the temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
