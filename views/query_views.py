from fastapi import FastAPI, APIRouter, Request, Body, Response
import json
from pyspark.sql.functions import col
from fastapi import FastAPI, APIRouter, HTTPException, Request, Body, Response
import json
from controller.download import DownloadMap
from controller.db import get_db
from fastapi import Depends
from sqlalchemy.orm import Session
from controller.manager import manager
from fastapi.encoders import jsonable_encoder
import datetime
from models.wfr_database import User

import zipfile
from io import BytesIO
from fastapi.responses import FileResponse, StreamingResponse
import orjson

import os
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, exc
from sqlalchemy.orm.session import sessionmaker
from dotenv.main import load_dotenv

from controller.defaults.queryDefaults import QueryDefaults
from controller.dynamic_query.query_builder import QueryFunctions

router = APIRouter(
    prefix='/v1/query_views',
    tags = ['query execution']
)

load_dotenv('fastapi/.env')

@router.get("/get_query", include_in_schema=False)
async def get_query(data: str, user=Depends(manager), db: Session = Depends(get_db)):

    if user == None: 
         raise HTTPException(status_code=500, detail="User Not Found")
    
    request =json.loads(data)

    datatable_data = {}
    for key in request:
        workspace, filename = key.split(':')
        if workspace == 'ITS':
            table = filename
        else:
            table = workspace.lower() + '_' + filename

        datatable_data[table] = request[key]

    datasetKeys = [x if 'million_acres' not in x.lower() else 'Million Acres Strategy Treatments' for x in list(request.keys())]
    datasetKeys = list(set(datasetKeys))

    result = QueryFunctions.query_results(datatable_data, db)

    if 'error' in result:
        return {'status': 'failed', 'message': result['error']}
    
    final = {
        'allFeatures':{
            'type': 'FeatureCollection',
            'crs': {
                    'type': 'name',
                    'properties': {
                        'name': 'EPSG:' + QueryDefaults().projection,
                    },
                },
            'features': result
        },
        "datasetNames" : datasetKeys
    }

    final_result = orjson.dumps(final, option=orjson.OPT_INDENT_2).decode()

    return Response(content=final_result, media_type="application/json")


@router.get("/download", include_in_schema=False)
async def download(data: str, fileName: str, user=Depends(manager), db: Session = Depends(get_db)):

    if user == None: 
         raise HTTPException(status_code=500, detail="User Not Found")
    
    request =json.loads(data)

    result = QueryFunctions.get_download_data(request, db)        
    
    if 'error' in result:
        return {'status': 'failed', 'message': result['error']}

    final = {
            'type': 'FeatureCollection',
            'crs': {
                    'type': 'name',
                    'properties': {
                        'name': 'EPSG:' + str(QueryDefaults().projection),
                    },
                },
            'features': result
        }
    
    final_result = orjson.dumps(final, option=orjson.OPT_INDENT_2).decode()

    return zipfiles(final_result, fileName)

@router.get("/create_shapefile", include_in_schema=False)
async def create_shapefile(data: str, filename: str, user=Depends(manager), db: Session = Depends(get_db)):

    if user == None: 
         raise HTTPException(status_code=500, detail="User Not Found")
    
    request =json.loads(data)

    query = DownloadMap.build_query(request, 'shapefile')

    engine = db.get_bind()
    host = engine.engine.url.host
    user = engine.engine.url.username
    port = engine.engine.url.port
    password = engine.engine.url.password
    database = engine.engine.url.database

    temp_path = DownloadMap.create_shapefile(query, host, port, user, password, database)

    zip_path = DownloadMap.compress(temp_path, filename)

    zip_filename = zip_path.split('/')[-1]

    CHUNK_SIZE = 1024 * 1024  # = 1MB - adjust the chunk size as desired
    headers = {'Content-Disposition': f'attachment; filename="{zip_filename}"'}

    return StreamingResponse(DownloadMap.iterfile(zip_path, CHUNK_SIZE), headers=headers, media_type='application/zip')



@router.post("/download_shapefile", include_in_schema=False)
async def download_shapefile(file_path:str, user=Depends(manager), db: Session = Depends(get_db)):

    if user == None: 
         raise HTTPException(status_code=500, detail="User Not Found")
    
    zip_filename = file_path.split('/')[-1]

    CHUNK_SIZE = 1024 * 1024  # = 1MB - adjust the chunk size as desired
    headers = {'Content-Disposition': f'attachment; filename="{zip_filename}"'}

    return StreamingResponse(DownloadMap.iterfile(file_path, CHUNK_SIZE), headers=headers, media_type='application/zip')



def zipfiles(final_result, fileName: str):
    io = BytesIO()
    zip_sub_dir = fileName
    zip_filename = "%s.zip" % zip_sub_dir    
    with zipfile.ZipFile(io, mode='w', compression=zipfile.ZIP_DEFLATED) as zip:
        zip.writestr(fileName + '.json', data=final_result)
        #close zip
        zip.close()
    response = StreamingResponse(
        iter([io.getvalue()]),
        media_type="application/x-zip-compressed",
        headers = { "Content-Disposition":f"attachment;filename=%s" % zip_filename,  "Content-Length": str(io.getbuffer().nbytes)}
    )

    io.close()

    return response