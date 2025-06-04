from fastapi import FastAPI,  APIRouter, Request, Body, Response
# import pandas as pd
import json
# import psycopg2
# from passlib.context import CryptContext
# import bcrypt
# import smtplib, ssl
from dotenv.main import load_dotenv
import os
import tempfile
import zipfile
import subprocess


load_dotenv('fastapi/env')

class DownloadMap:

    def iterfile(file_path, CHUNK_SIZE):
        with open(file_path, 'rb') as f:
            while chunk := f.read(CHUNK_SIZE):
                yield chunk
            else:
                DownloadMap.delete_file(file_path)

    def build_query(filters, output_file):

        raster_dbs = [x for x in filters.keys() if isinstance(filters[x], list)]
        vector_db = [x for x in filters.keys() if isinstance(filters[x], str)]
        
        select_polygon_clause = ''
        vector_base_filter = ''
        vector_table = ''
        cte_list = []
        
        if len(vector_db) == 1:
            vector_db = vector_db[0]

            raw_filter = filters[vector_db]

            # no filter is "N/A"
            if raw_filter == 'N/A':
                vector_filter = 'true'
            else:
                filter_list = filters[vector_db].split(' AND ')

                vector_filter = " AND ".join(['ca.' + x.strip() for x in filter_list])

            vector_dataset = vector_db.replace(':', "_")

            vector_base_filter = 'AND ST_Intersects(rrk.rast, bound.geometry)'
            vector_table = 'boundary as bound,'

            vector_cte = f"""
                boundary AS (
                    Select ST_Transform(ca.geom, 3310) as geometry
                    FROM fdh.{vector_dataset} as ca
                    where {vector_filter}
                )
            """

            cte_list.append(vector_cte)

            select_polygon_clause = '(ST_DumpAsPolygons(ST_Union(ST_Clip(rrk.rast, 1, ST_Buffer(bound.geometry, 0.0), nodata.nodatavalue, true)))).*'
        
        else:
            select_polygon_clause = '(ST_DumpAsPolygons(rrk.rast, 1)).*'
        
        for raster in raster_dbs:

            select_clause = None
            if output_file == 'shapefile':
                select_clause = DownloadMap.select_shapefile()
            elif output_file == 'geojson':
                select_clause = DownloadMap.select_geojson(raster)

            filter_vals = filters[raster]
            rrk_dataset = raster.replace(':', "_")

            if len(rrk_dataset) > 63:
                rrk_dataset = rrk_dataset[:63]

            rrk_ctes = f'''
                rrk_ids AS (
                    Select DISTINCT rrk_ids.rid
                    FROM (
                        Select (ST_ValueCount(rrk.rast, 1)).*, rrk.rid
                        FROM fdh.{rrk_dataset} as rrk
                    ) as rrk_ids
                    WHERE rrk_ids.value >= {filter_vals[0]}
                    AND rrk_ids.value <= {filter_vals[1]}
                    AND rrk_ids.count != 0
                ),
                nodata as (
                    Select nodatavalue 
                    FROM (
                        SELECT DISTINCT (ST_BandMetaData(rrk.rast)).nodatavalue
                        FROM fdh.{rrk_dataset} as rrk
                    ) as nodata
                    where nodata is not null
                    LIMIT 1
                ) 
            '''

            cte_list.append(rrk_ctes)

            query = f'''with {', '.join(cte_list)}
                {select_clause}
                From (
                    select {select_polygon_clause}
                    from fdh.{rrk_dataset} as rrk, {vector_table} nodata, rrk_ids
                    where rrk.rid = rrk_ids.rid
                    {vector_base_filter}
                ) as polygons
                where polygons.val >= {filter_vals[0]}
                AND polygons.val <= {filter_vals[1]}
                AND ST_IsValid(ST_Transform(polygons.geom, 4326))
            '''

            # query = f'''with boundary AS (
            #     Select ST_Transform(ca.geometry, 3310) as geometry
            #     FROM fdh.{vector_dataset} as ca
            #     where ca.{vector_filter}
            #     ),
            #     rrk_ids AS (
            #         Select DISTINCT rrk_ids.rid
            #         FROM (
            #             Select (ST_ValueCount(rrk.rast, 1)).*, rrk.rid
            #             FROM fdh.{rrk_dataset} as rrk
            #         ) as rrk_ids
            #         WHERE rrk_ids.value >= {filter_vals[0]}
            #         AND rrk_ids.value <= {filter_vals[1]}
            #         AND rrk_ids.count != 0
            #     ),
            #     nodata as (
            #         Select nodatavalue 
            #         FROM (
            #             SELECT DISTINCT (ST_BandMetaData(rrk.rast)).nodatavalue
            #             FROM fdh.{rrk_dataset} as rrk
            #         ) as nodata
            #         where nodata is not null
            #         LIMIT 1
            #     ) 
                #  {select_clause}
                # From (
                #     select (ST_DumpAsPolygons(ST_Union(ST_Clip(rrk.rast, 1, ST_Buffer(bound.geometry, 0.0), nodata.nodatavalue, true)))).*
                #     from fdh.{rrk_dataset} as rrk, boundary as bound, nodata, rrk_ids
                #     where rrk.rid = rrk_ids.rid
                #     AND ST_Intersects(rrk.rast, bound.geometry)
                # ) as polygons
                # where polygons.val >= {filter_vals[0]}
                # AND polygons.val <= {filter_vals[1]}
                # AND ST_IsValid(ST_Transform(polygons.geom, 4326))'''

            # query = query.strip()

            return query 
        

    def select_geojson(raster):
        return "SELECT '{" + '"type":"Feature","geometry":' + "' || ST_AsGeoJSON(ST_Transform(polygons.geom, 3310" \
                        + ")) || '" + ', "properties": { "name": "' + raster + '"' + ', "value": polygons.val' + "} }' as geometry "
    
    def select_shapefile():
        return 'SELECT ST_Transform(polygons.geom, 4326) as geometry, polygons.val as polygon_value'

    def create_geojson(query, db):

        try: 
            results = db.execute(query).fetchall()
        except Exception as error:
            return { 'error' : 'An error was encountered when performing the download.'}
        
        results_json = [json.loads(row.geom) for row in results]

        final = {
            'type': 'FeatureCollection',
            'crs': {
                    'type': 'name',
                    'properties': {
                        'name': 'EPSG:' + str(3310),
                    },
                },
            'features': results_json
        }

        return final 


    def create_shapefile(query, host, port, user, password, database):
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.shp')
        
        cmd = f'pgsql2shp -f {tmp.name} -h {host} -u {user} -P {password} {database} "{query}"'
        # cmd = f'pgsql2shp -f {tmp.name} -h {host} -p {str(port)} -u {user} -P {password} {database} "{query}"'

        os.system(cmd)

        return tmp.name
    

    def delete_file(file_path):

        file_names = DownloadMap.get_shapefile_paths(file_path)

        # add zip file path to delete
        file_names.append(file_path)

        for file_path in file_names:
            os.system(f'rm -rf {file_path}')


    def get_shapefile_paths(folder_path):
        file_name = folder_path.split('/')[-1]

        file_names = []
        for suffix in ['prj', 'shp', 'cpg', 'shx','dbf']:
            file_names.append(folder_path + '/' + file_name + "." + suffix)

        return file_names
        
    def compress(file_path, file_name):

        file_names = DownloadMap.get_shapefile_paths(file_path)
        # no_extention_path = file_path[:-4]
        compression = zipfile.ZIP_DEFLATED
        zip_path = file_path + ".zip"
        zf = zipfile.ZipFile(zip_path, mode="w")
        for path in file_names:
            ext = path.split('.')[-1]
            try: 
                zf.write(path, file_name + '.' + ext, compress_type=compression)              
            except: 
                continue

        zf.close()

        return zip_path










