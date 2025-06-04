from fastapi import FastAPI
import pandas as pd
import time
from pyspark.sql.functions import col
import json
from controller.defaults.queryDefaults import QueryDefaults, SparkStartUp
from dotenv.main import load_dotenv
from sqlalchemy.orm import Session
import os

class QueryFunctions:
    def __init__(self):
        self.env = load_dotenv('./fastApi.env')
        
    
    def its_query_builder(its_datasets):
        its_queries = []
        for key in its_datasets:
            filter = its_datasets[key][0]
            full_its_table_name = 'its.' + key
            query = f'''select geo_3310
                    from {full_its_table_name}
                    where {filter}
                '''
            its_queries.append(query)

        union_query = f'''
            its_polys AS (
                select geo_3310
                from (
                    {' UNION ALL '.join(its_queries)}
                ) polys
            )'''
        
        return union_query
        
    def its_rrk_query(idx, rrk_table_name, rrk_filter, its_query):
        full_rrk_table_name = 'fdh.' + rrk_table_name 

        rrk_min, rrk_max = rrk_filter
        query_parts = []
        rid_name = f'rids_{str(idx)}'

        query_parts.append(f'''{rid_name} AS (
                    Select DISTINCT rrk_ids.rid
                    FROM (
                        Select (ST_ValueCount(rrk.rast, 1)).*, rrk.rid
                        FROM {full_rrk_table_name} as rrk
                    ) as rrk_ids
                    WHERE rrk_ids.value >= {rrk_min}
                    AND rrk_ids.value <= {rrk_max}
                    AND rrk_ids.count != 0
                ), nodata_{str(idx)} as (
                    Select nodatavalue 
                    FROM (
                        SELECT (ST_BandMetaData(rrk.rast)).nodatavalue
                        FROM {full_rrk_table_name} as rrk
                    ) as nodata
                    where nodata is not null
                    LIMIT 1
                    )''')
        
        if idx == 0:
            query_parts.append(f""" 
                {its_query},
                clipped_{str(idx)} AS (
                select polygons.geom
                FROM (                    
                    select (ST_DumpAsPolygons(ST_Union(ST_Clip(original.rast, 1, poly.geo_3310, nodata_{str(idx)}.nodatavalue)))).*
                    from its_polys as poly, 
                    {full_rrk_table_name} as original, {rid_name}, nodata_{str(idx)}
                    where ST_Intersects(original.rast, poly.geo_3310)
                    AND {rid_name}.rid = original.rid
                    GROUP BY original.rast
                ) as polygons
                where polygons.val >= {rrk_min}
                AND polygons.val <= {rrk_max}
            ) """)
        else:
            clipped_name = f'clipped_{str(idx-1)}'
            query_parts.append(f''' 
                clipped_{str(idx)} AS (
                select polygons.geom
                FROM (
                    select (ST_DumpAsPolygons(ST_Union(ST_Clip(original.rast, 1, poly.geom, nodata_{str(idx)}.nodatavalue)))).* 
                    -- select (ST_DumpAsPolygons(ST_Union(ST_Clip(original.rast, 1, ST_Buffer(poly.geom, 0.0), nodata_{str(idx)}.nodatavalue)))).* 
                    from {clipped_name} as poly,  nodata_{str(idx)},
                    {full_rrk_table_name} as original, {rid_name}
                    where ST_Intersects(original.rast, poly.geom)
                    AND {rid_name}.rid = original.rid
                    GROUP BY original.rast
                ) as polygons
                where polygons.val >= {rrk_min}
                AND polygons.val <= {rrk_max}
            ) ''')

        return ', '.join(query_parts)
    
    def rrk_query(idx, rrk_table_name, rrk_filter):
        full_rrk_table_name = 'fdh.' + rrk_table_name 
        rid_name = f'rids_{str(idx)}'
        rrk_min, rrk_max = rrk_filter

        query_parts = f'''{rid_name} AS (
                    Select DISTINCT rrk_ids.rid
                    FROM (
                        Select (ST_ValueCount(rrk.rast, 1)).*, rrk.rid
                        FROM {full_rrk_table_name} as rrk
                    ) as rrk_ids
                    WHERE rrk_ids.value >= {rrk_min}
                    AND rrk_ids.value <= {rrk_max}
                    AND rrk_ids.count != 0
                ), nodata_{str(idx)} as (
                    Select nodatavalue 
                    FROM (
                        SELECT (ST_BandMetaData(rrk.rast)).nodatavalue
                        FROM {full_rrk_table_name} as rrk
                    ) as nodata
                    where nodata is not null
                    LIMIT 1
                    )'''
        
        if idx == 0:
            query_parts += f''', poly_{str(idx)} as (
                select polygons.geom
                FROM (
                    Select (ST_DumpAsPolygons(rrk.rast, 1)).*
                    FROM {full_rrk_table_name} as rrk, {rid_name}
                    where {rid_name}.rid = rrk.rid
                ) as polygons
                WHERE polygons.val >= {rrk_min}
                AND polygons.val <= {rrk_max}
            ) '''
        else:
            polygon_table = 'clipped_' + str(idx-1) if idx > 1 else 'poly_' + str(idx-1)
            query_parts += f''' 
                , clipped_{str(idx)} as (
                    select polygons.geom
                    From (
                        select (ST_DumpAsPolygons(ST_Union(ST_Clip(original.rast, 1, ST_Buffer({polygon_table}.geom, 0.0), nodata_{str(idx)}.nodatavalue, true)))).*
                        from {polygon_table}, {full_rrk_table_name} as original,
                        rids_{str(idx)}, nodata_{str(idx)}
                        where ST_Intersects(original.rast, {polygon_table}.geom)
                        and original.rid = rids_{str(idx)}.rid
                        Group by original.rast
                    ) as polygons
                    where polygons.val >= {rrk_min}
                    and polygons.val <= {rrk_max}
                )
            ''' 
        
        return query_parts
    
    ## (case when ST_isValid(poly_0.geom) then poly_0.geom else ST_MakeValid(poly_0.geom) end)
        
     
    def query_results(datasets, db: Session):

        all_results = []
        fullTime = time.time()
        # sparkStartUp = SparkStartUp()
        # spark = sparkStartUp.spark
        # spark_url = sparkStartUp.url
        queryDefaults = QueryDefaults()
        query_results = []

        # check if all datasets are on server
        for dataset_name in datasets:
            details = datasets[dataset_name]
            if dataset_name.lower().startswith('rrk'): 
                table_name = 'fdh.' + dataset_name
            else:
                table_name = 'its.' + dataset_name

            query = f'select 1 from {table_name}'
            
            try: 
                data = db.execute(query).fetchone()
            except Exception as error:
                return { 'error' : f'The data for {table_name} has not been uploaded to our server. Check back at a later date for access.'}


        # filter rrk and its datasets
        rrk_datasets = { key: datasets[key] for key in datasets.keys() if key.startswith('rrk')}
        its_datasets = { key: datasets[key] for key in datasets.keys() if key.startswith('its') }

        rrk_idx = 0
        query = []
        its_query = ''
        if len(its_datasets.items()) > 0:
            its_query = QueryFunctions.its_query_builder(its_datasets)

            for key, value in rrk_datasets.items():
                rrk_raster_cte = QueryFunctions.its_rrk_query(rrk_idx, key, value, its_query)
                query.append(rrk_raster_cte)
                rrk_idx += 1

            proj = QueryDefaults().projection
            final_select = "SELECT '{" + '"type":"Feature","geometry":' + "' || ST_AsGeoJSON(ST_Transform(clipped_" + str(rrk_idx-1) + ".geom, " + str(proj)\
                        + ")) || '" + ', "properties": { "name": "Intersection" } }' + "' as geom from clipped_" + str(rrk_idx-1)
            # final_select = "select '{" + '"type":"Feature","geometry":' + "' || ST_AsGeoJSON(ST_Transform(clipped_" + str(rrk_idx-1) + ".geom, " + str(proj) + "))'  }' as geom from clipped_" + str(rrk_idx-1)
            
            final_query = 'WITH ' + ', '.join(query) + final_select

        else:
            for key, value in rrk_datasets.items():
                rrk_raster_cte = QueryFunctions.rrk_query(rrk_idx, key, value)
                query.append(rrk_raster_cte)
                rrk_idx += 1
            
            proj = QueryDefaults().projection
            if(rrk_idx > 1):
                # final_select = f""" select ST_AsGeoJSON(ST_Transform(clipped_{(rrk_idx-1)}.geom, {proj})) as geom from clipped_{(rrk_idx-1)} where ST_IsValid(clipped_{(rrk_idx-1)}.geom)"""
                 final_select = "SELECT '{" + '"type":"Feature","geometry":' + "' || ST_AsGeoJSON(ST_Transform(clipped_" + str(rrk_idx-1) + ".geom, " + str(proj)\
                        + ")) || '" + ', "properties": { "name": "Intersection" } }' + "' as geom from clipped_" + str(rrk_idx-1) + ' where ST_IsValid(clipped_' + str(rrk_idx-1) + '.geom)'
                
            final_query = 'WITH ' + ', '.join(query) + final_select


        try: 
            results = db.execute(final_query).fetchall()
        except Exception as error:
            return { 'error' : 'An error was encountered when performing the intersection.'}
        
        final_results = []
        try:
            final_results.extend([json.loads(row.geom) for row in results])
        except:
            return {'error': 'Out of memory issue'}

        # spark.stop()

        return final_results
    

    def get_rrk_download(datasets, db:Session):
        

        return


    def get_download_data(datasets, db:Session):

        all_results = []
        fullTime = time.time()
        results = []
        proj = QueryDefaults().projection

        # check if all datasets are on server
        for dataset_name in datasets:
            details = datasets[dataset_name]
            # table_name = ''
            if dataset_name.lower().startswith('rrk'): 
                table_name = 'fdh.' + dataset_name
            else:
                table_name = 'its.' + dataset_name
            
            query = f'select 1 from {table_name}'
                # query = f'select geometry from fdh."{dataset_name} LIMIT 1"'

            # else:
            #     table_name = dataset_name.split(':')[1].lower()
            #     query = f"select 1 from fdh.{table_name}"

            try: 
                # query = f'select 1 from fdh.housingburdenpctl'
                data = db.execute(query).fetchone()               
            except Exception as error:
                return { 'error' : f'The data for {table_name} has not been uploaded to our server. Check back at a later date for access.'}

        for dataset_name in datasets:
            details = datasets[dataset_name]
            query = ''
            if dataset_name.lower().startswith('rrk') == False: # is its dataset
                table_name = dataset_name
                where_clause = details[0]

                # query = f'select ST_AsGeoJSON(poly.geo_3310) as geom from fdh."{table_name}" as poly' + \
                #         f" where to_date(to_timestamp(poly.{details[0]} / 1000)::TEXT, 'yyyy-mm-dd HH24:MI:SS') > '{details[2]}'" +\
                #         f" AND to_date(to_timestamp(poly.{details[0]} / 1000)::TEXT, 'yyyy-mm-dd HH24:MI:SS') <  '{details[3]}' "
                # query = f'select ST_AsGeoJSON(geo_3310) as geom from fdh."{table_name}" where ' + where_clause
                
                # added string for proper ArcGIS compatability
                query = "SELECT '{" + '"type":"Feature","geometry"' + ":' || ST_AsGeoJSON(ST_Transform(geo::geometry, " \
                    + proj + ')) || ' + "'," + '"properties": { "name": "Interagency Tracking System" } }' + "' as geom from " + f'its.{table_name} where ' + where_clause

            else: # is rrk
                table_name = dataset_name
                        # SELECT ST_AsGeoJSON(polygons.geom) as geom
                query = f'''WITH rids AS (
                            Select valCounts.rid
                            FROM (
                                Select (ST_ValueCount(tbl.rast, 1)).*, tbl.rid
                                FROM fdh.{table_name} as tbl
                            ) as valCounts
                            WHERE valCounts.value >= {details[0]}
                            AND valCounts.value <= {details[1]}
                        )''' + \
                        "SELECT '{" + '"type":"Feature","geometry":' + "' || ST_AsGeoJSON(ST_Transform(polygons.geom, " + proj\
                        + ")) || '" + ', "properties": { "name": "' + table_name + '"' + " } }' as geom " + f'''FROM (
                            SELECT (ST_DumpAsPolygons(tbl.rast)).* 
                            FROM fdh.{table_name} as tbl, rids
                            where rids.rid = tbl.rid
                            ) as polygons
                        WHERE polygons.val >= {details[0]}
                        AND polygons.val <= {details[1]}'''

            try: 
                table_results = db.execute(query).fetchall()               
            except Exception as error:
                return { 'error' : 'An error was encountered when performing the intersection.'}
        
            try:
                results.extend([json.loads(row.geom) for row in table_results])
            except:
                return {'error': 'Out of memory issue'}

        return results
