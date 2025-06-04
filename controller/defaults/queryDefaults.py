from fastapi import FastAPI
import pandas as pd
import numpy as np
import time
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DateType, LongType, DoubleType
from dotenv.main import load_dotenv
import os

class SparkStartUp:    
    def __init__(self):
        
        load_dotenv('./fastApi.env')

        self.spark = SparkSession.builder \
            .appName("PostgreSQL Connection with PySpark") \
            .config("spark.jars", os.environ['postgresql_jar_path']) \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .getOrCreate()

        self.url = "jdbc:postgresql://" + os.environ['base_host'] + ":5432/" + os.environ['db_name']

class QueryDefaults:
    def __init__(self):       

        self.projection = "4326"

        self.querySchema = StructType([
                        StructField("geom", StringType())
                    ])


