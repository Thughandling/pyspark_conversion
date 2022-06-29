"""
SparkSession -> DataFrame 생성을 위한
SparkContext -> RDD 생성을 위한


collect() → 해당 데이터 프레임의 모든 row 반환

- [sqlContext.read](http://sqlContext.read) → DataFrame읽을때 사용  [DataFrameReader를 반환]**

[sqlContext.read](http://sqlContext.read).format(’com.databricks.spark.csv’)  → csv파일을 읽어야 할때 사용 (in ver pyspark ver1.0)**


- data.withColumn(’date’, substring(data.TIMESTAMP,1,10)) →data에 “date”라는 이름으로 오른쪽 새로운 값으로 컬럼 생성/변환

⇒ df.loc[:,”date”] = substring(data.TIMESTAMP,1,10)


- data.registerTempTable("df_tmp") → data(RDD)를 “df_tmp”라는 임시 테이블로 카피(값 같음)

⇒ df.copy()


- sqlContext.sql → 데이터 프레임에 SQL 쿼리 날릴수 있음

⇒ data = sqlContext.sql("select * from df_tmp")


- data.dropDuplicates(['DriveSerialNumber', 'date']) 열값을 비교하여 중복되는 데이터 **행** 제거

+ keep = 'last' 옵션 → 맨마지막 값 남김


- spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()clear

emptyRDD = spark.sparkContext.emptyRDD()

→ Creates Empty RDD
"""
import datetime
import os as os
import time

## MTP / pyspark 라이브러리 호출
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql import types
from pyspark.sql.types import *
from pyspark.sql.functions import *
# from pyspark.sql.functions import lower, upper, col
# from pyspark.sql.functions import when
import numpy as np
from sklearn.neighbors import KernelDensity 
from sklearn import metrics
#import pandas as pd #as-is
from pyspark.sql import Row, context
import pandas as pd

spark = SparkSession.builder.appName('Basics').getOrCreate()

# pd.read_csv -> pyspark
read_pd = pd.read_csv('datasets_directory')
read_ps = spark.read.format('com.databricks.spark.csv').options(header='true', inferSchema='true').load('datasets_directory')


# Empty DataFrame
generate_empty_df =  pd.DataFrame()
#판다스
generate_empty_pyspark = spark.sparkContext.emptyRDD()
generate_empty_pyspark = spark.createDataFrame(generate_empty_pyspark) # empty RDD -> DataFrame 
#Pyspark 






