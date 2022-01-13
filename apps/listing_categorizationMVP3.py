from joblib.parallel import parallel_backend
import pyspark
import pandas as pd
import numpy as np
import dateparser
import datetime
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import resource
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
import sqlalchemy as sqla
import sklearn as skl
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import synapse
#import synapse.ml
#from synapse.ml.lightgbm import *
#from synapse.ml.lightgbm import LightGBMClassifier, LightGBMClassificationModel, LightGBMRegressor
#from synapse.ml.train import ComputeModelStatistics

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array
#from pyspark.sql.functions import from_unixtime, unix_timestamp, date_format, to_date, col
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, BooleanType, DateType, ArrayType

from joblibspark import register_spark
from joblib import Parallel, delayed

def init_spark():
    spark = SparkSession.builder\
        .appName("Revenue Models")\
        .config("spark.jars", "/opt/spark-apps/postgresql-42.2.22.jar")\
        .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5") \
        .getOrCreate()
    sc = spark.sparkContext
    return spark,sc

def main():
    register_spark()

    url = "jdbc:postgresql://demo-database:5432/mta_data"
    properties = {
    "user": "postgres",
    "password": "casa1234",
    "driver": "org.postgresql.Driver"
    }
    file = "/opt/spark-data/input_spark_1k.csv"
    spark,sc = init_spark()

    import synapse.ml
    clean_data = spark.read.load(file,format = "csv", inferSchema="true", header="true")


    #clean_data.printSchema()
    #clean_data.show()
    #clean_data.select('date').show()

    past_df = clean_data.filter(clean_data.date < pd.to_datetime('today'))
    future_df = clean_data.filter(clean_data.date > pd.to_datetime('today'))

    quantiles = [0.001, 0.2, 0.4, 0.6, 0.8, 0.999]
    predictions = {}

    past_df.printSchema()
    #for quantile in quantiles:
    #    qr = QuantileRegressor(quantile=quantile, alpha=0)
    #    y_pred = qr.fit(past_df, future_df).predict(past_df)
    #    predictions[quantile] = y_pred

    #df = spark.createDataFrame([past_df.select('bedrooms','month')],[past_df.select('price_string').alias("Y")])
    #vecAssembler = VectorAssembler(outputCol="price")
    #vecAssembler.setInputCols(['bedrooms','month'])
    #vecAssembler.transform(df).head().price

    vecAssemblerX = VectorAssembler(outputCol="X")
    vecAssemblerX.setInputCols(['month','bedrooms'])
    X = vecAssemblerX.transform(past_df).select(vector_to_array(F.col('X'))).show()
    #X = F.array(F.col(X))
    #X = F.array(vecAssemblerX.transform(past_df).select(vector_to_array(F.col('X'))))
    #past_df_new = X.withColumn(F.col('X'))
    #X = vecAssemblerX.getInputCols()
    #vecAssemblerY = VectorAssembler(outputCol="Y")
    #vecAssemblerY.setOutputCol(["price_string"])
    #vecAssemblerY.transform(past_df).show()
    print(type(X))

    #X = np.array(pd.to_numeric(past_df.select('bedrooms','month').collect(), errors='coerce'))
    #X = past_df.select('X').show()
    #Y = np.array(past_df.select('price_string').collect())
    #Y = past_df.select(vector_to_array(F.col('price_string'))).show()
    Y = past_df.select(F.array('price_string')).show()
    #Y.replace(to_replace=pd.NA, value=None, inplace=True)
    qr = QuantileRegressor(quantile=0.5,alpha=0)
    print(type(Y))
    #print(Y)

    #with parallel_backend('spark',n_jobs=2):
    #model = qr.fit(X,Y).predict(X)
    #print(model)

    #plt.scatter(Y,model, color = "red")
    #plt.scatter(Y, Y, color = "black", marker="+")
    #plt.suptitle('Price model', fontsize=20)
    #plt.xlabel('Price [$]', fontsize=16)
    #plt.ylabel('Price [$]', fontsize=16)
    #plt.show()
    #plt.savefig('/opt/spark-data/teste2.png')

    #cnt_cond = lambda cond: F.sum(F.when(cond, 1).otherwise(0))
    
    print('Number of apartment dates above model:', len(Y[(Y.flatten()>model)]))
    print('Number of apartment dates below model:', len(Y[(Y.flatten()<model)]))


if __name__ == '__main__':
  main()