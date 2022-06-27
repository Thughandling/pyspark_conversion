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

## KDEprocess_SMART_2021_v3 셀 수행

# 공용 parameter
path = None
target_path = None


## load_data_w_fw 함수: 파일 불러올 때 FW 정보 포함, FW: ERRORMOD 데이터 필터링
def load_data_w_fw(file):
    global path, target_path
    print(file)

    # 데이터 로드 시 FW 정보 관련 처리
    cols_fw = cols + ['FirmwareRevision']
    types_fw = {'FirmwareRevision': 'category'}
    try:
        types_fw.update(types_dict)
    except TypeError:  # types_dict None일 때 pass
        pass

    print('Loading {}: '.format(file) +
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    start = time.time()
    # 데이터 불러오기: KDE 작업 시 필요한 column들 + FW 정보

    ## MJS / pandas에서 csv 로드 부분을 pyspark로 변경
    # data = pd.read_csv(path + '/' + file, usecols=cols_fw, dtype=types_fw) #as-is
    # data['date'] = data.TIMESTAMP.str[:10] #as-is
    # # Category화를 통한 소요 메모리 절약 #as-is
    # data['date'] = data['date'].astype('category') #as-is
    # # Sort by: DriveSerialNumber, TIMESTAMP #as-is
    # data = data.sort_values(by=['DriveSerialNumber', 'TIMESTAMP']) #as-is

    data = SQLContext.createDataFrame()

    data = sqlContext.read.format('com.databricks.spark.csv')\
                    .options(header='true', inferSchema='true')\
                    .option("mode", "DROPMALFORMED")\
                    .load(file)
    
    # print(data.show(5))
    data = data.withColumn('date',substring(data.TIMESTAMP,1,10)) 
    data.registerTempTable("df_tmp")
    data = sqlContext.sql("select * from df_tmp")
    
    # Sort by: DriveSerialNumber, TIMESTAMP
    data = data.sort('DriveSerialNumber', 'TIMESTAMP')

    # FW: ERRORMOD인 데이터 별도 파일로 저장
    ## MTP / 기존 csv에서 DataFrame로 변경
    # df_err = data[data['FirmwareRevision'] == 'ERRORMOD'] #as-is
    # df_err.to_csv(target_path+ '/' +'FW_ERRORMOD_{}_{}'.format(logitem, file), index=False) #as-is
    data = sqlContext.sql("select * from df_tmp where FirmwareRevision = 'ERRORMOD'")

    # 불러온 데이터에서 FW: ERRORMOD 데이터 제외
    ## MTP / 기존 csv에서 DataFrame로 변경
    # data = data.drop(index=df_err.index) #as-is
    data = sqlContext.sql("select * from df_tmp where FirmwareRevision != 'ERRORMOD'")

    # 각 날짜별 마지막 data만 남김
     ## MTP / pandas to pyspark
    # data = data.drop_duplicates(subset=['DriveSerialNumber', 'date'], keep='last') #as-is
    data = data.dropDuplicates(['DriveSerialNumber', 'date']) 
    end = time.time()
    print('Loading data {} done - Elapsed time: '.format(file) +
          f'{(end-start):.3f} seconds')
    return data

## MTP / 변경 예정 - 시작
## get_kde_dist 함수: 데이터에 대한 분포 추정(KDE)
# numpy로 작업하여 dataframe으로 리턴
def get_kde_dist(array):
    tmp = array[~np.isnan(array)]
    
    # 아주 극단적인 분포에 대해 보다 나은 밀도 추정을 위해 데이터 범위 사전 조절 (0.01 ~ 99.99 percentile)
    lower_b = np.percentile(tmp, 0.01)
    upper_b = np.percentile(tmp, 99.99)
    tmp = tmp[(tmp >= lower_b) & (tmp <= upper_b)]
    tmp = tmp.reshape(-1, 1)
    tmp_avg = np.average(tmp)
    tmp_std = np.std(tmp)

    if tmp_std != 0:
        tmp_norm = (tmp - tmp_avg) / tmp_std
    else:
        tmp_norm = tmp

    x = np.linspace(np.min(tmp_norm), np.max(tmp_norm), num_of_density_layer)[:, np.newaxis]

    # x 값의 구간 첫 값과 마지막 값(데이터 0.01percentile / 99.99percentile)이 동일한 경우
    # KDE 수행하는 의미가 없고 시간만 과도하게 소요되므로 해당 경우 KDE 수행하지 않도록 설정
    if x[-1, 0] - x[0, 0] > 1e-3:  # 부동소수점 이슈 고려하여 설정
        if data_sampling:  # Data sampling 옵션 True일 때, 1/10으로 샘플링하여 KDE 수행
            tmp_norm_sample = tmp_norm.reshape(-1)
            np.random.seed(0)
            tmp_norm_sample = np.random.choice(tmp_norm_sample, len(tmp_norm_sample) // 10)
            tmp_norm_sample = tmp_norm_sample.reshape(-1, 1)
            kde = KernelDensity(kernel='gaussian').fit(tmp_norm_sample)
        else:
            kde = KernelDensity(kernel='gaussian').fit(tmp_norm)

        y_pdf = np.exp(kde.score_samples(x))

    else:
        x[:, 0] = x[-1, 0]  # 부동소수점 이슈 제거
        y_pdf = np.ones(num_of_density_layer)

    # x 생성 시 자연히 크기 순이므로 sort value 불필요 (sort 시 부동소수점 이슈로 인해 index 및 cdf 계산 결과 꼬일 수 있음)
    # as-is
    #density = pd.DataFrame({'x': x[:, 0], 'y_pdf': y_pdf})
    #density['y_cdf'] = [metrics.auc(density[:ind + 2].x, density[:ind + 2].y_pdf) for ind in density.index]
    #density['y_cdf'] = density['y_cdf'] / np.max(density['y_cdf'])
    #density['x'] = density['x'] * tmp_std + tmp_avg

    # aithe - extract density['x'] and density['y_cdf'] from dataframe pyspark
    col_x = [i for i in x[:, 0]]
    col_y_cdf = [j for j in y_pdf]
    density2 = [metrics.auc(col_x[:ind + 2], col_y_cdf[:ind + 2]) for ind in range(0,len(col_x))]
    col_y_pdf = density2 / np.max(col_y_cdf)
    col_x_new = x[:, 0] * tmp_std + tmp_avg
 
    # as-is
    #return density['x'], density['y_cdf']
    # aithe - 
    return col_x_new, col_y_pdf


# Crit list 출력
def get_crit_list(item, df_pivot):
    ## MTP / pandas to pyspark  
    # tmp_info = crit_warn_info.loc[item.lower()].str.split(',') #as-is
    tmp_info_LOW = crit_warn_info.filter(crit_warn_info.ITEM == item.lower())
    # dataframe에서 값을 가져올때는 collect를 사용한다.
    tmp_info_LOW = tmp_info_LOW.select(split(tmp_info_LOW.CRITICAL_LOW, ',', 2).alias('CRITICAL_LOW')).collect()
    tmp_info_LOW = tmp_info_LOW[0]
    
    tmp_info_HIGH = crit_warn_info.filter(crit_warn_info.ITEM == item.lower())
    tmp_info_HIGH = tmp_info_HIGH.select(split(tmp_info_HIGH.CRITICAL_HIGH, ',', 2).alias('CRITICAL_HIGH')).collect()
    tmp_info_HIGH = tmp_info_HIGH[0]

    spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
    #Creates Empty RDD
    emptyRDD = spark.sparkContext.emptyRDD()

    schema = StructType([
        StructField(item, StringType(), True)
        ])    

    # crit_list = pd.DataFrame().index #as-is
    # Convert empty RDD to Dataframe
    crit_list = emptyRDD.toDF(schema)
    #ss = SparkSession.builder.getOrCreate()

    for column in df_pivot.columns:
        ## MTP : pandas to pyspark
        # temp = df_pivot[column].dropna() #as-is
        temp = df_pivot.dropna(subset=column)
        ## MTP / 변경 중, *****pandas to pyspark 변경 어려움*****
        # as-is
        '''
        if (tmp_info.CRITICAL_LOW[0] == 'VALUE'):
            crit = temp[temp < float(tmp_info.CRITICAL_LOW[1])].index
            crit_list = crit_list.union(crit)
        if (tmp_info.CRITICAL_HIGH[0] == 'VALUE'):
            crit = temp[temp > float(tmp_info.CRITICAL_HIGH[1])].index
            crit_list = crit_list.union(crit)
        '''
        # aithe - extract crit_list from dataframe pyspark
        if column != 'DriveSerialNumber':
            if (tmp_info_LOW.CRITICAL_LOW[0] == 'VALUE'):
                # crit=temp.filter(temp[column]<float(tmp_info_LOW.CRITICAL_LOW[1])).collect()
                crit=temp.select('DriveSerialNumber').where(temp[column]<float(tmp_info_LOW.CRITICAL_LOW[1]))
                crit_list = crit_list.union(crit).distinct()

            if (tmp_info_HIGH.CRITICAL_HIGH[0] == 'VALUE') :
                # crit=temp.filter(temp[column]<float(tmp_info_HIGH.CRITICAL_HIGH[1])).collect()
                crit=temp.select('DriveSerialNumber').where(temp[column]>float(tmp_info_HIGH.CRITICAL_HIGH[1]))
                crit_list = crit_list.union(crit).distinct()

    return crit_list
    

## MTP / get_crit_list 변경되면 동일하게 적용 예정

## Warn list 출력
def get_warn_list(item, df_pivot):
    # tmp_info = crit_warn_info.loc[item.lower()].str.split(',') #as-is
    tmp_info_LOW = crit_warn_info.filter(crit_warn_info.ITEM == item.lower())
    tmp_info_LOW = tmp_info_LOW.select(split(tmp_info_LOW.WARNING_LOW, ',', 2).alias('WARNING_LOW')).collect()
    tmp_info_LOW = tmp_info_LOW[0]
    
    tmp_info_HIGH = crit_warn_info.filter(crit_warn_info.ITEM == item.lower())
    tmp_info_HIGH = tmp_info_HIGH.select(split(tmp_info_HIGH.WARNING_HIGH, ',', 2).alias('WARNING_HIGH')).collect()
    tmp_info_HIGH = tmp_info_HIGH[0]

    spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
    #Creates Empty RDD
    emptyRDD = spark.sparkContext.emptyRDD()

    schema = StructType([
        StructField(item, StringType(), True)
        ])    
    
    # crit_list = pd.DataFrame().index #as-is
    # Convert empty RDD to Dataframe
    crit_list = emptyRDD.toDF(schema)
    
    # Warning List
    for column in df_pivot.columns:
        #temp = df_pivot[column].dropna()
        temp = df_pivot.dropna(subset=column)
        # as-is
        '''
        if (tmp_info.WARNING_LOW[0] == 'PPM'):
            warn = temp[temp < np.percentile(temp, float(tmp_info.WARNING_LOW[1]) / 1000000 * 100)].index
            warn_list = warn_list.union(warn)
        elif tmp_info.WARNING_LOW[0] == 'VALUE':
            warn = temp[temp < float(tmp_info.WARNING_LOW[1])].index
            warn_list = warn_list.union(warn)
        if (tmp_info.WARNING_HIGH[0] == 'PPM'):
            warn = temp[temp > np.percentile(temp, 100 - float(tmp_info.WARNING_HIGH[1]) / 1000000 * 100)].index
            warn_list = warn_list.union(warn)
        elif tmp_info.WARNING_HIGH[0] == 'VALUE':
            warn = temp[temp > float(tmp_info.WARNING_HIGH[1])].index
            warn_list = warn_list.union(warn)
        '''
        # aithe - extract warn_list from dataframe pyspark
        if column != 'DriveSerialNumber':

            if (tmp_info_LOW.WARNING_LOW[0] == 'PPM'):
                p = float(tmp_info_LOW.WARNING_LOW[1])/ 1000000 * 100
                warn=temp.select('DriveSerialNumber').where(temp[column]<temp.selectExpr('percentile({column},{p})'))
                warn_list = crit_list.union(warn).distinct()
            elif tmp_info_LOW.WARNING_LOW[0] == 'VALUE':
                warn = temp.select('DriveSerialNumber').where(temp[column]<float(tmp_info_LOW.WARNING_LOW[1]))
                warn_list = warn_list.union(warn)

            if (tmp_info_HIGH.WARNING_HIGH[0] == 'PPM'):
                p = 100 - float(tmp_info.WARNING_HIGH[1]) / 1000000 * 100
                warn = temp.select('DriveSerialNumber').where(temp[column]>temp.selectExpr('percentile({column},{p})'))
                warn_list = warn_list.union(warn)
            elif tmp_info_HIGH.WARNING_HIGH[0] == 'VALUE':
                warn = temp.select('DriveSerialNumber').where(temp[column]>float(tmp_info_HIGH.WARNING_HIGH[1]))
                warn_list = warn_list.union(warn)

    return warn_list

## MTP / get_crit_list 변경되면 동일하게 적용 예정

## Warn list 및 boundary 출력
def get_warn_list_boundary(item, df_pivot):
    
    # tmp_info = crit_warn_info.loc[item.lower()].str.split(',')
    tmp_info_LOW = crit_warn_info.filter(crit_warn_info.ITEM == item.lower())
    tmp_info_LOW = tmp_info_LOW.select(split(tmp_info_LOW.WARNING_LOW, ',', 2).alias('WARNING_LOW')).collect()
    tmp_info_LOW = tmp_info_LOW[0]
    
    tmp_info_HIGH = crit_warn_info.filter(crit_warn_info.ITEM == item.lower())
    tmp_info_HIGH = tmp_info_HIGH.select(split(tmp_info_HIGH.WARNING_HIGH, ',', 2).alias('WARNING_HIGH')).collect()
    tmp_info_HIGH = tmp_info_HIGH[0]
    
    spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
    
    #Creates Empty RDD
    emptyRDD = spark.sparkContext.emptyRDD()

    schema = StructType([
        StructField(item, StringType(), True)
        ])    
    
    # crit_list = pd.DataFrame().index #as-is
    # Convert empty RDD to Dataframe
    warn_list = emptyRDD.toDF(schema)

    # aithe - create three emty lists for 
    lower_list = []
    upper_list = []
    row_lower_upper=[]
    
    # Warning List for upper_list, upper_list
    for column in df_pivot.columns:
        # temp = df_pivot[column].dropna()
        temp = df_pivot.dropna(subset=column)

        # as is 
        '''
        if (tmp_info.WARNING_LOW[0] == 'PPM'):
            lower_b = np.percentile(temp, float(tmp_info.WARNING_LOW[1]) / 1000000 * 100)
            lower_list.append(lower_b)
            warn = temp[temp < lower_b].index
            warn_list = warn_list.union(warn)
        elif tmp_info.WARNING_LOW[0] == 'VALUE':
            lower_b = float(tmp_info.WARNING_LOW[1])
            lower_list.append(lower_b)
            warn = temp[temp < lower_b].index
            warn_list = warn_list.union(warn)
        else:
            lower_list.append(np.nan)

        if (tmp_info.WARNING_HIGH[0] == 'PPM'):
            upper_b = np.percentile(temp, 100 - float(tmp_info.WARNING_HIGH[1]) / 1000000 * 100)
            upper_list.append(upper_b)
            warn = temp[temp > upper_b].index
            warn_list = warn_list.union(warn)
        elif tmp_info.WARNING_HIGH[0] == 'VALUE':
            upper_b = float(tmp_info.WARNING_HIGH[1])
            upper_list.append(upper_b)
            warn = temp[temp > upper_b].index
            warn_list = warn_list.union(warn)
        else:
            upper_list.append(np.nan)
        '''
        # aithe - extract upper_list, upper_list and warn_list from dataframe pyspark
        if column != 'DriveSerialNumber':
            tmp_set=['PPM','VALUE']
            if (tmp_info_LOW.WARNING_LOW[0] in tmp_set):
                if (tmp_info_LOW.WARNING_LOW[0] == 'PPM'):
                    temp1= temp.select(temp.columns[1]).collect()
   
                    output_temp1=[i[0] for i in temp1]
                    output_temp1=np.array(output_temp1, dtype=float)
                    p = float(tmp_info_LOW.WARNING_LOW[1])/ 1000000 * 100
                    
                    lower_b = np.percentile(output_temp1, p)
                    lower_list.append(lower_b)
                    # warn = temp[temp < lower_b].index
                    warn=temp.select('DriveSerialNumber').where(temp[column]<lower_b)
                    warn_list = warn_list.union(warn)
                elif tmp_info_LOW.WARNING_LOW[0] == 'VALUE':
                    lower_b = float(tmp_info_LOW.WARNING_LOW[1])
                    lower_list.append(lower_b)
                    # warn = temp[temp < lower_b].index
                    warn=temp.select('DriveSerialNumber').where(temp[column]<lower_b)
                    warn_list = warn_list.union(warn)
            else:
                lower_list.append(np.nan)

            if(tmp_info_HIGH.WARNING_HIGH[0] in tmp_set):
                if (tmp_info_HIGH.WARNING_HIGH[0] == 'PPM'):
                    temp1= temp.select(temp.columns[1]).collect()
   
                    output_temp1=[i[0] for i in temp1]
                    output_temp1=np.array(output_temp1, dtype=float)
                    p = 100 - float(tmp_info_HIGH.WARNING_HIGH[1]) / 1000000 * 100

                    upper_b = np.percentile(output_temp1, p)
                    # upper_b = temp.selectExpr('percentile({column},{p})') #aithe as is

                    upper_list.append(upper_b)
                    # warn = temp[temp > upper_b].index
                    warn=temp.select('DriveSerialNumber').where(temp[column]> upper_b)
                    warn_list = warn_list.union(warn)
                elif tmp_info_HIGH.WARNING_HIGH[0] == 'VALUE':
                    upper_b = float(tmp_info_HIGH.WARNING_HIGH[1])
                    upper_list.append(upper_b)
                    # warn = temp[temp > upper_b].index
                    warn=temp.select('DriveSerialNumber').where(temp[column]> upper_b)
                    warn_list = warn_list.union(warn)
            else:
                upper_list.append(np.nan)
            
            
    # as-is
    #warn_boundary = pd.DataFrame({'Lower': lower_list, 'Upper': upper_list})

    # aithe - create a new dataframe for warn_boundary that merges lower_list and upper_list
    spark = SparkSession.builder.appName('sparkdf').getOrCreate()
    for j in range(len(lower_list)):
        row_lower_upper.append((float(lower_list[j]), float(upper_list[j])))

    columns=["Lower", "Upper"]      
    warn_boundary= spark.createDataFrame(row_lower_upper, columns)

    return warn_list, warn_boundary


# process_selected_column 함수:
# 선택한 column에 대해 데이터 처리 작업
# (Critical / Warning SN list 및 데이터 분포 정보 csv 파일로 저장)
# 2021.09.08 수정: KDE 미수행 옵션 추가
def process_selected_column(selected_column, data, kde=True):
    global path, target_path
    print('{}: '.format(selected_column) +
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

    # Pivot : DriveSerialNumber x Date
    sub_df = data[['DriveSerialNumber', 'date', selected_column]]
    
    ## MTP / pandas to pyspark
    # as-is
    # sub_df_pivot = sub_df.pivot(index='DriveSerialNumber', columns='date', values=selected_column) #as-is
    
    # aithe - modifying sub_df_pivot
    sub_df_pivot = sub_df.groupBy("DriveSerialNumber").pivot("date").avg(selected_column)

    
    crit_list = get_crit_list(selected_column, sub_df_pivot)  # Critical List

    # ## MTP / 수정 예정
    # as-is
    # sub_df_pivot = sub_df_pivot.loc[~sub_df_pivot.index.isin(crit_list)]  # Critical 포함 SN 제외

    # aithe - modify sub_df_pivot
    crit_list1= crit_list.select(crit_list.columns[0]).collect()
    output_crit_list=[i[0] for i in crit_list1]
    sub_df_pivot=sub_df_pivot.filter(~col('DriveSerialNumber').isin(output_crit_list))

    
    warn_list, warn_boundary = get_warn_list_boundary(selected_column, sub_df_pivot)  # Warning List & Boundary
  
    # aithe - converts warn_boundary to list
    warn_list1= warn_list.select(warn_list.columns[0]).collect()
    output_warn_list=[i[0] for i in warn_list1]


    s = os.path.split(file)

    # ## MTP / csv 저장 대신 Sqlpool 사용으로 주석처리함
    # Data Save #as-is
    # if os.path.exists(target_path +'/' + selected_column) == False: #as-is
    #     os.mkdir(target_path +'/' + selected_column) #as-is

    # # pd.DataFrame(crit_list).to_csv(target_path +'/' + selected_column + '/Crit_' + file) #as-is
    # # pd.DataFrame(warn_list).to_csv(target_path +'/'+ selected_column + '/Warn_' + file) #as-is
    # # warn_boundary.to_csv(target_path +'/' + selected_column + '/Boundary_' + file) #as-is
    #df_crit_list = spark.createDataFrame(crit_list, StringType())
    crit_list.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").mode('overwrite').save(target_path +'/' + selected_column + '/Crit_' + s[1])
    warn_list.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").mode('overwrite').save(target_path +'/' + selected_column + '/Warn_' + s[1])
    warn_boundary.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").mode('overwrite').save(target_path +'/' + selected_column + '/Boundary_' + s[1])
    
    # ## MTP / 변경 예정
    if kde == True:
        # as-is
        # sub_df_pivot = sub_df_pivot.loc[~sub_df_pivot.index.isin(warn_list)]  # Warning 포함 SN 제외

        # aithe - modify sub_df_pivot to extract ary_normal
        sub_df_pivot1=sub_df_pivot.filter(~col('DriveSerialNumber').isin(output_warn_list))
       
        sub_df_pivot2=sub_df_pivot1.select(sub_df_pivot1.columns[1:len(sub_df_pivot1.columns)]).collect()
        ary_normal = np.array(sub_df_pivot2, dtype=float)
        
    #     # Density Estimation : KDE
        day = ary_normal.shape[1]
        SDist = np.zeros((day, num_of_density_layer))  ### Day by Day KDE Based CDF
        weight = np.zeros((day, num_of_density_layer))  ### Day by Day Weight (CDF Diff)
        for i in range(day):
            SDist[i,], weight[i,] = get_kde_dist(ary_normal[:, i])
    
    # as-is
    #     pd.DataFrame(SDist).to_csv(target_path +'/'+ selected_column + '/Value_' + file)
    #     pd.DataFrame(weight).to_csv(target_path +'/' + selected_column + '/Weight_' + file)
        # numpy를 Dataframe로 변환
        s2=[]
        if len(weight)>=0:
            for i in weight:
                s1=[j.item() for j in i]
                s2.append(s1)

        dfweight = spark.createDataFrame(s2)
        # 단일 CSV 파일로 저장
        dfweight.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").mode('overwrite').save(target_path +'/' + selected_column + '/Weight_' + s[1])

        s2=[]
        # numpy를 Dataframe로 변환
        if len(SDist)>=0:
            for i in SDist:
                s1=[j.item() for j in i]
                s2.append(s1)
        dfSDist = spark.createDataFrame(s2)
        # 단일 CSV 파일로 저장
        dfSDist.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").mode('overwrite').save(target_path +'/' + selected_column + '/Value_' + s[1])
        


def KDE_main(target_product, target_logitems, num_of_last_week):
    global path, target_path, product, logitem, system_info, cols, types_dict, crit_warn_info, file_list, data_sampling, num_of_density_layer, file

    product = target_product
    logitem = target_logitems

    print('Start making data for KDE')

    # System 정보
    system_info = ['TIMESTAMP', 'Cluster', 'NodeId', 'Generation', 'HwSkuId',
                   'DriveProductId', 'DriveSerialNumber', 'FirmwareRevision']

    # SMART 항목
    item_smart = ['CritWarning', 'Temperature', 'AvailableSpare', 'AvailSpareThreshold',
                  'PercentageUsed', 'DataUnitsRead', 'DataUnitsWritten',
                  'HostReadCommands', 'HostWriteCommands', 'ControllerBusyTime',
                  'PowerCycles', 'PowerOnHours', 'UnsafeShutdowns', 'MediaErrors',
                  'NumErrInfoLogEntries', 'WarnCompositeTempTime',
                  'CritCompositeTempTime', 'TempSensor1', 'TempSensor2', 'TempSensor3',
                  'TempSensor4', 'TempSensor5', 'TempSensor6', 'TempSensor7',
                  'TempSensor8']

    # Extended SMART 항목 전부 표기 - KDE 미수행 항목들은 뒤쪽에 표기
    item_ext_smart = ['Media_Units_Written', 'ECC_Iterations', 'Wear_Range_Delta',
                      'Unaligned_IO', 'Mapped_LBAs', 'Program_Fail_Count', 'Erase_Fail_Count',
                      'Capacitor_Health', 'Supported_Features', 'Power_Consumption',
                      'Temperature_Throttling']

    ## 제품 / Log item 별 item list 설정
    # 2021.09.08 수정: Critical Warning과 Capacitor Health도 Crit/Warn list 추출 및 KDE 작업 수행
    if logitem == 'SMART':
        if product == 'PM963':  # PM963: Temp Sensor 2까지 있음
            item_list = item_smart[0:3] + item_smart[4:19]
        elif product in ['PM983', 'PM1725b']:  # PM983 & PM1725b: Temp Sensor 3까지 있음
            item_list = item_smart[0:3] + item_smart[4:20]
        elif product == 'PM953':  # PM953: NVMe 1.1 spec - No. of Err Info Log Entries까지
            item_list = item_smart[0:3] + item_smart[4:15]
    elif logitem == 'Ext_SMART':
        if product in ['PM963', 'PM983']:
            item_list = item_ext_smart[:8]
        elif product == 'PM953':  # PM953: Ext SMART 지원 항목 적음
            item_list = [item_ext_smart[0]] + item_ext_smart[2:5]

    # 데이터 로드 시 가져올 columns : 필요한 column들만 사용
    cols = [system_info[0]] + [system_info[6]] + item_list

    # Telemetry 항목값의 Type 지정 : 필요 시 설정 (기본 None)
    types_dict = None

    ## MTP / pandas to pyspark
    # 제품별 Critical / Warning 정보 불러오기
    # crit_warn_info = pd.read_csv('Anomaly_Rulebase_{}.csv'.format(product), index_col=0) #as-is
    crit_warn_info = sqlContext.read.format('com.databricks.spark.csv')\
                .options(header='true', inferSchema='true')\
                .load(adls_path + '/datasets/Anomaly_Rulebase_{}.csv'.format(product))

    #aithe - add data type Y_MAX from int -> float
    crit_warn_info = crit_warn_info.withColumn("Y_MAX", crit_warn_info["Y_MAX"].cast('float'))

    
    ## MTP / pandas to pyspark시 os 함수 호출 불가, make_EDA_plot에서 file_list 호출함
    # # 파일 경로 및 파일 목록 #as-is
    # print('file path :', path)
    # file_list = os.listdir(path)
    # file_list.sort()
    
    # 동작 시간 단축을 위해 data를 sampling하여 사용할 지 옵션
    data_sampling = True

    num_of_density_layer =  100

    ### Density Estimation - All

    # file_list 범위 설정
    file_list = file_list[-num_of_last_week:]
    print(file_list)
    print('above_files')

    for file in file_list:
        print('Processing {}: '.format(file) +
              time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        start = time.time()
        data = load_data_w_fw(file)  # FW ERRORMOD 필터링 적용

        for selected_column in item_list:
            process_selected_column(selected_column, data)

        del data  # For memory

        end = time.time()
        print('Processing {} done - Elapsed time: '.format(file) +
              f'{(end-start):.3f} seconds')




########################### main conduct part#################################
## make_EDA_plot 셀 수행

## MTP / 노트북 통합으로 인한 주석
# import KDEprocess_SMART_2021_v3 as kde_f
# import Visualization_SMART_2021_v3 as viz_f

#공용 parameter
num_of_last_week = 2 #새로 KDE 추출할 데이터 갯수(Week 단위 기준)

## MTP / adls 경로 추출
# Primary storage info
account_name = 'aithe' # fill in your primary account name
container_name = 'dl-sec' # fill in your container name
#relative_path = 'EDA/Telemetry_raw/Daily/To_Weekly/PM963/SMART/' # fill in your relative folder path
relative_path = 'EDA/' # fill in your relative folder path#

adls_path = 'abfss://%s@%s.dfs.core.windows.net/'% (container_name, account_name)
print('Primary storage account path: ' + adls_path)

####### PM963

## Parameters 입력 : 제품 / Log 아이템
product = 'PM963' # PM963, PM983, PM1725b, PM953

#읽어올 파일 경로 및 파일 목록
#path = 'F:/Telemetry_raw/Daily/To_Weekly/{}/SMART'.format(product) #as-is

## MTP / 신규 추가
# path = adls_path + 'Telemetry_raw/Daily/To_Weekly/{}/SMARTAN'.format(product)
path = adls_path
# pathList = path + "*"
pathList = path
print('pathList: ' + pathList)
df_pathList = spark.read.format("csv").load(pathList)
file_list = df_pathList.inputFiles()  
print(file_list)
# files = df_pathList.inputFiles()    
# file_list = files.sort(reverse=True)

#KDE 시각화를 위해 생성되는 파일들 저장 위치
#target_path = 'F:/Telemetry_output/MSC/{}'.format(product) #as-is

## MTP / 변경
#target_path = adls_path + 'Telemetry_output/MSC/{}'.format(product)
target_path = adls_path+'MSC/{}'.format(product)
print('path : ' + path)
print('target_path : ' + target_path)

#file_list = os.listdir(path).sort() #as-is
### Make Data for KDE estimation 
#kde_f.path = path #SMART 데이터 위치 #as-is
#kde_f.target_path = target_path #KDE용 생성 데이터 저장 위치 #as-is

## MTP / 노트북 통합으로 kde_f 제거
# kde_f.KDE_main(product,'SMART', num_of_last_week) #as-is
# kde_f.KDE_main(product,'Ext_SMART', num_of_last_week) #as-is
KDE_main(product,'SMART', num_of_last_week)
# KDE_main(product,'Ext_SMART', num_of_last_week)
