import extracting_zip 
import get_kaggledata
import os
from checking import DataAnalyzer
from premodel_checking import classify_models, regress_models,main
import autogluon_automl
from preprocessing import preprocess_data_pipeline
from hyperimpute.plugins.imputers import Imputers
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
 
if not os.path.exists("C:\\Users\\maury\\Downloads\\playground-series-s4e9\\playground-series-s4e9.zip"):
    get_kaggledata.download_dataset("https://www.kaggle.com/competitions/playground-series-s4e9/data")
else:
    extracting_zip.extract_zip("C:\\Users\\maury\\Downloads\\playground-series-s4e9\\playground-series-s4e9.zip")


train_path = "playground-series-s4e9/train.csv"
test_path = "playground-series-s4e9/test.csv"
submission_path = "playground-series-s4e9/sample_submission.csv"

# Process the data
analyzer = DataAnalyzer(train_path, test_path, submission_path)
train, test,sub = analyzer.drop_columns(["id"])

# train_processed, test_processed = preprocess_data_pipeline(train, test, steps=['label_encode','simple_impute'],n_neighbors=2)
# print(train_processed.info())
# print(test_processed.info())
# print(train_processed)
# print(test_processed)

print(Imputers().list())
# ['sklearn_ice', 'gain', 'sinkhorn', 'nop', 'miracle', 'most_frequent', 'mice', 'softimpute', 'sklearn_missforest', 'median', 'ice',
#  'missforest', 'hyperimpute', 'miwae', 'mean', 'EM']

data=pd.concat([train,test],axis=0)

plugin = Imputers().get("hyperimpute")
data_fill=plugin.fit_transform(np.array(data))
data_fill=data_fill[:len(train)]

data_fill=data_fill.rename(columns={"0":"brand","1":"model","2":"model_year","3":"milage",
                          "4":"fuel_type","5":"engine","6":"transmission","7":"ext_col","8":"int_col","9":"accident","10":"clean_title","11":"price"})

print(data_fill.head())

# # print(test.head())
# y=train_processed.price
# X=train_processed.drop(columns=["price"])

# # autogluon_automl.autogluon_ensemble(train, test, sub,"price","root_mean_squared_error","best_quality")

# main(X,y)







