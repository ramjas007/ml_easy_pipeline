import extracting_zip 
import get_kaggledata
import os
from checking import DataAnalyzer
from premodel_checking import classify_models, regress_models,main
import autogluon_automl
 
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

y=train.price
X=train.drop(columns=["price"])

autogluon_automl.autogluon_ensemble(train, test, sub,"price","root_mean_squared_error","best_quality")







