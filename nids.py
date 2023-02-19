import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import difflib as dlib
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from pandas import factorize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
import sys



column_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto','state',	'dur','sbytes','dbytes','sttl','dttl',	'sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
'trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
'ct_srv_src','ct_srv_dst','ct_dst_ltm',	'ct_src_ ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','Label']
feature_cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto','state',	'dur','sbytes','dbytes','sttl','dttl',	'sloss','dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
'trans_depth','res_bdy_len','Sjit','Djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_srv_src','ct_srv_dst','ct_dst_ltm',	'ct_src_ ltm','ct_src_dport_ltm',
'ct_dst_sport_ltm','ct_dst_src_ltm']
label_cols = ['attack_cat', 'Label']

#process command line options with format <test file>, <selected classification method>, <task - label or attack_cat>, <optional load model name>
test_file_name = sys.argv[1]
class_selection = sys.argv[2]
task_selection = sys.argv[3]
load_pickles = False
if len(sys.argv) == 5:
    if sys.argv[4] == 'False':
        load_pickles = False
    elif sys.argv[4] == 'True':
        load_pickles = True
    else:
        print("invalid pickle option, not loading pickle")

pdata = pd.read_csv(test_file_name, header=None, names=column_names, skiprows=1)

df = pd.DataFrame(pdata, columns=column_names)

# delete the columns with null values
del df['ct_flw_http_mthd']
del df['is_ftp_login']
del df['ct_ftp_cmd']

# conversions
df['proto']=pd.factorize(df['proto'])[0]
df['state']=pd.factorize(df['state'])[0]
df['dsport']=pd.factorize(df['dsport'])[0]
df['srcip']=pd.factorize(df['srcip'])[0]
df['sport']=pd.factorize(df['sport'])[0]
df['dstip']=pd.factorize(df['dstip'])[0]
df['dur']=pd.factorize(df['dur'])[0]
df['service']=pd.factorize(df['service'])[0]

df["service"].replace('-','None')
df["attack_cat"].fillna('None', inplace = True)
# df["attack_cat"].value_counts()

#preprocessing attack_cat for clones and similar words
df["attack_cat"] = df["attack_cat"].str.strip()
attack_cat_uniques = df["attack_cat"].unique()
#string comparison loop for similar words
cat_list = []
cat_list.append(attack_cat_uniques[0])
for index in attack_cat_uniques:
    matches = dlib.get_close_matches(index, cat_list, n=1, cutoff=0.8)
    if not matches:
        cat_list.append(index)
    else:
        df["attack_cat"] = df["attack_cat"].replace(index, matches[0])
    
 
def label_class():
    y=df["Label"] #Target
    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)
    X_train.apply(LabelEncoder().fit_transform)
    X_test.apply(LabelEncoder().fit_transform)
    preprocessing.normalize(X_train)
#   Classification using KNN
    knnClassifier=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    metric_params=None, n_jobs=1, n_neighbors=5, p=2,
    weights='uniform')
    knnClassifier.fit(X_train, y_train)
    y_pred = knnClassifier.predict(X_validation)
    confusion_matrix(y_validation, y_pred)
    accuracy_score(y_validation, y_pred)
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
    print(metrics.classification_report(y_validation, y_pred))
    micro_f1 = f1_score(y_validation, y_pred, average='micro')
    print("Micro F1 Score:", micro_f1) 

def attack_class():
    y=df["attack_cat"]
    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)
    X_train.apply(LabelEncoder().fit_transform)
    X_test.apply(LabelEncoder().fit_transform)
    preprocessing.normalize(X_train)
#   classification using DecisionTree
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_validation)
    confusion_matrix(y_validation, y_pred)
    accuracy_score(y_validation, y_pred)
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
    print(metrics.classification_report(y_validation, y_pred))
    micro_f1 = f1_score(y_validation, y_pred, average='micro')
    print("Micro F1 Score:", micro_f1) 

def rfe_featureselection_label(task_selection):
    y_train = y_train[task_selection]
    rfe = RFECV(DecisionTreeClassifier())
    rfe = rfe.fit(X_train, y_train)
    print("Num features: ", rfe.n_features_)
    print("Selected features")
    print(X_train.columns[rfe.support_])
    return rfe, X_train.columns[rfe.support_]

def rfe_featureselection_attack_cat(task_selection):
    y_train = y_train[task_selection]
    preprocessing.normalize(X_train)
    rfe = RFECV(DecisionTreeClassifier())
    rfe = rfe.fit(X_train, y_train)
    print("Num features: ", rfe.n_features_)
    print("Selected features")
    print(X_train.columns[rfe.support_])
    return rfe, X_train.columns[rfe.support_]

def gnb_naive_bayes_label(task_selection, selected_features = [], load_pickle = False):
    if load_pickle:
        selected_features = ['srcip', 'sport', 'sbytes', 'dbytes', 'sttl', 'smeansz', 'Stime',
                            'Sintpkt', 'synack', 'ct_srv_dst']
    elif not selected_features:
        print("loading the whole thing")
        selected_features = feature_cols
    else:
        selected_features = selected_features
    y=df[task_selection]
    X=df[selected_features]
    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)
    preprocessing.normalize(X_train)
    gnb = GaussianNB()
    gnb = gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_validation)
    confusion_matrix(y_validation, y_pred)
    accuracy_score(y_validation, y_pred)
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
    print(metrics.classification_report(y_validation, y_pred))
    micro_f1 = f1_score(y_validation, y_pred, average='micro')
    print("Micro F1 Score:", micro_f1)

def gnb_naive_bayes_attack_cat(task_selection, selected_features = [], load_pickle = False):
    if load_pickle:
        selected_features = ['dsport', 'sbytes', 'sttl', 'service']
    elif not selected_features:
        selected_features = feature_cols
    else:
        selected_features = selected_features
    y=df[task_selection]
    X=df[selected_features]
    X_train, X_validation, y_train, y_validation = train_test_split( X, y, test_size = 0.20, random_state = 0)
    X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size = 0.10, random_state = 0)
    preprocessing.normalize(X_train)
    gnb = CategoricalNB(force_alpha=True)
    gnb = gnb.fit(X_train, y_train)
    CategoricalNB(force_alpha=True)
    y_pred = gnb.predict(X_validation)
    confusion_matrix(y_validation, y_pred)
    accuracy_score(y_validation, y_pred)
    print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_validation, y_pred)*100))
    print(metrics.classification_report(y_validation, y_pred))
    micro_f1 = f1_score(y_validation, y_pred, average='micro')
    print("Micro F1 Score:", micro_f1) 

print("Feature analysis and classification")

print("RFE Feature Selection")

#Reverse Feature Elimination and Naive Bayes Classification:
if class_selection == "CNB":
    if task_selection == "Label":
        #rfe_model, selected_colums = rfe_featureselection_label()
        selected_colums = ['srcip', 'sport', 'sbytes', 'dbytes', 'sttl', 'smeansz', 'Stime',
                            'Sintpkt', 'synack', 'ct_srv_dst']
        gnb_naive_bayes_label(task_selection, selected_features=selected_colums, load_pickle=load_pickles)
    elif task_selection == "attack_cat":
        #rfe_model, selected_colums = rfe_featureselection_attack_cat()
        selected_colums = ['dsport', 'sbytes', 'sttl', 'service']
        gnb_naive_bayes_attack_cat(task_selection, selected_features=selected_colums, load_pickle=load_pickles)
X=df.drop(["attack_cat"],axis=1)
y=df["Label"]
corr_data = np.abs(X.corrwith(y))
# Selecting features with the high correlation coefficients
selected = corr_data.nlargest(n=26).index
X= X[selected]
X=X.drop(["Label"],axis=1)
print("\nLabel classification")
#label_class()
print("\nattack_cat classification")
#attack_class()
print("done")
    
