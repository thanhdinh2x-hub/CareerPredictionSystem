import pandas as pd
from lazypredict.Supervised import LazyClassifier

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import  RandomForestClassifier
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2
import joblib

#TODO step1: cứ xem qua database đã bằng run file with python console
# ==================================================
data= pd.read_excel('../data/final_project.ods',dtype=str) # đọc tất cả dữ liệu rồi ép kiểu hết thành str
print(data.head())
print("____funtion___")
print(data["function"].unique())
#TODO step2: thống kê
# ======================================================


data = data.dropna(axis=0)
print(data.info())

    #FIXME: sắp xếp dữ liệu trước  cho bước tiền xử lý cột location
def location_sep( location ):
    if "," in location:
        return location[-2:]
    else:
        return location

data["location"] = data["location"].apply(location_sep)
print(data)

#TODO step3: tách cột
# ======================================================
target= "career_level"
x= data.drop(columns=[target])
y = data[target]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print("-----count_y-------")
print(y_train.value_counts())
#TODO step4:  tiền xử lý dữ liệu
# ======================================================

    # FIXME: cách 1 tiền xử lý từng bước cột(chỉ dể xem thôi nhé có thể xóa đi)
        # FIXME:TF_IDF đối với cột title
vectorizer = TfidfVectorizer()
x_train_title = vectorizer.fit_transform(x_train["title"])

output= vectorizer.vocabulary_
print(output)


# Chuyển toàn bộ ma trận(output của TfidfVectorizer) thành DataFrame
x_train_df_title = pd.DataFrame.sparse.from_spmatrix(x_train_title)
print(x_train_df_title)
        # FIXME:OneHot đối với cột location

encoder = OneHotEncoder()
x_train_location = encoder.fit_transform(x_train[["location"]])
print(x_train_location)
print(x_train_location.shape)
x_train_df_location = pd.DataFrame.sparse.from_spmatrix(x_train_location)


         # FIXME:TF_IDF đối với cột desciption
vectorizer1 = TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=0.01,max_df=0.99)
x_train_desciption = vectorizer1.fit_transform(x_train["description"])

output_description= vectorizer.vocabulary_
print(len(output_description))
print(x_train_desciption.shape)

#only uni-gram:66745
#uni-gram +bi-gram:847124

         # FIXME:OneHot đối với cột function

encoder1 = OneHotEncoder()
x_train_function = encoder1.fit_transform(x_train[["function"]])
print(x_train_function)
print("----test function với onehot---------")

print(x_train_function.shape)



# vectorizer21 = TfidfVectorizer()
# aaa = vectorizer21.fit_transform(x_train["function"])
# x_train_df_function = pd.DataFrame.sparse.from_spmatrix(aaa)
#
#
# output= vectorizer21.vocabulary_
# print(output)
# print("----test function với TF_IDF---------")

# print(aaa.shape)

         # FIXME:TF_IDF đối với cột industry
vectorizer2 = TfidfVectorizer()
x_train_industry1 = vectorizer2.fit_transform(x_train["industry"])

output= vectorizer2.vocabulary_
print(output)
print("----test industry với TF_IDF---------")

print(x_train_industry1.shape)

# encoder2 = OneHotEncoder()
# x_train_industry2 = encoder2.fit_transform(x_train[["industry"]])
# print(x_train_industry2)
# print("----test industry với onehot---------")
# print(x_train_industry2.shape)


    # FIXME: cách 2 dùng Pipeline scikit learn (nhưng do đã loại bỏ hết ô khuyết từ bandđầu nên ko cần pipeline nữa)
pass


#TODO Step5:  sau khi tiền xử lý thì gộp các dữ liệu đã xử lý lại thành 1 bộ train cuối cùng để nhét vào models
# ========================================================================

    #FIXME: cách 1 gộp chay các dữ liệu đã xử lý ở bước trên vào thành bộ train cuối cùng
pass

    #FIXME: cách 2 dùng ColumnTransformer để các feature của bộ train vào các tiền xủ lý tương ứng và xuất ra kết quả bộ train sau khi đi qua tiền xử lý
preprocessor = ColumnTransformer(transformers=[
    ("Tit", TfidfVectorizer(), "title"),
    ("Loc", OneHotEncoder(handle_unknown='ignore'), ["location"]), # phải có ignore vì 1 số location ở bộ test ko thấy trong bộ train
    ("Des",TfidfVectorizer(stop_words='english',ngram_range=(1,2),min_df=0.01,max_df=0.99),"description" ),
    ("func",OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("indus",TfidfVectorizer(), "industry"),

])
        #FIXME:sau khi định nghĩa xong các feature nào vào loại tiền xử lý trong ColumnTransformer thì fit_transform bộ train vào thôi
# transform train/test
x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)


x_train_processed = pd.DataFrame(
    x_train_processed.toarray()
)

x_test_processed = pd.DataFrame(
    x_test_processed.toarray()
)
result= preprocessor.fit_transform(x_train)

print("------------sau khi ddi qua columntransformer-------------")
print(result.shape)

#only uni-gram +bi-gram :(6458, 850827)
#uni-gram +bi-gram + min/max_df:(6458, 7901)


#TODO Step6: Chọn models và fit dữ liệu
# ====================================================================

    #FIXME:next step 1: tự chuyền bộ số hyperParameters vào mô hình nhưng ở đây đang truyền qua Pipeline
model = Pipeline(steps=[ #FIXME: khi dùng Pipeline thì đối với các estimator là transformer thì các bước transformer sẽ tự fit_transform rồi mới đưa xuống next step và last estimator là gì thì Pileline sẽ có toàn bộ thuộc tính cũng như phương thức của last estimator đó
    ('preprocessor', preprocessor),
    # ('selector', SelectKBest(score_func=chi2, k=800)),
    ('selector', SelectPercentile(score_func=chi2, percentile=10)),
    ('classifier', RandomForestClassifier(random_state=42,class_weight="balanced")), # thay cho cái trên vì muốn test với nhiều tham số hơn

])
    # xem matrix của slector
# output_selector_thr_pipeline = models.fit_transform(x_train,y_train) #xem matrix của slector do SelectKBest so sánh độ tuyến tính giữa feature với target nên phải truyền vào cả y_train nữa
# print(output_selector_thr_pipeline.shape)
    # end xem matrix của slector

model.fit(x_train, y_train)
pred = model.predict(x_test)
print(classification_report(y_test, pred))

 #FIXME: next step 2: dùng gridsearch để tìm mô hình có bộ số(hyperParameters) tối ưu nhất

param_grid= {
    "classifier__n_estimators" : [50,100,150,200], #FIXME: đang dùng nest params (regressor__)
    "classifier__max_depth" : [None,3,4,5],
    "classifier__criterion" : ["gini","entropy","log_loss"],
    "preprocessor__Des__min_df" : [ 0.01, 0.02,0.03,0.04 ],
    "preprocessor__Des__max_df" : [ 0.8,0.7,0.9,1],
    "selector__percentile" : [10,20,5,30],

}

# grs_model= GridSearchCV( # dùng gridSearch để tìm với những hyperParamerter nào thì mô hình có Scoring tối ưu nhất
#     estimator=models, # mô hình muốn test
#     param_grid=param_grid, # những hyperParameter cho mô hình test
#     n_jobs=3,  # bao nhiêu nhân hoạt động
#     cv=5,  # chia training datasets thành k fold_cross validation
#     verbose=1,  # muốn in thông tin test ra ngoài màn hình nhiều hay ít
#     scoring='recall_macro',  # muốn đánh giá mô hình theo tiêu chuẩn nào
# ) # FIXME: or dùng RandomizedSearchCV bên dưới nếu muốn tìm nhanh tập con của GridSearchCV

grs_model= RandomizedSearchCV( # dùng gridSearch để tìm với những hyperParamerter nào thì mô hình có Scoring tối ưu nhất
    estimator=model, # mô hình muốn test
    param_distributions=param_grid, # những hyperParameter cho mô hình test
    n_iter= 20, # trong 1 loạt trường hợp của GridSearchCV muốn lấy 1 tập con khoảng(30) trường hợp để search
    n_jobs=-1, # bao nhiêu nhân hoạt động
    cv=3, # chia training datasets thành k fold_cross validation
    verbose=2, # muốn in thông tin test ra ngoài màn hình nhiều hay ít
    scoring='precision_macro', # muốn đánh giá mô hình theo tiêu chuẩn nào
)

    #FIXME: đoạn dưới này dùng chung của cách 1 và 2

        #FIXME : Huấn luyện (train) mô hình.
        # Model học mối quan hệ giữa:
        #   x_train (đặc trưng đầu vào)
        # và
        #   y_train (đáp án đúng)
        # Sau bước này models đã biết cách dự đoán.
grs_model.fit(x_train, y_train) # do models là 1 biến(maybe gọi là object) đại diện cho Pipeline nên nó sẽ kế thữa last estimator đã nói ở trên
        #FIXME: in ra các hyperParam và thông số mô hình cho là tối ưu nhất
print(grs_model.best_params_) # cái này chỉ có khi dùng GridSearchCV
print(grs_model.best_score_)# cái này chỉ có khi dùng GridSearchCV
        #FIXME : Dùng models đã học để dự đoán Outcome của dữ liệu test.
        # Lưu ý: models KHÔNG học thêm ở bước này.
y_pred = grs_model.predict(x_test) #cũng giống như phương thức fit() ở trên thôi Pipeline cũng tự động transform raw dataset qua các transformer rồi cuối cùng mới predict

y_pred_serie = pd.Series(y_pred)
print(y_pred_serie.value_counts()) # xem % của dự đoán cho từng thuộc tính

print(classification_report(y_test, y_pred))
#
#       precision    recall  f1-score   support
#                         bereichsleiter       0.32      0.72      0.44       192
#          director_business_unit_leader       0.33      1.00      0.49        14
#                    manager_team_leader       0.55      0.24      0.34       534
# managing_director_small_medium_company       1.00      1.00      1.00         1
#   senior_specialist_or_project_manager       0.79      0.78      0.79       868
#                             specialist       0.09      0.83      0.17         6
#                               accuracy                           0.60      1615
#                              macro avg       0.51      0.76      0.54      1615
#                           weighted avg       0.65      0.60      0.59      1615




    # FIXME: next step 0 : dùng LazyPridict để xem các mô hình nào tốt nhất còn biết lấy để GridSearchCV cho ra chỉ số tốt nhất( xem xong thì lấy top5 để test với GridSearch)
# # transform train/test (cho lên trên)
# x_train_processed = preprocessor.fit_transform(x_train)
# x_test_processed = preprocessor.transform(x_test)
# x_train_processed = pd.DataFrame.sparse.from_spmatrix(
#     x_train_processed
# )
#
# x_test_processed = pd.DataFrame.sparse.from_spmatrix(
#     x_test_processed
# )
# reg = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = reg.fit(x_train_processed,x_test_processed,y_train,y_test)
# print(predictions)
# print(models)

#TODO step5: save the models to disk sau khi fit là dùng đc
# ======================================================
best_model = grs_model.best_estimator_

joblib.dump(
    best_model,
    "../models/career_level_model.pkl"
)