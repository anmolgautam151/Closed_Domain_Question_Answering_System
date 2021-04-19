from sklearn import svm, tree
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import pickle


def read_pickle(filepath):
    data_df = pd.read_pickle(filepath)
    data_df[["glove_cosine1", "glove_cosine2", "glove_cosine3", "glove_cosine4", "glove_cosine5", "glove_cosine6",
             "glove_cosine7", "glove_cosine8", "glove_cosine9", "glove_cosine10"]] = pd.DataFrame(
        data_df.glove_cosine.values.tolist(), index=data_df.index)
    data_df[["glove_euclidean1", "glove_euclidean2", "glove_euclidean3", "glove_euclidean4", "glove_euclidean5",
             "glove_euclidean6", "glove_euclidean7", "glove_euclidean8", "glove_euclidean9",
             "glove_euclidean10"]] = pd.DataFrame(data_df.glove_euclidean.values.tolist(), index=data_df.index)
    data_df[["tfidf_euclidean1", "tfidf_euclidean2", "tfidf_euclidean3", "tfidf_euclidean4", "tfidf_euclidean5",
             "tfidf_euclidean6", "tfidf_euclidean7", "tfidf_euclidean8", "tfidf_euclidean9",
             "tfidf_euclidean10"]] = pd.DataFrame(data_df.tfidf_euclidean.values.tolist(), index=data_df.index)
    data_df[["tfidf_cosine1", "tfidf_cosine2", "tfidf_cosine3", "tfidf_cosine4", "tfidf_cosine5", "tfidf_cosine6",
             "tfidf_cosine7", "tfidf_cosine8", "tfidf_cosine9", "tfidf_cosine10"]] = pd.DataFrame(
        data_df.tfidf_cosine.values.tolist(), index=data_df.index)
    data_df[["dependency1", "dependency2", "dependency3", "dependency4", "dependency5", "dependency6", "dependency7",
             "dependency8", "dependency9", "dependency10"]] = pd.DataFrame(data_df.dependency.values.tolist(),
                                                                           index=data_df.index)
    data_df.drop(
        columns=["glove_cosine", "glove_euclidean", "tfidf_euclidean", "tfidf_cosine", "dependency"],
        inplace=True)
    labels = data_df["target"]
    data_df.drop(columns=["target"], inplace=True)
    return data_df.to_numpy(), labels.to_numpy()


if __name__ == '__main__':
    train_path = "./data/preprocess_ml_train.pkl"
    dev_path = "./data/preprocess_ml_dev.pkl"
    train_X, train_Y = read_pickle(train_path)

    scalar = RobustScaler()
    train_X_LR = scalar.fit_transform(train_X)

    dev_X, dev_Y = read_pickle(dev_path)
    dev_X_LR = scalar.fit_transform(dev_X)
    models = []
    models.append(("DT", tree.DecisionTreeClassifier(max_depth=10)))
    models.append(("RF", RandomForestClassifier(min_samples_leaf=8, n_estimators=60)))
    cla = LogisticRegression(multi_class='ovr', solver='newton-cg')
    models.append(("SVM", SVC(kernel='poly', degree=10, max_iter=50000)))
    models.append(("GB", GradientBoostingClassifier(min_samples_leaf=8, n_estimators=60)))
    y_axis1, y_axis2 = [], []
    x_axis = ["DT", "RF", "SVM", "GB", "LR"]
    y_axis1_f1, y_axis2_f1 = [], []
    for name, model in models:
        print(name + ' Training')
        model.fit(train_X, train_Y)
        filename = "./model/" + name + ".sav"
        pickle.dump(model, open(filename, 'wb'))

    print('LR Training')
    cla.fit(train_X_LR, train_Y)
    filename = './model/LR.sav'
    pickle.dump(cla, open(filename, 'wb'))

    for name, model1 in models:
        print(name + ' Predicting')
        filename = "./model/" + name + ".sav"
        model = pickle.load(open(filename, 'rb'))
        y_pred = model.predict(train_X)
        train_accu = accuracy_score(train_Y, y_pred)
        y_axis1.append(train_accu)
        y_axis1_f1.append(f1_score(train_Y, y_pred, average='macro'))
        y_pred = model.predict(dev_X)
        test_accu = accuracy_score(dev_Y, y_pred)
        y_axis2.append(test_accu)
        y_axis2_f1.append(f1_score(dev_Y, y_pred, average='macro'))

    print("LR Predicting")
    filename = './model/LR.sav'
    model = pickle.load(open(filename, 'rb'))
    ypred = model.predict(train_X_LR)
    y_axis1.append(accuracy_score(train_Y, ypred))
    y_axis1_f1.append(f1_score(train_Y, ypred, average='micro'))
    ypred = model.predict(dev_X_LR)
    y_axis2.append(accuracy_score(dev_Y, ypred))
    y_axis2_f1.append(f1_score(dev_Y, ypred, average='micro'))

    plt.plot(x_axis, y_axis1, color='b', label='Train')
    plt.plot(x_axis, y_axis2, color='r', label='Dev')
    plt.title("Accuracy")
    plt.legend()
    plt.show()

    plt.plot(x_axis, y_axis1_f1, color='b', label='Train')
    plt.plot(x_axis, y_axis2_f1, color='r', label='Dev')
    plt.title("F1 score")
    plt.legend()
    plt.show()
