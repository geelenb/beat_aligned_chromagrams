from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

def confusion_matrix_for_problem(X, y, model=None):
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, random_state=0, stratify=y
    )

    if model is None:
        model = RidgeClassifier()
    model.fit(train_X, train_y)
    prediction_y = model.predict(test_X)

    return confusion_matrix(test_y, prediction_y)


def cross_val_predict_confusion_matrix_for_problem(X, y, model=None):
    if model is None:
        model = RidgeClassifier

    prediction_y = cross_val_predict(model, X, y)
    return confusion_matrix(y, prediction_y)


def accuracy_from_confusion_matrix(confusion_matrix):
    return confusion_matrix.diagonal().sum() / confusion_matrix.sum()


def dictionary_of_models():
    # base models
    models = {
        "Ridge": RidgeClassifier(),
        # "LinearSVC": LinearSVC(dual=False),
        # "LinearSVC dual": LinearSVC(dual=True),
        # "SVM RBF": SVC(kernel='rbf'),
        # "SVM sigmoid": SVC(kernel='sigmoid'),
        "KNN minkowski": KNeighborsClassifier(5),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=None
        ),
        # "RandomForest max_depth = 16": RandomForestClassifier(
        #     n_estimators=10000, max_depth=16
        # ),
        "XGBoost 1000": XGBClassifier(n_estimators=1000, n_jobs=-1),
        # "XGBoost 20": XGBClassifier(n_estimators=1000, n_jobs=-1, max_depth=20, learning_rate=.1), # 0.672
        # "XGBoost 50": XGBClassifier(n_estimators=50, n_jobs=-1, max_depth=50, learning_rate=.1), # 0.672
        # "XGBoost 100": XGBClassifier(n_estimators=1000, n_jobs=-1, max_depth=100, learning_rate=.1), # 0.672
    }

    models = {
        # **{
        #     f"Poly, {name}": make_pipeline(PolynomialFeatures(), model)
        #     for name, model in models.items()
        # },
        **{
            f"StandardScaler, {name}": make_pipeline(StandardScaler(), model)
            for name, model in models.items()
        },
        # **models,
    }

    # Add dummy models
    models["Dummy stratified"] = DummyClassifier(random_state=0)
    models["Dummy most_frequent"] = DummyClassifier(
        strategy="most_frequent", random_state=0
    )

    return models
