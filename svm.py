'''
Baseline algorithm (paraphrase binary classifier) Decision Tree and SVM
'''
import statistics

from data import *
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.model_selection import KFold

dataset = Dataset("Dataset.csv")



x = list(map(lambda x: dict(a = x), dataset.sentence[:800]))
X = list(map(lambda x, y: [x, y], x, dataset.label[:800]))


y = list(map(lambda x: dict(a = x), dataset.sentence[800:]))

ps = []
rs = []
acs = []
for x in range(1, 11):
    clf_svm = SklearnClassifier(make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto'))) # Support Vector Machine
    clf_svm = clf_svm.train(X)
    dt_preds = clf_svm.classify_many(y)

    print("#", x, " ROUND: Precision Score for Baseline Algorithm is: ", p := precision_score(dataset.label[800:], dt_preds, average= 'macro'))
    print("#", x, " ROUND: Recall Score for Baseline Algorithm is: ", r := recall_score(dataset.label[800:], dt_preds, average= 'macro'))
    print("#", x, " ROUND: Accuracy Score for Baseline Algorithm is: ", a := accuracy_score(dataset.label[800:], dt_preds))
    ps.append(p)
    rs.append(r)
    acs.append(a)

print("Average Precision: ", statistics.mean(ps))
print("Average Recall: ", statistics.mean(rs))
print("Average Accuracy: ", statistics.mean(acs))