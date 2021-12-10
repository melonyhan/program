from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import data_load
import argparse
import torch

#import our data
parser = argparse.ArgumentParser(description='SVM')
parser.add_argument("network_name", type=str,
                        help="get network name")

parser.add_argument("--label_rate", type=float, default=0.05,    
                        help="label rate")
parser.add_argument("--label_ratio", type=float, default=2,    
                        help="ratio between negative and positive labels")

parser.add_argument("--train_rate", type=float, default=0.7,
                        help="train set rate")
parser.add_argument("--test_rate", type=float, default=0.3,
                        help="test set rate")
parser.add_argument("--kernel", type=str, default='linear',
                        help="select different type of kernel function('rbf', 'linear', 'poly')")                        

args = parser.parse_args()


g = data_load.data_load(args.network_name, args.label_rate, args.train_rate, args.test_rate, args.label_ratio)
features = torch.tensor(g.ndata['feat'])
labels = torch.tensor(g.ndata['label'])
train_nid = g.ndata['train_mask'].nonzero().squeeze()
test_nid = g.ndata['test_mask'].nonzero().squeeze()
#split the data

X_train = features[train_nid]
y_train = labels[train_nid]
X_test = features[test_nid]
y_test = labels[test_nid]

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("""accuracy score: %f
        precision score: %f
        recall score: %f
        fi score: %f"""
        %(acc, pre, recall, f1))
