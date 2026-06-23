import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
data=pd.read_csv("heartdisease.txt")

data = data.drop(columns=['slope', 'ca', 'thal'])
# Numériques → médiane (robuste aux outliers)
data['trestbps'] = data['trestbps'].fillna(data['trestbps'].median())
data['chol']     = data['chol'].fillna(data['chol'].median())
data['thalch']   = data['thalch'].fillna(data['thalch'].median())
data['oldpeak']  = data['oldpeak'].fillna(data['oldpeak'].median())

# Catégorielles → mode
data['fbs']      = data['fbs'].fillna(data['fbs'].mode()[0])
data['restecg']  = data['restecg'].fillna(data['restecg'].mode()[0])
data['exang']    = data['exang'].fillna(data['exang'].mode()[0])
data['target'] = (data['num'] > 0).astype(int)
data = pd.get_dummies(data, columns=['sex', 'cp', 'restecg'], drop_first=True)
data = data.drop(columns=['num', 'id', 'dataset'])
X = data.drop(columns=['target']).values.astype(float)
y = data['target'].values.astype(float).reshape(-1, 1)
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]
X_min = X_train.min(axis=0)
X_max = X_train.max(axis=0)
X_train = (X_train - X_min) / (X_max - X_min + 1e-8)
X_test  = (X_test  - X_min) / (X_max - X_min + 1e-8)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def log_loss(y,h):
    eps = 1e-8 
    return -np.mean((np.log(h+eps)*y+(1-y)*np.log(1-h+eps)))
def initialize(n_features):
    W = np.zeros((n_features, 1))
    b = 0.0
    return W, b
def forward(X,W,b):

    c= X@W+b
    return sigmoid(c)
def gradients(X,y,h):
    N=len(X)
    dw=1/N *X.T@(h-y)
    db=1/N *np.sum((h-y))
    return dw,db
def update(W, b, dw, db, a):
    W=W-a*dw
    b=b-a*db
    return W,b
def train(X,y,a,n_epoch):
    W,b=initialize(n_features=X.shape[1])
    losses=[]
    for i in range(n_epoch):
        z=forward(X,W,b)
        Loss=log_loss(y,z)
        losses.append(Loss)
        dw,db=gradients(X,y,z)
        W,b=update(W,b,dw,db,a)
    return W,b,losses
def predict(X, W, b, threshold=0.5):
    h = forward(X, W, b)
    return (h >= threshold).astype(int)
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[TN, FP], [FN, TP]])
def precision_recall_f1(y_true, y_pred):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1
def roc_auc(y_true, y_prob):
    thresholds = np.linspace(0, 1, 200)
    tprs, fprs = [], []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        TP = np.sum((pred == 1) & (y_true == 1))
        TN = np.sum((pred == 0) & (y_true == 0))
        FP = np.sum((pred == 1) & (y_true == 0))
        FN = np.sum((pred == 0) & (y_true == 1))
        tprs.append(TP / (TP + FN + 1e-8))
        fprs.append(FP / (FP + TN + 1e-8))
    fprs  = np.array(fprs)
    tprs  = np.array(tprs)
    idx   = np.argsort(fprs)
    auc   = np.trapz(tprs[idx], fprs[idx])
    return auc, fprs[idx], tprs[idx]
# ===================== TRAIN =====================
W, b, losses = train(X_train, y_train, a=0.01, n_epoch=2000)

# ===================== PREDICT =====================
y_pred = predict(X_test, W, b)
y_prob = forward(X_test, W, b)

# ===================== MÉTRIQUES =====================
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()
y_prob_flat = y_prob.flatten()

print("Accuracy  :", round(accuracy(y_test_flat, y_pred_flat), 4))
cm = confusion_matrix(y_test_flat, y_pred_flat)
print("Confusion matrix :\n", cm)
p, r, f1 = precision_recall_f1(y_test_flat, y_pred_flat)
print(f"Precision : {p:.4f} | Recall : {r:.4f} | F1 : {f1:.4f}")
auc, fprs, tprs = roc_auc(y_test_flat, y_prob_flat)
print("AUC-ROC   :", round(auc, 4))

# ===================== COURBES =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(losses)
ax1.set_title("Convergence — Log Loss")
ax1.set_xlabel("Itération")
ax1.set_ylabel("Loss")

ax2.plot(fprs, tprs, label=f"AUC = {auc:.4f}")
ax2.plot([0,1], [0,1], '--', color='gray')
ax2.set_title("Courbe ROC")
ax2.set_xlabel("FPR")
ax2.set_ylabel("TPR")
ax2.legend()

plt.tight_layout()
plt.show()
    
    

