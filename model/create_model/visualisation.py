from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_confusion_matrix(y_true, y_pred, labels, save=False, save_dir=None, filename=None):
    """绘制归一化混淆矩阵"""

    confusion_mtx = confusion_matrix(y_true, y_pred)
    precision_confusion_mtx = confusion_mtx.T / (confusion_mtx.sum(axis=1)).T
    recall_confusion_mtx = confusion_mtx / confusion_mtx.sum(axis=0)

    fig = plt.figure(figsize=(21, 6))

    plt.subplot(1, 3, 1)
    _ = sns.heatmap(confusion_mtx, annot=True, cmap="Blues", fmt="", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")

    plt.subplot(1, 3, 2)
    _ = sns.heatmap(precision_confusion_mtx, annot=True, cmap="Blues", fmt='.3f', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Precision Matrix")

    plt.subplot(1, 3, 3)
    _ = sns.heatmap(recall_confusion_mtx, annot=True, cmap="Blues", fmt='.3f', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Recall Matrix")

    fig.tight_layout()
    
    if save:
        fig.savefig(os.path.join(save_dir, filename))


def plot_roc_curve(y_test, y_score, labels, save=False, save_dir=None, filename=None):
    """绘制ROC曲线"""

    n_classes = y_score.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 求平均，计算ROC包围的面积
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # 绘制ROC曲线
    fig = plt.figure(figsize=(14, 10))
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.4f})'.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.4f})'.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.grid()

    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, filename))


def plot_precision_recall_curve(y_test, y_score, labels, save=False, save_dir=None, filename=None):
    """绘制 精确率 (precision）与 召回率  (recall)  曲线"""

    n_classes = y_score.shape[1]

    precision = dict()
    recall = dict()

    fig = plt.figure(figsize=(14, 10))
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], lw=2, label='Precision-Recall for {} class)'.format(labels[i]))

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs. Recall curve')
    plt.legend(loc="best")
    plt.grid()
    
    fig.tight_layout()
    
    if save:
        fig.savefig(os.path.join(save_dir, filename))
