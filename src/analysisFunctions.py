import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=15)
        plt.yticks(tick_marks, target_names, fontsize=15)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 10 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=30)
    plt.show()


def showResultsOC(y_pred, y_true):
    """
    Print statistics for One Class problems
    """
    from sklearn.metrics import (
        accuracy_score, recall_score,
        precision_score, f1_score,
        confusion_matrix
    )
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    print("Accuracy: {:.4f}".format(accuracy_score(y_true, y_pred)))
    print("Precision: {:.4f}".format(precision_score(y_true, y_pred)))
    print("Recal: {:.4f}".format(recall_score(y_true, y_pred)))
    print("F1-score: {:.4f}".format(f1_score(y_true, y_pred)))

    cm = confusion_matrix(y_pred, y_true).T
    df_cm = pd.DataFrame(cm, ['Other', 'Marvin'], ['Other', 'Marvin'])

    fig = plt.figure(figsize=(6, 6))
    _ = sns.heatmap(df_cm, cmap='Blues', fmt='g', annot=True, annot_kws={"size": 16}, cbar=False)
    plt.title('Confusion matrix', fontsize=20)
    plt.ylabel('True class', fontsize=15)
    plt.xlabel('Predicted class', fontsize=15)

    plt.tight_layout()
    plt.show()
