#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import graphviz

from sklearn import tree, datasets
from sklearn.model_selection import (train_test_split, cross_val_score,
                                    LearningCurveDisplay, ShuffleSplit)
from tp_arbres_source import (rand_gauss, rand_bi_gauss, rand_tri_gauss,
                              rand_checkers, rand_clown,
                              plot_2d, frontiere)


rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {'axes.labelsize': 6,
          'font.size': 12,
          'legend.fontsize': 12,
          'text.usetex': False,
          'figure.figsize': (5, 6)}
plt.rcParams.update(params)

sns.set_context("poster")
sns.set_palette("colorblind")
sns.set_style("white")
_ = sns.axes_style()

#%%
############################################################################
# Question 2
############################################################################

np.random.seed(1)

# Training data
n1 = 114
n2 = 114
n3 = 114
n4 = 114
data = rand_checkers(n1, n2, n3, n4)

plt.ion()
plt.title('Data set')
plot_2d(data[:, :2], data[:, 2], w=None)

X_train = data[:, :2]
Y_train = data[:, 2].astype(int)

# Decision trees and errors
dmax = 12
error_entropy = np.zeros(dmax)
error_gini = np.zeros(dmax)

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = i + 1)
    dt_entropy.fit(X_train, Y_train)
    error_entropy[i] = 1 - dt_entropy.score(X_train, Y_train)

    dt_gini = tree.DecisionTreeClassifier(criterion = "gini", max_depth = i + 1)
    dt_gini.fit(X_train, Y_train)
    error_gini[i] = 1 - dt_gini.score(X_train, Y_train)

plt.figure()
plt.plot(error_entropy * 100, 'g')
plt.plot(error_gini * 100, 'r')
plt.xlabel('Profondeur maximale')
plt.ylabel('Pourcentage d\'erreur')
plt.draw()

#%%
############################################################################
# Question 3
############################################################################

# Best calssification
dt_entropy.max_depth = np.where(error_entropy == min(error_entropy))[0][0] + 1
dt_entropy.fit(X_train, Y_train)

plt.figure()
frontiere(lambda x: dt_entropy.predict(x.reshape((1, -1))), X_train, Y_train, step=100)
plt.title("Best frontier with entropy criterion")
plt.draw()

#%%
############################################################################
# Question 4
############################################################################

# Exporting decision tree
dot_data = tree.export_graphviz(dt_entropy, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("Arbre")

#%%
############################################################################
# Question 5
############################################################################

# Test data
data_test = rand_checkers(40, 40, 40, 40)
X_test = data_test[:, :2]
Y_test = data_test[:, 2].astype(int)

# Decision trees and errors
dmax = 12
error_entropy = np.zeros(dmax)
error_gini = np.zeros(dmax)

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = i + 1)
    dt_entropy.fit(X_train, Y_train)
    error_entropy[i] = 1 - dt_entropy.score(X_test, Y_test)

    dt_gini = tree.DecisionTreeClassifier(criterion = "gini", max_depth = i + 1)
    dt_gini.fit(X_train, Y_train)
    error_gini[i] = 1 - dt_gini.score(X_test, Y_test)

plt.figure()
plt.plot(error_entropy * 100, 'g')
plt.plot(error_gini * 100, 'r')
plt.xlabel('Profondeur maximale')
plt.ylabel('Pourcentage d\'erreur')
plt.draw()
#%%
############################################################################
# Question 6
############################################################################

# Dataset
digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Training and testing data
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

X_train, X_test, Y_train, Y_test = train_test_split(data, digits.target, test_size = 0.8)

# Decision tree and errors on training data
dmax = 12
error_entropy = np.zeros(dmax)
error_gini = np.zeros(dmax)

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = i + 1)
    dt_entropy.fit(X_train, Y_train)
    error_entropy[i] = 1 - dt_entropy.score(X_train, Y_train)

    dt_gini = tree.DecisionTreeClassifier(criterion = "gini", max_depth = i + 1)
    dt_gini.fit(X_train, Y_train)
    error_gini[i] = 1 - dt_gini.score(X_train, Y_train)

plt.figure()
plt.plot(error_entropy * 100, 'g')
plt.plot(error_gini * 100, 'r')
plt.xlabel('Profondeur maximale')
plt.ylabel('Pourcentage d\'erreur')
plt.draw()

# Best classification examples
dt_entropy.max_depth = np.where(error_entropy == min(error_entropy))[0][0] + 1
dt_entropy.fit(X_train, Y_train)
predict = dt_entropy.predict(X_train)

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_train, predict):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

# Exporting decision tree
dot_data = tree.export_graphviz(dt_entropy, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("Arbre_digits")

# Errors on test data
dmax = 12
error_entropy = np.zeros(dmax)
error_gini = np.zeros(dmax)

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion = "entropy", max_depth = i + 1)
    dt_entropy.fit(X_train, Y_train)
    error_entropy[i] = 1 - dt_entropy.score(X_test, Y_test)

    dt_gini = tree.DecisionTreeClassifier(criterion = "gini", max_depth = i + 1)
    dt_gini.fit(X_train, Y_train)
    error_gini[i] = 1 - dt_gini.score(X_test, Y_test)

plt.figure()
plt.plot(error_entropy * 100, 'g')
plt.plot(error_gini * 100, 'r')
plt.xlabel('Profondeur maximale')
plt.ylabel('Pourcentage d\'erreur')
plt.draw()

#%%
############################################################################
# Question 7
############################################################################

# Dataset
X = digits.data
Y = digits.target

# Scores
dmax = 12
N = 5
score_entropy = np.zeros((dmax,N))
score_gini = np.zeros((dmax,N))

for i in range(dmax):
    dt_entropy = tree.DecisionTreeClassifier(criterion = "entropy",
    max_depth = i + 1)
    score_entropy[i] = cross_val_score(dt_entropy, X, Y, cv=N)

    dt_gini = tree.DecisionTreeClassifier(criterion = "gini",
    max_depth = i + 1)
    score_gini[i] = cross_val_score(dt_gini, X, Y, cv=5)

score_mean_entropy = score_entropy.mean(axis=1)
score_mean_gini = score_gini.mean(axis=1)

print("Moyennes des scores avec l'entropie : \n", np.round(score_mean_entropy, 3))
print("Moyennes des scores avec l'indice de Gini : \n", np.round(score_mean_gini, 3))

plt.figure()
plt.plot(score_mean_entropy, 'g')
plt.plot(score_mean_gini, 'r')
plt.xlabel('Profondeur maximale')
plt.ylabel('Score')
plt.draw()

# Max depth
dmax_entropy = np.where(score_mean_entropy == max(score_mean_entropy))[0][0] + 1
dmax_gini = np.where(score_mean_gini == max(score_mean_gini))[0][0] + 1

print("Profondeur maximale avec l'entropie :", dmax_entropy,
"\n Score :", max(score_mean_entropy))
print("Profondeur maximale avec l'indice de Gini :", dmax_gini,
"\n Score :", max(score_mean_gini))
#%%
############################################################################
# Question 8
############################################################################

# Learning curves
dt_entropy = tree.DecisionTreeClassifier(criterion = "entropy",
max_depth = dmax_entropy)
dt_gini = tree.DecisionTreeClassifier(criterion = "gini",
max_depth = dmax_gini)

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)

common_params = {
    "X": X,
    "y": Y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
    "score_type": "both",
    "n_jobs": 4,
    "line_kw": {"marker": "o"},
    "std_display_style": "fill_between",
    "score_name": "Score",
}

for ax_idx, estimator in enumerate([dt_entropy, dt_gini]):
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
    handles, label = ax[ax_idx].get_legend_handles_labels()
    ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
    ax[ax_idx].set_title(f"Courbe d'apprentissage {estimator.criterion}")