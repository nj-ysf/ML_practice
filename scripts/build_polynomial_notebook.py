"""Construit le notebook pédagogique à partir des fonctions réellement présentes."""
import ast
import json
from pathlib import Path
import textwrap
import uuid

ROOT = Path(__file__).resolve().parents[1]
TARGET = next(ROOT.glob('*polynomiale*.ipynb'))
cells = []


def md(s):
    cells.append(dict(cell_type='markdown', id=uuid.uuid4().hex[:8], metadata={}, source=textwrap.dedent(s).strip()))


def code(s, explanation):
    cells.append(dict(cell_type='code', id=uuid.uuid4().hex[:8], metadata={}, source=textwrap.dedent(s).strip(), execution_count=None, outputs=[]))
    md(explanation)


md(r'''
# 1. Régression polynomiale multivariée : Advertising

Prédire **Sales** à partir de **TV, Radio, Newspaper**, comparer une régression
linéaire multiple et trois modèles polynomiaux (sans pénalité, LASSO, RIDGE).
Toutes les transformations et sélections sont ajustées sur l'entraînement.
Le test de 20 % est réservé à l'évaluation. Exécuter les cellules dans l'ordre,
depuis le dossier du projet. Dépendances : NumPy, pandas, matplotlib, scikit-learn,
IPython et threadpoolctl. Scikit-learn fournit seulement la séparation et la standardisation.

La sélection utilise des validations manuelles reproductibles. Les coefficients
sont appris sans estimateur de régression prêt à l'emploi ni PolynomialFeatures.
''')
md(r'''
## 2. Analyse des fonctions existantes

Inspection des cellules Python et Markdown effectuée avant écriture : le notebook
cible ne contenait que `print("hello world ")`. Les notebooks Advertising, TP3,
TP4 et `multiple_linear_regression_from_scratch.ipynb` contiennent déjà les briques utiles.
La source retenue est **tp4_multiple_linear_regression_scaling.ipynb**, dont la
standardisation est ajustée sur le train et le biais ajouté en dernière colonne.
Les cinq fonctions ci-dessous sont recopiées **sans modification de leur corps**
depuis cette source pour conserver un notebook autonome ; elles ne sont pas réimplémentées.

| Fonction / outil existant | Rôle | Réutilisation / adaptation |
|---|---|---|
| `model(X, theta)` (cellule 24, indice Python) | X @ theta | Toutes les prédictions |
| `cost_function` (26) | somme des carrés / (2m) | MSE = 2 × coût ; base des pénalités |
| `grad` (30) | Xᵀ(Xθ−y)/m | Descente existante et gradient LASSO multiplié par 2 |
| `gradient_descent` (32) | Descente à pas fixe + historique | Modèles non pénalisés, dans des coordonnées QR bien conditionnées |
| `coef_determination` (40) | R² | Toutes les évaluations |
| `StandardScaler` (18) | (X−μ)/σ | Même outil, réajusté dans chaque fold |
| `train_test_split` (16) | Partition 80/20, graine 42 | Appelé une seule fois |
| MSE et R² sklearn (49 et suivantes) | Évaluation | MSE existant via coût, R² via fonction ; ajout RMSE et MAE |

Conventions : A : (m, p+1), y : (m, 1), θ : (p+1, 1), biais θ[-1].
Le notebook `multiple_linear_regression_from_scratch` place au contraire le biais
en première colonne : ne pas mélanger ces conventions. La classe du notebook
`advertising_linear_regression_gradient_descent` calcule à tort le carré de la
somme des résidus ; son coût n'est pas réutilisé. Aucune fonction de construction
polynomiale ou de pénalisation compatible n'a été trouvée dans les notebooks inspectés.
''')
code('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations_with_replacement
from math import comb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import display, Markdown
from threadpoolctl import threadpool_limits

SEED = 42
np.random.seed(SEED)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)
plt.rcParams.update({'figure.figsize': (8, 4), 'axes.grid': True})
''', 'Les outils externes ne construisent ni les monômes ni les modèles. Les graines rendent les partitions reproductibles.')
source = json.loads((ROOT / 'tp4_multiple_linear_regression_scaling.ipynb').read_text(encoding='utf-8'))
names = {'model', 'cost_function', 'grad', 'gradient_descent', 'coef_determination'}
original = []
for c in source['cells']:
    if c['cell_type'] == 'code':
        s = ''.join(c['source'])
        try:
            tree = ast.parse(s)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                original.append(ast.get_source_segment(s, node))
assert len(original) == 5
code('\n\n'.join(original), r'''Ces fonctions originales restent inchangées. Le coût vaut MSE/2, et non MSE.
La colonne de biais est ajoutée ailleurs ; `model` ne l'ajoute jamais automatiquement.
R² peut être négatif. Pour une cible constante, il est indéfini ; l'évaluation le signale par NaN.''')
md('## 3. Chargement et exploration des données')
code('''
data_path = Path('data/Advertising.csv')
if not data_path.exists():
    raise FileNotFoundError('Ouvrir le notebook depuis le dossier ML_Algos contenant data/.')
dataset = pd.read_csv(data_path)
features = ['TV', 'Radio', 'Newspaper']
columns = features + ['Sales']
assert set(columns).issubset(dataset.columns)
display(dataset.head())
display(pd.DataFrame({'type': dataset.dtypes.astype(str), 'manquants': dataset.isna().sum()}))
display(dataset[columns].describe())
X = dataset[features].to_numpy(dtype=float)
y = dataset[['Sales']].to_numpy(dtype=float)
assert np.isfinite(X).all() and np.isfinite(y).all(), 'Valeurs manquantes/non finies : traiter dans les folds.'
print('Dataset :', dataset.shape, 'X :', X.shape, 'y :', y.shape)
print('Colonnes exclues :', sorted(set(dataset.columns) - set(columns)))
''', 'La sélection explicite des trois budgets exclut toute colonne d’index éventuelle. Les valeurs non finies entraînent un arrêt explicite : aucune imputation n’est ajustée sur le test.')
md(r'''
## 4. Prétraitement et séparation train/test

Une seule séparation 80/20. Dans chaque fold : standardiser les trois budgets,
générer leurs monômes, standardiser ces monômes, puis ajouter une seule constante.
La première standardisation limite les grandes puissances ; la seconde rend les
pénalités comparables. Les monômes sont donc exprimés dans les budgets centrés-réduits.
Pour un modèle complet non pénalisé, ce changement de coordonnées conserve l'espace
polynomial ; pour les modèles pénalisés, il définit explicitement la base de pénalisation.

Les fonctions techniques sont définies ici car la sélection du degré en a besoin.
La section 10 en détaille et vérifie la construction sans redéfinition.
''')
code('''
train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED)
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]
assert not set(train_indices) & set(test_indices)
print('Train :', X_train.shape, y_train.shape, 'Test :', X_test.shape, y_test.shape)
''', 'Les indices sont fixés une seule fois et partagés par tous les modèles. Le test ne participe ni au choix du degré ni au réglage de λ.')
code('''
def polynomial_features(X, degree):
    """Construire tous les monômes non constants de degré total 1 à degree.

    Paramètres : X, ndarray fini (m, q) ; degree, entier >= 1.
    Retour : P, ndarray (m, comb(q+degree, degree)-1), et exposants (p, q).
    Formule : P[i,j] = produit_k X[i,k] ** exposants[j,k].
    Étapes : combinaisons d'indices avec répétition, comptage des exposants,
    puis produit vectorisé. Complexité O(m*p*q), mémoire O(m*p*q).
    Réutilisation : NumPy ; aucune fonction polynomiale existante disponible.
    Éviter : constante dupliquée, puissances seules sans interactions,
    grands budgets non standardisés et exposants de degré total trop grand.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('X doit être une matrice finie (m, q).')
    if not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError('degree doit être un entier positif.')
    q = X.shape[1]
    powers = np.array([np.bincount(c, minlength=q)
                       for d in range(1, degree + 1)
                       for c in combinations_with_replacement(range(q), d)])
    P = np.prod(X[:, None, :] ** powers[None, :, :], axis=2)
    assert P.shape == (len(X), comb(q + degree, degree) - 1)
    assert np.isfinite(P).all()
    return P, powers


def design_matrix(X, degree, state=None):
    """Ajuster ou appliquer le prétraitement et ajouter le biais final.

    Paramètres : X float (m,3), degree int ; state None pour ajuster sur train,
    ou tuple (StandardScaler budgets, StandardScaler monômes) déjà ajusté.
    Retour : A (m,p+1) et state (tuple), p=comb(3+degree,degree)-1.
    Formule : A = [standardiser(P(standardiser(X))), 1].
    Étapes : budgets, monômes, monômes standardisés, constante.
    Réutilise StandardScaler du TP4 et polynomial_features ; O(m*p*3).
    Éviter : fit sur validation/test, standardiser le biais, changer degree
    lors de l'application d'un état déjà ajusté.
    """
    if state is None:
        raw = StandardScaler().fit(X)
        P, _ = polynomial_features(raw.transform(X), degree)
        expanded = StandardScaler().fit(P)
        state = (raw, expanded)
    else:
        raw, expanded = state
        P, _ = polynomial_features(raw.transform(X), degree)
    A = np.column_stack((expanded.transform(P), np.ones(len(X))))
    assert np.isfinite(A).all()
    return A, state


def evaluate(A, y, theta):
    """Évaluer un modèle avec les métriques dans l'unité originale de Sales.

    Paramètres : A float (m,p+1), y float (m,1), theta float (p+1,1).
    Retour : dict de quatre scalaires MSE, RMSE, MAE, R2.
    Formules : MSE=2*cost_function ; RMSE=sqrt(MSE) ; MAE=mean(|y-pred|).
    Étapes : contrôler les dimensions, prédire, agréger les erreurs.
    Réutilise model, cost_function, coef_determination ; O(m*p).
    Éviter : y (m,) face à pred (m,1), qui produit un broadcasting erroné.
    R² renvoie NaN pour une cible constante.
    """
    assert y.shape == (len(A), 1) and theta.shape == (A.shape[1], 1)
    pred = model(A, theta)
    assert np.isfinite(pred).all()
    mse = 2 * cost_function(A, y, theta)
    return {'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': np.mean(np.abs(y-pred)),
            'R2': coef_determination(y, pred) if np.var(y) > 0 else np.nan}


def fit_unregularized(A, y):
    """Réutiliser la descente originale après un changement de coordonnées QR.

    Paramètres : A float (m,p+1), de rang plein avec m>=p+1 ; y float (m,1).
    Retour : theta (p+1,1), historique demi-MSE (60,), diagnostic dict.
    Formules : A=QR, Z=sqrt(m)*Q, theta=sqrt(m)*R^-1*gamma.
    ZᵀZ/m=I : le pas 0.5 est stable et réduit l'erreur des paramètres par 2.
    Étapes : vérifier rang, QR, gradient_descent sur Z, retour aux coefficients
    originaux via solve ; contrôle du gradient et du coût dans A.
    Réutilise gradient_descent, grad, cost_function, model. O(m*p² + p³).
    Éviter : appliquer QR séparément au test, pénaliser gamma au lieu de theta,
    inverser AᵀA ou ignorer une déficience de rang. QR ne change pas le modèle.
    """
    assert y.shape == (len(A), 1)
    if np.linalg.matrix_rank(A) < A.shape[1]:
        raise ValueError('Matrice de rang insuffisant : réduire le degré.')
    Q, R = np.linalg.qr(A, mode='reduced')
    Z = np.sqrt(len(A)) * Q
    gamma, history = gradient_descent(Z, y, np.zeros((A.shape[1], 1)), 0.5, 60)
    theta = np.linalg.solve(R, np.sqrt(len(A)) * gamma)
    residual_gradient = np.max(np.abs(grad(A, y, theta)))
    assert np.isfinite(history).all() and np.isfinite(theta).all()
    assert np.max(np.diff(history)) < 1e-9
    assert residual_gradient < 1e-6
    return theta, history, {'iterations': 60, 'gradient_inf': residual_gradient,
                            'condition': np.linalg.cond(A)}


def make_folds(n, k=5, seed=SEED):
    """Créer un K-fold mélangé, manuel et reproductible.

    Paramètres : n int nombre d'observations, k int dans [2,n], seed int.
    Retour : liste de k tuples (indices train, indices validation), arrays 1D.
    Principe : permutation puis k blocs disjoints ; train = complément du bloc.
    Étapes : mélanger, découper, construire les compléments. O(k*n).
    Réutilisation : aucune fonction K-fold maison trouvée ; NumPy seulement.
    Éviter : utiliser les indices du test ou recréer les folds selon le degré.
    """
    if not 2 <= k <= n:
        raise ValueError('Il faut 2 <= k <= n.')
    blocks = np.array_split(np.random.default_rng(seed).permutation(n), k)
    return [(np.concatenate([b for j, b in enumerate(blocks) if j != i]), b)
            for i, b in enumerate(blocks)]


def degree_cv(X, y, degrees, folds):
    """Comparer les degrés sur les mêmes folds sans fuite de prétraitement.

    Paramètres : X (n,3), y (n,1), degrees séquence d'entiers, folds liste
    de tuples d'indices locaux à X. Retour : DataFrame, une ligne par degré/fold.
    Formule : MSE_validation = mean((y_validation - A_validation*theta)**2).
    Étapes : ajuster design_matrix sur train du fold, transformer validation,
    fit_unregularized, evaluate sur les deux parties ; O(D*K*n*p²).
    Réutilise design_matrix, fit_unregularized et evaluate (donc fonctions TP4).
    Éviter : scaler global et mélange des indices de folds avec le test final.
    """
    rows = []
    for degree in degrees:
        for fold, (tr, va) in enumerate(folds):
            A, state = design_matrix(X[tr], int(degree))
            V, _ = design_matrix(X[va], int(degree), state)
            theta, _, info = fit_unregularized(A, y[tr])
            rows.append({'Degré': degree, 'Fold': fold,
                         'MSE entraînement': evaluate(A, y[tr], theta)['MSE'],
                         'MSE validation': evaluate(V, y[va], theta)['MSE'],
                         'Gradient inf': info['gradient_inf']})
    return pd.DataFrame(rows)
''', r'''Ces fonctions ajoutent seulement les éléments absents. Le QR est un préconditionnement
de la descente existante, pas un nouvel estimateur : la fonction `gradient_descent`
reste appelée. Avec ZᵀZ/m = I, 60 mises à jour à pas 0,5 suffisent à la précision
numérique. Cela évite de confondre mauvaise convergence et mauvais degré pour les
monômes corrélés. Le test est toujours prédit avec A et θ, jamais avec un QR ajusté au test.
Les pénalités seront appliquées aux coefficients standardisés θ, et non aux coordonnées QR.''')
md('## 5. Modèle de référence : régression linéaire multiple')
code('''
A_linear, state_linear = design_matrix(X_train, 1)
theta_linear, history_linear, info_linear = fit_unregularized(A_linear, y_train)
print('Matrice avec intercept :', A_linear.shape, 'theta :', theta_linear.shape)
print('Diagnostics :', info_linear)
display(pd.DataFrame([evaluate(A_linear, y_train, theta_linear)], index=['Linéaire train']))
''', 'Le degré 1 contient exactement les trois variables et le biais. L’évaluation du test est reportée après la sélection.')
md(r'''
## 6. Première technique : K-fold fixe multivarié

K=5, soit 128 observations d'ajustement et 32 de validation par fold pour ce dataset.
Les degrés 1 à 6 contiennent respectivement 3, 9, 19, 34, 55 et 83 monômes, hors
intercept. Même le degré 6 reste ajustable, mais risque une variance élevée.
L'écart-type est calculé entre folds (ddof=1) : c'est un indicateur de dispersion,
pas un intervalle de confiance indépendant.
''')
code('''
degrees = list(range(1, 7))
folds = make_folds(len(y_train))
with threadpool_limits(limits=1):
    cv_fixed = degree_cv(X_train, y_train, degrees, folds)
degree_table = cv_fixed.groupby('Degré').agg(**{
    'MSE entraînement': ('MSE entraînement', 'mean'),
    'MSE validation moyen': ('MSE validation', 'mean'),
    'Écart-type validation': ('MSE validation', 'std')})
degree_table['Caractéristiques'] = [comb(3+d, d)-1 for d in degrees]
degree_fixed = int(degree_table['MSE validation moyen'].idxmin())
display(degree_table)
print('Minimum du K-fold fixe :', degree_fixed)
''', 'Les MSE d’entraînement sont les moyennes des erreurs sur les sous-ensembles d’ajustement des folds, et non sur les 160 observations réunies.')
md(r'''
## 7. Deuxième technique : K-fold multivarié répété

« Validation croisée multivariée » n'est pas un algorithme différent du premier
K-fold : la méthode 1 utilise déjà les trois variables **et leurs interactions**.
La distinction retenue est donc un **5-fold répété sur trois permutations**
(graines 101, 202, 303), pour examiner la sensibilité au découpage.
Les trois répétitions partagent les mêmes partitions pour tous les degrés.
Le degré final sera celui qui minimise le MSE moyen répété ; le K-fold fixe sert
de comparaison. Cette règle est fixée avant de consulter le test.
''')
code('''
repeated_folds = [fold for seed in [101, 202, 303]
                  for fold in make_folds(len(y_train), seed=seed)]
with threadpool_limits(limits=1):
    cv_repeated = degree_cv(X_train, y_train, degrees, repeated_folds)
repeat_table = cv_repeated.groupby('Degré').agg(**{
    'MSE entraînement': ('MSE entraînement', 'mean'),
    'MSE validation moyen': ('MSE validation', 'mean'),
    'Écart-type validation': ('MSE validation', 'std')})
selected_degree = int(repeat_table['MSE validation moyen'].idxmin())
display(repeat_table)
per_repeat = cv_repeated.assign(Répétition=cv_repeated['Fold']//5).groupby(
    ['Répétition', 'Degré'])['MSE validation'].mean().unstack()
display(per_repeat)
print('Degrés gagnants par répétition :', per_repeat.idxmin(axis=1).tolist())
''', 'Les 15 scores ne sont pas indépendants puisque les observations réapparaissent. La répétition mesure une sensibilité aux partitions, sans constituer une validation externe imbriquée.')
md('## 8. Troisième technique : analyse empirique biais–variance')
code('''
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, table, title in zip(axes, [degree_table, repeat_table], ['K-fold fixe', 'K-fold répété']):
    ax.plot(table.index, table['MSE entraînement'], 'o-', label='Entraînement')
    ax.plot(table.index, table['MSE validation moyen'], 'o-', label='Validation')
    ax.set(xlabel='Degré', ylabel='MSE (échelle logarithmique)', title=title, yscale='log')
    ax.legend()
plt.tight_layout()
plt.show()
''', r'''Un modèle trop simple présente des erreurs train et validation élevées : sous-apprentissage.
Un modèle qui réduit l'erreur train mais augmente l'erreur validation surapprend.
Le compromis se situe près du minimum de validation, avec sa dispersion à examiner
dans les tableaux. Cette lecture est un diagnostic complémentaire des scores CV,
pas une estimation formelle séparée du biais et de la variance statistique.''')
md('## 9. Choix final du degré')
code('''
display(Markdown(f"Le K-fold fixe choisit **{degree_fixed}** ; le K-fold répété choisit "
                 f"**{selected_degree}**, retenu selon la règle annoncée. "
                 f"Son MSE CV répété est **{repeat_table.loc[selected_degree, 'MSE validation moyen']:.4f}**, "
                 f"avec un écart-type entre folds de **{repeat_table.loc[selected_degree, 'Écart-type validation']:.4f}**. "
                 f"Il comporte **{comb(3+selected_degree, selected_degree)-1}** caractéristiques hors biais. "
                 "Le minimum d'erreur d'entraînement ne décide pas du choix."))
''', 'Un désaccord entre partitions doit être interprété comme une incertitude du choix. La moyenne répétée fournit ici la règle de décision préétablie, sans intervention du test.')
md(r'''
## 10. Construction des caractéristiques polynomiales from scratch

Pour chaque degré total t, les combinaisons avec répétition des indices de variables
énumèrent chaque monôme exactement une fois. Le vecteur (a,b,c) produit
$z_{TV}^a z_{Radio}^b z_{Newspaper}^c$, avec $1\le a+b+c\le d$.
Il y a $p=\binom{3+d}{d}-1$ colonnes non constantes. Au degré 2 : les trois
variables, TV², TV×Radio, TV×Newspaper, Radio², Radio×Newspaper, Newspaper².
L'ordre diffère de l'énoncé mais l'ensemble est identique. Une seule constante
est ajoutée par `design_matrix`, après standardisation.
''')
code('''
example = np.array([[2., 3., 5.]])
P2, powers2 = polynomial_features(example, 2)
assert P2.shape == (1, 9)
assert np.array_equal(P2[0], [2, 3, 5, 4, 6, 10, 9, 15, 25])
assert len(np.unique(powers2, axis=0)) == 9
display(pd.DataFrame(powers2, columns=['Exposant TV', 'Exposant Radio', 'Exposant Newspaper']))
print('Valeurs des neuf monômes sur [2,3,5] :', P2)
''', 'Ce contrôle vérifie réellement les interactions et leur ordre. La génération est celle définie en section 4 ; aucune seconde implémentation n’est introduite.')
md('## 11. Entraînement du modèle polynomial')
code('''
A_poly, state_poly = design_matrix(X_train, selected_degree)
with threadpool_limits(limits=1):
    theta_poly, history_poly, info_poly = fit_unregularized(A_poly, y_train)
print('A :', A_poly.shape, 'theta :', theta_poly.shape, 'y :', y_train.shape)
print('Diagnostics :', info_poly)
plt.semilogy(2*history_linear, label='Linéaire')
plt.semilogy(2*history_poly, label='Polynomial')
plt.xlabel('Itération de descente dans les coordonnées QR')
plt.ylabel('MSE entraînement')
plt.legend()
plt.show()
''', 'Le gradient résiduel et la décroissance du coût contrôlent la convergence. Les monômes corrélés rendent la descente directe lente ; le QR préconditionne le problème tout en réutilisant la boucle du notebook d’origine.')
md('## 12. Comparaison avec la régression linéaire multiple')
code('''
T_linear, _ = design_matrix(X_test, 1, state_linear)
T_poly, _ = design_matrix(X_test, selected_degree, state_poly)
initial_rows = []
for name, A, T, theta in [('Linéaire', A_linear, T_linear, theta_linear),
                           ('Polynomial', A_poly, T_poly, theta_poly)]:
    scores = evaluate(T, y_test, theta)
    initial_rows.append({'Modèle': name, 'MSE train': evaluate(A, y_train, theta)['MSE'],
                         **{k+' test': v for k, v in scores.items()}})
initial_comparison = pd.DataFrame(initial_rows).set_index('Modèle')
display(initial_comparison)
''', 'Le test est consulté après le choix du degré. Ses résultats ne seront pas utilisés pour choisir λ ; les grilles et critères ci-dessous sont fixés indépendamment.')
md(r'''
## 13. LASSO : implémentation et réglage

On minimise $J_1(\theta)=\mathrm{MSE}+\lambda\sum_{j<p}|\theta_j|$,
avec MSE = somme des carrés / m. Le biais, dernier coefficient, n'est pas pénalisé.
La fonction coût originale représente la moitié du MSE : il faut donc multiplier
son coût **et son gradient par deux**. Sa boucle de descente ne sait pas appliquer
le proximal L1 : une boucle spécialisée est nécessaire, tout en réutilisant `grad`.

Le proximal est le seuillage doux $S_t(u)=\operatorname{sign}(u)\max(|u|-t,0)$.
On emploie le gradient proximal accéléré (FISTA), pas une dérivée arbitraire en zéro.
Pas $1/L$, avec $L=2\lambda_{max}(A^TA/m)$ ; arrêt sur les conditions KKT,
pas sur une simple petite variation de coût. Budget maximal : 100 000 itérations.
Un échec de convergence arrête le calcul au lieu de produire des scores trompeurs.

Procédure simplifiée : degré fixé par CV répétée, λ choisi sur le K-fold fixe
de l'entraînement. La même grille logarithmique 10⁻⁵ à 10¹ est utilisée pour les
deux pénalités. Il ne s'agit **pas** d'une CV imbriquée : les meilleurs scores CV
peuvent être optimistes après sélection du degré et de λ. Le test indépendant
sert d'évaluation finale. Une CV externe imbriquée serait utile pour quantifier
la performance de toute la procédure sur plusieurs partitions, au prix de nombreux
réajustements. Ici, on privilégie la lisibilité et on annonce cette limite.
''')
code('''
def penalized_cost(A, y, theta, lam, penalty):
    """Calculer MSE + lambda*norme, sans pénaliser le biais final.

    Paramètres : A (m,p+1), y (m,1), theta (p+1,1), lam float >=0,
    penalty str 'LASSO' ou 'RIDGE'. Retour : float scalaire.
    Formules : 2*cost_function + lam*sum(abs(theta[:-1])) pour L1,
    ou +lam*sum(theta[:-1]**2) pour L2.
    Étapes : contrôler les arguments, calculer coût et pénalité ; O(m*p).
    Réutilise cost_function, donc model. Éviter : pénaliser theta[-1],
    confondre MSE et demi-MSE, comparer les objectifs entre lambda différents.
    """
    if lam < 0 or penalty not in ('LASSO', 'RIDGE'):
        raise ValueError('Pénalité inconnue ou lambda négatif.')
    weights = theta[:-1]
    norm = np.abs(weights).sum() if penalty == 'LASSO' else (weights**2).sum()
    return 2*cost_function(A, y, theta) + lam*norm


def fit_lasso(A, y, lam, tol=1e-6, max_iter=100000):
    """Optimiser L1 par gradient proximal accéléré (FISTA).

    Paramètres : A float (m,p+1), y float (m,1), lam float >0,
    tol float tolérance KKT absolue, max_iter int budget.
    Retour : theta (p+1,1), historique objectif (k,2) [itération,coût],
    diagnostic dict (iterations, kkt_inf).
    Formules : u=z-(2/L)*grad(A,y,z), theta[:-1]=S_(lam/L)(u[:-1]).
    Accélération : t_new=(1+sqrt(1+4*t²))/2 ; z=theta+(t-1)/t_new*(theta-old).
    KKT : g_j+lam*sign(theta_j)=0 si non nul, |g_j|<=lam sinon ; g_biais=0.
    Étapes : pas spectral, proximal, extrapolation, test KKT tous les 25 pas.
    Réutilise grad et penalized_cost, donc model/cost_function. O(iter*m*p).
    Éviter : seuil sur le biais, gradient demi-MSE, arrêt avant convergence.
    FISTA peut avoir un coût non monotone ; le critère est la stationnarité KKT.
    """
    assert lam > 0 and y.shape == (len(A), 1)
    L = 2*np.linalg.eigvalsh(A.T @ A / len(A))[-1]
    theta = np.zeros((A.shape[1], 1))
    z, t = theta.copy(), 1.
    history = []
    for iteration in range(1, max_iter+1):
        u = z - 2*grad(A, y, z)/L
        new = u.copy()
        new[:-1] = np.sign(u[:-1])*np.maximum(np.abs(u[:-1])-lam/L, 0)
        t_new = (1+np.sqrt(1+4*t*t))/2
        z = new + (t-1)/t_new*(new-theta)
        theta, t = new, t_new
        if iteration % 25 == 0:
            g = 2*grad(A, y, theta)
            w = theta[:-1]
            violation = np.where(w != 0, np.abs(g[:-1]+lam*np.sign(w)),
                                 np.maximum(np.abs(g[:-1])-lam, 0))
            kkt = max(float(violation.max()), float(abs(g[-1, 0])))
            objective = penalized_cost(A, y, theta, lam, 'LASSO')
            assert np.isfinite(objective)
            history.append((iteration, objective))
            if kkt < tol:
                return theta, np.array(history), {'iterations': iteration, 'kkt_inf': kkt}
    raise RuntimeError(f'LASSO non convergé : lambda={lam}, KKT={kkt}')


def tune_penalty(X, y, degree, folds, lambdas, solver):
    """Sélectionner lambda sur le train avec une fonction d'optimisation fournie.

    Paramètres : X (n,3), y (n,1), degree int, folds liste d'indices,
    lambdas array positif (l,), solver callable(A,y,lambda).
    Retour : résumé DataFrame (l,3), lambda optimal float, détails DataFrame.
    Formule : lambda*=argmin moyenne_folds MSE_validation(lambda).
    Étapes : ajuster les scalers une fois par fold, entraîner pour chaque lambda,
    évaluer sans ajouter de pénalité au score prédictif, agréger. Coût K*l fits.
    Réutilise design_matrix, evaluate et solver, donc fonctions d'origine.
    Éviter : choisir lambda sur test ou comparer MSE+penalité entre modèles.
    """
    rows = []
    for fold, (tr, va) in enumerate(folds):
        A, state = design_matrix(X[tr], degree)
        V, _ = design_matrix(X[va], degree, state)
        for lam in lambdas:
            theta, _, info = solver(A, y[tr], float(lam))
            rows.append({'lambda': lam, 'fold': fold,
                         'MSE train': evaluate(A, y[tr], theta)['MSE'],
                         'MSE validation': evaluate(V, y[va], theta)['MSE'], **info})
    details = pd.DataFrame(rows)
    table = details.groupby('lambda').agg(**{'MSE train': ('MSE train', 'mean'),
                 'MSE validation': ('MSE validation', 'mean'),
                 'Écart-type': ('MSE validation', 'std')})
    return table, float(table['MSE validation'].idxmin()), details
''', 'La pénalité L1 produit des zéros exacts grâce au proximal. Le contrôle KKT inclut le biais non pénalisé. Le réglage mesure le MSE prédictif, jamais la fonction objectif pénalisée.')
code('''
lambdas = np.logspace(-5, 1, 13)
LAMBDA = 1.0  # lambda imposé pour LASSO et RIDGE (la CV reste affichée à titre indicatif)
with threadpool_limits(limits=1):
    lasso_cv, _, lasso_details = tune_penalty(
        X_train, y_train, selected_degree, folds, lambdas, fit_lasso)
    lambda_lasso = lambda_ridge = LAMBDA
    theta_lasso, history_lasso, info_lasso = fit_lasso(A_poly, y_train, lambda_lasso)
display(lasso_cv)
print('Lambda LASSO :', lambda_lasso, 'Diagnostics :', info_lasso)
print('Pire résidu KKT en CV :', lasso_details['kkt_inf'].max())
print('Coefficients exactement nuls (hors biais) :', np.sum(theta_lasso[:-1] == 0))
''', 'Le degré reste fixé. Les paramètres de standardisation sont appris séparément dans chaque fold, puis réajustés sur le train complet pour le modèle final.')
md(r'''
## 14. RIDGE : implémentation et réglage

$J_2=\mathrm{MSE}+\lambda\sum_{j<p}\theta_j^2$.
Avec $D=\operatorname{diag}(1,\ldots,1,0)$, le gradient vaut
$2\,\mathrm{grad}(A,y,\theta)+2\lambda D\theta$.
La solution satisfait $(A^TA+m\lambda D)\theta=A^Ty$.
On résout le problème équivalent augmenté $[A;\sqrt{m\lambda}D]$ et $[y;0]$
avec `numpy.linalg.lstsq`, sans inversion explicite ni estimateur sklearn.
Cette solution analytique est un ajout spécialisé justifié pour RIDGE : elle
évite une nouvelle boucle de gradient et les lenteurs des directions corrélées.
Les modèles non régularisés continuent d'utiliser la descente d'origine.
''')
code('''
def fit_ridge(A, y, lam):
    """Résoudre RIDGE par moindres carrés augmentés, sans pénaliser le biais.

    Paramètres : A float (m,p+1), y float (m,1), lam float >=0.
    Retour : theta (p+1,1), historique None (solution directe), diagnostic dict.
    Formule : min ||[A;sqrt(m*lam)D]theta-[y;0]||², D[-1,-1]=0.
    Étapes : construire D, empiler, résoudre lstsq, vérifier stationnarité.
    Réutilise grad pour le contrôle du gradient et penalized_cost pour le coût.
    Complexité O((m+p)*p²). Éviter : inverse de AᵀA, oubli du facteur m,
    pénalité du biais ou confusion entre solution directe et descente itérative.
    """
    assert lam >= 0 and y.shape == (len(A), 1)
    D = np.eye(A.shape[1])
    D[-1, -1] = 0
    augmented = np.vstack((A, np.sqrt(len(A)*lam)*D))
    target = np.vstack((y, np.zeros((A.shape[1], 1))))
    theta = np.linalg.lstsq(augmented, target, rcond=None)[0]
    g = 2*grad(A, y, theta) + 2*lam*(D@theta)
    residual = float(np.max(np.abs(g)))
    assert residual < 1e-6 and np.isfinite(theta).all()
    return theta, None, {'gradient_inf': residual,
                          'objectif': penalized_cost(A, y, theta, lam, 'RIDGE')}
''', 'Les colonnes corrélées sont stabilisées par la pénalité quadratique. Contrairement à LASSO, RIDGE ne cherche pas à annuler exactement les poids.')
code('''
with threadpool_limits(limits=1):
    ridge_cv, _, ridge_details = tune_penalty(
        X_train, y_train, selected_degree, folds, lambdas, fit_ridge)
    theta_ridge, _, info_ridge = fit_ridge(A_poly, y_train, lambda_ridge)
display(ridge_cv)
print('Lambda RIDGE :', lambda_ridge, 'Diagnostics :', info_ridge)
print('Pire gradient en CV :', ridge_details['gradient_inf'].max())
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, table, best, name in zip(axes, [lasso_cv, ridge_cv], [lambda_lasso, lambda_ridge], ['LASSO', 'RIDGE']):
    ax.semilogx(table.index, table['MSE train'], 'o-', label='Train')
    ax.errorbar(table.index, table['MSE validation'], yerr=table['Écart-type'], fmt='o-', label='Validation ± écart-type')
    ax.axvline(best, color='black', linestyle='--', label='λ choisi')
    ax.set(title=name, xlabel='λ', ylabel='MSE')
    ax.legend()
plt.tight_layout()
plt.show()
plt.plot(history_lasso[:, 0], history_lasso[:, 1])
plt.xlabel('Itération LASSO')
plt.ylabel('MSE + pénalité L1')
plt.title('Convergence du modèle LASSO final')
plt.show()
''', 'Une pénalité trop forte augmente le biais. Les barres montrent la dispersion des folds, sans prétendre fournir des intervalles de confiance. Les deux grilles ont la même convention MSE, mais les valeurs de λ des pénalités L1 et L2 ne sont pas directement équivalentes.')
md('## 15. Comparaison globale des quatre modèles')
code('''
models = {'Linéaire': (1, 0., A_linear, T_linear, theta_linear),
          'Polynomial': (selected_degree, 0., A_poly, T_poly, theta_poly),
          'LASSO': (selected_degree, lambda_lasso, A_poly, T_poly, theta_lasso),
          'RIDGE': (selected_degree, lambda_ridge, A_poly, T_poly, theta_ridge)}
# Scores CV comparables : mêmes cinq folds, même MSE, degré polynomial fixé.
cv_scores = {'Linéaire': degree_table.loc[1, 'MSE validation moyen'],
             'Polynomial': degree_table.loc[selected_degree, 'MSE validation moyen'],
             'LASSO': lasso_cv.loc[lasso_cv.index[np.isclose(lasso_cv.index, lambda_lasso)][0], 'MSE validation'],
             'RIDGE': ridge_cv.loc[ridge_cv.index[np.isclose(ridge_cv.index, lambda_ridge)][0], 'MSE validation']}
recommended = min(cv_scores, key=cv_scores.get)
rows = []
for name, (degree, lam, A, T, theta) in models.items():
    scores = evaluate(T, y_test, theta)
    rows.append({'Modèle': name, 'Degré': degree, 'λ': lam,
                 'MSE train': evaluate(A, y_train, theta)['MSE'],
                 'MSE test': scores['MSE'], 'RMSE test': scores['RMSE'],
                 'MAE test': scores['MAE'], 'R² test': scores['R2'],
                 'MSE CV': cv_scores[name], 'Nombre de monômes': A.shape[1]-1,
                 'Poids non nuls': int(np.count_nonzero(theta[:-1]))})
comparison = pd.DataFrame(rows).set_index('Modèle')
display(comparison)
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for j, (name, (_, _, _, T, theta)) in enumerate(models.items()):
    pred = model(T, theta).ravel()
    axes[0,j].scatter(y_test.ravel(), pred, alpha=.7)
    lo, hi = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    axes[0,j].plot([lo,hi], [lo,hi], 'r--')
    axes[0,j].set(title=name, xlabel='Sales réel', ylabel='Sales prédit')
    axes[1,j].scatter(pred, y_test.ravel()-pred, alpha=.7)
    axes[1,j].axhline(0, color='red', linestyle='--')
    axes[1,j].set(xlabel='Sales prédit', ylabel='Résidu réel − prédit')
plt.tight_layout()
plt.show()
comparison[['MSE train', 'MSE test']].plot.bar(rot=0, ylabel='MSE', title='Erreurs sur les mêmes partitions')
plt.tight_layout()
plt.show()
_, powers = polynomial_features(np.ones((1,3)), selected_degree)
labels = [' '.join(f'{name}^{exp}' for name, exp in zip(features, row) if exp) for row in powers]
coef_table = pd.DataFrame({'Polynomial': theta_poly[:-1,0], 'LASSO': theta_lasso[:-1,0],
                          'RIDGE': theta_ridge[:-1,0]}, index=labels)
display(coef_table)
coef_table.plot.bar(figsize=(max(10, len(labels)*.38), 5), ylabel='Coefficient standardisé')
plt.tight_layout()
plt.show()
''', 'Les coefficients portent sur les mêmes monômes standardisés : leurs amplitudes sont comparables entre les trois modèles polynomiaux. Les noms TV, Radio et Newspaper désignent ici les budgets centrés-réduits. Le modèle recommandé est choisi par CV, indépendamment du classement test.')
md('## 16. Conclusion et recommandations')
code('''
linear_mse = comparison.loc['Linéaire', 'MSE test']
poly_mse = comparison.loc['Polynomial', 'MSE test']
gain = 100*(linear_mse-poly_mse)/linear_mse
test_best = comparison['MSE test'].idxmin()
zeros = int(np.sum(theta_lasso[:-1] == 0))
reg_effect = '\\n'.join(
    f"- **{name}** : MSE test {comparison.loc[name, 'MSE test']:.4f}, "
    f"écart test−train {comparison.loc[name, 'MSE test']-comparison.loc[name, 'MSE train']:.4f} ; "
    f"variation du MSE test par rapport au polynomial non pénalisé "
    f"{comparison.loc[name, 'MSE test']-poly_mse:+.4f}."
    for name in ['LASSO', 'RIDGE'])
display(Markdown(f"""
Le polynomial de degré **{selected_degree}** obtient un MSE test de **{poly_mse:.4f}**,
contre **{linear_mse:.4f}** pour le linéaire : réduction relative de **{gain:.1f} %**.
Il utilise **{A_poly.shape[1]-1} monômes**, contre trois pour le linéaire.
{'Le gain prédictif observé justifie cette complexité pour la prédiction sur ce domaine.' if gain > 10 else 'Le gain observé est limité ou absent ; la simplicité doit peser fortement dans la décision.'}
Ce jugement pratique n'est pas un test de significativité.

**Recommandation selon la validation : {recommended}**, MSE CV **{cv_scores[recommended]:.4f}**.
Le meilleur résultat sur ces 40 observations test appartient à **{test_best}**.
{'Les deux classements concordent.' if recommended == test_best else 'Les classements diffèrent : on conserve le choix annoncé par validation et on signale cette incertitude.'}
Le test sert à juger la généralisation, sans relancer le réglage des hyperparamètres.

L'avantage CV de LASSO est faible par rapport à la dispersion des folds : ce
classement ne prouve pas sa supériorité statistique. RIDGE choisit la borne basse
de la grille dans cette exécution ; si ce diagnostic apparaît ci-dessus, cela
indique une préférence pour une pénalité presque nulle, et non un optimum intérieur
bien identifié. Le polynomial non pénalisé constitue alors une alternative simple.

{reg_effect}

LASSO annule exactement **{zeros} coefficients sur {len(theta_lasso)-1}** (biais exclu).
Un coefficient nul retire un monôme dans cette base ; cela ne signifie pas toujours
qu'une variable brute disparaît de toutes les interactions. Avec des termes corrélés,
LASSO peut choisir des représentants différents selon l'échantillon. RIDGE réduit les
poids et stabilise les directions corrélées, sans assurer une sélection parcimonieuse.
Cette stabilité est une propriété de la pénalité ; elle n'a pas été mesurée ici par bootstrap.

La remontée du MSE de validation aux degrés élevés, malgré une faible erreur train,
signale le surapprentissage. Le seul écart test−train d'une partition de 40 observations
ne suffit pas à mesurer sa gravité ni à prouver une amélioration significative.
La régularisation est utile si elle améliore la validation et réduit la sensibilité
aux directions peu informatives ; elle n'améliore pas automatiquement le test.

Ces recommandations concernent des données similaires à Advertising, supposées
indépendantes et identiquement distribuées. Pour une priorité d'interprétabilité,
le modèle linéaire reste plus simple ; pour des budgets hors du domaine observé,
l'extrapolation polynomiale est fragile. Avant déploiement, vérifier sur de nouvelles
données ou par validation imbriquée la robustesse du degré, de λ et du classement.
"""))
''', 'Toutes les valeurs de la conclusion proviennent des calculs exécutés. Le degré et les pénalités sont conditionnels aux partitions et à la grille choisies, sans garantie universelle de supériorité.')

notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                    'language_info': {'name': 'python', 'version': '3.13'}}, nbformat=4, nbformat_minor=5)
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Notebook construit : {TARGET.name}, {len(cells)} cellules')
