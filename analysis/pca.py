import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA

from featurisation.feature_transformers import get_data, get_fransen_data

def apply_pca(X, df):
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    # Add PCA results to the DataFrame
    df['PC1'] = pca_result[:, 0]
    df['PC2'] = pca_result[:, 1]
    print(len(pca_result))
    print(len(df['PC1']))
    print(pca.explained_variance_ratio_)

    return pca_result


# GET TARGET DATA ----------------------------------------------------
mols_new, y = get_data()

# GET SOURCE DATA ----------------------------------------------------
mols, base_y = get_fransen_data()

# get model info
best_model = load('../models/inhouse.joblib')
fpSize = best_model.named_steps['fingerprint'].fpSize
maxPath = best_model.named_steps['fingerprint'].maxPath
variance_threshold_step = best_model.named_steps['feature_selection']
selected_features_mask = variance_threshold_step.get_support()
feature_names = [str(feature) for feature, selected in zip(range(fpSize), selected_features_mask) if selected]
feature_names = [int(bit) for bit in feature_names]
feature_names_np = np.array([int(bit) for bit in feature_names])


# ANALYSIS ----------------------------------------------------
# fingerprint generation
fpgen = AllChem.GetRDKitFPGenerator(fpSize=fpSize, maxPath=maxPath)
fps_new = []
for i in range(len(mols_new)):
    fp = fpgen.GetFingerprint(mols_new[i])
    fps_new.append(fp)
fps = []
for i in range(len(mols)):
    fp = fpgen.GetFingerprint(mols[i])
    fps.append(fp)

df = pd.DataFrame()
df['fp'] = fps + fps_new
X = df['fp'].apply(np.array)
X = np.array([np.array(fingerprint) for fingerprint in X])

pca_result = apply_pca(X, df)


# PCA plot
plt.scatter(df['PC1'][:len(fps)], df['PC2'][:len(fps)], c='k', label="Fransen et al.", s=15)
plt.scatter(df['PC1'][len(fps):], df['PC2'][len(fps):], c='orangered', label="In-house", s=15)
plt.xlabel('Principal Component 1', fontsize=13)
plt.ylabel('Principal Component 2', fontsize=13)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.legend(fontsize=11, frameon=False)
plt.tight_layout()
plt.show()
