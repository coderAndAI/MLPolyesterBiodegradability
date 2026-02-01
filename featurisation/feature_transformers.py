import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import load

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def _parse_smiles(smiles):
    """Parse SMILES string to RDKit mol, trying sanitized first then unsanitized."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        logger.warning("Failed to parse SMILES: %s", smiles)
    return mol


def get_data():
    df = pd.read_csv(DATA_DIR / 'inhouse_data.csv')
    y = df['Biodegradability_yes']
    y = LabelEncoder().fit_transform(y)
    y = np.array(y)

    mols = df['Trimer'].apply(_parse_smiles)
    mols = np.array(mols)

    return mols, y


def get_fransen_data():
    df = pd.read_csv(DATA_DIR / 'fransen_data.csv')

    # Remove rows with NaN values in biodeg column, and duplicates
    df = df.dropna(subset=['Biodegradability_yes'], ignore_index=True)
    df = df.drop(
        index=[277, 405, 66, 97, 98, 227, 443, 96, 92, 109, 99, 454, 446, 287, 436, 437, 51, 431, 342, 129, 120, 274,
               475, 464])
    df = df.reset_index(drop=True)

    # get y, convert bool to numerical, put in np array
    y = df['Biodegradability_yes']
    y = LabelEncoder().fit_transform(y)
    y = np.array(y)

    mols = df['Trimer'].apply(_parse_smiles)
    mols = np.array(mols)

    return mols, y


def inside_validity_domain(test_mol):
    mols_new, _ = get_data()

    # get model info
    best_model = load(MODELS_DIR / 'inhouse.joblib')

    features = best_model.named_steps['fingerprint'].transform(mols_new)
    test_features = best_model.named_steps['fingerprint'].transform(
        np.array([Chem.MolFromSmiles(test_mol)])
    )

    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    out_of_range = (test_features < min_vals) | (test_features > max_vals)
    f = out_of_range.sum(axis=1)

    return f.sum() == 0


class getFingerprints(BaseEstimator, TransformerMixin):  # custom transformer for the pipeline
    def __init__(self, fpSize=512, maxPath=7):
        self.fpSize = fpSize
        self.maxPath = maxPath

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mols = pd.Series(X)

        # set up fingerprint generation
        fpgen = AllChem.GetRDKitFPGenerator(fpSize=self.fpSize, maxPath=self.maxPath)

        fps = [fpgen.GetFingerprint(mol) for mol in mols]

        # get X
        X = np.array(fps)
        X = np.array([np.array(fingerprint) for fingerprint in X])

        return X


class getMolDescriptors(BaseEstimator, TransformerMixin):  # custom transformer for the pipeline
    def __init__(self, missingVal=None):
        self.missingVal = missingVal

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mols = pd.Series(X)

        all_descriptors = []
        for i in range(len(mols)):
            res = []
            for nm, fn in Descriptors._descList:
                try:
                    val = fn(mols[i])
                except Exception:
                    logger.debug("Descriptor %s failed for molecule %d", nm, i, exc_info=True)
                    val = self.missingVal
                res.append(val)
            res = np.clip(res, a_min=None, a_max=np.finfo(np.float32).max)
            all_descriptors.append(res)

        X = np.array(all_descriptors, dtype=np.float64)
        return X


class FeatureMaskTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, descriptors_removed=None):
        self.descriptors_removed = descriptors_removed

    def fit(self, X, y=None):
        # Nothing to fit, as the mask is predefined
        return self

    def transform(self, X):
        descriptors = np.array([desc[0] for desc in Descriptors._descList])
        feature_mask = np.array([(desc not in self.descriptors_removed) for desc in descriptors])

        # Apply the feature mask to remove listed features
        return X[:, feature_mask]


class getFP_WithPred(BaseEstimator, TransformerMixin): # custom transformer for the pipeline
    def __init__(self, fpSize=512, maxPath=7):
        self.fpSize = fpSize
        self.maxPath = maxPath

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        mols = pd.Series(X)

        best_model = load(MODELS_DIR / 'chain_step1.joblib')
        pred = best_model.predict_proba(mols)

        # set up fingerprint generation
        fpgen = AllChem.GetRDKitFPGenerator(fpSize=self.fpSize, maxPath=self.maxPath)

        fps = []
        for i in range(len(mols)):
            fp = fpgen.GetFingerprint(mols[i])
            fps.append(fp)

        # get X
        X = np.array(fps)
        X = np.hstack((X, pred[:, 1][:, np.newaxis]))
        X = np.array([np.array(fingerprint) for fingerprint in X])

        return X
