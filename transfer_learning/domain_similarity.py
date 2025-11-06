from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from featurisation.feature_transformers import get_data, get_fransen_data, getFingerprints
from otdd.pytorch.distance import DatasetDistance
import torch
from torch.utils.data import Dataset, DataLoader


# GET TARGET DATA ----------------------------------------------------
mols_new, y = get_data()

# preprocess
pipeline = Pipeline([
    ('fingerprint', getFingerprints(fpSize=2048, maxPath=6)),
    ('threshold', VarianceThreshold())
])

X = pipeline.fit_transform(mols_new)

# GET SOURCE DATA ----------------------------------------------------
mols, base_y = get_fransen_data()

base_X = pipeline.transform(mols)



class MakeDataset(Dataset):
    def __init__(self, X, y):
        # Convert to float32 and long tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        # Ensure y is 1D, even if (n,1)
        self.y = torch.tensor(y, dtype=torch.long).view(-1)
        self.targets = self.y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


src_dataset = MakeDataset(base_X, base_y)
tgt_dataset = MakeDataset(X, y)

src_loader = DataLoader(src_dataset, batch_size=16, shuffle=True)
tgt_loader = DataLoader(tgt_dataset, batch_size=16, shuffle=True)


# Instantiate distance
dist = DatasetDistance(src_loader, tgt_loader,
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device='cpu')
dist_src = DatasetDistance(src_loader, src_loader,
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device='cpu')
dist_tgt = DatasetDistance(tgt_loader, tgt_loader,
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device='cpu')


d = dist.distance(maxsamples = 48)
print(f'OTDD(src,tgt)={d}')

d_src = dist_src.distance(maxsamples = 48)
print(f'OTDD(src,src)={d_src}')

d_tgt = dist_tgt.distance(maxsamples = 48)
print(f'OTDD(tgt,tgt)={d_tgt}')

d_norm = d / (d_src * d_tgt)**0.5
print(f'Normalised Distance = {d_norm}')