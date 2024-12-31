##########################################################
#Author:          Yufeng Liu
#Create time:     2024-09-28
#Description:               
##########################################################

LOCAL_FEATS = [
    'N_stem',
    'Soma_surface',
    #'Average Contraction',
    'Average Bifurcation Angle Local',
    'Average Bifurcation Angle Remote',
    'Average Parent-daughter Ratio',
]

REG2LOBE = {
    '(X)FG': 'FL',
    'BN.L': 'BN',
    'CC.L': 'C2',   # Note, it is corpus callosum, not Cingulate cortex!
    'FL.L': 'FL',
    'FL.R': 'FL',
    'FP.L': 'FL',
    'FP.R': 'FL',
    'FT.L': 'FL',
    'IFG': 'FL',
    'IFG.R': 'FL',
    'IPL-near-AG': 'PL',
    'IPL.L': 'PL',
    'M(I)FG.L': 'FL',
    'MFG': 'FL',
    'MFG.L': 'FL',
    'MFG.R': 'FL',
    'MTG': 'TL',
    'MTG.L': 'TL',
    'MTG.R': 'TL',
    'OL.L': 'OL',
    'OL.R': 'OL',
    'PL': 'PL',
    'PL.L': 'PL',
    'PL.L_OL.L': 'PL',
    'S(M)FG.R': 'FL',
    'S(M)TG.L': 'TL',
    'S(M)TG.R': 'TL',
    'S(M,I)TG': 'TL',
    'SFG': 'FL',
    'SFG.L': 'FL',
    'SFG.R': 'FL',
    'STG': 'TL',
    'STG-AP': 'TL',
    'TL.R': 'TL',
    'TL.L': 'TL',
    'TP': 'TL',
    'TP.L': 'TL',
}

def standardize_features(dfc, feat_names, epsilon=1e-8, inplace=True):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.mean()) / (fvalues.std() + epsilon)
    if inplace:
        dfc.loc[:, feat_names] = fvalues.values
    else:
        dfcc = dfc.copy()
        dfcc.loc[:, feat_names] = fvalues.values
        return dfcc

def normalize_features(dfc, feat_names, epsilon=1e-8, inplace=True):
    fvalues = dfc[feat_names]
    fvalues = (fvalues - fvalues.min()) / (fvalues.max() - fvalues.min() + epsilon)
    if inplace:
        dfc.loc[:, feat_names] = fvalues.values
    else:
        dfcc = dfc.copy()
        dfcc.loc[:, feat_names] = fvalues.values
        return dfcc

