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

