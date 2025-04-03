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

LOCAL_FEATS2 = [
    'N_stem',
    'Soma_surface',
    'Average Contraction',
    'Average Bifurcation Angle Local',
    'Average Bifurcation Angle Remote',
    'Average Parent-daughter Ratio',
]

mRMR_FEATS = [  # Top 10 features by mRMR
    'Overall Width',
    'N_stem',
    'Average Fragmentation',
    'Average Bifurcation Angle Local',
    'Overall Depth',
    'Average Diameter',
    'Overall Height',
    'Average Contraction',
    'Soma_surface',
    'Max Branch Order',
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
    'FL_TL.L': 'FL',    # map to this region
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
    'STG.R': 'TL',
    'STG-AP': 'TL',
    'TL.R': 'TL',
    'TL.L': 'TL',
    'TP': 'TL',
    'TP.L': 'TL',
}


def region_mapper():
    reg_mapper = {}
    for reg in REG2LOBE.keys():
        reg_lr = reg.replace('.L', '').replace('.R', '')
        reg_lr = reg_lr.replace('-near-AG', '')
        if reg_lr == 'PL_OL': reg_lr = 'P(O)L'
        elif reg_lr == 'FL_TL': reg_lr = 'F(T)L'

        reg_mapper[reg] = reg_lr

    return reg_mapper

def to_PID5(pid):
    # standardize the patitent id to 'P' + '05d' format
    if len(pid) == 4:
        return f'{pid[0]}00{pid[1:]}'
    elif len(pid) == 6:
        return pid
    else:
        raise ValueError

def to_TID3(tid):
    # standardize the tissue/sample id to 'T' + '03d' format
    if len(tid) == 3:
        return f'{tid[0]}0{tid[1:]}'
    elif len(tid) == 4:
        return tid
    else:
        return ValueError


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

