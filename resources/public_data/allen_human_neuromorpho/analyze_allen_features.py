##########################################################
#Author:          Yufeng Liu
#Create time:     2024-09-27
#Description:               
##########################################################
import os
import sys
import random
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn import svm
from xgboost import XGBClassifier

sys.path.append('../../../src')
from config import standardize_features

LOCAL_FEATS = [
    'Stems',
    'SomaSurface',
    'AverageContraction',
    'AverageBifurcationAngleLocal',
    'AverageBifurcationAngleRemote',
    'AverageParent-daughterRatio',
]

def load_features(gf_file, meta_file, col_reg='layer', min_neurons=5, standardize=False, remove_na=True, surface_sqrt=True):
    # Loading the data
    df = pd.read_csv(gf_file, index_col=0)[LOCAL_FEATS]
    
    meta = pd.read_csv(meta_file, index_col=0)

    # merge the features and meta information 
    df = df.merge(meta[col_reg], left_on=df.index, right_on=meta.index)

    if remove_na:
        df = df[df.isna().sum(axis=1) == 0]
    df.set_index('key_0', inplace=True) # use the neuron name as index
    # rename the meta columns
    df.rename(columns={col_reg:'region'}, inplace=True)

    # 
    sns.set_theme(style='ticks', font_scale=1)
    # filter brain regions with number of neurons smaller than `min_neurons`
    rs, rcnts = np.unique(df.region, return_counts=True)
    rs_filtered = rs[rcnts >= min_neurons]
    dff = df[df.region.isin(rs_filtered)]

    if surface_sqrt:
        df['SomaSurface'] = np.sqrt(df.SomaSurface)
    
    # standardize column-wise
    if standardize:
        standardize_features(dff, LOCAL_FEATS, inplace=True)
    return dff


def feature_distributions(gf_file, meta_file, col_reg='layer', boxplot=True, min_neurons=5):
    df = load_features(gf_file, meta_file, col_reg=col_reg, min_neurons=min_neurons)
    sregions = sorted(np.unique(df.region))
    for feat in LOCAL_FEATS:
        dfi = df[[feat, 'region']]
        if boxplot:
            sns.boxplot(data=dfi, x='region', y=feat, hue='region', order=sregions)
            prefix = 'boxplot'
        else:
            sns.stripplot(data=dfi, x='region', y=feat, s=3, alpha=0.5, hue='region', order=sregions)
            prefix = 'stripplot'
        plt.xticks(rotation=90, rotation_mode='anchor', ha='right', va='center')
        if feat.startswith('AverageBifurcationAngle'):
            plt.ylim(30, 110)
        elif feat.startswith('AverageParent'):
            plt.ylim(0.5, 1.2)
        elif feat.startswith('AverageContraction'):
            plt.ylim(0.80, 0.96)


        plt.subplots_adjust(bottom=0.28)
        plt.savefig(f'{prefix}_{feat}.png', dpi=300)
        plt.close()

def joint_distributions(gf_file, meta_file, min_neurons=5, feature_reducer='UMAP'):
    sns.set_theme(style='ticks', font_scale=1.5)

    df = load_features(gf_file, meta_file, min_neurons=min_neurons, standardize=True)

    cache_file = f'cache_{feature_reducer.lower()}.pkl'
    # map to the UMAP space
    if os.path.exists(cache_file):
        print(f'--> Loading existing {feature_reducer} file')
        with open(cache_file, 'rb') as fp:
            emb = pickle.load(fp)
    else:
        if feature_reducer == 'UMAP':
            reducer = umap.UMAP(random_state=1024)
        elif feature_reducer == 'PCA':
            reducer = PCA(n_components=2)

        emb = reducer.fit_transform(df[LOCAL_FEATS])
        if feature_reducer == 'PCA':
            print(reducer.explained_variance_ratio_)
        with open(cache_file, 'wb') as fp:
            pickle.dump(emb, fp)
    
    key1 = f'{feature_reducer}1'
    key2 = f'{feature_reducer}2'
    df[[key1, key2]] = emb
    #sregions = sorted(np.unique(df.region))
    sregions = ['layer 1', 'layer 2', 'layer 3']

    if feature_reducer == 'UMAP':
        xlim, ylim = (-2,8), (4, 11)
    elif feature_reducer == 'PCA':
        xlim, ylim = (-6,6), (-6,6)

    cur_df = df[df.region.isin(sregions)]
    sns.jointplot(data=cur_df, x=key1, y=key2, kind='scatter', xlim=xlim, ylim=ylim, 
                  hue='region', marginal_kws={'common_norm': False, 'fill': False},
                  joint_kws={'alpha':1.},
                  )
    plt.legend(frameon=False, ncol=1, handletextpad=0, markerscale=1.5)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{feature_reducer.lower()}.png', dpi=300)
    plt.close()


    # also for all neurons
    sns.jointplot(
        data=df, x=key1, y=key2, kind='scatter', xlim=xlim, ylim=ylim,
        joint_kws={'s': 5, 'alpha': 0.5},
        marginal_kws={'common_norm':False, 'fill': False, }
    )
    plt.xticks([]); plt.yticks([])
    plt.savefig(f'{feature_reducer.lower()}_all.png', dpi=300)
    plt.close()

def predict_layers(gf_file, meta_file, col_reg='layer', min_neurons=5):
    random.seed(1024)

    print('--> Loading the data')
    df = load_features(gf_file, meta_file, col_reg=col_reg, min_neurons=min_neurons, standardize=True, surface_sqrt=False)

    players = ['layer 1', 'layer 2', 'layer 3']
    pmapper = dict(zip(players, np.arange(len(players))))

    df = df[df.region.isin(players)]
    # convert the layers to numeric
    df['region'] = df.region.map(pmapper)

    # do training
    test_ratio = 0.2
    ntest = int(test_ratio * df.shape[0])
    # split the data
    all_ids = np.arange(df.shape[0]).tolist()
    test_ids = random.sample(all_ids, ntest)
    test_set = df.iloc[test_ids]
    train_set = df.iloc[list(set(all_ids) - set(test_ids))]
    
    print('Initializing the classifier')
    clf = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    #clf = svm.SVC(kernel='rbf')
    
    print('==> Fit and predict')
    cls_feats = [feat for feat in LOCAL_FEATS if feat not in ['AverageContraction']]
    clf.fit(train_set[cls_feats], train_set.region)
    preds = clf.predict(test_set[cls_feats])
    gt = test_set.region.values
    print(f'Layer prediction accuracy: {100.0 * (preds == gt).sum() / gt.shape[0]:.2f}%')
    


if __name__ == '__main__':
    gf_file = 'intermediates/gf_one_point_soma_150um.csv'
    meta_file = 'intermediates/meta_regions_types.csv'

    if 0:
        feature_distributions(gf_file, meta_file, col_reg='layer', boxplot=True)
        #joint_distributions(gf_file, meta_file, feature_reducer='UMAP')

    if 1:
        # test the prediction of layers according to morphological features
        predict_layers(gf_file, meta_file)

