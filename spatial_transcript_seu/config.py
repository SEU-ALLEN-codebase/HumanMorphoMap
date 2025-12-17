##########################################################
#Author:          Yufeng Liu
#Create time:     2025-06-10
#Description:               
##########################################################

LAYER_CODES = {
    'L1': 1,
    'L2': 2,
    'L3': 3,
    'L4': 4,
    'L5': 5,
    'L6': 6,
    'WM': 7,
    'L2-L3': 8,
    'L5-L6': 9,
}

LAYER_CODES_REV = dict(zip(LAYER_CODES.values(), LAYER_CODES.keys()))


PYRAMIDAL_SUPERCLUSTERS = (
    'Deep-layer near-projecting',
    'Deep-layer corticothalamic and 6b',
    'Upper-layer intratelencephalic',
    'Deep-layer intratelencephalic'
)

#### ANNOTATED Anchor points, (W, H)
LAYER_ANCHORS = {
    'P00083': {
        'L2-L3': ((913,1516), (1923,497)),
        'L4':    ((1092,1523), (1925,633)),
        'L5-L6': ((1253,1519), (1885,862))
    }
}


