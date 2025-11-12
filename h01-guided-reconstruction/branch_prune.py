##########################################################
#Author:          Yufeng Liu
#Create time:     2025-10-25
#Description:               
##########################################################

import os
import glob
import math
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from ml.stats_utils import my_mannwhitneyu

from swc_handler import parse_swc, write_swc
from morph_topo.morphology import Morphology, Topology

######## Branch feature calculation ############
class BranchFeatures:

    def __init__(self, in_swc, epsilon=1e-8):
        self._load_data(in_swc)
        self.epsilon = epsilon
                
    def _load_data(self, in_swc):
        # load the data and initialize the topological tree
        print(f'--> Processing {os.path.split(in_swc)[-1]}')
        tree = parse_swc(in_swc)
        self.morph = Morphology(tree)
        topo_tree, self.seg_dict = self.morph.convert_to_topology_tree()
        self.topo = Topology(topo_tree)

        return True

    def calc_features(self, scale_r=1.0, length_thres=10):
        # initialize the coordinate dict for later use
        coords_dict = {}
        for node in self.morph.tree:
            coords_dict[node[0]] = np.array(node[2:5])

        # iterate over all features
        fseg_dict = {}
        sxyz = coords_dict[self.topo.idx_soma]
        srad = self.topo.pos_dict[self.topo.idx_soma][5]
        for sid, seg_ids in self.seg_dict.items():
            # Check if the branch is salient
            # level >= 2
            current = self.topo.pos_dict[sid]
            pid = current[6]
            if pid == -1:
                continue
            
            parent = self.topo.pos_dict[pid]
            if parent[6] == -1:
                continue

            pid2 = parent[6]
            parent2 = self.topo.pos_dict[pid2]

            # outside of the soma
            xyz_cur = coords_dict[sid]
            xyz_par = coords_dict[pid]
            dist_s_cur = np.linalg.norm(sxyz - xyz_cur)
            dist_s_par = np.linalg.norm(sxyz - xyz_par)
            if (dist_s_cur < srad*scale_r) or (dist_s_par < srad*scale_r):
                continue

            # length > length_thres
            euc_length = np.linalg.norm(xyz_cur - xyz_par)
            if euc_length < length_thres:
                continue

            # calculate the features
            seg_ids_all = [sid] + seg_ids + [pid]
            vn = xyz_cur - xyz_par  # parent-to-current
            vn /= (np.linalg.norm(vn) + self.epsilon)
            # radius
            # radius for current branch
            radii = [self.morph.pos_dict[nid][5] for nid in seg_ids_all]
            radius_curr = np.median(radii) if len(radii) >= 3 else np.mean(radii)
            max_radius_curr = max(radii)
            # radius for parent branch
            seg_ids_par = [pid] + self.seg_dict[pid] + [pid2]
            radii = [self.morph.pos_dict[nid][5] for nid in seg_ids_par]
            radius_par = np.median(radii) if len(radii) >= 3 else np.mean(radii)
            max_radius_par = max(radii)
            
            radius = radius_curr / (radius_curr + radius_par + self.epsilon)
            max_radius = max_radius_curr / (max_radius_curr + max_radius_par + self.epsilon)
            
            # soma-parent-current orientation
            vsp = xyz_par - sxyz
            vsp /= (np.linalg.norm(vsp) + self.epsilon)
            cos_ang1 = np.dot(vsp, vn)
            ang_s1 = np.arccos(cos_ang1)
            
            # parent-current-branch angle
            xyz_par2 = coords_dict[pid2]
            vn1 = xyz_par - xyz_par2
            vn1 /= (np.linalg.norm(vn1) + self.epsilon)
            cos_ang2 = np.dot(vn1, vn)
            ang_s2 = np.arccos(cos_ang2)

            fseg_dict[sid] = (radius, max_radius, ang_s1, ang_s2)

        print(f'   {len(fseg_dict)} branches in {len(self.seg_dict)} branches')

        return fseg_dict

def dataset_features(in_swc_dir, out_feat_file):
    data_list = []
    num_processed = 0
    for in_swc in glob.glob(os.path.join(in_swc_dir, '*.swc')):
        swc_name = os.path.split(in_swc)[-1][:-4]
        bf = BranchFeatures(in_swc)
        fseg_dict = bf.calc_features()
        
        # iterate over all sids for a swc file
        for sid, features in fseg_dict.items():
            radius, max_radius, ang_s1, ang_s2 = features
            data_list.append({
                'swc_name': swc_name,
                'sid': sid,
                'radius': radius,
                'max_radius': max_radius,
                'ang_s1': ang_s1,
                'ang_s2': ang_s2
            })

        num_processed += 1
        if num_processed % 10 == 0:
            print(f'---> {num_processed}...')

    # 创建DataFrame
    df = pd.DataFrame(data_list)

    # 设置index为swc_name
    df.set_index('swc_name', inplace=True)
    df.to_csv(out_feat_file, index=True)


########## Modeling and detection ##############
def check_distribution(feat_file, dataset):
    from verify_gmm import features_on_umap

    figname = f'umap_of_branch_features_{dataset}.png'
    feat_names = ['radius', 'max_radius', 'ang_s1', 'ang_s2']
    features_on_umap(feat_file, feat_names, figname, scale=5)

def compare_features(feat_file_auto, feat_file_h01):
    # load the features for datasets
    fauto = pd.read_csv(feat_file_auto, low_memory=False, index_col=0)
    fh01 = pd.read_csv(feat_file_h01, low_memory=False, index_col=0)

    sns.set_theme(style='ticks', font_scale=2.7)
    # 合并数据
    df1_copy = fauto[['radius', 'max_radius', 'ang_s1', 'ang_s2']].copy()
    df2_copy = fh01[['radius', 'max_radius', 'ang_s1', 'ang_s2']].copy()

    df1_copy['source'] = 'auto'
    df2_copy['source'] = 'h01'

    combined_df = pd.concat([df1_copy, df2_copy])

    # 绘制小提琴图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    numeric_columns = ['radius', 'max_radius', 'ang_s1', 'ang_s2']

    for i, col in enumerate(numeric_columns):
        # 创建小提琴图 - 使用灰色边框，不填充
        sns.violinplot(x='source', y=col, data=combined_df, ax=axes[i],
                      palette=['magenta', 'blue'],  # 深灰和浅灰
                      #color='slategray',
                      hue='source',
                      saturation=1, cut=0, fill=True, 
                      alpha=0.25,
                      linewidth=3, inner=None)  # 设置边框线宽，不显示内部图形

        # 添加简化的箱线图 - 使用黑色，更简洁
        sns.boxplot(x='source', y=col, data=combined_df, ax=axes[i],
                   width=0.25, palette=['magenta', 'blue'],
                   hue='source',
                   boxprops=dict(alpha=1., linewidth=3),
                   whiskerprops=dict(linewidth=3),
                   capprops=dict(linewidth=3),
                   medianprops=dict(linewidth=3),  # 中位数用红色突出
                   fill=False, 
                   showfliers=False)  # 不显示异常值

        # 计算p-value
        data1 = fauto[col].dropna()
        data2 = fh01[col].dropna()

        # 使用Mann-Whitney U检验（非参数检验，不假设正态分布）
        stat, p_value, cles, significance = my_mannwhitneyu(data1, data2, size_correction=True)
        # 打印详细的统计结果
        print("=" * 80)
        print(f"= Statistical Test Results (Mann-Whitney U Test) for {col}")
        print(f"= -- p-value={p_value} and effect size={cles:.4f}, Significance={significance}")
        print("=" * 80)
        print('\n')

        # 在图上添加p-value和显著性标识
        y_max = max(data1.max(), data2.max())
        y_min = min(data1.min(), data2.min())
        y_range = y_max - y_min

        # 添加显著性线和标识
        axes[i].plot([0, 1], [y_max + 0.05 * y_range, y_max + 0.05 * y_range],
                    color='black', linewidth=2)
        axes[i].text(0.5, y_max + 0.06 * y_range,
                    f'{significance}',
                    ha='center', va='bottom')

        # 调整y轴限制以容纳显著性标识
        axes[i].set_ylim(y_min - 0.1 * y_range, y_max + 0.18 * y_range)

        #axes[i].set_title(f'{col} Distribution (Violin Plot)')
        axes[i].set_xlabel('')
        #axes[i].set_ylabel('')

        for spine in axes[i].spines.values():
            spine.set_linewidth(3)
        axes[i].tick_params(axis='both', which='major', width=3)

    plt.tight_layout()
    plt.savefig('comp_auto_h01_branch_features.png', dpi=300, bbox_inches='tight')
    plt.close()


    

# 2. 基于BIC选择最佳n_components
def select_best_components(data, max_components=60):
    """自动选择最佳GMM组件数量"""
    bic_values = []
    n_components_range = range(1, max_components+1)

    for n in n_components_range:
        gmm = GaussianMixture(n_components=n,
                             covariance_type='diag',
                             random_state=1024)
        gmm.fit(data)
        bic_score = gmm.bic(data)
        bic_values.append(bic_score)
        print(f'--> BIC value: {bic_score:.4f} for n_components={n}')

    '''
    bic_values = [1790137.7886, 1578491.3097, 1508877.9561, 1463786.6712, 1443058.9821, 
              1418821.0883, 1411634.8235, 1403412.8721, 1393991.6256, 1393682.9624,
              1383801.7050, 1382008.8262, 1370939.0590, 1366462.9641, 1367344.7835,
              1358747.0563, 1330621.1163, 1329828.8896, 1330405.6472, 1328431.0205,
              1356828.9049, 1354313.0866, 1351972.6735, 1350173.1669, 1341393.4601,
              1344705.3382, 1311441.7344, 1310469.5788, 1313511.9890, 1309137.9267,
              1304901.3992, 1309628.8721, 1310312.8357, 1304315.0915, 1304215.5268,
              1304397.0367, 1300781.3924, 1330806.5518, 1300681.1702, 1301897.9539,
              1304285.7438, 1302752.3214, 1301565.0043, 1295709.2344, 1297295.1227,
              1295257.8866, 1295644.9617, 1295270.1197, 1328400.5590, 1332519.1858,
              1327748.1472, 1292989.0569, 1294489.9937, 1328332.7093, 1292577.5177,
              1291386.9978, 1292775.7695, 1294090.7804, 1319205.2391, 1290268.5785]
    '''

    # 可视化BIC曲线
    sns.set_theme(style='ticks', font_scale=1.6)
    plt.figure(figsize=(6,6))
    plt.plot(n_components_range, bic_values, 'o-')
    plt.xlim(0, max_components)
    #plt.ylim(-9e5,-7e5)
    #plt.yticks([-9e5,-8e5,-7e5])

    plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.xlabel('Number of Components')
    plt.ylabel('BIC Score')
    plt.subplots_adjust(bottom=0.15)
    plt.title('BIC for GMM Model Selection')
    plt.savefig('gmm_bic_branch_features.png', dpi=600)
    plt.close()

    best_n = np.argmin(bic_values) + 1  # +1因为从0开始索引
    print(f"自动选择的最佳组件数: {best_n}")
    return best_n

def plot_outlier_distribution(auto_scores, threshold):

    ########### Helper function ###########
    def ceil_to_n_significant_digits(num, n=5):
        if num == 0:
            return 0.0
        # 计算科学计数法的指数
        exponent = math.floor(math.log10(abs(num)))
        # 将数字缩放到 1eX 范围内，然后乘以 10^(n-1) 进行ceil操作
        scaled = num / 10**exponent  # 例如 123.456 → 1.23456
        multiplier = 10 ** (n - 1)
        ceil_scaled = math.ceil(scaled * multiplier) / multiplier  # 1.23456 → 1.2346 (若n=5)
        # 恢复原始数量级
        return ceil_scaled * 10**exponent


    sns.set_theme(style='ticks', font_scale=1.8)
    plt.figure(figsize=(6, 6))

    # to avoid overlapping bins caused by float precision
    threshold = ceil_to_n_significant_digits(threshold, 5)
    print(threshold)


    # 创建明确的分组标签数组
    hue_labels = np.where(auto_scores > threshold, 'Anomaly', 'Normal')

    ###### helper function ########
    def make_aligned_bins(scores, threshold, n_bins=50):
        """生成与threshold对齐的分箱边界"""
        min_val = np.min(scores)
        max_val = np.max(scores)

        if max_val > 10000: # in case of outliers
            v5, v25, v50, v75, v95 = np.percentile(auto_scores, [5, 25, 50, 75, 95])
            max_val = v95 + (v95 - v5) / 0.9 * 0.05

        # 计算threshold两侧需要的bin数量比例
        left_ratio = (threshold - min_val) / (max_val - min_val)
        right_ratio = 1 - left_ratio

        # 计算两侧的实际bin数（保持总数≈n_bins）
        left_bins = max(1, int(np.round(n_bins * left_ratio)))
        right_bins = max(1, int(np.round(n_bins * right_ratio)))

        # 生成分段线性分箱
        left_edges = np.linspace(min_val, threshold, left_bins + 1, endpoint=False)
        right_edges = np.linspace(threshold, max_val, right_bins + 1)
        # remove possible duplicates
        if left_edges[-1] == right_edges[0]:
            left_edges = left_edges[:-1]

        # 合并并去重（threshold会出现在两个数组中）
        return np.unique(np.concatenate([left_edges, right_edges]))


    # 使用示例
    aligned_bins = make_aligned_bins(auto_scores, threshold, n_bins=40)

    # 绘制直方图
    #import ipdb; ipdb.set_trace()
    ax = sns.histplot(x=auto_scores,
                     bins=aligned_bins,
                     stat='probability',
                     kde=False,
                     hue=hue_labels,
                     palette={'Normal': 'skyblue', 'Anomaly': 'tomato'},
                     edgecolor='white',
                     linewidth=0.5)

    # 标记阈值线
    threshold_line = ax.axvline(threshold, color='darkred', linestyle='--', linewidth=2)

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='skyblue', lw=4, label='Normal'),
                       Line2D([0], [0], color='tomato', lw=4, label='Anomaly'),
                       threshold_line]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False)

    # 添加统计标注
    if threshold > 0:
        scalef = 1.1
    else:
        scalef = 0.99

    ax.annotate(f'{np.mean(auto_scores > threshold):.1%} anomalies',
                xy=(threshold, 0),
                xytext=(threshold*scalef, ax.get_ylim()[1]*0.5))#,
                #arrowprops=dict(arrowstyle='->'),
                #bbox=dict(boxstyle='round', fc='white'))

    # 格式调整
    ax.set(xlabel='Anomaly Score',
           ylabel='Proportion',
           title='Predicted Anomaly Distribution\nusing GMM model')
    sns.despine()
    plt.xlim(aligned_bins[0], aligned_bins[-1])
    plt.tight_layout()
    plt.savefig('gmm_predicted_initial_branch_features.png', dpi=600)
    plt.close()

####### Pruning utilities ##########
def pruning(df_auto, in_swc_dir, out_swc_dir):

    nprocessed = 0
    npruned = 0
    for swc_file in glob.glob(os.path.join(in_swc_dir, '*swc')):
        swc_name = os.path.split(swc_file)[-1][:-4]
        out_swc_file = os.path.join(out_swc_dir, f'{swc_name}.swc')
        
        if os.path.exists(out_swc_file):
            continue
        nprocessed += 1
        
        cur_status = df_auto.loc[swc_name]
        if cur_status.gmm_label.sum() == 0:
            shutil.copy(swc_file, out_swc_dir)
            #os.system(f'cp {swc_file} {out_swc_file}')
            continue
        
        npruned += 1
        if nprocessed % 5 == 0:
            print(f'===========> [pruned/processed]: {npruned}/{nprocessed}')

        tree = parse_swc(swc_file)
        morph = Morphology(tree)
        topo_tree, seg_dict = morph.convert_to_topology_tree()
        
        cur_pos = cur_status[cur_status.gmm_label == 1].sid
        # iterative pruning
        nodes_to_remove = set()
        for ipos in cur_pos:
            if ipos in nodes_to_remove:
                continue
            # remove current branch
            nodes_to_remove.update(seg_dict[ipos])
            # also all childrens from this point
            # 使用栈或队列找到所有后代节点
            stack = [ipos]
            branch_nodes = set()
            
            while stack:
                current = stack.pop()
                if current in branch_nodes:
                    continue
                branch_nodes.add(current)
                
                # 将当前节点的子节点加入栈中
                if current in morph.child_dict:
                    stack.extend(morph.child_dict[current])
            
            # 将整个分支的节点加入待删除集合
            nodes_to_remove.update(branch_nodes)

        # update the tree
        new_tree = []
        for node in tree:
            if node[0] not in nodes_to_remove:
                new_tree.append(node)

        write_swc(new_tree, out_swc_file)
        
        print(f'--> {swc_name}: processed={len(new_tree)}, original={len(tree)}')
        

def detect_outlier_stems(
        dataset_dict, anomaly_file, best_n=None, max_iter=10, visualize=True
):

    if not os.path.exists(anomaly_file):
        # Load the data
        feat_names = ['radius', 'max_radius', 'ang_s1', 'ang_s2']
        df_h01 = pd.read_csv(dataset_dict['h01']['feat_file'], index_col=0)
        feats_h01 = df_h01[feat_names]
        df_auto = pd.read_csv(dataset_dict['auto']['feat_file'], index_col=0)
        feats_auto = df_auto[feat_names]

        # feature standardize
        scaler = StandardScaler()
        feats_h01_scaled = scaler.fit_transform(feats_h01)

        # training GMM model
        best_n = best_n or select_best_components(feats_h01_scaled)
        gmm = GaussianMixture(n_components=best_n, covariance_type='diag', random_state=1024)
        gmm.fit(feats_h01_scaled)

        # get the threshold for anomaly detection
        threshold = np.percentile(-gmm.score_samples(feats_h01_scaled), 95)

        # outlier detection
        # processing and prediction
        feats_auto_scaled = scaler.transform(feats_auto)
        auto_scores = -gmm.score_samples(feats_auto_scaled)
        auto_labels = (auto_scores > threshold).astype(int)
        anomaly_pct = auto_labels.mean()
        print(f'Percentage of anomaly: {anomaly_pct:.2%} based on threshold={threshold:.2f}, Top scores: {np.sort(auto_scores)[-5:][::-1]}')

        # push the data to original dataframe
        df_auto['gmm_score'] = auto_scores
        df_auto['gmm_label'] = auto_labels

        if visualize:
            # overall score distribution
            plot_outlier_distribution(auto_scores, threshold)

        # saving the file
        df_auto.to_csv(anomaly_file, index=True)
    else:
        # load the pred-calculated data directly
        df_auto = pd.read_csv(anomaly_file, index_col=0, low_memory=False)
    
    # Do actual pruning based on detected anomalies
    in_swc_dir = dataset_dict['auto']['in_swc_dir']
    out_swc_dir = dataset_dict['auto']['out_swc_dir']
    os.makedirs(out_swc_dir, exist_ok=True)
    pruning(df_auto, in_swc_dir, out_swc_dir)

    

if __name__ == '__main__':
    dataset_dict = {
        'h01': {
            'in_swc_dir': 'data/H01_resample1um_prune25um',
            'feat_file': 'data/H01_resample1um_prune25um_branch_features.csv'
        },
        'auto': {
            'in_swc_dir': 'data/auto8.4k_0510_resample1um_mergedBranches0712',
            'feat_file': 'data/auto8.4k_0510_resample1um_mergedBranches0712_branch_features.csv',
            'out_swc_dir': 'data/auto8.4k_0510_resample1um_mergedBranches0712_branchPruned1029'
        }
    }

    if 0:   # calculate the branch features for datasets
        dataset = 'h01'
        in_swc_dir = dataset_dict[dataset]['in_swc_dir']
        out_feat_file = dataset_dict[dataset]['feat_file']
        dataset_features(in_swc_dir, out_feat_file)

    if 1:
        # prune
        #dataset = 'auto'
        #check_distribution(dataset_dict[dataset]['feat_file'], dataset=dataset)   # continuous distribution, GMM is preferred

        compare_features(dataset_dict['auto']['feat_file'], dataset_dict['h01']['feat_file'])
        
        best_n = 17
        #anomaly_file = 'data/auto8.4k_0510_resample1um_mergedBranches0712_branch_features_anomaly.csv'
        #detect_outlier_stems(dataset_dict, anomaly_file, best_n=best_n)
        
