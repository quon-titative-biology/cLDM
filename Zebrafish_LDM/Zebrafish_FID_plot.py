import sys
import time
import socket
import os
import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from torchvision import transforms as T
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster as cl
import scipy.spatial as sp

BASE_PATH = '/group/gquongrp/workspaces/rmvaldar/'

#Loading dfs
FID_perturb_df = pd.read_csv(os.join(BASE_PATH,'FID_Perturbed_EMBED_comb_domain.csv'), index_col=[0])
color_maps = pd.read_csv(os.join(BASE_PATH,'colors_EXP2.csv'), index_col=[0])
color_dict = color_maps.to_dict()['color']
date_color_maps = pd.read_csv(BASE_PATH + 'color_dates.csv', index_col=[0])
date_color_dict = date_color_maps.to_dict()['color']

#Unique values
Ages = [3,5]
Count = []
Family_groups = [['pdzk1KO','PDZK1P1','PDZK1'],
                ['ARHGAP11A', 'ARHGAP11B', 'arhgap11KO', 'arhgap11MO', 'controlMO'],
                ['ncf1KO','NCF1A','NCF1C'],
                ['srgap2KO', 'SRGAP2A', 'SRGAP2C'],
                ['FAM72B', 'fam72KO'],
                ['FRMPD2A', 'FRMPD2B', 'frmpd2KO'],
                ['ptpn20KO','PTPN20', 'PTPN20CP'],
                ['NPY4R','npy4rKO'],
                ['GPR89B', 'gpr89KO'],
                ['scrambled'],
                ['eGFP'],
                ['uninjected'],
                ['hydinKO']]

Family_groups = [['pdzk1KO1', 'pdzk1KO2', 'pdzk1KO3', 'pdzk1KO4', 'pdzk1KO5','pdzk1KOT','PDZK1P11', 'PDZK1P12', 'PDZK1P13','PDZK1P1T', 'PDZK11', 'PDZK12', 'PDZK13', 'PDZK1T'],
                ['ARHGAP11A1', 'ARHGAP11A2', 'ARHGAP11A3','ARHGAP11AT', 'ARHGAP11B1', 'ARHGAP11B2', 'ARHGAP11B3','ARHGAP11BT', 'arhgap11KO1', 'arhgap11KO2','arhgap11KOT', 'arhgap11MO1', 'arhgap11MO2', 'arhgap11MO3', 'arhgap11MO4','arhgap11MOT', 'controlMO1', 'controlMO2', 'controlMO3', 'controlMO4','controlMOT'],
                ['ncf1KO1', 'ncf1KO2', 'ncf1KO3', 'ncf1KO4','ncf1KOT', 'NCF1A1', 'NCF1A2', 'NCF1A3','NCF1AT', 'NCF1C1', 'NCF1C2', 'NCF1C3', 'NCF1CT'],
                ['srgap2KO1', 'srgap2KO2', 'srgap2KO3','srgap2KOT', 'SRGAP2A1', 'SRGAP2A2', 'SRGAP2A3','SRGAP2AT', 'SRGAP2C1', 'SRGAP2C2', 'SRGAP2C3', 'SRGAP2CT'],
                ['FAM72B1', 'FAM72B2', 'FAM72B3','FAM72BT', 'fam72KO1', 'fam72KO2', 'fam72KO3', 'fam72KO4', 'fam72KOT'],
                ['FRMPD2A1', 'FRMPD2A2', 'FRMPD2A3','FRMPD2AT', 'FRMPD2B1', 'FRMPD2B2', 'FRMPD2B3','FRMPD2BT', 'frmpd2KO1', 'frmpd2KO2', 'frmpd2KO3', 'frmpd2KO4', 'frmpd2KOT'],
                ['ptpn20KO1', 'ptpn20KO2', 'ptpn20KO3', 'ptpn20KO4', 'ptpn20KOT', 'PTPN201', 'PTPN202', 'PTPN203', 'PTPN20T', 'PTPN20CP1', 'PTPN20CP2', 'PTPN20CP3', 'PTPN20CPT'],
                ['NPY4R1', 'NPY4R2', 'NPY4R3', 'NPY4RT', 'npy4rKO1', 'npy4rKO2', 'npy4rKO3', 'npy4rKO4', 'npy4rKOT'],
                ['GPR89B1', 'GPR89B2', 'GPR89B3', 'GPR89BT', 'gpr89KO1', 'gpr89KO2', 'gpr89KO3', 'gpr89KO4', 'gpr89KOT'],
                ['scrambled'],
                ['eGFP'],
                ['uninjected'],
                ['hydinKO1','hydinKO2','hydinKO3','hydinKO4','hydinKOT']]
Control_family = ['scrambled','eGFP','uninjected']
'(4.936, 13.0]' 
ranges=['(4.936, 13.0]' ,'(13.0, 21.0]' ,'(21.0, 29.0]',
                                           '(29.0, 37.0]' , '(37.0, 45.0]' , '(45.0, 53.0]' ,
                                           '(53.0, 61.0]' , '(61.0, 69.0]']
uniq_lb, uniq_lb_count = np.unique(age_plt_mut_comb,return_counts=True)
#uniq_plates = np.unique(np.asarray([label.split('_')[0] + '_' + label.split('_')[1] for label in uniq_lb]))


#Counts
count_dict = dict(zip(uniq_lb, uniq_lb_count))


#With Controls and all values:
All_counts = [count for count in count_dict.values()]
All_labels = [label for label in count_dict.keys()]



Sample_ages = [batch.split('_')[0] for batch in FID_perturb_df.columns]
Sample_labels = [batch.split('_')[2].split('-')[0] for batch in FID_perturb_df.columns]
Sample_dates = [label.split('_')[0] + '_' + label.split('_')[1] for label in FID_perturb_df.columns]
Sample_counts = pd.cut([count for count in All_counts],8, labels=ranges).tolist()



order = ['ARHGAP11A1', 'ARHGAP11B1', 'arhgap11KO1', 'arhgap11MO1', 'controlMO1', 'FAM72B1', 
'fam72KO1', 'FRMPD2A1', 'FRMPD2B1', 'frmpd2KO1', 'GPR89B1', 'gpr89KO1', 'hydinKO1', 'NCF1A1', 
'NCF1C1', 'ncf1KO1', 'NPY4R1', 'npy4rKO1', 'PDZK11', 'pdzk1KO1', 'PDZK1P11', 'PTPN201', 
'PTPN20CP1', 'ptpn20KO1', 'scrambled', 'SRGAP2A1', 'SRGAP2C1', 'srgap2KO1', 'uninjected','eGFP']

date_order = ['3_2021.10.11', '3_2021.10.19','5_2021.10.13','3_2021.10.20', 
             '3_2021.12.23','3_2022.09.29', '3_2022.09.30', '3_2022.10.01', 
             '3_2022.10.02','5_2022.10.17','5_2022.10.19', '5_2022.10.20',
             '3_2022.11.05', '3_2022.11.07', '3_2022.11.08','3_2022.11.22',
             '5_2022.11.11', '5_2022.11.15','5_2022.11.17','5_2022.11.19',
             '3_2022.12.02', '3_2022.12.05',  '5_2022.12.05','3_2023.09.26']

val_matrix = FID_perturb_df.to_numpy()
full_data = [Sample_ages,Sample_labels,Sample_dates,Sample_counts]
uniq_data = [Ages, Family_groups,ranges]


def make_colormap(map_arr, data,df,palette):
    pal = sns.color_palette(palette=palette, n_colors=len(map_arr))
    lut = dict(zip(map(str, map_arr), pal))
    colors = pd.Series(data, df.iloc[:,0:].columns).map(lut)
    return lut, colors


def make_colormap_labels(data,df,dct):
    colors = pd.Series(data, df.iloc[:,0:].columns).map(dct)
    return dct, colors

def generate_heatmap(matrix,name,class_names,uniq_data, full_data, singular_paletes,label_palettes,plate_plattes, df,order,date_order):
    #Age Side Bar
    Age_lut, Age_colors = make_colormap(uniq_data[0],full_data[0],df,singular_paletes[0])
    Label_lut, Label_colors = make_colormap_labels(full_data[1],df,label_palettes)
    Plate_lut, Plate_colors = make_colormap_labels(full_data[2],df,plate_plattes)
    Count_lut, Count_colors = make_colormap(uniq_data[2],full_data[3],df,singular_paletes[1])
    dist_metric = 'euclidean'
    linkage_method = 'ward'
    matrix_for_hc = sp.distance.pdist(matrix, dist_metric);
    Z = cl.hierarchy.linkage(matrix_for_hc,linkage_method)
    leaf_order = cl.hierarchy.leaves_list(Z);
    plot_mat = pd.DataFrame(matrix, columns = class_names, index = class_names)

    hx = sns.clustermap(plot_mat,
               figsize = (18,18),
               method = 'ward',
               cmap='Reds',
               col_cluster=True,
               row_cluster = True,
               row_colors=[Age_colors,Label_colors,Count_colors],
               col_colors=[Age_colors,Label_colors,Count_colors],
               col_linkage = Z,
               row_linkage = Z,
               xticklabels=1, yticklabels=1)
    hx.fig.suptitle(name)
    for age in list(set(full_data[0])):
        hx.ax_col_dendrogram.bar(0, 0, color=Age_lut[age],
                            label=age, linewidth=0)
    hx.ax_col_dendrogram.legend(loc="center left", ncol=5)
    for labels in order:
        hx.ax_row_dendrogram.bar(0, 0, color=Label_lut[labels],
                            label=labels, linewidth=0)
    hx.ax_row_dendrogram.legend(loc="center", ncol=2)
    for counts in uniq_data[2]:
        hx.ax_col_dendrogram.bar(0, 0, color=Count_lut[counts],
                            label=counts, linewidth=0)
    hx.ax_col_dendrogram.legend(loc="center left", ncol=2)
    plt.savefig(name +'.png')
    return hx, plot_mat


singular_palletes = ['flare','crest']
plot, val_df = generate_heatmap(val_matrix, 'SSRT_Combined_Domain_FID',
                            [label for label in All_labels],uniq_data,
                            full_data,singular_palletes,color_dict,date_color_dict,FID_perturb_df,order,date_order)