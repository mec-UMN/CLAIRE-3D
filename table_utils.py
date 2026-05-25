from prettytable import PrettyTable
import networkx as nx

from optimize_fns import *

def PPA_table_info(table_group, model_list, PPA_curr, PPA_curr_custom, PPA_curr_opt):
    #import pdb;pdb.set_trace()
    if not table_group:
        table_group = PrettyTable()
        table_group.field_names = ['Algorithm','A-Gen', 'A-Cstm','A-Opt','A-%', 'L-Gen','L-Cstm','L-Opt','L-%',  'E-Gen','E-Cstm','E-Opt','E-%', 'pD-Gen', 'pD-Cstm','pD-Opt','pD-%', 'c2c-Opt-L', 'c2c-Cstm-L', 'noc_c2c diff-L', 'c2c-Opt-E','c2c-Cstm-E', 'network_c2c diff-E', 'Debug-Gen', 'Debug-Cstm', 'Debug-Opt', 'RE-Gen', 'RE-Cstm', 'RE-Opt', 'DDR Latency-Gen', 'DDR Latency-Cstm', 'DDR Latency-Opt', "DDR latency- error between gen and opt", "DDR latency- error between cstm and opt", "noc latency- cstm", "noc latency- opt", "noc latency- error between cstm and opt", "noc energy- cstm", "noc energy- opt", "noc energy- error between cstm and opt"]
        for column in table_group.field_names[1:]:
            table_group.align[column] = "r" 
        table_group.add_row([
            "Training set", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "","", "","", "","", "", "","", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
        ])
    else:
        table_group.field_names = ['Algorithm','A-Gen', 'A-Cstm','A-Opt','A-%', 'L-Gen','L-Cstm','L-Opt','L-%',  'E-Gen','E-Cstm','E-Opt','E-%', 'pD-Gen', 'pD-Cstm','pD-Opt','pD-%', 'c2c-Opt-L', 'c2c-Cstm-L', 'noc_c2c diff-L', 'c2c-Opt-E','c2c-Cstm-E', 'network_c2c diff-E', 'Debug-Gen', 'Debug-Cstm', 'Debug-Opt', 'RE-Gen', 'RE-Cstm', 'RE-Opt', 'DDR Latency-Gen', 'DDR Latency-Cstm', 'DDR Latency-Opt', "DDR latency- error between gen and opt", "DDR latency- error between cstm and opt", "noc latency- cstm", "noc latency- opt", "noc latency- error between cstm and opt", "noc energy- cstm", "noc energy- opt", "noc energy- error between cstm and opt"]
        for column in table_group.field_names[1:]:
            table_group.align[column] = "r"
        table_group.add_row([
            "Test set", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "","", "","", "","", "", "","", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""
        ])
        #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    for idx in range(len(PPA_curr_custom[0])):
        #import pdb;pdb.set_trace()
        table_group.add_row([
        model_list[idx].split('.')[1],
        f"{float(PPA_curr[2][0]):.2f}",
        f"{float(PPA_curr_custom[2][idx]):.2f}",
        f"{float(PPA_curr_opt[2][idx]):.2f}",
        str(f"{(float(PPA_curr_opt[2][idx])-float(PPA_curr_custom[2][idx]))/float(PPA_curr_custom[2][idx])*100:.3f}")+"%",
        f"{float(PPA_curr[0][idx]):.3f}",
        f"{float(PPA_curr_custom[0][idx]):.3f}",
        f"{float(PPA_curr_opt[0][idx]):.3f}",
        str(f"{(float(PPA_curr_opt[0][idx])-float(PPA_curr_custom[0][idx]))/float(PPA_curr_custom[0][idx])*100:.3f}")+"%",
        f"{float(PPA_curr[1][idx]):.3f}",
        f"{float(PPA_curr_custom[1][idx]):.3f}",
        f"{float(PPA_curr_opt[1][idx]):.3f}",
        str(f"{(float(PPA_curr_opt[1][idx])-float(PPA_curr_custom[1][idx]))/float(PPA_curr_custom[1][idx])*100:.3f}")+"%",
        f"{float(PPA_curr[1][idx]/PPA_curr[0][idx]/PPA_curr[2][0]):.3f}",
        f"{float(PPA_curr_custom[1][idx]/PPA_curr_custom[0][idx]/PPA_curr_custom[2][idx]):.3f}",
        f"{float(PPA_curr_opt[1][idx]/PPA_curr_opt[0][idx]/PPA_curr_opt[2][idx]):.3f}",
        str(f"{(float(PPA_curr_opt[1][idx]/PPA_curr_opt[0][idx]/PPA_curr_opt[2][idx])-float(PPA_curr_custom[1][idx]/PPA_curr_custom[0][idx]/PPA_curr_custom[2][idx]))/float(PPA_curr_custom[1][idx]/PPA_curr_custom[0][idx]/PPA_curr_custom[2][idx])*100:.3f}")+"%",
        f"{sum(PPA_curr_opt[4][idx][1:]):.5f}",
        f"{sum(PPA_curr_custom[4][idx][1:]):.5f}",
        str(f"{(float(sum(PPA_curr_opt[4][idx]))-float(sum(PPA_curr_custom[4][idx])))-(float(PPA_curr_opt[0][idx])-float(PPA_curr_custom[0][idx])):.3f}"),
        f"{sum(PPA_curr_opt[5][idx][1:]):.5f}",
        f"{sum(PPA_curr_custom[5][idx][1:]):.5f}",
        str(f"{((float(sum(PPA_curr_opt[5][idx]))-float(sum(PPA_curr_custom[5][idx])))-(float(PPA_curr_opt[1][idx])-float(PPA_curr_custom[1][idx]))):.3f}"),
        [[node,f"{float(PPA_curr[6][idx].nodes[node]['second_comment'].split(':')[1])*1e+6:.3f}", f"{float(PPA_curr[6][idx].nodes[node]['fifth_comment'])*1e+6:.3f}"] for node in PPA_curr[6][idx].nodes()],
        [[node,f"{float(PPA_curr_custom[6][idx].nodes[node]['second_comment'].split(':')[1])*1e+6:.3f}", f"{float(PPA_curr_custom[6][idx].nodes[node]['fifth_comment'])*1e+6:.3f}"] for node in PPA_curr_custom[6][idx]],
        [[node,f"{float(PPA_curr_opt[6][idx].nodes[node]['second_comment'].split(':')[1])*1e+6:.3f}", f"{float(PPA_curr_opt[6][idx].nodes[node]['fifth_comment'])*1e+6:.3f}"] for node in PPA_curr_opt[6][idx]],
        f"{PPA_curr[10][0]:.3f}",
        f"{PPA_curr_custom[10][idx]:.3f}",
        f"{PPA_curr_opt[10][idx]:.3f}",
        f"{sum(PPA_curr[11][idx])*1e-6:.3f}",
        f"{sum(PPA_curr_custom[11][idx])*1e-6:.3f}",
        f"{sum(PPA_curr_opt[11][idx])*1e-6:.3f}",
        f"{(float(sum(PPA_curr_opt[11][idx]))-float(sum(PPA_curr[11][idx])))/float(sum(PPA_curr_opt[11][idx]))*100:.3f}%",
        f"{(float(sum(PPA_curr_opt[11][idx]))-float(sum(PPA_curr_custom[11][idx])))/float(sum(PPA_curr_opt[11][idx]))*100:.3f}%",
        f"{float(PPA_curr_custom[4][idx][0]):.3f}",
        f"{float(PPA_curr_opt[4][idx][0]):.3f}",
        str(f"{(float(PPA_curr_opt[4][idx][0])-float(PPA_curr_custom[4][idx][0]))/float(PPA_curr_custom[4][idx][0])*100:.3f}")+"%",
        f"{float(PPA_curr_custom[5][idx][0]):.3f}",
        f"{float(PPA_curr_opt[5][idx][0]):.3f}",
        str(f"{(float(PPA_curr_opt[5][idx][0])-float(PPA_curr_custom[5][idx][0]))/float(PPA_curr_custom[5][idx][0])*100:.3f}")+"%",
        ])

    return table_group

def config_table_info(table_group_config, model_list, coverage_common, coverage_custom,coverage_opt , config_common, config_opt_up_custom, config_opt_up_opt, max_cluster_idx):
    if not table_group_config:
        table_group_config=PrettyTable()
        table_group_config.field_names = ['Algorithm','ANC-Gen', 'ANC-Cstm', 'ANC-Opt',  'AEC-Gen', 'AEC-Cstm','AEC-Opt', 'CNU-Gen', 'CNU-Cstm', 'CNU-Opt', 'CEU-Gen', 'CEU-Cstm', 'CEU-Opt', 'Config-Gen','Config-Cstm', 'Config-Opt']
        for column in table_group_config.field_names[1:]:
            table_group_config.align[column] = "r" 
        table_group_config.add_row([
            "Training set", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
        ])  
        train=True
    else:
        table_group_config.field_names = ['Algorithm','ANC-Gen', 'ANC-Cstm', 'ANC-Opt',  'AEC-Gen', 'AEC-Cstm','AEC-Opt', 'CNU-Gen', 'CNU-Cstm', 'CNU-Opt', 'CEU-Gen', 'CEU-Cstm', 'CEU-Opt', 'Config-Gen','Config-Cstm', 'Config-Opt']
        for column in table_group_config.field_names[1:]:
            table_group_config.align[column] = "r"   
        table_group_config.add_row([
            "Test set", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "",
        ]) 
        train=False

    for idx in range(len(model_list)):
        #import pdb;pdb.set_trace()
        #print(config_opt_up_opt[1][max_cluster_idx[idx]][0])
        table_group_config.add_row([
        model_list[idx].split('.')[1],
        f"{coverage_common[0][idx]:.3f}",
        f"{coverage_custom[0][idx]:.3f}",
        f"{coverage_opt[0][idx]:.3f}",
        f"{coverage_common[1][idx]:.3f}",
        f"{coverage_custom[1][idx]:.3f}",
        f"{coverage_opt[1][idx]:.3f}",
        f"{coverage_common[2][idx]:.3f}",
        f"{coverage_custom[2][idx]:.3f}",
        f"{coverage_opt[2][idx]:.3f}",
        f"{coverage_common[3][idx]:.3f}",
        f"{coverage_custom[3][idx]:.3f}",
        f"{coverage_opt[3][idx]:.3f}",
        f"{config_common[0]}\n{config_common[1]}",
        f"{config_opt_up_custom[0][idx]}\n{config_opt_up_custom[1][idx]}",
        f"{config_opt_up_opt[0][max_cluster_idx[idx]]}\n{config_opt_up_opt[1][max_cluster_idx[idx]][0].nodes()}"  if train else f"{config_opt_up_opt[0][max_cluster_idx[idx]]}\n{config_opt_up_opt[1][max_cluster_idx[idx]]}"
        ])
        table_group_config.max_width=55
    return table_group_config
        
def cluster_table_info(table_group_cluster, model_list, cluster_common, cluster_custom_gl, cluster_opt, max_cluster_idx):

    if not table_group_cluster:
        table_group_cluster=PrettyTable()
        table_group_cluster.field_names = ['Algorithm','A_common','A_cstm','A_opt','E_common','E_cstm', 'E_opt', 'L_common','L_cstm', 'L_opt', 'Util_common','Util_cstm','Util_opt','chip_common', 'chip_cstm', 'chip_opt']
        for column in table_group_cluster.field_names[1:]:
            table_group_cluster.align[column] = "r" 
        table_group_cluster.add_row([
            "Training set", "", "", "",  "", "", "",  "", "", "", "", "", "", '','','',
        ]) 
    else:
        table_group_cluster.field_names = ['Algorithm','A_common','A_cstm','A_opt','E_common','E_cstm', 'E_opt', 'L_common','L_cstm', 'L_opt', 'Util_common','Util_cstm','Util_opt','chip_common', 'chip_cstm', 'chip_opt']
        for column in table_group_cluster.field_names[1:]:
            table_group_cluster.align[column] = "r" 
        table_group_cluster.add_row([
            "Test set", "", "", "",  "", "", "",  "", "", "", "", "", "", '','','',
        ]) 
    #import pdb;pdb.set_trace()
    
    Areaofnodes_cm=[[node,f"{float(cluster_common[0].nodes[node]['third_comment'].split(':')[1])*1e+6:.3f}"] for node in cluster_common[0].nodes()]
    Energyofnodes_cm=[[node,f"{float(cluster_common[0].nodes[node]['second_comment'].split(':')[1])*1e+6:.3f}"] for node in cluster_common[0].nodes()]
    Latencyofnodes_cm=[[node,f"{float(cluster_common[0].nodes[node]['comment'].split(':')[1])*1e+6:.3f}"] for node in cluster_common[0].nodes()]
    Utilofnodes_cm=[[node,f"{float(cluster_common[0].nodes[node]['fifth_comment']):.3f}"] for node in cluster_common[0].nodes()]
    Areaofnodes_cstm=[[[node,f"{float(item[0].nodes[node]['third_comment'].split(':')[1])*1e+6:.3f}"] for node in item[0].nodes()] for item in cluster_custom_gl]
    Energyofnodes_cstm=[[[node,f"{float(item[0].nodes[node]['second_comment'].split(':')[1])*1e+6:.3f}"] for node in item[0].nodes()] for item in cluster_custom_gl]
    Latencyofnodes_cstm=[[[node,f"{float(item[0].nodes[node]['comment'].split(':')[1])*1e+6:.3f}"] for node in item[0].nodes()] for item in cluster_custom_gl]
    Utilofnodes_cstm=[[[node,f"{float(item[0].nodes[node]['fifth_comment']):.3f}"]  for node in item[0].nodes()] for item in cluster_custom_gl]
    Areaofnodes_opt=[[[node,f"{float(item[0].nodes[node]['third_comment'].split(':')[1])*1e+6:.3f}"] for node in item[0].nodes()] for item in cluster_opt]
    Energyofnodes_opt=[[[node,f"{float(item[0].nodes[node]['second_comment'].split(':')[1])*1e+6:.3f}"] for node in item[0].nodes()] for item in cluster_opt]
    Latencyofnodes_opt=[[[node,f"{float(item[0].nodes[node]['comment'].split(':')[1])*1e+6:.3f}"] for node in item[0].nodes()] for item in cluster_opt]
    Utilofnodes_opt=[[[node,f"{float(item[0].nodes[node]['fifth_comment']):.3f}"]  for node in item[0].nodes()] for item in cluster_opt]
    for idx in range(len(model_list)):
        #import pdb;pdb.set_trace()
        table_group_cluster.add_row([
        model_list[idx].split('.')[1],
        Areaofnodes_cm,
        Areaofnodes_cstm[idx],
        Areaofnodes_opt[max_cluster_idx[idx]],
        Energyofnodes_cm,
        Energyofnodes_cstm[idx],
        Energyofnodes_opt[max_cluster_idx[idx]],
        Latencyofnodes_cm,
        Latencyofnodes_cstm[idx],
        Latencyofnodes_opt[max_cluster_idx[idx]],
        Utilofnodes_cm,
        Utilofnodes_cstm[idx],
        Utilofnodes_opt[max_cluster_idx[idx]],
        list(nx.connected_components(cluster_common[0])),
        list(nx.connected_components(cluster_custom_gl[idx][0])), 
        list(nx.connected_components(cluster_opt[max_cluster_idx[idx]][0]))
        ])
        table_group_cluster.max_width=55
    return table_group_cluster


def plot_stat_PPA(PPA_curr, PPA_curr_custom, PPA_curr_opt, config_custom, opt, directory_name, train_len, training_set):
    bar_width = 0.4
    relative_path = os.path.join(directory_name, 'custom.json')
    write_json_file(config_custom, relative_path)
    r1 = np.arange(len(PPA_curr_custom[0]))*(bar_width + 0.1)*4
    r2 = [x + bar_width+0.1 for x in r1]
    r3 = [x + bar_width+0.1 for x in r2]
    #import pdb;pdb.set_trace() 

    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_yscale('log')

    bars1 = ax.bar(r1, PPA_curr_custom[0], color='blue', width=bar_width, edgecolor='grey', label='Custom')
    if opt:
        bars2 = ax.bar(r2, PPA_curr_opt[0], color='green', width=bar_width, edgecolor='grey', label='Optimal')
    bars3 = ax.bar(r3, PPA_curr[0], color='red', width=bar_width, edgecolor='grey', label='Generic')
    if opt:
        bars_list=[bars1, bars2, bars3]
    else:
        bars_list=[bars1, bars3]

    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, 
                height, 
                f'{height:.2f}', 
                ha='center', 
                va='bottom', 
                fontweight='bold',
                fontsize=7.5
            )

    ax.set_xlabel('Categories', fontweight='bold')
    ax.set_ylabel('Latency(ms)', fontweight='bold')
    ax.set_title('Latency Comparison between Generic, Optimal and Custom configurations- Training set')
    ax.set_xticks([r*(bar_width + 0.1)*4 + bar_width for r in range(train_len)])
    #import pdb;pdb.set_trace() 
    ax.set_xticklabels(training_set, rotation=45)

    ax.legend()
    plt.tight_layout()
    plt.savefig(directory_name+f'cluster_opt_generic_design_latency.png')
    plt.show()
    plt.close()


    r1 = np.arange(len(PPA_curr_custom[1]))*(bar_width + 0.1)*4
    r2 = [x + bar_width+0.1 for x in r1]
    r3 = [x + bar_width+0.1 for x in r2]
    #import pdb;pdb.set_trace() 

    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_yscale('log')

    bars1 = ax.bar(r1, PPA_curr_custom[1], color='blue', width=bar_width, edgecolor='grey', label='Custom')
    if opt:
        bars2 = ax.bar(r2, PPA_curr_opt[1], color='green', width=bar_width, edgecolor='grey', label='Optimal')
    bars3 = ax.bar(r3, PPA_curr[1], color='red', width=bar_width, edgecolor='grey', label='Generic')
    if opt:
        bars_list=[bars1, bars2, bars3]
    else:
        bars_list=[bars1, bars3]
    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, 
                height, 
                f'{height:.2f}', 
                ha='center', 
                va='bottom', 
                fontweight='bold',
                fontsize=7.5
            )

    ax.set_xlabel('Categories', fontweight='bold')
    ax.set_ylabel('Energy(uJ)', fontweight='bold')
    ax.set_title('Energy Comparison between Generic, Optimal and Custom configurations- Training set')
    ax.set_xticks([r*(bar_width + 0.1)*4 + bar_width for r in range(train_len)])
    ax.set_xticklabels(training_set, rotation=45)

    ax.legend()
    plt.tight_layout()
    plt.savefig(directory_name+f'cluster_opt_generic_design_energy.png')
    plt.show()
    plt.close()

    r1 = np.arange(len(PPA_curr_custom[2]))*(bar_width + 0.1)*4
    r2 = [x + bar_width+0.1 for x in r1]
    r3 = [x + bar_width+0.1 for x in r2]
    #import pdb;pdb.set_trace() 

    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_yscale('log')

    bars1 = ax.bar(r1, PPA_curr_custom[2], color='blue', width=bar_width, edgecolor='grey', label='Custom')
    if opt:
        bars2 = ax.bar(r2, PPA_curr_opt[2], color='green', width=bar_width, edgecolor='grey', label='Optimal')
    bars3 = ax.bar(r3, PPA_curr[2]*train_len, color='red', width=bar_width, edgecolor='grey', label='Generic')
    if opt:
        bars_list=[bars1, bars2, bars3]
    else:
        bars_list=[bars1, bars3]
    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, 
                height, 
                f'{height:.2f}', 
                ha='center', 
                va='bottom', 
                fontweight='bold',
                fontsize=7.5
            )

    ax.set_xlabel('Models', fontweight='bold')
    ax.set_ylabel('Area (mm2)', fontweight='bold')
    ax.set_title('Area Comparison between Generic, Optimal and Custom configurations- Training set')
    ax.set_xticks([r*(bar_width + 0.1)*4 + bar_width for r in range(train_len)])
    ax.set_xticklabels(training_set, rotation=45)

    ax.legend()
    plt.tight_layout()
    plt.savefig(directory_name+f'cluster_opt_generic_design_area.png')
    plt.show()
    plt.close()
