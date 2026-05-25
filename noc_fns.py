import json
import os
import math
import matplotlib.pyplot as plt
from json_utils import *
from optimize_fns import eval_noc
noc_dse_exec=False

def run_noc_dse(directory_name,  cluster_common,config_common, cluster_custom, config_custom, cluster_opts, config_opts, ind_graphs_common,  ind_graphs_list_custom, ind_graphs_list_opt, PPA_common_in, PPA_custom_in, PPA_opts_in):
    os.system("rm -r "+directory_name)
    os.system("mkdir "+directory_name)
    #Optimal/Libray synthesized dse
    PD_limit=1500 #mW/mm2
    Nocdse_BW_opts=[]
    cluster_opts_noc=[]
    opt_widths=[]
    PPA_opt=[]
    if cluster_opts:
        for cluster_idx, cluster_opt in enumerate(cluster_opts):
            #import pdb;pdb.set_trace() 
            result=noc_dse(cluster_opt, directory_name, config_opts[0][cluster_idx], ind_graphs_list_opt[cluster_idx])
            Nocdse_BW_opts.append(result[0])
            cluster_opts_noc.append(result[1])
            #opt_widths.append(result[2])
            PPA_opt.append(eval_noc(result[1][1],result[0], ind_graphs_list_opt[cluster_idx], PPA_opts_in[cluster_idx]))
            #import pdb;pdb.set_trace()

    Nocdse_BW_common=[]
    cluster_common_noc=[]
    PPA_common=[]
    if cluster_common:
        #generic dse
        Nocdse_BW_common, cluster_common_noc=noc_dse(cluster_common, directory_name, config_common[0], ind_graphs_common)
        PPA_common=eval_noc(cluster_common_noc[1],Nocdse_BW_common, ind_graphs_common,PPA_common_in)


    Nocdse_BW_custom=[]
    cluster_custom_noc=[]
    opt_width_custom=[]
    PPA_custom=[[],[],[]]
    #Custom dse
    for cluster_idx, cluster in enumerate(cluster_custom):
        result=noc_dse(cluster, directory_name, config_custom[0][cluster_idx], ind_graphs_list_custom[cluster_idx])
        Nocdse_BW_custom.append(result[0])
        cluster_custom_noc.append(result[1])
        #opt_width_custom.append(result[2])
        PPA_temp=eval_noc(result[1][1], result[0], ind_graphs_list_custom[cluster_idx], [[item[cluster_idx]] for item in PPA_custom_in])
        for idx,item in enumerate(PPA_temp):
            #print(PPA_custom,item)
            PPA_custom[idx].append(item[0])
    #import pdb;pdb.set_trace() 

    return Nocdse_BW_common, cluster_common_noc, Nocdse_BW_custom, cluster_custom_noc,  Nocdse_BW_opts, cluster_opts_noc, PPA_common, PPA_custom, PPA_opt

def noc_dse(universal_cluster, directory_name, config, ind_graphs):
    model_list=universal_cluster[1]
    #print("run_noc", model_list)
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
    init_noc_BW=PPA_config['BW_noc']
    if noc_dse_exec:
        start, finish=1, 2000
        throughput_semiideal, rcc_ideal, throughput_ideal=[],[],[]
        flops_ideal, L_compute_ideal =0,0
        noc_widths=[1*i for i in range(start, finish)]
        #Ideal Graph
        for node in universal_cluster[0].nodes():
            L_compute_ideal+=float(universal_cluster[0].nodes[node]['comment'].split(':')[1])*1e9
            flops_ideal+=2*float(universal_cluster[0].nodes[node]['sixth_comment'].split(':')[1])
        for idx, channel_width in enumerate(noc_widths):  
            #import pdb;pdb.set_trace() 
            noc_BW=channel_width*PPA_config['num_bits_per_link_noc']*PPA_config["Throughput_noc_router_max"] # 8 bits per 1 link
            L_comm, L_comm_ideal =0,0
            for edge in universal_cluster[0].edges():
                #import pdb;pdb.set_trace() 
                node1, node2=edge
                ideal_Qout=max(float(universal_cluster[0].nodes[node1]['seventh_comment'].split(':')[1]),float(universal_cluster[0].nodes[node2]['seventh_comment'].split(':')[1]))
                L_comm_ideal+=math.ceil(ideal_Qout/noc_BW*1e9)

            rcc_ideal.append(L_compute_ideal/L_comm_ideal)
            throughput_semiideal.append(flops_ideal*1e-3/(L_comm_ideal+L_compute_ideal)) #TFLOPS/s
            throughput_ideal.append(flops_ideal*1e-3/max(L_comm_ideal,L_compute_ideal)) #TFLOPS/s
            if rcc_ideal[idx]>1:
                break
        #if any("vgg" in item for item in model_list):
            #import pdb;pdb.set_trace() 
        
        x3, y3 =rcc_ideal, throughput_ideal

        # For all groups in the group
        start, finish=1, 1000
        noc_widths=[1*i for i in range(start, finish)]
        current_dir = os.path.dirname(__file__)
        relative_path = os.path.join(current_dir, 'params.json')
        write_json_file(config, relative_path)
        #ind_graphs, _,_,_ =construct_ind_graphs(model_list, False)
        opt_width=plot_roofline_actual(model_list,ind_graphs, noc_widths,init_noc_BW, directory_name, False, x3, y3)
        opt_noc_BW=opt_width*PPA_config['num_bits_per_link_noc']*PPA_config["Throughput_noc_router_max"]

        #import pdb;pdb.set_trace() 


        #print(model_list, best_noc_BW, min_decision)
        #print(opt_noc_BW, opt_width)`
    
    else:
        opt_noc_BW=init_noc_BW
        
    for edge in universal_cluster[0].edges():
        universal_cluster[0].edges[edge]['weight']=math.ceil(universal_cluster[0].edges[edge]['weight']*init_noc_BW/opt_noc_BW)
    
    if noc_dse_exec:
        plot_roofline_actual(model_list,ind_graphs, [opt_width],init_noc_BW, directory_name, True,x3,y3)
    return opt_noc_BW, universal_cluster

def plot_roofline_actual(model_list,ind_graphs, noc_widths,init_noc_BW, directory_name, plot, x3, y3):
    if plot:
        plt.plot(x3, y3, color='orange', marker='s', linestyle='-', label="Ideal")
    plt.savefig(directory_name+'roofline_nocdse_'+"_".join([model.split(".")[1] for model in model_list])+'.png')
     
    #import pdb;pdb.set_trace() 

    opt_width=0
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
    #print("construct_graph_noc",construct_ind_graphs(model_list, False))
    cmap = plt.get_cmap("viridis")
    #plot_models=['vision.densenet121','vision.mobilenetv2',  'vision.resnet18']
    plot_models=model_list
    for model_idx, graph in enumerate(ind_graphs):
        if model_list[model_idx] in plot_models:
            crossing_point=0
            rcc, throughput=[],[]
            flops, L_compute=0,0
            for node in graph.nodes():
                L_compute+=graph.nodes[node]['weight']
                flops+=2*graph.nodes[node]['fifth_comment'] if node.startswith('CONV') or node.startswith('LINEAR') else 0
            #print("flops, lcmput", model_idx, flops, L_compute)
            for idx, channel_width in enumerate(noc_widths):  
                noc_BW=channel_width*PPA_config['num_bits_per_link_noc']*PPA_config["Throughput_noc_router_max"] # 8 bits per 1 link 
                L_comm=0
                for edge in graph.edges():
                    weight=graph.edges[edge]['weight']
                    L_comm+=math.ceil(weight*init_noc_BW/noc_BW)
                
                rcc.append(L_compute/L_comm)
                throughput.append(flops*1e-3/max(L_comm, L_compute)) #TFLOPS/s
                if rcc[idx]>1:
                    crossing_point=idx
                    #print("flops, lcmput", model_idx, flops, L_compute, L_comm)
                    #print("rcc", rcc[-1], throughput[-1])
                    break
            if plot:
                #import pdb;pdb.set_trace() 

                x1, y1, labels =rcc, throughput, [str(x) for x in noc_widths]
                # Create first scatter plot
                plt.scatter(x1, y1, color=cmap(model_idx/len(model_list)), marker='o', label="CLAIRE"+model_list[model_idx])

                # Create second scatter plot
                #plt.plot(x2, y2, color='green', marker='s', linestyle='-', label="Semiideal")

                # Add labels for first dataset
                for i in range(len(x1)):
                    plt.text(x1[i], y1[i], labels[i], fontsize=12, ha='right', va='bottom', color='blue')
            if rcc[-1]>1:
                #print(rcc, noc_widths[crossing_point])
                opt_width=max(opt_width,noc_widths[crossing_point])
                #import pdb;pdb.set_trace() 
                #print(f"RCC crosses 1 at index {crossing_point}, noc_width = {noc_widths[crossing_point]}, RCC = {rcc[crossing_point]}")
            else:
                print("RCC does not cross 1 in the given data.")
                import pdb;pdb.set_trace() 
            #import pdb;pdb.set_trace() 
        if plot:
            plt.xlabel("rcc (Compute Latency/Communciation latency)")
            plt.ylabel("Throughput (TFLOPS/s)")
            plt.title("Roofline model when NoC links are changed")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)  # Horizontal grid lines
            plt.xticks(rotation=45)  # Rotate x-axis labels if needed
            plt.savefig(directory_name+'roofline_nocdse_'+"_".join([model.split(".")[1] for model in model_list])+'.png')
        #import pdb;pdb.set_trace()        
        # Labels and title
    plt.close()
    return opt_width
