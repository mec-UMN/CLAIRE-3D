from Utils import *
import math
from itertools import combinations, product
from json_utils import *
import numpy as np
import pandas as pd
import time
plot_nop_stats=False

def plot_xy(x, y, filename, title, x_label, y_label):
    plt.figure(figsize=(6, 4))
    bars = plt.bar(x, y, color='skyblue', edgecolor='black')

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', 
                 ha='center', va='bottom', fontsize=10, color='black')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Horizontal grid lines
    #y-axis in log scale
    #plt.yscale('log')
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.savefig(filename)
    plt.close()
    
    #import pdb;pdb.set_trace() 

def area_chiplet_io_nop(nop_mode, nop_width, nop_idx, clusters,graph_or_cluster, A_nop_modes):
    cluster_count=0
    A_cluster,A_cluster_mod, A_network, A_d2d_mod, n_pins=[], [], [0,0], [], []
    for cluster in clusters:
        #print([float(graph_or_cluster.nodes[item]['third_comment'].split(':')[1]) for item in cluster])
        A_IO, A_IO_mod, n_pins_cluster=area_io(nop_mode,nop_width)
        A_nop, A_nop_mod=area_nop(nop_mode, nop_width)
        A_network[1]+=A_IO
        A_network[0]+=A_nop
        A_cluster.append(sum([float(graph_or_cluster.nodes[item]['third_comment'].split(':')[1]) for item in cluster])+A_nop+A_IO)
        A_cluster_mod.append(sum([float(graph_or_cluster.nodes[item]['fourth_comment'].split(':')[1]) for item in cluster]))
        A_d2d_mod.append(A_nop_mod+A_IO_mod)
        n_pins.append(n_pins_cluster)
        cluster_count+=1
        if nop_idx is not None:
            A_nop_modes[nop_idx]+=A_nop*1e+6
    #import pdb;pdb.set_trace()
    return A_nop_modes, A_cluster, A_cluster_mod, cluster_count, A_network, A_d2d_mod, n_pins

def l_e_chiplet_io_nop(nop_mode, nop_width, nop_idx, clusters,graph_or_cluster, ind_graphs, L_nop_modes, E_nop_modes, noc_channel_BW):
    #update ind_graphs and calculate L,E simualtenosuly
    L, E, E_network, L_network, flops, L_ddr=[], [], [], [], [],[]
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
    #print(ind_graphs)
    for idx_graph, graph in enumerate(ind_graphs):
        #print(graph.edges())
        L_node={}
        L_ddr_node={}
        for node in graph.nodes():
            node_name=node[:-1] if node.endswith('1') or node.endswith('2') else node
            L_node[node_name] = max(L_node.get(node_name, 0), graph.nodes[node]['weight'])
            if node_name not in L_ddr_node:
                L_ddr_node[node_name] = 0
            L_ddr_node[node_name] += float(graph.nodes[node]['ddr_latency']) 
        L.append(sum([L_node[node] for node in L_node]))
        L_ddr.append([L_ddr_node[node] for node in L_ddr_node])
        #import pdb; pdb.set_trace()
        #E.append(sum([float(graph.nodes[node]['second_comment'].split(':')[1])*graph.nodes[node]['weight']*1E-9/float(graph.nodes[node]['comment'].split(':')[1]) for node in graph.nodes()]))
        E.append(sum([float(graph.nodes[node]['second_comment'].split(':')[1])*graph.nodes[node]['fifth_comment'] for node in graph.nodes()]))
        #print("Total Compute Latency", L[idx_graph], "Total Energy", E[idx_graph])
        E_network.append([0,0,0])
        L_network.append([0,0,0])
        flops.append(sum([2*graph.nodes[node]['fifth_comment'] if node.startswith('CONV') or node.startswith('LINEAR') else 0 for node in graph.nodes()]))
        #print("flops, lcmput", idx_graph, sum([2*graph.nodes[node]['fifth_comment'] if node.startswith('CONV') or node.startswith('LINEAR') else 0 for node in graph.nodes()]), sum([graph.nodes[node]['weight'] for node in graph.nodes()]))
        #print(E)
       # Initialize per-edge dicts before the loop
        L_noc_dict = {}
        L_IO_dict = {}
        E_noc_dict = {}
        E_IO_dict = {}
        E_nop_dict = {}
        L_nop_dict = {}

        for edge in graph.edges():
            edge_weight = graph.edges[edge]['weight']
            if edge not in graph_or_cluster.edges():
                edgeclusters_identified = 0
                node_a = edge[0]
                node_b = edge[1]
                for idx, cluster in enumerate(clusters):
                    if node_a in cluster:
                        cluster_a = idx
                        edgeclusters_identified += 1
                    if node_b in cluster:
                        cluster_b = idx
                        edgeclusters_identified += 1
                    if edgeclusters_identified == 2:
                        break
                if edgeclusters_identified < 2:
                    import pdb; pdb.set_trace()

                BW_nop, init_BW_noc, E_nop, BW_IO, E_IO = update_nop_width(edge_weight, nop_mode, noc_channel_BW, nop_width, PPA_config)
                L_nop = math.ceil(edge_weight * init_BW_noc / BW_nop)
                L_IO = math.ceil(edge_weight * init_BW_noc / BW_IO)

                # Record per-edge values in dicts
                L_nop_dict[edge] = L_nop
                L_IO_dict[edge] = L_IO
                E_nop_dict[edge] = E_nop
                E_IO_dict[edge] = E_IO

            else:
                BW_noc, init_BW_noc, E_noc = update_noc_width(graph, edge, noc_channel_BW)
                L_noc = math.ceil(edge_weight * init_BW_noc / BW_noc)

                # Record per-edge values in dicts
                L_noc_dict[edge] = L_noc
                E_noc_dict[edge] = E_noc

        # --- Outside the loop: accumulate into L, E, L_network, E_network ---
        #Ensure parallelism for edges in same cluster by taking max of their latencies instead of sum
        L_noc_parallel_dict = {}
        for key, value in L_noc_dict.items():
            #import pdb; pdb.set_trace()
            updated_edge_name = tuple(item[:-1] if item.endswith('1') or item.endswith('2') else item for item in key)
            L_noc_parallel_dict[updated_edge_name] = max(value, L_noc_parallel_dict.get(updated_edge_name, 0))
        L[idx_graph] += sum(L_noc_parallel_dict.values()) + sum(L_nop_dict.values()) + sum(L_IO_dict.values())
        E[idx_graph] += sum(E_noc_dict.values()) + sum(E_nop_dict.values()) + sum(E_IO_dict.values())

        E_network[idx_graph][0] += sum(E_noc_dict.values()) * 1e+6
        L_network[idx_graph][0] += sum(L_noc_parallel_dict.values()) * 1e-6
        E_network[idx_graph][1] += sum(E_nop_dict.values()) * 1e+6
        L_network[idx_graph][1] += sum(L_nop_dict.values()) * 1e-6
        E_network[idx_graph][2] += sum(E_IO_dict.values()) * 1e+6
        L_network[idx_graph][2] += sum(L_IO_dict.values()) * 1e-6               
        #print(BW_noc)
        #print("flops, lcmput", idx_graph, sum([2*graph.nodes[node]['fifth_comment'] if node.startswith('CONV') or node.startswith('LINEAR') else 0 for node in graph.nodes()]), sum([graph.nodes[node]['weight'] for node in graph.nodes()]), L_network[idx_graph][0] )
                #update graph weight
                #print("E", E[idx_graph], E_noc)
        #print("flops, lcmput", idx_graph, sum([2*graph.nodes[node]['fifth_comment'] if node.startswith('CONV') or node.startswith('LINEAR') else 0 for node in graph.nodes()]), sum([graph.nodes[node]['weight'] for node in graph.nodes()]))
        #print( "Total Network Latency", L_network[idx_graph][0], "Total Network Energy", E_network[idx_graph][0])
    if nop_idx is not None:
        #L_nop_modes.append(sum(row[1] for row in L_network))
        L_nop_modes[nop_idx]+=L_network[0][1]
        #E_nop_modes.append(sum(row[1] for row in E_network))
        E_nop_modes[nop_idx]+=E_network[0][1]

    return L_nop_modes, E_nop_modes, L, E, L_network, E_network, flops, ind_graphs, L_ddr

def nop_topology_dse(nop_modes, nop_widths, clusters, graph_or_cluster, ind_graphs, directory_name, noc_channel_BW, model_list):
    A_nop_modes, E_nop_modes, L_nop_modes=[0]*len(nop_modes), [0]*len(nop_modes) , [0]*len(nop_modes)       
    #map algorithm onto cluster and check if edge not in cluster
    for nop_idx, nop_mode in enumerate(nop_modes):
        nop_width= nop_widths[nop_idx]
        A_nop_modes, _,_,_, _, _, _=area_chiplet_io_nop(nop_mode, nop_width, nop_idx, clusters,graph_or_cluster, A_nop_modes)
        L_nop_modes, E_nop_modes, _,_, _, _, _,_,_=l_e_chiplet_io_nop(nop_mode, nop_width, nop_idx, clusters,graph_or_cluster, ind_graphs, L_nop_modes, E_nop_modes, noc_channel_BW)
    #import pdb;pdb.set_trace() 
    
    #print(directory_name)
    if plot_nop_stats:
        plot_xy(nop_modes, L_nop_modes , directory_name+ 'L_nop_'+ model_list[0]+'.png', 'Latency of Interconnect Topologies', 'Interconnection type', 'Latency of the NoP interface in ms')
        plot_xy(nop_modes, E_nop_modes, directory_name+'E_nop_'+ model_list[0]+'.png', 'Energy of Interconnect Topologies', 'Interconnection type', 'Energy of the NoP interface in uJ')
        plot_xy(nop_modes, A_nop_modes, directory_name+'A_nop_'+ model_list[0]+'.png', 'Area of Interconnect Topologies', 'Interconnection type', 'Area of the NoP interface in mm2')
        #import pdb;pdb.set_trace() 
    PPA=[(L or 1) * (E or 1) * (A or 1)for L,E,A in zip(L_nop_modes, E_nop_modes, A_nop_modes)]
    nop_top_idx=PPA.index(min(PPA))
    #print("nop_top_idx", nop_top_idx)
    #import pdb;pdb.set_trace() 
    return nop_top_idx

def nop_dse(mode, sample_width, predefined_nop_width,clusters, graph_or_cluster, ind_graphs, directory_name, noc_channel_BW, model_list):
    start_time = time.time()
    nop_mode=mode[0]
    if len(clusters)>1:
        #nop_width=width[0]
        #noc_BWs=[1*i for i in range(int(noc_channel_BW), int(200*noc_channel_BW), int(noc_channel_BW))]
        
        start, finish=1,20000
        nop_widths=[1*i for i in range(start, finish, 10)]
    
        throughput_semiideal, rcc_ideal, throughput_ideal, ideal_Qout_t=[],[],[],[]
        flops_ideal, L_chip_ideal, L_noc_ideal =0,0,0
        #Ideal Graph
        #import pdb;pdb.set_trace() 
        for cluster_idx, cluster in enumerate(clusters):
            ideal_Qout_t.append(0)
            for node in cluster:
                L_chip_ideal+=float(graph_or_cluster.nodes[node]['comment'].split(':')[1])*1e9
                flops_ideal+=2*float(graph_or_cluster.nodes[node]['sixth_comment'].split(':')[1]) 

        #print("time taken for nop dse ideal setup:", time.time()-start_time)
        current_dir = os.path.dirname(__file__)
        relative_path = os.path.join(current_dir, 'PPA_config.json')
        with open(relative_path) as f:
            PPA_config = json.load(f)
        for idx, nop_width in enumerate(nop_widths):  
            L_noc_ideal=0
            nop_BW,_=beachhead(nop_mode, nop_width, PPA_config)
            _, _, _, BW_IO, _=update_nop_width(None, nop_mode, 0, nop_width, PPA_config)  
            
            for edge in graph_or_cluster.edges():
                #import pdb;pdb.set_trace() 
                node1, node2=edge
                ideal_Qout=max(float(graph_or_cluster.nodes[node1]['seventh_comment'].split(':')[1]),float(graph_or_cluster.nodes[node2]['seventh_comment'].split(':')[1]))
                L_noc_ideal+=math.ceil(ideal_Qout/nop_BW*1e9)  #placing noc_width=nop_width for smoother graph
        
            L_nop_ideal=0
            L_io_ideal=0 
            #import pdb;pdb.set_trace() 
            #ideal_Qout_nop=max(ideal_Qout_t)
            all_pairs = []
            for s1, s2 in combinations(clusters, 2):
                #all_pairs.extend(product(s1, s2))
                for pair in product(s1, s2):
                    node1, node2=pair
                    ideal_Qout_nop=max(float(graph_or_cluster.nodes[node1]['seventh_comment'].split(':')[1]),float(graph_or_cluster.nodes[node2]['seventh_comment'].split(':')[1]))
                    L_nop_ideal+=math.ceil(ideal_Qout_nop/nop_BW*1e9) 
                    L_io_ideal+=math.ceil(ideal_Qout_nop/BW_IO*1e9)
            #import pdb;pdb.set_trace() 
            #L_nop_ideal=math.ceil(ideal_Qout_nop/nop_BW*1e9)
            #L_io_ideal=math.ceil(ideal_Qout_nop/BW_IO*1e9)
            #print(noc_channel_BW*1e-9, nop_BW*1e-9, BW_IO*1e-9, L_io_ideal, L_noc_ideal, L_nop_ideal)   
            rcc_ideal.append((L_chip_ideal)/max(L_io_ideal, L_noc_ideal, L_nop_ideal))
            throughput_semiideal.append(flops_ideal*1e-3/(L_chip_ideal+L_io_ideal+L_noc_ideal+L_nop_ideal)) #TFLOPS/s
            throughput_ideal.append(flops_ideal*1e-3/max(L_chip_ideal,L_io_ideal, L_noc_ideal, L_nop_ideal)) #TFLOPS/s
            #print(nop_width)
            if rcc_ideal[-1]>15:
                #print(rcc_ideal, throughput_ideal)
                break
        #print("time taken for nop dse rcc and throughput calc_1:", time.time()-start_time)
        x3, y3 =rcc_ideal, throughput_ideal
        if plot_nop_stats:
            plt.plot(x3, y3, color='orange', marker='s', linestyle='-', label="Ideal")
        df = pd.DataFrame({"x3": x3, "y3": y3})
        if plot_nop_stats:
            cmap = plt.get_cmap("viridis")
        
        if predefined_nop_width:
            nop_widths=[predefined_nop_width]
        else:
            start, finish=1,25
            nop_widths=[1*i for i in range(start, finish)]
        non_zero_nop_comm_models=[]
        rcc, throughput=[],[]
        L_network=l_e_chiplet_io_nop(nop_mode, sample_width[0], None, clusters,graph_or_cluster, ind_graphs, None, None, noc_channel_BW)[4]
        for model_idx, model in enumerate(model_list):
            if L_network[model_idx][1]!=0:
                non_zero_nop_comm_models.append(model)
                rcc.append([])
                throughput.append([]) #TFLOPS/s
        nop_count=[]
        current_dir = os.path.dirname(__file__)
        relative_path = os.path.join(current_dir, 'PPA_config.json')
        with open(relative_path) as f:
            PPA_config = json.load(f)
        for nop_idx, nop_width in enumerate(nop_widths):
            nop_BW, max_length=beachhead(nop_mode, nop_width, PPA_config)
            _, _, _, BW_IO, _ =update_nop_width(None, nop_mode, 0, nop_width, PPA_config)  
            #if nop_BW<noc_channel_BW or BW_IO<noc_channel_BW:
            if nop_BW<noc_channel_BW:
                #print("No of nop channels:", nop_width, "nop bandwidth:" , nop_BW*1e-9, "noc bandwidth:" , noc_channel_BW*1e-9, "io bandwidth:" , BW_IO*1e-9)
                continue
            _, A_cluster, A_cluster_mod, cluster_count, A_network, A_d2d_mod, n_pins =area_chiplet_io_nop(nop_mode, nop_width, None, clusters,graph_or_cluster, None)
            A_chiplet_min=min(A_cluster)
            if(max_length>math.sqrt(A_chiplet_min)):
                aspect_ratio=max_length**2/A_chiplet_min #length/width
                #import pdb;pdb.set_trace() 
                #break
            _, _, L, _, L_network, _, flops, ind_graphs, _ =l_e_chiplet_io_nop(nop_mode, nop_width, None, clusters,graph_or_cluster, ind_graphs, None, None, noc_channel_BW)
            #import pdb;pdb.set_trace() 
            model_count=0
            #import pdb;pdb.set_trace() 
            for model_idx, model in enumerate(model_list):
                if model in non_zero_nop_comm_models:
                    #import pdb;pdb.set_trace() 
                    L_chip=L[model_idx]-sum(L_network[model_idx])*1e6
                    #print(L_chip, sum(L_network[model_idx])*1e6, flops[model_idx])
                    rcc_temp=L_chip/(max(L_network[model_idx])*1e6)
                    throughput_temp=flops[model_idx]*1e-3/max(max(L_network[model_idx])*1e6, L_chip)
                    rcc[model_count].append(rcc_temp)
                    throughput[model_count].append(throughput_temp) #TFLOPS/s
                    model_count+=1
            #import pdb;pdb.set_trace() 
            if not predefined_nop_width:
                if min([row[-1] for row in rcc])>1:
                    #print(rcc[-1], throughput[-1])
                    crossing_point=nop_idx
                    nop_count.append(nop_widths[nop_idx])
                    #import pdb;pdb.set_trace() 
                    break
            nop_count.append(nop_widths[nop_idx])
            
        #import pdb;pdb.set_trace() 
        #print("time taken for nop dse rcc and throughput calc_2:", time.time()-start_time)  
        for model_idx, _ in enumerate(non_zero_nop_comm_models):
            if predefined_nop_width:
                noc_width=width_noc(noc_channel_BW)
                x1, y1, labels =rcc[model_idx], throughput[model_idx], ["NoP count:"+str(x)+"\n Noc count:"+str(noc_width)+"\n rcc:"+str(round(rcc[model_idx][idx],2)) for idx,x in enumerate(nop_count)]
            else:
                x1, y1, labels =rcc[model_idx], throughput[model_idx], [str(x) for idx,x in enumerate(nop_count)]
            # Create first scatter plot
            if plot_nop_stats:
                plt.scatter(x1, y1, color=cmap(model_idx/len(non_zero_nop_comm_models)), marker='o', label="CLAIRE"+non_zero_nop_comm_models[model_idx])
            df_new = pd.DataFrame({"x1": x1, "y1": y1, "Model":non_zero_nop_comm_models[model_idx]})
            df = pd.concat([df, df_new], ignore_index=True)
            # Create second scatter plot
            #plt.plot(x2, y2, color='green', marker='s', linestyle='-', label="Semiideal")

            # Add labels for first dataset
            if plot_nop_stats:
                for i in range(len(x1)):
                    plt.text(x1[i], y1[i], labels[i], fontsize=12, ha='right', va='top', color='blue')
        if plot_nop_stats:
            plt.xlabel("rcc (Compute Latency/Communciation latency)")
            plt.ylabel("Throughput (TFLOPS/s)")
            plt.title("Roofline model when NoP links are changed")
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)  # Horizontal grid lines
            plt.xticks(rotation=45)  # Rotate x-axis labels if needed
            plt.savefig( directory_name+'roofline_nopdse_'+"_".join([model.split(".")[1] for model in model_list])+'.png')
            #import pdb;pdb.set_trace() 
            plt.close()
        #print(rcc)
        if not predefined_nop_width:
            if rcc[0]:
                if min([row[-1] for row in rcc])>1:
                    #print(rcc)
                    opt_width=nop_widths[crossing_point]
                    #print(f"RCC crosses 1 at index {crossing_point}, noc_width = {noc_widths[crossing_point]}, RCC = {rcc[crossing_point]}")
                else:
                    print("RCC does not cross 1 in the given data.")
                    import pdb;pdb.set_trace() 
                    opt_width=nop_widths[-1]
            else:
                print("Error! NoP constraints not met")
                opt_width=nop_widths[-1]
                import pdb;pdb.set_trace() 
        df.to_excel(directory_name+"plot_data_nop"+"_".join([model.split(".")[1] for model in model_list])+".xlsx", index=False)
    else:
        opt_width=sample_width[0]
    #print("time taken for nop dse setup:", time.time()-start_time)
    #import pdb;pdb.set_trace()
    if predefined_nop_width:
        opt_width=predefined_nop_width
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
    opt_nop_BW, _=beachhead(nop_mode, opt_width, PPA_config)
    _, A_cluster, A_cluster_mod, cluster_count, A_network, A_d2d_mod, n_pins =area_chiplet_io_nop(nop_mode, opt_width, None, clusters,graph_or_cluster, None)
    _, _, L, E, L_network, E_network, flops, ind_graphs, L_ddr =l_e_chiplet_io_nop(nop_mode, opt_width, None, clusters,graph_or_cluster, ind_graphs, None, None, noc_channel_BW)
    #print("time taken for nop dse l_e and area:", time.time()-start_time)
    L_noio_nonop, L_io_nonop, L_io_nop,E_noio_nonop, E_io_nonop, E_io_nop, A_noio_nonop, A_io_nonop, A_io_nop, A_noio_nop=[],[],[],[],[],[],[],[],[], []
    for model_idx, _ in enumerate(model_list):
        L_noio_nonop.append(L[model_idx]*1e-6-sum(L_network[model_idx][1:]))
        L_io_nonop.append(L[model_idx]*1e-6-sum(L_network[model_idx][1:2]))
        L_io_nop.append(L[model_idx]*1e-6)
        E_noio_nonop.append(E[model_idx]*1e9-sum(E_network[model_idx][1:])*1e3)
        E_io_nonop.append(E[model_idx]*1e9-sum(E_network[model_idx][1:2])*1e3)
        E_io_nop.append(E[model_idx]*1e9)
    A_noio_nonop.append(sum(A_cluster)*1e6-sum(A_network[0:])*1e6)
    A_io_nonop.append(sum(A_cluster)*1e6-sum(A_network[0:1])*1e6)
    A_noio_nop.append(sum(A_cluster)*1e6-sum(A_network[1:2])*1e6)
    A_io_nop.append(sum(A_cluster)*1e6)

    if plot_nop_stats:
        plot(model_list, L_noio_nonop,L_io_nonop,L_io_nop, "Model List", "Latency(ms)", "Impact of NoP and IO on Latency", 
            directory_name+"Latency_overhead_"+"_".join([model.split(".")[1] for model in model_list])+'.png', 
            "W/o IO and W/o NoP","With IO and W/o NoP","With IO and With NoP")
        plot(["_".join([model.split(".")[1] for model in model_list])], A_noio_nonop,A_io_nonop,A_io_nop, "Model List", "Area(mm2)", "Impact of NoP and IO on Area", 
            directory_name+"Area_overhead_"+"_".join([model.split(".")[1] for model in model_list])+'.png', 
            "W/o IO and W/o NoP","With IO and W/o NoP","With IO and With NoP")
        plot(["_".join([model.split(".")[1] for model in model_list])], A_noio_nonop,A_noio_nop,A_io_nop, "Model List", "Area(mm2)", "Impact of NoP and IO on Area", 
            directory_name+"Area_overhead_updated_"+"_".join([model.split(".")[1] for model in model_list])+'.png', 
            "W/o IO and W/o NoP","W/o IO and with NoP","With IO and With NoP")
        plot(model_list, E_noio_nonop,E_io_nonop,E_io_nop, "Model List", "Energy(mJ)", "Impact of NoP and IO on Energy", 
            directory_name+"Energy_overhead_"+"_".join([model.split(".")[1] for model in model_list])+'.png', 
            "W/o IO and W/o NoP","With IO and W/o NoP","With IO and With NoP")
    

    #print("time taken for nop dse:", time.time()-start_time)
    #import pdb;pdb.set_trace()
    return A_cluster, A_cluster_mod, L, E, L_network, E_network, A_network, cluster_count, opt_nop_BW, ind_graphs, A_d2d_mod, n_pins, L_ddr

def plot(x_list, list1,list2,list3, xlabel, ylabel, title, filename, label1,label2,label3):
    bar_width = 0.4
    r1 = np.arange(len(list1))*(bar_width + 0.1)*4
    r2 = [x + bar_width+0.1 for x in r1]
    r3 = [x + bar_width+0.1 for x in r2]
        #import pdb;pdb.set_trace() 

    fig, ax = plt.subplots(figsize=(24, 16))
    ax.set_yscale('log')

    bars1 = ax.bar(r1, list1, color='blue', width=bar_width, edgecolor='grey', label=label1)
    bars2 = ax.bar(r2, list2, color='green', width=bar_width, edgecolor='grey', label=label2)
    bars3 = ax.bar(r3, list3, color='red', width=bar_width, edgecolor='grey', label=label3)
    bars_list=[bars1, bars2, bars3]

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

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title)
    ax.set_xticks([r*(bar_width + 0.1)*4 + bar_width for r in range(len(x_list))])
    #import pdb;pdb.set_trace() 
    ax.set_xticklabels(x_list, rotation=45)

    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    #import pdb;pdb.set_trace() 

    plt.close()

def nre_cost_evaluate(A, A_mod, cluster_count):
    cost_mode='CATCH'
    if cost_mode=="chiplet_acturary":
        #NRECost_total = sum_i(K_chip*A_chiplet_i+K_mod*A_mod_i)=sum(A_chiplet)*K_chip+sum(A_mod)*K_mod
        #K_chip=0.3*5.1e7/300, K_mod = 0.5*5.1e7/300 - https://arxiv.org/abs/2203.12268v4
        NRECost=[(A[0]*0.3/300+A_mod[0]*0.5/300)*51] #in 10^6 terms
        #NRECost=[0.2*5.1e7]
        #print(NRECost)
    elif cost_mode=="CATCH":
        #NRECost_total=NRE_backend_cost+NRE_frontend_cost+nre_mask_cost
        with open("PPA_config.json", "r") as file:
            data = json.load(file)

        # Filter keys starting with 'nre' and assign them dynamically
        nre_variables = {key: value for key, value in data.items() if key.startswith("NRE")}

        # Assign to global variables
        globals().update(nre_variables)
        #import pdb;pdb.set_trace() 
        nre_l=NRE_backend_cost_logic+NRE_frontend_cost_logic
        nre_m=NRE_backend_cost_mem+NRE_frontend_cost_mem
        A_logic=1
        #Pl=A_logic/A[0] #PL, PM, and PA refer to the percentage of the design that falls into the logic, memory, or analog categories
        Pl = 1
        #Preticleshare is the percentage of the reticle that is taken by the design
        p_r_share=1
        NRECost=[A[0]*(nre_l*Pl+ nre_m*(1-Pl))+cluster_count*NRE_mask_cost*p_r_share] 
    return NRECost