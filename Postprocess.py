from torchprofile import profile_macs
import torch.nn as nn
import torch
import numpy as np
import textwrap
import itertools
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.algorithms.community import girvan_newman, label_propagation_communities, kernighan_lin_bisection
import time
import hashlib
import networkx as nx
try:
    from leidenalg import find_partition, ModularityVertexPartition
    _has_leiden = True
except ImportError:
    _has_leiden = False
try:
    import metis
    _has_metis = True
except ImportError:
    _has_metis = False
from prettytable import PrettyTable
from collections import OrderedDict, defaultdict,  Counter
import matplotlib.pyplot as plt
from Utils import print_file

from models import SimpleConv, SimpleLinear, SimpleRELU, SimpleGELU, SimpleDropout, SimpleAdaptPool, SimpleMaxPool, SimpleEncoder

import community as community_louvain
from sklearn.cluster import KMeans,  SpectralClustering

plot_process_stats=False

def stats(grouped_layers, num_bits, G):
    desired_width=30
    flops_total=0
    params_total=0
    table_group = PrettyTable()
    table_group.field_names = ['Layer Type', 'Frequency', 'Input_Volume (MB)','Output_Volume (MB)','Input Degree-Connectivity', 'Output Degree-Connectivity', 'Promixity Factor', 'GFLOPs', 'Parameter_size(MB)',]
    for idx, (key, layer_numbers) in enumerate(grouped_layers.items(), start=1):
        #import pdb;pdb.set_trace() 
        params=no_params(key, num_bits)
        flops =macs(key)
        wrapped_layer_names = textwrap.fill(str(key[0])+'_'+str(idx), width=desired_width//2)  
        in_bw=key[1][0]*key[1][0]*key[3]*num_bits*1e-6/8     
        out_bw=key[2][0]*key[2][0]*key[4]*num_bits*1e-6/8
        prox=(max(layer_numbers)-min(layer_numbers))/(len(layer_numbers))
        layer_name=wrapped_layer_names.split('_')[0]+"_"+str(key[1][0])+"_"+str(key[2][0])+"_"+str(key[3])+"_"+str(key[4])
        table_group.add_row([
            layer_name,
            len(layer_numbers),
            round(in_bw,4),
            round(out_bw,4),
            G.to_directed().in_degree(idx),
            G.to_directed().out_degree(idx),
            round(prox,0),
            round(flops*2*1e-9,4),
            round(params,4),
        ])
        params_total+= round(params,4)
        flops_total+=round(flops*2*1e-9,4)
    #print("flops",flops_total)
    table_group.add_row([
        "Total",
        "",
        "",
        "",
        "",
        "",
        "",
        round(flops_total,3),
        round(params_total,3),
    ])
    return table_group

def no_params(curr, num_bits):
    #import pdb;pdb.set_trace() 
    if curr[0].startswith('CONV'):
        return (int(curr[5][0])*int(curr[5][0])*curr[3]+1)*curr[4]*num_bits*1e-6/8 
    elif curr[0].startswith('LINEAR'):
        return curr[4]*(curr[3]+1)*num_bits*1e-6/8 
    else:
        return 0

def macs(curr):
    if curr[0].startswith('CONV'):
        inputs= torch.rand(1, curr[3], curr[1][0], curr[2][0])
        model= SimpleConv(curr)
    elif curr[0].startswith('LINEAR'):
        inputs= torch.rand(1, curr[3])
        model= SimpleLinear(curr)
    elif curr[0].startswith('RELU'):
        inputs= torch.rand(1, curr[3], curr[1][0], curr[2][0])
        model= SimpleRELU(curr)
    elif curr[0].startswith('GELU'):
        inputs= torch.rand(1, curr[3], curr[1][0], curr[2][0])
        model= SimpleGELU(curr)
    elif curr[0].startswith('DROPOUT'):
        inputs= torch.rand(1, curr[3], curr[1][0], curr[2][0])
        model= SimpleDropout(curr)
    elif curr[0].startswith('ADAPTIVEAVGPOOL'):
        inputs= torch.rand(1, curr[3], curr[1][0], curr[2][0])
        model= SimpleAdaptPool(curr)
    elif curr[0].startswith('MAXPOOL'):
        inputs= torch.rand(1, curr[3], curr[1][0], curr[2][0])
        model= SimpleMaxPool(curr)
    elif curr[0].startswith('NONDYNAMIC'):
        inputs= torch.rand(1, curr[3], curr[1][0], curr[2][0])
        model= SimpleEncoder(curr)
    else:
        inputs= torch.rand(1, 1, 28, 28)
        model = nn.Identity()
    #print(model)
    macs = profile_macs(model, inputs)
    #print(macs)
    return macs

def graphs_analyze(G):
    #config = "CONV2D"
    block_frequency = {}
    degree={}
    in_degree={}
    out_degree={}


    for graph in G:
        for node in graph.nodes():
            #import pdb;pdb.set_trace() 
            #if graph.nodes[node]['label'].startswith(config):
            block = graph.nodes[node]['label'].split('_')[0].replace("\n", "")+'_'+graph.nodes[node]['second_comment']+graph.nodes[node]['third_comment']
            graph.nodes[node]['label']=block
            if block in block_frequency:
                block_frequency[block][0] = block_frequency.get(block, 0)[0] + 1
                block_frequency[block][1] = block_frequency.get(block, 0)[1] + int(graph.nodes[node]['comment'].split(':')[1])
            else:
                block_frequency[block]=[1]
                block_frequency[block].append(int(graph.nodes[node]['comment'].split(':')[1]))
            #import pdb;pdb.set_trace() 

            if block not in degree:
                degree[block]=[graph.to_undirected().degree(node)]
                in_degree[block]=[graph.to_directed().in_degree(node)]
                out_degree[block]=[graph.to_directed().out_degree(node)]
            else:
                degree[block].append(graph.to_undirected().degree(node))
                in_degree[block].append(graph.to_directed().in_degree(node))
                out_degree[block].append(graph.to_directed().out_degree(node))

            
    
    #print("Inter-algorithm Frequency of Computational Blocks:", block_frequency)
    #print("Degree of Connectivity of each Node:", degree)

    sorted_dict = dict(sorted(block_frequency.items()))
    desired_width=30
    table_group = PrettyTable()
    table_group.field_names = ['Node Name', 'In_Channels','Out_Channels','Kernel','Stride','In Size','Out Size','Inter Node Frequency','Total Nodes','Degree of Connectivity ']
    
    for idx, (key, value) in enumerate(sorted_dict.items(), start=1):
        key_val= key.split('_')
        table_group.add_row([
           key_val[0],
           key_val[1],
           key_val[2],
           key_val[3] if len(key_val)>5 else '-' ,
           key_val[4] if len(key_val)>5 else '-',
           key_val[5] if len(key_val)>5 else key_val[3] ,
           key_val[6] if len(key_val)>5 else key_val[4],
           value[0],
           value[1],
           degree[key]
        ])
        
    return table_group
    #print(table_group)

def graphs_analyze_1(G):
    #config = "CONV2D"
    block_frequency = {}
    degree={}
    common_nodes={}
    high_frequency_nodes={}
    count=-1
    for idx, graph in enumerate(G):
        for node in graph.nodes():
            #if graph.nodes[node]['label'].startswith(config):
            #block = graph.nodes[node]['label'].split('_')[0].replace("\n", "")+'_'+graph.nodes[node]['second_comment']
            block = graph.nodes[node]['label'].split('_')[0].replace("\n", "")
            graph.nodes[node]['label']=block
            edge_weight=[]
            if block in block_frequency:
                block_frequency[block][0] = block_frequency.get(block, 0)[0] + 1
                block_frequency[block][1][0] = block_frequency.get(block, 0)[1][0] + int(graph.nodes[node]['weight'])
                block_frequency[block][1][5] = block_frequency.get(block, 0)[1][5] + float(graph.nodes[node]['fifth_comment'])
                block_frequency[block][2].append(idx+1)
                block_frequency[block][3].append(node)
                neighbors=list(graph.successors(node))
                edge_nodes=sorted(set(neighbors))
                block_frequency[block][4].append([graph.nodes[edge_node]['label'] for edge_node in edge_nodes])
                for edge_node in edge_nodes:
                    count=0
                    edge_data=graph.get_edge_data(node, edge_node)
                    if edge_data is not None:
                        #print(edge_data)
                        count+=edge_data['weight']
                    edge_weight.append(count)
                block_frequency[block][5].append(edge_weight)
                #block_frequency[block][5].append()
            else:
                block_frequency[block]=[1]
                block_frequency[block].append([int(graph.nodes[node]['weight']),float(graph.nodes[node]['comment'].split(':')[1]),float(graph.nodes[node]['second_comment'].split(':')[1]),float(graph.nodes[node]['third_comment'].split(':')[1]),float(graph.nodes[node]['fourth_comment'].split(':')[1]), float(graph.nodes[node]['fifth_comment']), float(graph.nodes[node]['sixth_comment'].split(':')[1]), float(graph.nodes[node]['seventh_comment'].split(':')[1]), float(graph.nodes[node]['eighth_comment'].split(':')[1]), float(graph.nodes[node]['ninth_comment'].split(':')[1]), float(graph.nodes[node]['ddr_latency'])])
                block_frequency[block].append([idx+1])
                block_frequency[block].append([node])
                neighbors=list(graph.successors(node))
                edge_nodes=sorted(set(neighbors))
                block_frequency[block].append([[graph.nodes[edge_node]['label'] for edge_node in edge_nodes]])
                for edge_node in edge_nodes:
                    count=0
                    edge_data=graph.get_edge_data(node, edge_node)
                    if edge_data is not None:
                        #print(edge_data)
                        count+=edge_data['weight']
                    edge_weight.append(count)
                block_frequency[block].append([edge_weight])
            #import pdb;pdb.set_trace() 

            if block not in degree:
                degree[block]=[graph.to_undirected().degree(node)]
            else:
                degree[block].append(graph.to_undirected().degree(node))               
    
    for block in block_frequency:
        block_frequency[block].append(degree[block])
        if len(set(block_frequency[block][2]))>1:
            common_nodes[block]=block_frequency[block]
        elif block_frequency[block][1][0]>5:
            high_frequency_nodes[block]=block_frequency[block]
    #import pdb;pdb.set_trace() 
    #print("Inter-algorithm Frequency of Computational Blocks:", block_frequency)
    #print("Degree of Connectivity of each Node:", degree)

    sorted_dict = dict(sorted(block_frequency.items()))
    desired_width=30
    table_group = PrettyTable()
    #table_group.field_names = ['Node Name','In_Channels','Out_Channels','Kernel','Stride','Inter Node Frequency','Total Nodes','Degree of Connectivity ']
    table_group.field_names = ['Node Name','Inter Node Frequency','Total Nodes','Degree of Connectivity ',  'Latency of a single operation', 'Energy of a single operation','Area']
    
    for idx, (key, value) in enumerate(sorted_dict.items(), start=1):
        key_val= key.split('_')
        table_group.add_row([
           key_val[0],
           value[0],
           value[1][0],
           degree[key],
           value[1][1],
           value[1][2],
           value[1][3],
        ])
        """
        table_group.add_row([
           key_val[0],
           key_val[1],
           key_val[2],
           key_val[3] if len(key_val)>3 else '-' ,
           key_val[4] if len(key_val)>4 else '-',
           value[0],
           value[1],
           degree[key]
        ])
        """
    #import pdb;pdb.set_trace() 
    
    return table_group, common_nodes, high_frequency_nodes, block_frequency
    #print(table_group)
def edges_analyze(block_frequency):
    edge_frequency={}
    node_frequency={}
    for node in block_frequency:
        edge_nodes=list(itertools.chain.from_iterable(block_frequency[node][4]))
        edge_weights=list(itertools.chain.from_iterable(block_frequency[node][5]))
        for idx, edge_node in enumerate(edge_nodes):
            edge={node, edge_node}
            #print(edge)
            if frozenset(edge) not in edge_frequency:
                #edge_frequency[frozenset(edge)] = [1]
                edge_frequency[frozenset(edge)] = [edge_weights[idx]]
            else:
                #edge_frequency[frozenset(edge)][0]+= 1
                edge_frequency[frozenset(edge)][0]+= edge_weights[idx]
            
            if node not in node_frequency:
                node_frequency[node]=[block_frequency[node][1][0], block_frequency[node][1][1],block_frequency[node][1][2],block_frequency[node][1][3],block_frequency[node][1][4], block_frequency[node][1][5], block_frequency[node][1][6], block_frequency[node][1][7], block_frequency[node][1][8], block_frequency[node][1][9], block_frequency[node][1][10]]
                
            if edge_node not in node_frequency:
                node_frequency[edge_node]=[block_frequency[edge_node][1][0], block_frequency[edge_node][1][1],block_frequency[edge_node][1][2],block_frequency[edge_node][1][3], block_frequency[edge_node][1][4], block_frequency[edge_node][1][5], block_frequency[edge_node][1][6],block_frequency[edge_node][1][7],  block_frequency[edge_node][1][8], block_frequency[edge_node][1][9], block_frequency[edge_node][1][10]]

        #import pdb;pdb.set_trace() 
    
    desired_width=30
    table_group = PrettyTable()
    table_group.field_names = ['Edge Node 1','Edge Node 2','Total Edges']
    
    for idx, (key, value) in enumerate(edge_frequency.items(), start=1):
        key_list=list(set(key))
        table_group.add_row([
           key_list[0],
           key_list[1] if len(key_list)>1 else key_list[0],
           value[0],
        ])
    #import pdb;pdb.set_trace() 

    return table_group, edge_frequency, node_frequency

def cluster_generate(G, edge_frequency, node_frequency, directory_name, model_list, cluster_opti):
    start_time = time.time()
    bar_graph_image_name = directory_name+"edge_stats.png"
    edge_weight_stats(edge_frequency, bar_graph_image_name)
    bar_graph_image_name = directory_name+"node_stats.png"
    node_weight_stats(node_frequency, bar_graph_image_name)
    universal_cluster_graph_image_name = directory_name+"universal_graph.png"
    unique_nodes, unique_edges,universal_cluster_G = universal_cluster_graph(edge_frequency, node_frequency, universal_cluster_graph_image_name)
    #import pdb;pdb.set_trace() 
    #print("Universal cluster graph generation time taken:", time.time() - start_time)
    
    #print(len(unique_nodes), len(unique_edges))
    table_cluster_coverage,_,_,_,_=cluster_analyze(G, unique_nodes, unique_edges, model_list)
    cluster_coverage_file_name = directory_name+"universal_cluster_coverage.txt"
    print_file(cluster_coverage_file_name, table_cluster_coverage)
    #print("Cluster generation time taken:", time.time() - start_time)
    if cluster_opti:
        cluster_graph_G, cluster_nodes, cluster_edges= cluster(universal_cluster_G,directory_name)
        table_cluster_coverage,_,_, _,_=cluster_analyze(G, cluster_nodes, cluster_edges, model_list)
        cluster_coverage_file_name = directory_name+"cluster_coverage.txt"
        print_file(cluster_coverage_file_name, table_cluster_coverage)
        return cluster_graph_G,table_cluster_coverage
    else:
        return unique_nodes, unique_edges,universal_cluster_G,table_cluster_coverage

def cluster_old(G, directory_name):
    method = 2 # 1-girvan_newman, 2- lovain method, 3- Spectral Clustering, 4- label propagation, 5 - Leiden community detection
    #6-metis, #7-KL
    color_map = {}
    colors = ['orange', 'skyblue', 'lightgreen', 'red', 'purple', 'yellow']
    min_area=0.2e-6
    #import pdb;pdb.set_trace()
    match method:
        case 1:
            comp = girvan_newman(G)
            clusters = next(comp) 
            for i, cluster in enumerate(clusters):
                for node in cluster:
                    color_map[node] = colors[i % len(colors)]
            #import pdb;pdb.set_trace() 
            
            subgraphs = [G.subgraph(cluster) for cluster in clusters]
            subgraph_colors=[color_map[node] for node in cluster for cluster in clusters]
        case 2:
            clusters = {}
            #import pdb;pdb.set_trace()
            partition = community_louvain.best_partition(G, weight='dummy_key', resolution =2.1, random_state=42)
            for node, community in partition.items():
                color_map[node] = colors[community % len(colors)]
                if community not in clusters:
                    clusters[community] = []
                clusters[community].append(node)
                 # Compute the area of each cluster
            cluster_areas = {
                c: sum(float(G.nodes[node]["third_comment"].split(':')[1]) for node in nodes)
                for c, nodes in clusters.items()
            }
            #import pdb;pdb.set_trace() 

            # Identify clusters below the minimum area
            small_clusters = {c: clusters[c] for c, area in cluster_areas.items() if area < min_area}

            # Merge small clusters into the cluster they are most connected to
            for c, nodes in small_clusters.items():
                # Calculate total edge weight to all other clusters
                connectivity = {}
                for node in nodes:
                    for neighbor in G.neighbors(node):
                        neighbor_community = partition[neighbor]
                        if neighbor_community != c:  # Ignore connections within the same cluster
                            if neighbor_community not in connectivity:
                                connectivity[neighbor_community] = 0
                            connectivity[neighbor_community] += G[node][neighbor].get('weight', 1.0)  # Use edge weight or default to 1.0

                # Find the cluster with the highest connectivity
                if connectivity:
                    best_cluster = max(connectivity, key=connectivity.get)
                    
                    # Merge the small cluster into the best-connected cluster
                    clusters[best_cluster].extend(nodes)
                    cluster_areas[best_cluster] += cluster_areas[c]
                    
                    # Remove the small cluster
                    del clusters[c]
                    del cluster_areas[c]
                    
                    # Update partition mapping
                    for node in nodes:
                        partition[node] = best_cluster
            
            """
            #Merge all smaller clusters to second cluster such that total number of clusters is reduced to 2. The largest cluster is kept as it is.
            largest_cluster= max(cluster_areas, key=cluster_areas.get)
            second_cluster_sel=False
            for c, nodes in list(clusters.items()):
                if c != largest_cluster:
                    if not second_cluster_sel:
                        second_cluster = c
                        second_cluster_sel = True
                    else:
                        clusters[second_cluster].extend(nodes) 
                        cluster_areas[second_cluster] += cluster_areas[c]
                        del clusters[c]
                        del cluster_areas[c]
                        for node in nodes:
                            partition[node] = second_cluster
            #import pdb;pdb.set_trace()
            """

            for node, community in partition.items():
                color_map[node] = colors[community % len(colors)]
            subgraphs = [G.subgraph(cluster) for cluster in clusters.values()]
            subgraph_colors=[color_map[cluster[0]] for cluster in clusters.values()]

        case 3: 
            clusters = {}
            adj_matrix = nx.to_numpy_array(G)
            sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
            labels = sc.fit_predict(adj_matrix)
            for node, label in zip(G.nodes(), labels):
                color_map[node] = colors[label % len(colors)]
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(node)

            subgraphs = [G.subgraph(cluster) for cluster in clusters.values()]
            subgraph_colors=[color_map[cluster[0]] for cluster in clusters.values()]


        case 4:
            communities = label_propagation_communities(G)
            for i, community in enumerate(communities):
                for node in community:
                    color_map[node] = colors[i % len(colors)]
            #import pdb;pdb.set_trace() 
            subgraphs = [G.subgraph(community) for community in communities]
            subgraph_colors=[color_map[node] for node in community for community in communities]
        case 5:
            clusters = {}
            partition_leiden = find_partition(G, ModularityVertexPartition)
            for node, community in partition.items():
                color_map[node] = colors[community % len(colors)]
                if community not in clusters:
                    clusters[community] = []
                clusters[community].append(node)
            subgraphs = [G.subgraph(cluster) for cluster in clusters.values()]
            subgraph_colors=[color_map[node] for cluster in clusters.values()]
        
        case 6:
            num_parts = 2  # Change as needed
            edgecuts, parts = metis.part_graph(G, num_parts)

            clusters = {i: [] for i in range(num_parts)}
            for node, part in zip(G.nodes(), parts):
                clusters[part].append(node)

            for i, cluster in clusters.items():
                for node in cluster:
                    color_map[node] = colors[i % len(colors)]

            subgraphs = [G.subgraph(cluster) for cluster in clusters.values()]
            subgraph_colors = [color_map[cluster[0]] for cluster in clusters.values()]
        case 7:
            num_parts = 2  # Typically, FM does bi-partitioning
            cluster_1, cluster_2 = kernighan_lin_bisection(G, weight="weight")
            
            clusters = {0: cluster_1, 1: cluster_2}
            
            for i, cluster in clusters.items():
                for node in cluster:
                    color_map[node] = colors[i % len(colors)]
            
            subgraphs = [G.subgraph(cluster) for cluster in clusters.values()]
            subgraph_colors = [color_map[list(cluster)[0]] for cluster in clusters.values()]
            
    pos = nx.spring_layout(G)
    node_colors = [color_map[node] for node in G.nodes()]
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    
    if plot_process_stats:
        fig_size=15
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1400, edgelist=edges, edge_color=weights,font_size=10, font_weight='bold', width=2)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')
        for node in G.nodes():
            if G.has_edge(node, node):
                x, y = pos[node]
                plt.text(x +0.002*fig_size, y+0.005*fig_size, str(edge_labels[(node, node)]), fontsize=9, color='red', verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.1', edgecolor='none', facecolor='white', alpha=0.7))
        """
        for node, (x, y) in pos.items():
            comment = G.nodes[node]['comment']
            plt.text(x + 0.0003*fig_size, y-0.0003*fig_size, comment, fontsize=9, verticalalignment='center',weight='bold', bbox=dict(boxstyle='round,pad=0.01', edgecolor='none', facecolor='white', alpha=0.7))
            second_comment = G.nodes[node]['second_comment']
            plt.text(x + 0.005, y-0.002, second_comment, fontsize=9, verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='white', alpha=0.7))
        """
        plt.title('Graph Clusters based on Edge Weights', fontsize=16, fontweight='bold')
        plt.savefig(directory_name+'cluster_graph.png')
        plt.show()
        plt.close()
        plt.clf()
    
    for i, subgraph in enumerate(subgraphs):
        edges_with_weights  = nx.get_edge_attributes(subgraph,'weight')
        if edges_with_weights:
            edges, weights = zip(*edges_with_weights.items())
        else:
            edges, weights = [], []
        #import pdb;pdb.set_trace() 
        #print(i, subgraph.nodes(), sum([float(subgraph.nodes[node]["third_comment"].split(':')[1]) for node in subgraph.nodes()])*1e+6)
        pos = nx.spring_layout(subgraph)
        num_nodes=len(subgraph)

        if plot_process_stats:
            fig_size=max(6, num_nodes)
            plt.figure(figsize=(fig_size, fig_size))
            nx.draw(subgraph, pos, with_labels=True, node_color=subgraph_colors[i],  edgelist=edges, edge_color=weights, node_size=1400, font_size=8, font_weight='bold')
            edge_labels_sub = nx.get_edge_attributes(subgraph, 'weight')
            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels_sub, font_color='blue')
            for node in subgraph.nodes():
                if subgraph.has_edge(node, node):
                    x, y = pos[node]
                    plt.text(x +0.0025*fig_size, y+0.01*fig_size, str(edge_labels_sub[(node, node)]), fontsize=9, color='red', verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.1', edgecolor='none', facecolor='white', alpha=0.7))
            """
            for node, (x, y) in pos.items():
                comment = G.nodes[node]['comment']
                plt.text(x + 0.0003*fig_size, y-0.0003*fig_size, comment, fontsize=9, verticalalignment='center',weight='bold', bbox=dict(boxstyle='round,pad=0.01', edgecolor='none', facecolor='white', alpha=0.7))
                second_comment = G.nodes[node]['second_comment']
                plt.text(x + 0.005, y-0.002, second_comment, fontsize=9, verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='white', alpha=0.7))
            """
            plt.title(f'Cluster {i+1} using Louvain Method')
            plt.savefig(directory_name+f'cluster_subgraph_{i}.png')
            plt.show()
            plt.close()


    inter_subgraph_weights = {}
    """
    #print(partition)
    for u, v, data in G.edges(data=True):
        #import pdb;pdb.set_trace() 
        u_cluster = partition[u]
        v_cluster = partition[v]
        if u_cluster != v_cluster:
            if u_cluster > v_cluster:
                u_cluster, v_cluster = v_cluster, u_cluster
            set_clus=(u_cluster, v_cluster) 
            if set_clus not in inter_subgraph_weights:
                inter_subgraph_weights[set_clus] = 0
            inter_subgraph_weights[set_clus] += data['weight']
    """

    #for (cluster1, cluster2), weight_sum in inter_subgraph_weights.items():
        #print(f"Sum of edge weights between Cluster {cluster1 + 1} and Cluster {cluster2 + 1}: {weight_sum}")
    for i, g in enumerate(subgraphs):
        nx.set_node_attributes(
            g,
            {n: i for n in g.nodes()},
            name="subgraph_id"
        )

    G_compose = nx.compose_all(subgraphs)
    pos = nx.spring_layout(G_compose)
    node_colors = [color_map[node] for node in G_compose.nodes()]
    edges,weights = zip(*nx.get_edge_attributes(G_compose,'weight').items())
    edge_labels_compose = nx.get_edge_attributes(G_compose, 'weight')
    
    if plot_process_stats:
        fig_size=15
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        nx.draw(G_compose, pos, with_labels=True, node_color=node_colors, node_size=1400, edgelist=edges, edge_color=weights,font_size=10, font_weight='bold', width=2)

        nx.draw_networkx_edge_labels(G_compose, pos, edge_labels=edge_labels_compose, font_color='blue')
        for node in G_compose.nodes():
            if G_compose.has_edge(node, node):
                x, y = pos[node]
                plt.text(x +0.002*fig_size, y+0.005*fig_size, str(edge_labels_compose[(node, node)]), fontsize=9, color='red', verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.1', edgecolor='none', facecolor='white', alpha=0.7))
        """
        for node, (x, y) in pos.items():
            comment = G.nodes[node]['comment']
            plt.text(x + 0.0003*fig_size, y-0.0003*fig_size, comment, fontsize=9, verticalalignment='center',weight='bold', bbox=dict(boxstyle='round,pad=0.01', edgecolor='none', facecolor='white', alpha=0.7))
            second_comment = G.nodes[node]['second_comment']
            plt.text(x + 0.005, y-0.002, second_comment, fontsize=9, verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='white', alpha=0.7))
        """
        plt.title('Combined Graph with Three Independent Subgraphs')
        plt.savefig(directory_name+f'cluster_subgraphs.png')
        plt.show()
        plt.close()
    edge_data = {frozenset(key): value for key, value in edge_labels_compose.items()}
    #import pdb;pdb.set_trace() 
    return G_compose, set(G_compose.nodes()),edge_data


def cluster(G, directory_name, model_name_list):
    """
    Runs ALL clustering methods (1-7), produces a comparison plot, then returns
    the Louvain (method 2) result.  Inputs/outputs are unchanged.
    """

    colors = ['orange', 'skyblue', 'lightgreen', 'red', 'purple', 'yellow']
    min_area = 0.2e-6

    # ------------------------------------------------------------------
    # USER CHOICE: edge attribute used for clustering across ALL methods.
    # Must be a numeric edge attribute present in G.
    # Common options: 'weight', 'weightforclustering'
    # ------------------------------------------------------------------
    clustering_weight_attr = 'None'


    #print(f"[cluster] Using edge attribute '{clustering_weight_attr}' for all clustering methods."
    #f"({'unweighted' if clustering_weight_attr is None else 'weighted'}).")
    # ------------------------------------------------------------------
    # Unified runner: returns (clusters_dict, partition_dict, label_str)
    # clusters_dict: {cluster_id: [node, ...]}
    # partition_dict: {node: cluster_id}
    # Returns (None, None, label) if the method is unavailable or fails.
    # All methods use clustering_weight_attr for edge weights.
    # ------------------------------------------------------------------
    def _run_case(method_id, G):
        try:
            if method_id == 1:
                # Girvan-Newman uses most-valuable-edge; weight the betweenness
                # by clustering_weight_attr by building a view with 'weight' remapped.
                def most_valuable_edge(G):
                    betweenness = nx.edge_betweenness_centrality(
                        G, weight=clustering_weight_attr)
                    return max(betweenness, key=betweenness.get)
                comp = girvan_newman(G, most_valuable_edge=most_valuable_edge)
                raw_clusters = next(comp)
                clusters = {i: list(c) for i, c in enumerate(raw_clusters)}
                partition = {node: i for i, nodes in clusters.items() for node in nodes}
                return clusters, partition, "Girvan-Newman"

            elif method_id == 2:
                if clustering_weight_attr is None:
                    # Louvain requires a string weight kwarg; uniform weights = unweighted
                    nx.set_edge_attributes(G, 1.0, name='_unweighted')
                    louvain_weight = '_unweighted'
                else:
                    louvain_weight = clustering_weight_attr
                partition = community_louvain.best_partition(
                    G, weight=louvain_weight, resolution=2.1, random_state=42)
                clusters = {}
                for node, community in partition.items():
                    clusters.setdefault(community, []).append(node)
 
                # small-cluster merging (preserved exactly from original)
                cluster_areas = {
                    c: sum(float(G.nodes[node]["third_comment"].split(':')[1]) for node in nodes)
                    for c, nodes in clusters.items()
                }
                small_clusters = {c: clusters[c] for c, area in cluster_areas.items() if area < min_area}
                for c, nodes in small_clusters.items():
                    connectivity = {}
                    for node in nodes:
                        for neighbor in G.neighbors(node):
                            nc = partition[neighbor]
                            if nc != c:
                                connectivity[nc] = connectivity.get(nc, 0) + \
                                    G[node][neighbor].get(clustering_weight_attr, 1.0)
                    if connectivity:
                        best_cluster = max(connectivity, key=connectivity.get)
                        clusters[best_cluster].extend(nodes)
                        cluster_areas[best_cluster] += cluster_areas[c]
                        del clusters[c], cluster_areas[c]
                        for node in nodes:
                            partition[node] = best_cluster
 
                # rebuild partition after merging
                partition = {node: c for c, nodes in clusters.items() for node in nodes}
                return clusters, partition, "Louvain"

            elif method_id == 3:
                adj_matrix = nx.to_numpy_array(G, weight=clustering_weight_attr)
                sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=42)
                labels = sc.fit_predict(adj_matrix)
                clusters = {}
                partition = {}
                for node, label in zip(G.nodes(), labels):
                    partition[node] = int(label)
                    clusters.setdefault(int(label), []).append(node)
                return clusters, partition, "Spectral"

            elif method_id == 4:
                communities = list(label_propagation_communities(G))
                # label_propagation_communities does not accept a weight argument;
                # for weighted clustering, consider async_lpa_communities instead.
                clusters = {i: list(c) for i, c in enumerate(communities)}
                partition = {node: i for i, nodes in clusters.items() for node in nodes}
                return clusters, partition, "Label Prop."

            elif method_id == 5:
                if not _has_leiden:
                    return None, None, "Leiden (N/A)"
                import igraph as ig
                mapping = {n: i for i, n in enumerate(G.nodes())}
                edge_weights = [
                    G[u][v].get(clustering_weight_attr, 1.0) for u, v in G.edges()]
                ig_G = ig.Graph(
                    n=G.number_of_nodes(),
                    edges=[(mapping[u], mapping[v]) for u, v in G.edges()])
                ig_G.es['weight'] = edge_weights
                leiden_part = find_partition(
                    ig_G, ModularityVertexPartition, weights='weight')
                clusters = {}
                partition = {}
                node_list_ordered = list(G.nodes())
                for cid, members in enumerate(leiden_part):
                    node_names = [node_list_ordered[m] for m in members]
                    clusters[cid] = node_names
                    for node in node_names:
                        partition[node] = cid
                return clusters, partition, "Leiden"

            elif method_id == 6:
                if not _has_metis:
                    return None, None, "METIS (N/A)"
                # METIS reads 'weight' from the graph; copy chosen attr to 'weight'
                G_metis = G.copy()
                for u, v, data in G_metis.edges(data=True):
                    data['weight'] = data.get(clustering_weight_attr, 1.0)
                edgecuts, parts = metis.part_graph(G_metis, 2)
                clusters = {}
                partition = {}
                for node, part in zip(G.nodes(), parts):
                    partition[node] = part
                    clusters.setdefault(part, []).append(node)
                return clusters, partition, "METIS"

            elif method_id == 7:
                from networkx.algorithms.community import kernighan_lin_bisection
                c1, c2 = kernighan_lin_bisection(G, weight=clustering_weight_attr)
                clusters = {0: list(c1), 1: list(c2)}
                partition = {node: 0 for node in c1}
                partition.update({node: 1 for node in c2})
                return clusters, partition, "Kernighan-Lin"

        except Exception as e:
            #print(f"[cluster] method {method_id} failed: {e}")
            return None, None, f"Method {method_id} (err)"

        return None, None, f"Method {method_id}"

    # ------------------------------------------------------------------
    # Run all methods
    # ------------------------------------------------------------------
    all_results = {}
    for mid in range(2, 3):
        c, p, lbl = _run_case(mid, G)
        all_results[mid] = (c, p, lbl)
        n_c = len(c) if c is not None else "N/A"
        #print(f"[cluster] method {mid} ({lbl}): {n_c} clusters")

    # ------------------------------------------------------------------
    # Comparison plot: one subplot per method, fixed layout for fairness
    # ------------------------------------------------------------------
    pos_fixed = nx.spring_layout(G, seed=42)
    ncols = 4
    nrows = 2          # 7 methods -> 4+3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes_flat = axes.flatten()

    for idx, mid in enumerate(range(2, 3)):
        ax = axes_flat[idx]
        clusters_m, partition_m, label_m = all_results[mid]

        if partition_m is None:
            ax.set_title(label_m, fontsize=10)
            ax.axis('off')
            continue

        node_colors_m = [colors[partition_m[n] % len(colors)] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos_fixed, node_color=node_colors_m,
                               node_size=300, ax=ax)
        nx.draw_networkx_labels(G, pos_fixed, font_size=6, ax=ax)
        edge_attrs = nx.get_edge_attributes(G, clustering_weight_attr)
        if edge_attrs:
            edges_list, wts = zip(*edge_attrs.items())
            nx.draw_networkx_edges(G, pos_fixed, edgelist=edges_list,
                                   edge_color=wts, edge_cmap=plt.cm.Blues,
                                   ax=ax, width=1.2)
        else:
            nx.draw_networkx_edges(G, pos_fixed, ax=ax)

        n_clusters = len(clusters_m)
        ax.set_title(f"{label_m}  ({n_clusters} clusters)", fontsize=10, fontweight='bold')
        ax.axis('off')

    # hide unused subplot (8th cell)
    for idx in range(7, len(axes_flat)):
        axes_flat[idx].axis('off')

        fig.suptitle(f'Clustering Method Comparison - {model_name_list}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    models_str = ",".join(model_name_list)
    short_hash = hashlib.md5(models_str.encode()).hexdigest()[:8]

    plt.savefig(directory_name + f'cluster_method_comparison_{short_hash}.png', bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    plt.clf()

    #plot area of each clusters for each method in separate bar graphs
    area_data = {}
    for mid in range(2, 3):
        clusters_m, partition_m, label_m = all_results[mid]
        if clusters_m is None:
            continue
        area_data[label_m] = []
        for c_id, nodes in clusters_m.items():
            area = sum(float(G.nodes[node]["third_comment"].split(':')[1]) for node in nodes)
            area_data[label_m].append((c_id, area))
    # Plotting
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes_flat = axes.flatten()
    for idx, mid in enumerate(range(2, 3)):
        ax = axes_flat[idx]
        label_m = all_results[mid][2]
        if label_m not in area_data:
            ax.set_title(label_m, fontsize=10)
            ax.axis('off')
            continue
        c_ids, areas = zip(*area_data[label_m])
        ax.bar(c_ids, areas, color='skyblue')
        ax.set_title(f"{label_m} Cluster Areas", fontsize=10, fontweight='bold')
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Total Area") 
        ax.set_xticks(c_ids)
        ax.set_xticklabels([str(cid) for cid in c_ids], rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    # hide unused subplot (8th cell)
    for idx in range(7, len(axes_flat)):
        axes_flat[idx].axis('off')
    plt.suptitle(f'Cluster Area Comparison - {model_name_list}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(directory_name + f'cluster_area_comparison_{short_hash}.png', bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    
      # ------------------------------------------------------------------
    # M metric bar chart: area-normalized inter-cluster / total edge weight
    # M = (sum_inter w_ij / sum_inter (a_i+a_j)) / (sum_all w_ij / sum_all (a_i+a_j))
    # ------------------------------------------------------------------
    def _node_area(node):
        try:
            return float(G.nodes[node]["third_comment"].split(':')[1])
        except (KeyError, IndexError, ValueError):
            return 1.0
 
    metric_M = {}
    for mid in range(2, 3):
        clusters_m, partition_m, label_m = all_results[mid]
        if partition_m is None:
            continue
 
        #inter_w = 0.0;  inter_a = 0.0
        #total_w = 0.0;  total_a = 0.0
        inter_norm = 0.0; total_norm = 0.0
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1.0)
            a = _node_area(u) + _node_area(v)
            #total_w += w
            #total_a += a
            total_norm += w / a if a > 0 else 0.0
            if partition_m[u] != partition_m[v]:
                #inter_w += w
                #inter_a += a
                inter_norm += w / a if a > 0 else 0.0
 
        #inter_norm = inter_w / inter_a if inter_a > 0 else 0.0
        #total_norm = total_w / total_a if total_a > 0 else 0.0
        metric_M[label_m] = inter_norm / total_norm if total_norm > 0 else 0.0
 
    labels_M = list(metric_M.keys())
    values_M = list(metric_M.values())
 
    fig, ax = plt.subplots(figsize=(max(8, len(labels_M) * 1.4), 6))
    bars = ax.bar(labels_M, values_M, color='salmon', edgecolor='black', linewidth=0.8)
    for bar, val in zip(bars, values_M):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                f"{val:.7f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(f'Normalized Inter-Cluster Communication Metric - {model_name_list}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Clustering Method')
    ax.set_ylabel(r'$\mathcal{M}$ (lower is better)')
    ax.set_xticks(range(len(labels_M)))
    ax.set_xticklabels(labels_M, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(directory_name + f'cluster_metric_M_comparison_{short_hash}.png', bbox_inches='tight', dpi=150)
    plt.show()
    plt.close()
    plt.clf()

    # ------------------------------------------------------------------
    # From here: use Louvain (method 2) exactly as the original case 2
    # ------------------------------------------------------------------
    clusters, partition, _ = all_results[2]
    color_map = {node: colors[partition[node] % len(colors)] for node in G.nodes()}

    subgraphs = [G.subgraph(nodes) for nodes in clusters.values()]
    subgraph_colors = [color_map[nodes[0]] for nodes in clusters.values()]

    # ---- full-graph plot ----
    display_weight_attr = 'weight'
    pos = nx.spring_layout(G)
    node_colors = [color_map[node] for node in G.nodes()]
    edges, weights = zip(*nx.get_edge_attributes(G, display_weight_attr).items())

    if plot_process_stats:
        fig_size = 15
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1400,
                edgelist=edges, edge_color=weights, font_size=10, font_weight='bold', width=2)
        edge_labels = nx.get_edge_attributes(G, display_weight_attr)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')
        for node in G.nodes():
            if G.has_edge(node, node):
                x, y = pos[node]
                plt.text(x + 0.002 * fig_size, y + 0.005 * fig_size,
                         str(edge_labels[(node, node)]), fontsize=9, color='red',
                         verticalalignment='center', weight='bold',
                         bbox=dict(boxstyle='round,pad=0.1', edgecolor='none',
                                   facecolor='white', alpha=0.7))
        plt.title('Graph Clusters based on Edge Weights', fontsize=16, fontweight='bold')
        plt.savefig(directory_name + 'cluster_graph.png')
        plt.show()
        plt.close()
        plt.clf()

    # ---- per-subgraph plots ----
    for i, subgraph in enumerate(subgraphs):
        edges_with_weights = nx.get_edge_attributes(subgraph, display_weight_attr)
        if edges_with_weights:
            edges, weights = zip(*edges_with_weights.items())
        else:
            edges, weights = [], []
        pos = nx.spring_layout(subgraph)
        num_nodes = len(subgraph)

        if plot_process_stats:
            fig_size = max(6, num_nodes)
            plt.figure(figsize=(fig_size, fig_size))
            nx.draw(subgraph, pos, with_labels=True, node_color=subgraph_colors[i],
                    edgelist=edges, edge_color=weights, node_size=1400,
                    font_size=8, font_weight='bold')
            edge_labels_sub = nx.get_edge_attributes(subgraph, display_weight_attr)
            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels_sub, font_color='blue')
            for node in subgraph.nodes():
                if subgraph.has_edge(node, node):
                    x, y = pos[node]
                    plt.text(x + 0.0025 * fig_size, y + 0.01 * fig_size,
                             str(edge_labels_sub[(node, node)]), fontsize=9, color='red',
                             verticalalignment='center', weight='bold',
                             bbox=dict(boxstyle='round,pad=0.1', edgecolor='none',
                                       facecolor='white', alpha=0.7))
            plt.title(f'Cluster {i + 1} using Louvain Method')
            plt.savefig(directory_name + f'cluster_subgraph_{i}.png')
            plt.show()
            plt.close()

    # ---- composed graph ----
    for i, g in enumerate(subgraphs):
        nx.set_node_attributes(g, {n: i for n in g.nodes()}, name="subgraph_id")

    G_compose = nx.compose_all(subgraphs)
    pos = nx.spring_layout(G_compose)
    node_colors = [color_map[node] for node in G_compose.nodes()]
    edges, weights = zip(*nx.get_edge_attributes(G_compose, display_weight_attr).items())
    edge_labels_compose = nx.get_edge_attributes(G_compose, display_weight_attr)

    if plot_process_stats:
        fig_size = 15
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        nx.draw(G_compose, pos, with_labels=True, node_color=node_colors, node_size=1400,
                edgelist=edges, edge_color=weights, font_size=10, font_weight='bold', width=2)
        nx.draw_networkx_edge_labels(G_compose, pos, edge_labels=edge_labels_compose, font_color='blue')
        for node in G_compose.nodes():
            if G_compose.has_edge(node, node):
                x, y = pos[node]
                plt.text(x + 0.002 * fig_size, y + 0.005 * fig_size,
                         str(edge_labels_compose[(node, node)]), fontsize=9, color='red',
                         verticalalignment='center', weight='bold',
                         bbox=dict(boxstyle='round,pad=0.1', edgecolor='none',
                                   facecolor='white', alpha=0.7))
        plt.title('Combined Graph with Three Independent Subgraphs')
        plt.savefig(directory_name + f'cluster_subgraphs.png')
        plt.show()
        plt.close()

    edge_data = {frozenset(key): value for key, value in edge_labels_compose.items()}
    return G_compose, set(G_compose.nodes()), edge_data

def edge_weight_stats(edge_frequency, bar_graph_image_name):
    plt.close()
    plot_graph=False
    sorted_data = dict(sorted(edge_frequency.items(), key=lambda item: item[1], reverse=True))
    categories=['\n'.join(map(str, key)) if len(key)>1 else ''.join(map(str, key)) +'\n loop' for key in sorted_data.keys()]
    values=list(sorted_data.values())
    print_edge_stats=False
    if print_edge_stats:
        print("+------------------+-------+")
        print("|   Combination    | Value |")
        print("+------------------+-------+")

        # Add rows
        for combo, value in sorted_data.items():
            combination_str = ", ".join(combo)
            print(f"| {combination_str:<16} | {value[0]:<5} |")

        # Print table footer
        print("+------------------+-------+")
    #print(sorted_data)
    #import pdb;pdb.set_trace() 
    if not plot_graph:
        return
    figsize=6
    if plot_process_stats:
        plt.figure(figsize=(figsize*3, figsize))
        num_val=10
        fig, ax = plt.subplots(figsize=(figsize*3, figsize))
        bars = ax.bar(categories[:num_val], [value[0] for value in values[:num_val]], color=['skyblue', 'orange', 'green', 'red'])
        ax.set_title('Edge Weights Stats', fontsize=16, fontweight='bold')
        ax.set_xlabel('Edge Combination', fontsize=14)
        ax.set_ylabel('Edge Weights', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, yval, ha='center', va='bottom', fontsize=12)

        # Display the plot
        plt.savefig(bar_graph_image_name)
        plt.show()
        plt.close()


def node_weight_stats(node_frequency, bar_graph_image_name):
    plt.close()
    plot_graph=False
    sorted_data = dict(sorted(node_frequency.items(), key=lambda item: item[1], reverse=True))
    #categories=['\n'.join(map(str, key)) if len(key)>1 else ''.join(map(str, key)) +'\n loop' for key in sorted_data.keys()]
    categories=[key for key in sorted_data.keys()]
    values=list(sorted_data.values())
    print_node_stats=False
    if print_node_stats:
        print("+------------------+-------+")
        print("|   Combination    | Value |")
        print("+------------------+-------+")

        # Add rows
        for combo, value in sorted_data.items():
            combination_str=combo
            #combination_str = ", ".join(combo)
            print(f"| {combination_str:<16} | {value[0]:<5} |")

        # Print table footer
        print("+------------------+-------+")
    #print(sorted_data)
    #import pdb;pdb.set_trace() 
    if not plot_graph:
        return
    if plot_process_stats:
        figsize=6
        plt.figure(figsize=(figsize*3, figsize))
        num_val=10
        fig, ax = plt.subplots(figsize=(figsize*3, figsize))
        bars = ax.bar(categories[:num_val], [value[0] for value in values[:num_val]], color=['skyblue', 'orange', 'green', 'red'])

        ax.set_title('Node Weights/Frequency Stats', fontsize=16, fontweight='bold')
        ax.set_xlabel('Node Combination', fontsize=14)
        ax.set_ylabel('Node Weights/Freuqency', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, yval, ha='center', va='bottom', fontsize=12)

        # Display the plot
        plt.savefig(bar_graph_image_name)
        plt.show()
        plt.close()

    
def cluster_analyze(G, cluster_nodes, cluster_edges, model_list):
    coverage=[]
    coverage_edge=[]
    node_utli_cluster=[]
    edge_util_cluster=[]
    for idx, graph in enumerate(G):
        count_nodes=0
        total_count_nodes=0
        count_edges=0
        total_count_edges=0
        nodes_in_cluster=0
        checked=set()
        nodes_in_cluster=set()
        edges_in_cluster=set()
        #import pdb;pdb.set_trace() 
        edge_weights=nx.get_edge_attributes(graph, 'weight')
        #rename graph nodes if they end with '1' or '2' to restore to original names
        for node in graph.nodes():
            if graph.nodes[node]['label'] in cluster_nodes:
                nodes_in_cluster.add(node)
                count_nodes+=int(graph.nodes[node]['weight'])
            total_count_nodes+=int(graph.nodes[node]['weight'])
            for item in list(graph.successors(node)):
                #import pdb;pdb.set_trace() 
                node1=graph.nodes[node]['label']
                node2=graph.nodes[item]['label']
                if (node1,node2) not in checked and (node2, node1) not in checked:
                    edge=frozenset({node1,node2})
                    #print(idx, node, item, edge_weights, cluster_edges, node1, node2, edge)
                    if edge in cluster_edges:
                        edges_in_cluster.add(edge)
                        count_edges+=edge_weights[(node,item)]
                        if (item, node) in edge_weights and item!=node:
                            count_edges+=edge_weights[(item, node)]
                    checked.add((node1,node2))
                total_count_edges+=edge_weights[(node,item)]
                #print(edge, checked, count_edges, total_count_edges)
        #print(count_edges,total_count_edges)
        coverage.append(count_nodes/total_count_nodes*100)
        #if total_count_edges==0:
            #import pdb;pdb.set_trace()
        coverage_edge.append(count_edges/total_count_edges*100)
        node_utli_cluster.append(len(nodes_in_cluster)/len(cluster_nodes))
        #print(checked, cluster_edges)
        edge_util_cluster.append(len(edges_in_cluster)/len(cluster_edges))
    #import pdb;pdb.set_trace() 
    
    desired_width=30
    table_cluster_coverage = PrettyTable()
    table_cluster_coverage.field_names = ['Algorithm', 'Cluster coverage- node', 'Cluster coverage-edge', 'Cluster Utilization-Node', 'Cluster Utilization-Edge']
    for idx, (key) in enumerate(model_list):
        item=key.split('.')[1]
        table_cluster_coverage.add_row([
           item,
           round(coverage[idx],2),
           round(coverage_edge[idx],2),
           round(node_utli_cluster[idx],2),
           round(edge_util_cluster[idx],2)
        ])
    #import pdb;pdb.set_trace() 

    return table_cluster_coverage, coverage, coverage_edge, node_utli_cluster, edge_util_cluster

def universal_cluster_graph(edge_frequency, node_frequency, cluster_graph_image_name):
    start_time = time.time()
    plot_graph=False
    unique_edges_list=[]
    unique_nodes=set()
    G_directed = nx.DiGraph()
    #import pdb;pdb.set_trace() 
    comment={}
    node_value={}
    for idx, (key, value) in enumerate(edge_frequency.items()):
        weight=value[0]
        key_list=list(set(key))
        node=key_list[0]
        connection=key_list[1] if len(key_list)>1 else key_list[0]
        weightforclustering=weight*1e-6/(min(node_frequency[node][3], node_frequency[connection][3])) #weight normalized by minimum area of the two nodes
        unique_edges_list.append([node, connection,{'weight':weight, 'weightforclustering': weightforclustering}])
        #unique_edges_list.append([node, connection])
        unique_nodes.add(node)
        unique_nodes.add(connection)
    
    #import pdb;pdb.set_trace() 

    for node in unique_nodes:
        G_directed.add_node(node,weight=node_frequency[node][0], comment="L:"+f"{node_frequency[node][1]}", second_comment="E:"+f"{node_frequency[node][2]}", third_comment="A:"+f"{node_frequency[node][3]}",fourth_comment="A_mod:"+f"{node_frequency[node][4]}", fifth_comment=node_frequency[node][5], sixth_comment="HW_MACs:"+f"{node_frequency[node][6]}", seventh_comment="Q_out_max:"+f"{node_frequency[node][7]}",  eighth_comment="Q_in_max:"+f"{node_frequency[node][8]}", ninth_comment="Q_wg_max:"+f"{node_frequency[node][9]}", ddr_latency="DDR_cycles:"+f"{node_frequency[node][10]}")
    G_directed.add_edges_from(unique_edges_list)
    G=G_directed.to_undirected()

    if plot_graph:
        #import pdb;pdb.set_trace() 
        plt.clf()
        num_nodes = len(G)
        num_edges = len(G.edges())
        min_fig_size = 15
        edge_scale_factor = 0.2

        fig_size = max(min_fig_size, num_nodes * 0.4 + num_edges * edge_scale_factor)
        # Seed for reproducibility
        seed_value = 42
        np.random.seed(seed_value)

        pos = nx.kamada_kawai_layout(G, scale=0.2) 
        edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
        #labels = {n: str(n) + '\n' + str(G.nodes[n]['weight']) for n in G.nodes}
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        plt.subplots_adjust(top=0.97,bottom=0)  #
        nx.draw(
            G,
            pos,
            with_labels=True,
            #labels=labels,
            font_weight='bold',
            node_size=1000,
            node_color='skyblue',
            font_size=8,
            edgelist=edges, 
            edge_color=weights,
            width=2.0,
        )
        
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9, font_weight='bold',label_pos=0.2)

        for node in G.nodes():
            if G.has_edge(node, node):
                x, y = pos[node]
                plt.text(x +0.0005*fig_size, y+0.0005*fig_size, str(edge_labels[(node, node)]), fontsize=9, color='red', verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.1', edgecolor='none', facecolor='white', alpha=0.7))
        """
        for node, (x, y) in pos.items():
            comment = G.nodes[node]['comment']
            plt.text(x + 0.0003*fig_size, y-0.0003*fig_size, comment, fontsize=9, verticalalignment='center',weight='bold', bbox=dict(boxstyle='round,pad=0.01', edgecolor='none', facecolor='white', alpha=0.7))
            second_comment = G.nodes[node]['second_comment']
            plt.text(x + 0.005, y-0.002, second_comment, fontsize=9, verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='white', alpha=0.7))
        """

        ax.set_title("Universal Cluster Graph", fontsize=24, fontweight='bold')
        plt.savefig(cluster_graph_image_name)
        plt.show()
        plt.close()
    
    end_time = time.time()
    #print(f"Time taken to plot universal cluster graph: {end_time - start_time} seconds")


    return unique_nodes, edge_frequency, G

def cluster_graph(high_frequency_nodes, cluster_nodes, cluster_edges, cluster_graph_image_name):
    #import pdb;pdb.set_trace() 

    for node in high_frequency_nodes:
        cluster_nodes.add(node)
        for row in high_frequency_nodes[node][4]:
            for item in row:
                if item in cluster_nodes:
                    cluster_edges.add((node, item))
    G_directed = nx.DiGraph()
    G_directed.add_nodes_from(cluster_nodes)
    G_directed.add_edges_from(cluster_edges, length=1)
    plt.clf()
    G=G_directed.to_undirected()
    num_nodes = len(G)
    num_edges = len(G.edges())
    min_fig_size = 15
    edge_scale_factor = 0.2

    fig_size = max(min_fig_size, num_nodes * 0.4 + num_edges * edge_scale_factor)
    # Seed for reproducibility
    seed_value = 42
    np.random.seed(seed_value)
    if plot_process_stats:
        pos = nx.kamada_kawai_layout(G, scale=0.2) 
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        plt.subplots_adjust(top=0.97,bottom=0)  #
        nx.draw(
            G,
            pos,
            with_labels=True,
            #labels=nx.get_node_attributes(G, 'label'),
            font_weight='bold',
            node_size=10000,
            node_color='skyblue',
            font_size=16,
            edge_color='gray',
            width=5,
        )
        
        ax.set_title("Cluster Graph based on probabilities and intra-graph node frequency", fontsize=24, fontweight='bold')
        plt.savefig(cluster_graph_image_name)
        plt.show()
        plt.close()

    high_frequency_nodes
    #import pdb;pdb.set_trace() 
    return  cluster_nodes, cluster_edges