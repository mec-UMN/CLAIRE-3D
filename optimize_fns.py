import copy
from json_utils import *
import os
import sys
import networkx as nx
import more_itertools
import itertools
from prettytable import PrettyTable
from Preprocess import *
from Postprocess import *
import heapq
import textwrap
from Utils import print_file, update_noc_width, grid_lego, plot_jacc, round_up
from Cost import *
from nop_fns import *
from scipy.cluster.hierarchy import linkage, fcluster
from collections import defaultdict
import time

print_table = True
def construct_ind_graphs_sub_fn(model_name, modified_file=False):
    start_time = time.time()
    result_list = []
    result_prev={}
    item=model_name.split('.')[1]
    platform=model_name.split('.')[0]
    #print('model',item)
    if item.startswith('peanut') :
        image_size=[480,640]
        #image_size=[960,960]
    elif item.startswith('whisper'):
        image_size=[3000, 1]
        #image_size=[256, 1]
    elif platform.startswith ('audio'):
        image_size=[175,95]
    elif item.startswith ('ast'):
        image_size=[175,95]
    elif platform.endswith('llm'):
        image_size=[128,1]
    elif item.startswith('inception'):
        image_size=[299,299]
    elif item.startswith('t5') or item.startswith('chronos'):
        image_size=[512,1]
    elif item.startswith('convtasnet'):
        image_size=[32000,1]
    else:
        image_size=[224,224]
    layer_number=0
    if platform=='vision':
        file_path='./Inputs/modelvision.'+item+'.txt'
    elif platform.startswith('hugging'):
        file_path='./Inputs/modelhugging.'+item+'.txt'
    elif platform.startswith('sr'):
        file_path='./Inputs/modelsr.'+item+'.txt'
    elif platform.startswith('audio'):
        file_path='./Inputs/modelaudio.'+item+'.txt'
    else:
        file_path='./Inputs/model_'+item+'.txt'
    
    #if modified_file:
        #file_path=file_path.replace('.txt','_modif.txt')
    count_prev=1
    end_loop=0
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            #Extraction logic for each line
            #print(line_number)
            result = extract_info(line.strip(), result_prev, image_size, line_number, layer_number, item, count_prev, end_loop)
            #print(image_size)
            if result is not None:
                #print(result)
                if 'out_channel/features' in result:
                    layer_number+=1
                    result_prev=result.copy()
                    if modified_file:
                        #split result into two layers
                        result_updated_1=result.copy()
                        result_updated_2=result.copy()

                        if result['count']<2:
                            result_updated_1['out_channel/features']=int(result['out_channel/features']//2) 
                            result_updated_2['out_channel/features']= result['out_channel/features'] - result_updated_1['out_channel/features']
                        else:
                            result_updated_1['count']=int(result['count']//2)
                            result_updated_2['count']= result['count'] - result_updated_1['count']
                        result_updated_1['layer_type']=result['layer_type']+'1'
                        result_list.append(result_updated_1)
                        result_updated_2['layer_type']=result['layer_type']+'2'
                        result_list.append(result_updated_2)
                        #import pdb;pdb.set_trace()
                    else:
                        result_list.append(result)
                    count_prev= result['count']
                    end_loop=0
                elif 'loop' in result:
                    end_loop=result['loop']
                    if end_loop==3:
                        count_prev=1
                        end_loop=0
                elif 'count' in result:
                    count_prev= result['count']
                else:
                    result_prev['out_channel/features']=int(result) 
                #print("count:",count_prev)
                #print(line_number, line, result_prev)
    #if item == 'chronos':
        #import pdb;pdb.set_trace() 
    #print(result_list)
    #print("Sub fn: Constructed Individual Layers for", model_name, "time taken:", time.time()-start_time)
    return result_list, item

def construct_ind_graphs(model_list, plot, modified_file=False):
    ind_graphs=[]
    L,A,E=[],0, []
    for model_name in model_list:
        start_time = time.time()
        result_list, item=construct_ind_graphs_sub_fn(model_name, modified_file=modified_file[model_name] if isinstance(modified_file, dict) else modified_file)
        #print("Constructed Individual Layers for", model_name, "time taken:", time.time()-start_time)
        if item=='effnet':
            # List of layers, similar to the one you've provided
            layers=result_list
            # Iterate over the layers and check for the pattern
            for i in range(len(layers) - 4):
                # Check if the pattern matches: adaptiveavgpool2d -> conv2d -> conv2d -> silu -> sigmoid
                if (layers[i]['layer_type'] == 'adaptiveavgpool2d' and
                    layers[i+1]['layer_type'] == 'conv2d' and
                    layers[i+2]['layer_type'] == 'conv2d' and
                    layers[i+3]['layer_type'] == 'silu' and
                    layers[i+4]['layer_type'] == 'sigmoid'):
                    
                    # Update input and output sizes for these layers
                    for j in range(i, i + 5):
                        layers[j]['in_size'] = (1, 1)
                        layers[j]['out_size'] = (1, 1)

                    i+=5
        if item=="mobilevit":
            layers=result_list
            # Iterate over the layers and check for the pattern of 5 linear layers + silu + linear
            for i in range(len(layers) - 6):  # Ensure there's space for 5 linear layers, silu, and 1 more linear layer
                # Check if the pattern matches: linear -> linear -> linear -> linear -> linear -> silu -> linear
                if (layers[i]['layer_type'] == 'linear' and
                    layers[i+1]['layer_type'] == 'linear' and
                    layers[i+2]['layer_type'] == 'linear' and
                    layers[i+3]['layer_type'] == 'linear' and
                    layers[i+4]['layer_type'] == 'linear' and
                    layers[i+5]['layer_type'] == 'silu' and
                    layers[i+6]['layer_type'] == 'linear'):
                    
                    # Update input and output sizes for these layers by dividing them by 16
                    for j in range(i, i + 7):  # Update the 5 linear layers, silu, and the last linear layer
                        layers[j]['in_size'] = tuple(math.ceil(dim / 16) for dim in layers[j]['in_size'])
                        layers[j]['out_size'] = tuple(math.ceil(dim / 16) for dim in layers[j]['out_size'])
                    i+=6
                    

        #print("Constructed Individual Layers for", model_name, "time taken:", time.time()-start_time)
        if print_table:
            table = PrettyTable()
            table.field_names = list(result_list[0].keys())
            for row in result_list:
                table.add_row(list(row.values()))
            folder_path="./Summary/"
            os.makedirs(folder_path, exist_ok=True)
            summary_file_name = folder_path+"model"+str(item)+".txt"
            if print_file:
                print_file(summary_file_name, table)
        #print("Constructed Individual Layers for", model_name, "time taken:", time.time()-start_time)
        #Group Layers
        #print("Model:", model_name)
        grouped_layers = group_layers(result_list)
        #import pdb;pdb.set_trace() 
        
        #print("Constructed Grouped Layers for", model_name, "time taken:", time.time()-start_time)
        if print_table:
            desired_width=30
            table_group = PrettyTable()
            #table_group.field_names = ['Layer Type', 'In Size', 'Out Size', 'In Channel/Features', 'Out Channel/Features',
            #                     'Kernel Size', 'Stride', 'Padding', 'Layer Numbers']
            table_group.field_names = ['Layer Type', 'Layer Numbers']
            for idx, (key, layer_numbers) in enumerate(grouped_layers.items(), start=1):
                layer_info = result_list[layer_numbers[0][0][0]]
                wrapped_layer_numbers = textwrap.fill(', '.join(map(str, layer_numbers[0][0])), width=desired_width)
                #wrapped_layer_names = textwrap.fill(str(key[0])+'_'+str(idx), width=desired_width//2)        
                wrapped_layer_names = textwrap.fill(str(key), width=desired_width//2)        

                table_group.add_row([
                    wrapped_layer_names,
                    wrapped_layer_numbers
                ])

                """   
                table_group.add_row([
                    wrapped_layer_names,
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    key[5],
                    key[6],
                    key[7],
                    wrapped_layer_numbers
                ])
                """
            

            #import pdb;pdb.set_trace() 
            folder_path="./Grouping/"
            os.makedirs(folder_path, exist_ok=True)
            text_file_name = folder_path+"model"+str(item)+".txt"
            print_file(text_file_name, table_group)
        #print("Constructed Individual Layers and Grouped Layers for", model_name, "time taken:", time.time()-start_time)
        folder_path="./Graphs/"
        os.makedirs(folder_path, exist_ok=True)
        graph_image_name = folder_path + "/model" + str(item) + "_graph.png"
        #print(grouped_layers)
        G_temp,L_temp,E_temp,A_temp= create_graph(grouped_layers, graph_image_name, item, plot)
        #import pdb;pdb.set_trace() 
        #find_perm_cluster(G_temp)
        L.append(L_temp)
        E.append(E_temp)
        A+=A_temp
        ind_graphs.append(G_temp)
        #print(item, A_temp)
        #print("Model:", model_name, "Model Edges:", G_temp.edges())
        #print("Constructed Individual Graphs for", model_name, "time taken:", time.time()-start_time)
        #import pdb;pdb.set_trace() 
        #print("Constructed Individual Graphs, time taken:", time.time()-start_time)
    
    return ind_graphs,L,E,[A]

def construct_univ_cluster(model_list,ind_graphs, folder_path):
    #folder_path="./Stats_Clusters/"
    start_time = time.time()
    _, edge_frequency, node_frequency=edges_analyze(graphs_analyze_1(ind_graphs)[3])
    unique_nodes, unique_edges,universal_cluster_G, table_cluster_coverage=cluster_generate(ind_graphs, edge_frequency,node_frequency, folder_path, model_list,False)
    #import pdb;pdb.set_trace() 
    universal_cluster=[universal_cluster_G, model_list,unique_nodes, unique_edges]
    #print("Constructed Universal Cluster, time taken:", time.time()-start_time)
    return universal_cluster

def generate_partitions(nodes):
    if not nodes:
        yield []
    else:
        first = nodes[0]
        rest = nodes[1:]
        for partition in generate_partitions(rest):
            for i, subset in enumerate(partition):
                yield partition[:i] + [subset + [first]] + partition[i+1:]
            yield [[first]] + partition


def find_all_partitions(universal_cluster_G):
    #import pdb;pdb.set_trace() 
    nodes = list(universal_cluster_G.nodes())
    
    for partition in generate_partitions(nodes):
        # Check if each subset forms a strongly connected component
        valid_partition = True
        for subset in partition:
            subgraph = universal_cluster_G.subgraph(subset)
            if isinstance(universal_cluster_G, nx.DiGraph):
                if not nx.is_strongly_connected(subgraph) and not nx.is_weakly_connected(subgraph):
                    valid_partition = False
                    break
            else:
                if not nx.is_connected(subgraph):
                    valid_partition = False
                    break
        if valid_partition:
            yield partition
    #import pdb;pdb.set_trace() 

def find_perm_cluster(universal_cluster):
    universal_cluster_G=universal_cluster[0]
    all_valid_partitions=list(find_all_partitions(universal_cluster_G))
    if not all_valid_partitions:
        print("No valid partitions found.")
        return None
    else:
        print("Nodes:", list(universal_cluster_G.nodes()))
        print("\nGenerated graphs from valid partitions:")
        
        for partition in all_valid_partitions:
            subgraphs =[universal_cluster_G.subgraph(cluster).copy() for cluster in partition]
            print("\nPartition:", partition)
            for idx, subgraph in enumerate(subgraphs):
                print(f"Subgraph {idx + 1}:")
                print("Nodes:", subgraph.nodes())
                print("Edges:", subgraph.edges())
    import pdb;pdb.set_trace() 
    
    return subgraphs

def run_DSE_cluster(model_ind_graphs, clusters, config_cluster, opt, mode, num_opt_clusters, directory_name, dse_count, modified_file=False):
    result_dict={}
    result_out=[]
    config_custom=[[]]
    config_common=[]
    approach=2
    limit_check=True
    unique_nodes = set()
    if len(model_ind_graphs)>1:
        os.system("rm -r "+directory_name)
        os.system("mkdir "+directory_name)
        #import pdb;pdb.set_trace()
        for idx, graph in enumerate(model_ind_graphs[1]):
            unique_nodes.update(graph.nodes)
            #result_out.append(run_DSE([model_ind_graphs[0][idx]],None, False, True, limit_check,directory_name, dse_count, set(graph.nodes))[0:2])
            if modified_file:
                modified_file_flag = modified_file.get(model_ind_graphs[0][idx], False)
            else:                
                modified_file_flag = False
            result_out.append(run_DSE([model_ind_graphs[0][idx]],None, False, True, limit_check,directory_name, dse_count, None, modified_file_flag))
            #import pdb;pdb.set_trace()

        #import pdb;pdb.set_trace()
        result_list=[]
        for idx_grph, graph in enumerate(result_out):
            result_dict_custom={}
            for item in graph[0]:
                #import pdb;pdb.set_trace()
                config=tuple(item[1].items())
                ind_graph_nx=item[2][0]
                result_dict_custom[config]=item[0]
            result_list.append([(model_ind_graphs[0][idx_grph], config, result_dict_custom[config], ind_graph_nx) for config in result_dict_custom])
            for config in result_dict_custom:
                if mode == "EDAP":
                    result_dict_custom[config]=numpy.prod(result_dict_custom[config])
                elif mode == "high_performance":
                    result_dict_custom[config]=round_up(result_dict_custom[config][0][0], 2)
                elif mode == "compact":
                    result_dict_custom[config]=result_dict_custom[config][2]
                   
            #import pdb;pdb.set_trace()
            
            #Make Debug folder if it doesn't exist
            if not os.path.exists(directory_name + "Debug/"):
                os.makedirs(directory_name + "Debug/")
            #import pdb;pdb.set_trace()
            #save result_dict_custom onto a file for debugging
            #with open(directory_name + f"Debug/debug_result_dict_custom_{model_ind_graphs[0][idx_grph]}.txt", "w") as f:
                #for config, value in result_dict_custom.items():
                    #f.write(f"{config}: {value}\n")
            config_custom[0].append(dict(min({k: v for k, v in reversed(result_dict_custom.items()) if v != 0}, key=result_dict_custom.get)))
            #import pdb;pdb.set_trace()
        del result_dict_custom
        #Generic config
        #import pdb;pdb.set_trace()
        #result_out=run_DSE(model_ind_graphs[0],None, False, True, limit_check, directory_name, dse_count, unique_nodes)[0:2]
        result_out=run_DSE(model_ind_graphs[0],None, False, True, limit_check, directory_name, dse_count, None)[0:2]
        for item in result_out[0]:
            #import pdb;pdb.set_trace()
            config=tuple(item[1].items())
            ind_graph_nx=item[2][0]
            result_dict[config]=item[0]
        #import pdb;pdb.set_trace()

        for config in result_dict:
            if mode == "EDAP":
                result_dict[config]=numpy.prod(result_dict[config])
            elif mode == "high_performance":
                result_dict[config]=[round_up(x, 2) for x in result_dict[config][0]]
            elif mode == "compact":
                result_dict[config]=result_dict[config][2]
            #result_dict[config]=numpy.prod(result_dict[config][:2])

        #print(result_dict.values())
        #save result_dict onto a file for debugging
        #with open(directory_name + f"Debug/debug_result_dict_generic.txt", "w") as f:
            #for config, value in result_dict.items():
                #f.write(f"{config}: {value}\n")
        
        values = np.array(list(result_dict.values()), dtype=float)
        norm = (values - values.min(axis=0)) / (values.max(axis=0) - values.min(axis=0) + 1e-8)
        norm_means = dict(zip(result_dict.keys(), norm.mean(axis=1)))

        config_common_t = min(
            (k for k, v in reversed(result_dict.items()) if any(x != 0 for x in v)),
            key=lambda k: norm_means[k])        
        #import pdb;pdb.set_trace()
        config_common.append(dict(config_common_t))
        del result_dict

        #print("Generic Configuration")
        #import pdb;pdb.set_trace()
        PPA_common, cluster_common, ind_graphs_common=eval(model_ind_graphs[0],config_common[0], False, directory_name)
        config_common.append(list(cluster_common[0].nodes()))
        #import pdb;pdb.set_trace()
        #PPA_common[2]=[sum([float(cluster_common[0].nodes[item]['third_comment'].split(':')[1]) for item in cluster_common[0]])*1e+6]
        #import pdb;pdb.set_trace() 

        #import pdb;pdb.set_trace()
        PPA_custom=[[],[],[]]
        cluster_custom,ind_graphs_list_custom=[], []
        config_custom.append([])
        for idx_config, config in enumerate(config_custom[0]):
            #print(model_ind_graphs[0][idx])
            #print("Custom Configuration", idx_config)
            PPA_temp, cluster_temp, ind_graphs_temp=eval([model_ind_graphs[0][idx_config]],config, False, directory_name, modified_file)
            #import pdb;pdb.set_trace()
            for idx,item in enumerate(PPA_temp):
                #print(PPA_custom,item)
                PPA_custom[idx].append(item[0])
            
            #PPA_custom.append(PPA_temp)
            ind_graphs_list_custom.append(ind_graphs_temp)
            cluster_custom.append(cluster_temp)
            #import pdb;pdb.set_trace()
            #PPA_custom[2][idx_config]=sum([float(cluster_temp[0].nodes[item]['third_comment'].split(':')[1]) for item in cluster_temp[0]])*1e+6
            config_custom[1].append(list(cluster_temp[0].nodes()))

        if opt:
            #optimal configuration
            filtered_data=filter_data(result_list, PPA_custom, model_ind_graphs[0], False)

            #segregated_data,model_opt_list = segregate_into_ranges(filtered_data, 3)
            segregated_data,model_opt_list = segregate_into_ranges_nodes(filtered_data, model_ind_graphs[0] , num_opt_clusters, directory_name)
            
            #import pdb;pdb.set_trace()
            print("Optimal Config_Clusters -", model_opt_list)
            
            #import pdb;pdb.set_trace()
            model_graph_dict={item: model_ind_graphs[1][idx] for idx, item in enumerate(model_ind_graphs[0])}
            if approach ==1:
                #model_opt_list=[]
                config_opt=[[]]
                #print(result_list, segregated_data)
                for bin_idx, item in enumerate(segregated_data):
                    model_dict_opt={}
                    result_dict_opt={}
                    result_dict_opt_pre={}
                    for bin in item:
                        #print(len(segregated_data), model_opt_list, bin_idx)
                        if bin[0][0] in model_opt_list[bin_idx]:
                            if bin[1] in result_dict_opt_pre:
                                result_dict_opt_pre[bin[1]]=[[x + y for x, y in zip(row1, row2)] for row1, row2 in zip(result_dict_opt_pre[bin[1]], bin[2])]
                                #model_dict_opt[bin[1]].append(bin[0])
                            else:
                                result_dict_opt_pre[bin[1]]=bin[2]
                                #model_dict_opt[bin[1]]=[bin[0]]
                            #print(bin[0])
                    #import pdb;pdb.set_trace()
                    
                    for config in result_dict_opt_pre:
                        if mode == "EDAP":
                            result_dict_opt[config]=numpy.prod(result_dict_opt_pre[config])
                        elif mode == "high_performance":
                            result_dict_opt[config]=result_dict_opt_pre[config][0]
                        elif mode == "compact":
                            result_dict_opt[config]=result_dict_opt_pre[config][2]
                #print(result_dict_opt.values())
                    config_opt_t = min(result_dict_opt, key=result_dict_opt.get)
                    #import pdb;pdb.set_trace()
                    config_opt[0].append(dict(config_opt_t))
                    #model_opt_list.append(model_dict_opt[config_opt_t])
                
                config_opt.append([])
                #import pdb;pdb.set_trace()
                PPA_opt=[]
                cluster_opt, ind_graphs_list_opt=[], []
                for bin_idx, config in enumerate(config_opt):
                    #import pdb;pdb.set_trace()
                    out_temp=eval(model_opt_list[bin_idx],config, False, directory_name)
                    PPA_opt.append(out_temp[0])
                    cluster_opt.append(out_temp[1])
                    ind_graphs_list_opt.append(out_temp[2])
                    #PPA_opt[bin_idx][2]=[sum([float(out_temp[1][0].nodes[item]['third_comment'].split(':')[1]) for item in out_temp[1][0]])*1e+6]
                    config_opt[1].append(list(cluster_opt[0].nodes()))
            else:
                reset_config()
                result_out_opt=[]
                for bin_idx, model_opt in enumerate(model_opt_list):
                    algo_graph_dict = dict(zip(model_ind_graphs[0], model_ind_graphs[1]))
                    filtered_graphs = [algo_graph_dict[algo] for algo in model_opt_list[bin_idx] if algo in algo_graph_dict]
                    unique_nodes = set()
                    for graph in filtered_graphs:
                        unique_nodes.update(graph.nodes)
                    #import pdb;pdb.set_trace()
                    #result_out_opt.append(run_DSE(model_opt, None, False, True, limit_check,directory_name, dse_count, unique_nodes)[0:2])
                    result_out_opt.append(run_DSE(model_opt, None, False, True, limit_check,directory_name, dse_count, None)[0:2])
                #import pdb;pdb.set_trace()
                PPA_opt=[]
                cluster_opt,ind_graphs_list_opt=[], []
                config_opt=[[], []]
                for bin_idx, result_opt in enumerate(result_out_opt):
                    result_dict_opt={}
                    for item in result_opt[0]:
                        #import pdb;pdb.set_trace()
                        config=tuple(item[1].items())
                        result_dict_opt[config]=item[0]
                    #import pdb;pdb.set_trace()
                    
                    for config in result_dict_opt:
                        if mode == "EDAP":
                            result_dict_opt[config]=result_dict_opt[config]
                            result_dict_opt[config]=numpy.prod(result_dict_opt[config])
                        elif mode == "high_performance":
                            result_dict_opt[config]= [round_up(x, 2) for x in result_dict_opt[config][0]]
                        elif mode == "compact":
                            result_dict_opt[config]=result_dict_opt[config][2]
                        #result_dict[config]=numpy.prod(result_dict[config][:2])

                    #save result_dict_opt onto a file for debugging
                    #with open(directory_name + f"Debug/debug_result_dict_opt_{model_opt_list[bin_idx]}.txt", "w") as f:
                        #for config, value in result_dict_opt.items():
                            #f.write(f"{config}: {value}\n")
                    #print(result_dict_opt.values())
                    values_opt = np.array(list(result_dict_opt.values()), dtype=float)
                    norm_opt = (values_opt - values_opt.min(axis=0)) / (values_opt.max(axis=0) - values_opt.min(axis=0) + 1e-8)
                    norm_means_opt = dict(zip(result_dict_opt.keys(), norm_opt.mean(axis=1)))

                    config_opt_t = min(
                        (k for k, v in reversed(result_dict_opt.items()) if any(x != 0 for x in v)),
                        key=lambda k: norm_means_opt[k]
                    )                    
                    #import pdb;pdb.set_trace()
                    config_opt_temp=dict(config_opt_t)
                    
                    PPA_opt_temp, cluster_opt_temp, ind_graphs_temp=eval(model_opt_list[bin_idx],config_opt_temp, False, directory_name)
                    #import pdb;pdb.set_trace()
                    #PPA_opt_temp[2]=[sum([float(cluster_opt_temp[0].nodes[item]['third_comment'].split(':')[1]) for item in cluster_opt_temp[0]])*1e+6]
                    #import pdb;pdb.set_trace() 
                    PPA_opt.append(PPA_opt_temp)
                    cluster_opt.append(cluster_opt_temp)
                    ind_graphs_list_opt.append(ind_graphs_temp)
                    config_opt[0].append(config_opt_temp)
                    config_opt[1].append(list(cluster_opt_temp[0].nodes()))

              
                #import pdb;pdb.set_trace()
                
        else:
            PPA_opt, config_opt,cluster_opt, ind_graphs_list_opt=[],[],[],[]
        #print(result_dict.values())
        #import pdb;pdb.set_trace()   

        return PPA_common, config_common, cluster_common, PPA_custom, config_custom,cluster_custom,  PPA_opt, config_opt,cluster_opt,  ind_graphs_common,  ind_graphs_list_custom, ind_graphs_list_opt

    else:
        #print("Debug", model_ind_graphs[0])
        #import pdb;pdb.set_trace()
        result_out.append(run_DSE(model_ind_graphs[0], [clusters[0][0]] + clusters[1:], config_cluster, False, False, directory_name, dse_count, None, modified_file))
        PPA_opt, config_opt, ind_G=result_out[0][0][0]
        #PPA_opt, config_opt, cluster_opt=min(result_out, key=lambda x: numpy.prod(x[0]))
        #import pdb;pdb.set_trace()
        table_cluster_coverage, coverage, coverage_edge, node_utli_cluster, edge_util_cluster=cluster_analyze(ind_G, clusters[0][1], clusters[0][2], model_ind_graphs[0])
        
        return PPA_opt, config_opt,[coverage, coverage_edge, node_utli_cluster, edge_util_cluster]

def eval(model_ind_graphs,config, plot, directory_name, modified_file=False):

    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'params.json')
    write_json_file(config, relative_path)
    ind_graphs, L,E,A =construct_ind_graphs(model_ind_graphs, plot, modified_file)
    cluster_opt=construct_univ_cluster(model_ind_graphs,ind_graphs, directory_name)
    A= [sum([float(cluster_opt[0].nodes[item]['third_comment'].split(':')[1]) for item in cluster_opt[0]])*1e+6] 
    for idx_graph, graph in enumerate(ind_graphs):
        #print(model_list[idx_graph], graph.edges())
        for edge in graph.edges():
            edge_weight=graph.edges[edge]['weight']
            L[idx_graph]-=edge_weight*1e-6
    #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    #A=area_cluster(config)
    PPA_opt = [L,E,A] 
    return PPA_opt, cluster_opt, ind_graphs

def eval_noc(model_ind_graphs, channel_BW, ind_graphs, PPA_in):
    
    L,E,A=PPA_in[0:3]
    #print(PPA_in)
    #print(model_ind_graphs,L, E,A)
    initial_L=sum(L)
    for idx_graph, graph in enumerate(ind_graphs):
        #print(model_list[idx_graph], graph.edges())
        for edge in graph.edges():
            edge_weight=graph.edges[edge]['weight']
            BW_noc, init_BW_noc, E_noc=update_noc_width(graph, edge, channel_BW)
            E[idx_graph]+=E_noc*1e+6
            L[idx_graph]+=(edge_weight*init_BW_noc/BW_noc)*1e-6
    #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    #A=area_cluster(config)
    #noc_L_overhead=(sum(L)-initial_L)/initial_L*100
    #print("NOC Latency Overhead (%): ", noc_L_overhead)
    PPA_opt = [L,E,A] 
    return PPA_opt


def run_DSE(model_list, graph_or_cluster, config, dse, limit_check,directory_name, dse_count, unique_nodes, modified_file=False):
    #print("run_DSE", model_list)
    start_time = time.time()
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'params.json')
    if config:
        original_data=config
        write_json_file(config, relative_path)
    else:
        original_data = read_json_file(relative_path)
    modified_data,result_out = modify_and_run(model_list, graph_or_cluster, original_data, relative_path,dse, limit_check, directory_name, dse_count, unique_nodes, modified_file)
    if modified_data!=original_data:
        print("Error")
        sys.exit()
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(f"DSE Time taken: {elapsed_time:.2f} seconds")
    return result_out

def modify_and_run(model_list, graph_or_cluster, data, file_path, dse, limit_check, directory_name,dse_count, unique_nodes, modified_file=False):
    result_out=[]
    start_time = time.time()
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
        
    #import pdb;pdb.set_trace()
    iterations=1 if not dse else dse_count
    stack_mode = "iso-silicon_area"
    if stack_mode =="iso-footprint_area":
        PD_limit=1500/2 #mW/mm2
        A_limit_max=250 #mm2
    elif stack_mode =="iso-silicon_area":
        PD_limit=1500 #mW/mm2
        A_limit_max=125 #mm2
    #Max_Mem_size=1024*320*4 # in bits
    #print("Debug: Model_list" , model_list, iterations)

    original_data = copy.deepcopy(data)
    #print("data", data)
    #import pdb;pdb.set_trace()
    #print(model_list, modified_file)
    modified_keys = {"n_SA", "no_modules_act", "no_modules_pool"}
    ranges = {
        key: sorted([int(value * (2 ** i)) for i in range(iterations)])
        if any(key.endswith(suffix) for suffix in unique_nodes) or key.endswith("PE") or key.startswith("no_modules")
        else (sorted([int(value * 2 ** (i-1)) for i in range(iterations)]) if key == "n_SA" and modified_file else [value])
        for key, value in data.items()
    } if unique_nodes is not None else {
        key: sorted([int(value * (2 ** i)) for i in range(iterations)])
        if not (key in modified_keys and modified_file)
        else sorted([int(value * 2 ** (i-1)) for i in range(iterations)])
        for key, value in data.items()
    }

    combinations = list(itertools.product(*ranges.values()))
    all_configs = [dict(zip(ranges.keys(), combo)) for combo in combinations]
    #if iterations==1:
        #print(all_configs)
    # Keep only symmetric configurations
    if 'row_PE' in ranges and 'col_PE' in ranges:
        all_configs = [cfg for cfg in all_configs if cfg['row_PE'] == cfg['col_PE']]
    #print(all_configs)
    for config in all_configs:
        #import pdb;pdb.set_trace() 
        #print(config)
        write_json_file(config, file_path)
        PPA_result=list(run_PPA(model_list, graph_or_cluster, config, not dse, directory_name, modified_file))
        if limit_check: 
            count_in_limit=0
            #for idx,_ in enumerate(model_list):
                #print(PPA_result[0][2], idx)
            #Mem_size_required= config['row_PE']* config['col_PE']*config['n_SA']*PPA_config["num_bits_act"]    
            L=PPA_result[0][0][0]
            E=PPA_result[0][1][0]
            A=PPA_result[0][2][0]
            if E/L/A<PD_limit and A<A_limit_max:
                count_in_limit+=1
            #print(model_list,count_in_limit ,PPA_result[0])
            #if count_in_limit==len(model_list):
                result_out.append(PPA_result)
            #import pdb;pdb.set_trace()
        else:
            result_out.append(PPA_result)
        #print(len(result_out))

    write_json_file(original_data, file_path)
    #import pdb;pdb.set_trace() 
    #print(result_out)
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(f"Modify_run Evaluation Time taken: {elapsed_time:.2f} seconds") 
    return data, [result_out]


def run_PPA(model_list, graph_or_cluster_list, config, nop, directory_name, modified_file=False):
    PPA=[]
    start_time = time.time()
    mode_nop='dse'
    channel_BW=None
    if nop:
        PPA_config=read_json_file(os.path.join(os.path.dirname(__file__), 'PPA_config.json'))   
        if mode_nop == 'dse':
            #nop_modes=['grs','aib','aib']
            #nop_widths=[10, 2, 2]
            nop_modes=['grs', 'ucie3d', 'aib']
            nop_widths=[10, 2, 2]
        elif mode_nop == 'fixed_3d':
            nop_modes=['ucie3d']
            nop_widths=[math.floor(PPA_config['BW_m2_ucie3d']*1e-6*49/PPA_config['BW_ucie3d'])]
        elif mode_nop == 'fixed_aib':
            nop_modes=['aib']
            nop_widths=[math.floor(PPA_config['BW_m_aib']*1e-3*math.sqrt(49)/PPA_config['BW_aib'])]
        #import pdb;pdb.set_trace()
        #print("run_nop", model_list)
        graph_or_cluster=graph_or_cluster_list[0]
        noc_channel_BW=graph_or_cluster_list[1]
        if isinstance(graph_or_cluster, nx.DiGraph):
            num_components = nx.number_weakly_connected_components(graph_or_cluster)
        else:
            num_components = nx.number_connected_components(graph_or_cluster)
        current_dir = os.path.dirname(__file__)
        relative_path = os.path.join(current_dir, 'params.json')
        write_json_file(config, relative_path)
        #if nodes have postfix, enable modified_file
        modified_file={item: False for item in model_list}
        for node in graph_or_cluster.nodes():
            if node[-1]=='1' or node[-1]=='2':
                #print("Modified File Detected: ", node)
                modified_file={item: True for item in model_list}
                #import pdb;pdb.set_trace()
                break
        ind_graphs=construct_ind_graphs(model_list, False, modified_file)[0] 
        #import pdb;pdb.set_trace()
        #print("L",construct_ind_graphs(model_list, False, modified_file)[1])
        clusters=list(nx.connected_components(graph_or_cluster))
        #print("construct_graph_noc",construct_ind_graphs(model_list, False))
        #import pdb;pdb.set_trace()
        if graph_or_cluster_list[2]:
            nop_top_idx=graph_or_cluster_list[3]
            #import pdb;pdb.set_trace() 
            predefined_nop_widths=width_nop(nop_modes[nop_top_idx], graph_or_cluster_list[2])
        else:
            predefined_nop_widths=None
            #if len(model_list)!=1 and len(model_list)<24:
            nop_top_idx= nop_topology_dse(nop_modes, nop_widths, clusters, graph_or_cluster, ind_graphs, directory_name, noc_channel_BW, model_list)
            #else:
            #nop_top_idx=-1
            predefined_nop_widths=nop_widths[nop_top_idx] 
        #print("Selected NOP Topology: ", nop_top_idx)
        A_cluster, A_cluster_mod, L, E, L_network, E_network, A_network, cluster_count,Nopdse_BW, ind_graphs, A_d2d_mod, n_pins, L_ddr =nop_dse([nop_modes[nop_top_idx]], [nop_widths[nop_top_idx]], predefined_nop_widths, clusters, graph_or_cluster, ind_graphs, directory_name, noc_channel_BW, model_list)
        #import pdb;pdb.set_trace()
        A=[sum(A_cluster)] 
        A_mod=[sum(A_cluster_mod)]

        #print("nop bandwidths: ", model_list, Nopdse_BW )
        #import pdb;pdb.set_trace()
        L=[x*1e-6 for x in L]
        E=[x*1e+6 for x in E]
        A=[grid_lego(x*1e+6) for x in A]
        #L_ddr=[x*1e-6 for x in L_ddr]
        #L_network=[x*1e-6 for x in L_network]
        #E_network=[x*1e+6 for x in E_network]
        A_mod=[x*1e+6 for x in A_mod]
        cost_params=read_json_file(os.path.join(os.path.dirname(__file__), 'Cost.json'))
        if nop_modes[nop_top_idx]=='ucie3d':
            stack_area_dict={0: max(A_cluster)*1e+6} # maximum of area of cluster is area of stack
            stack_ids={0: [i for i in range(cluster_count)]} # all clusters in one stack
            n_pins_dict={"die2die":{0: sum(n_pins)}} # all pins for die2die
        else:
            stack_area_dict={i: A_cluster[i]*1e+6 for i in range(cluster_count)} # each cluster in its own stack
            stack_ids={i: [i] for i in range(cluster_count)} # each cluster in its own stack
            n_pins_dict={"die2interposer":{i: n_pins[i] for i in range(cluster_count)}} # pins per cluster

        cost_res_dir=directory_name+"/Cost_Results/"
        os.makedirs(cost_res_dir, exist_ok=True)
        #import pdb;pdb.set_trace()
        #NRECost=nre_cost_evaluate(A, A_mod, cluster_count)
        total_cost, cost_breakdown, costperdie, NRECost, RECost = cost_main_fn(''.join(m.split('.')[-1] for m in model_list), cost_params, [grid_lego(x*1e+6) for x in A_cluster], [x*1e6 for x in A_cluster_mod], clusters, [x*1e6 for x in A_d2d_mod], stack_area_dict, stack_ids, n_pins_dict, cost_res_dir)
        #import pdb;pdb.set_trace()
        PPA=[L,E,A, NRECost, L_network, E_network, ind_graphs, [A_network], [Nopdse_BW], [nop_top_idx], RECost, L_ddr]
             
    else:
        ind_graphs, L,E,A=construct_ind_graphs(model_list, False, modified_file)
        #print("Constructed Individual Graphs_time taken:", time.time()-start_time)
        cluster=construct_univ_cluster(model_list,ind_graphs, directory_name)
        A= [sum([float(cluster[0].nodes[item]['third_comment'].split(':')[1]) for item in cluster[0]])*1e+6]
        PPA=[L,E,A]
        #import pdb;pdb.set_trace() 
    end_time = time.time()
    elapsed_time = end_time - start_time
    #print(f"PPA Evaluation Time taken: {elapsed_time:.2f} seconds")  
    return PPA, copy.deepcopy(config),ind_graphs
    #print(model_list, PPA)
    #import pdb;pdb.set_trace() 


def plot_scatter(x, y, filename, title, x_label, y_label, labels):
    plt.scatter(x, y, color='blue', marker='o', label="Data Points")

    # Add labels to each point
    for i in range(len(x)):
        plt.text(x[i], y[i], labels[i], fontsize=12, ha='right', va='bottom', color='red')

    # Labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)  # Horizontal grid lines
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.savefig(filename)
    plt.close()

def find_optimal_cluster(universal_cluster, folder_path):
    #folder_path="./Stats_Clusters/"
    #import pdb;pdb.set_trace() 
    #print("opt_cluster", universal_cluster[1])
    cluster_graph=cluster(universal_cluster[0],folder_path, universal_cluster[1])
    #import pdb;pdb.set_trace() 
    return cluster_graph

def combine_similar_clusters(cluster_graph_list_in, folder_path):
    cluster_graph_list_out = [[] for _ in range(len(cluster_graph_list_in))]
    cluster_sim_matrix = {}
    for idx_1, cluster_graph_1 in enumerate(cluster_graph_list_in):
        clusters_1 = list(nx.connected_components(cluster_graph_1[0]))
        split_graph_1 = True if len(clusters_1) == 1 else False
        cluster_subgraphs_1 = [cluster_graph_1[0].subgraph(cluster).copy() for cluster in clusters_1]
        for idx_2,cluster_graph_2 in enumerate(cluster_graph_list_in):
            clusters_2 = list(nx.connected_components(cluster_graph_2[0]))
            split_graph_2 = True if len(clusters_2) == 1 else False
            cluster_subgraphs_2 = [cluster_graph_2[0].subgraph(cluster).copy() for cluster in clusters_2]

            if idx_1 < idx_2:

                for sg_idx1, subgraph_1 in enumerate(cluster_subgraphs_1):

                    for sg_idx2, subgraph_2 in enumerate(cluster_subgraphs_2):
                        weighted_intersection, weighted_union = jacc_sim(subgraph_1, subgraph_2,"recompose", split_graph_1, split_graph_2 )

                        jacc_similarity = (weighted_intersection / weighted_union if weighted_union else 0)

                        if jacc_similarity > 0.8:
                            #print("Similar clusters found between", list(subgraph_1.nodes()), "and", list(subgraph_2.nodes()),"JS =", jacc_similarity)

                            # ---- MERGE ----
                            merged = nx.Graph()
                            merged.add_nodes_from(subgraph_1.nodes(data=True))
                            merged.add_edges_from(subgraph_1.edges(data=True))
                            merged.add_nodes_from(subgraph_2.nodes(data=True))
                            merged.add_edges_from(subgraph_2.edges(data=True))
                        
                            if (idx_1, sg_idx1) in cluster_sim_matrix:
                                if len(cluster_sim_matrix[(idx_1, sg_idx1)]) >= len(merged):
                                    cluster_sim_matrix[(idx_2, sg_idx2)] = cluster_sim_matrix[(idx_1, sg_idx1)]
                                    break
                            if (idx_2, sg_idx2) in cluster_sim_matrix:
                                if len(cluster_sim_matrix[(idx_2, sg_idx2)]) >= len(merged):
                                    cluster_sim_matrix[(idx_1, sg_idx1)] = cluster_sim_matrix[(idx_2, sg_idx2)]
                                    break
                            cluster_sim_matrix[(idx_1, sg_idx1)] = merged
                            cluster_sim_matrix[(idx_2, sg_idx2)] = merged
    #based on cluster_sim_matrix, create new cluster_graph_list_out
    for idx, cluster_graph in enumerate(cluster_graph_list_in):
        clusters = list(nx.connected_components(cluster_graph[0]))
        subgraphs = []
        for sg_idx, cluster in enumerate(clusters):
            if (idx, sg_idx) in cluster_sim_matrix:
                new_graph = cluster_sim_matrix[(idx, sg_idx)].copy()
                #update node attributes same as original subgraph
                original_subgraph = cluster_graph[0].subgraph(cluster)
                for node in new_graph.nodes():
                    if node in original_subgraph.nodes():
                        for attr in original_subgraph.nodes[node]:
                            new_graph.nodes[node][attr] = original_subgraph.nodes[node][attr]
                subgraphs.append(new_graph)
            else:
                subgraph = cluster_graph[0].subgraph(cluster).copy()
                subgraphs.append(subgraph)
            
        #if there are common nodes between subgraphs, rename a duplicate node from one of the subgraphs as node+'_dup'
        all_nodes = set()
        for i in range(len(subgraphs)):
            for node in subgraphs[i].nodes():
                if node in all_nodes:
                    new_node = node + '_dup'
                    mapping = {node: new_node}
                    subgraphs[i] = nx.relabel_nodes(subgraphs[i], mapping)
                all_nodes.add(node)
        
        graph=nx.compose_all(subgraphs)
        cluster_graph_list_out[idx].append(graph)  
        cluster_graph_list_out[idx].append(dict(graph.nodes()))
        #dict with key as frozen edge set and value as edge attributes
        edge_attr_dict = {}
        for edge in graph.edges(data=True):
            edge_key = frozenset([edge[0], edge[1]])
            edge_attr_dict[edge_key] = edge[2]['weight']
        cluster_graph_list_out[idx].append(edge_attr_dict)

        #print([subgraph.nodes() for subgraph in subgraphs])
        #print(list(nx.connected_components(cluster_graph_list_out[idx][0])))         

    cluster_all=[list(nx.connected_components(cluster_graph_list_out[idx][0])) for idx in range(len(cluster_graph_list_out))]
        
    #import pdb; pdb.set_trace()
    return cluster_graph_list_out


def segregate_into_ranges(data, num_opt_clusters=5):
    #import pdb;pdb.set_trace() 
    #flat_data = [sublist[2][2][0] for row in data for sublist in row]
    flat_data=[row[2][2][0] for row in data]
    min_value, max_value = np.min(flat_data), np.max(flat_data)
    bins = np.logspace(np.log10(min_value-1), np.log10(max_value+1), num_opt_clusters + 1)
    if min_value==max_value:
        num_opt_clusters=1
        bins=[min_value/2, 3*min_value/2]
    
    #import pdb;pdb.set_trace() 
    #print(bins)
    # Initialize segregated data list
    segregated_data_1 = [[] for _ in range(num_opt_clusters)]
    model_count_1=[{}]
    for i in range(num_opt_clusters-1):
        model_count_1.append({})
    # Assign each element to the appropriate bin
    for item in data:
        model=item[0]
        config=item[1]
        values=item[2]
        for i in range(num_opt_clusters):
            #import pdb;pdb.set_trace() 
            if bins[i] <= values[2][0] < bins[i + 1]:
                distance_to_bin_i = abs(values[2][0] - bins[i])
                distance_to_bin_i_plus_1 = abs(values[2][0] - bins[i + 1])

                if distance_to_bin_i < distance_to_bin_i_plus_1 and i!=0:
                    segregated_data_1[i-1].append((model, config, values))
                    if model in model_count_1[i-1]:
                        model_count_1[i-1][model] += 1
                    else:
                        model_count_1[i-1][model] = 1
                else:
                    segregated_data_1[i].append((model, config, values))
                    if model in model_count_1[i]:
                        model_count_1[i][model] += 1
                    else:
                        model_count_1[i][model] = 1
                break
        #print(model_count_1, model, config, values)
    segregated_data = [sublist for sublist in segregated_data_1 if sublist]
    model_count=[model for model in model_count_1 if model]
    max_values = defaultdict(int)

    #import pdb;pdb.set_trace() 


    for d in model_count:
        for key, value in d.items():
            if value > max_values[key]:
                max_values[key] = value

    updated_list = []
    out_data = []
    for bin_idx, d in enumerate(model_count):
        new_dict = {key for key, value in d.items() if max_values[key] == value}
        if new_dict:  
            updated_list.append(new_dict)
            #import pdb;pdb.set_trace() 
            out_data.append(segregated_data[bin_idx])
    
    #import pdb;pdb.set_trace() 
    
    return out_data, updated_list


def segregate_into_ranges_nodes(data, model_list, num_opt_clusters, directory_name):
    #import pdb;pdb.set_trace() 
    model_len=len(model_list)
    #print(data[0:model_len])
    start_time=time.time()
    similarity=[[0]*model_len for _ in range(model_len)]
    for i in range(model_len):
        for j in range(i+1, model_len):
            G1 = None
            G2 = None
            for k in range(len(data)):
                if data[k][0][0]==model_list[i]:
                    G1=data[k][3]
                    break
            for k in range(len(data)):
                if data[k][0][0]==model_list[j]:
                    G2=data[k][3]
                    break
            if G1 is not None and G2 is not None:
                weighted_intersection, weighted_union=jacc_sim(G1,G2)
                similarity[i][j] = weighted_intersection / weighted_union if weighted_union else 0
            else:
                similarity[i][j]=0
            similarity[j][i]=similarity[i][j]
            #print(model_list[i],model_list[j], similarity[i][j])
        similarity[i][i]=1

    jacc_similarity=np.array(similarity)
    # Plotting
    model_name=[item.split('.')[1].upper() for item in model_list]
    plot_jacc(jacc_similarity, model_name, model_name, directory_name, 'train', None)
    distance_matrix = 1 - jacc_similarity
    linkage_matrix = linkage(distance_matrix, method='ward')
    if num_opt_clusters is not None:
        clusters = fcluster(linkage_matrix, num_opt_clusters, criterion='maxclust')
    else:
        threshold = 0.5
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        num_opt_clusters=max(clusters)

    updated_list=[[] for _ in range(num_opt_clusters)] 
    out_data = [[] for _ in range(num_opt_clusters)]
    print("clusters", clusters)
    for data_item in data:
        model=data_item[0][0]
        model_idx=data_item[0][1]
        config=data_item[1]
        values=data_item[2]
        out_data[clusters[model_idx]-1].append((model, config, values))

    for model_idx, cluster_idx in enumerate(clusters):
        updated_list[cluster_idx-1].append(model_list[model_idx])
    
    end_time=time.time()
   # print("Time for clustering models based on Jaccard similarity: ", end_time-start_time)
    #import pdb;pdb.set_trace() 

    return out_data, updated_list

def jacc_sim(G1, G2, mode="weighted", split_graph_1=False, split_graph_2=False):
    weights_G1 = {n: float(d['third_comment'].split(':')[1])*1e+5 for n, d in G1.nodes(data=True)} if not split_graph_1 else {n: float(d['third_comment'].split(':')[1])*1e+5/2 for n, d in G1.nodes(data=True)}
    weights_G2 = {n: float(d['third_comment'].split(':')[1])*1e+5 for n, d in G2.nodes(data=True)} if not split_graph_2 else {n: float(d['third_comment'].split(':')[1])*1e+5/2 for n, d in G2.nodes(data=True)}
    common_nodes = set(weights_G1.keys()).intersection(set(weights_G2.keys()))

    if mode =="recompose":
        # identify nodes in common_nodes that have same weight in both graphs
        common_seg_nodes = {n for n in common_nodes if weights_G1[n] == weights_G2[n]}
        weighted_intersection = sum(weights_G1[n] for n in common_seg_nodes)
       
    else:
        # Compute weighted intersection
        weighted_intersection = sum(min(weights_G1[n], weights_G2[n]) for n in common_nodes)

    # Compute weighted union
    weighted_union = sum(max(weights_G1.get(n, 0), weights_G2.get(n, 0)) for n in set(weights_G1) | set(weights_G2))
    #import pdb;pdb.set_trace() 

    return weighted_intersection, weighted_union

def filter_data(data, PPA_custom, model_list, filter):
    #L_fac=0.5
    #PD_limit=1500 #mW/mm2
    #A_limit_max=150 #mm2
    filtered_data=[]
    #import pdb;pdb.set_trace() 

    for row in data:
        for item in row:
            model=item[0]
            config=item[1]
            L=item[2][0][0]
            for model_idx, model1 in enumerate(model_list):
                if model1==model:
                    break
            L_limit=PPA_custom[0][model_idx]
            E=item[2][1][0]
            A=item[2][2][0]
            #import pdb;pdb.set_trace() 
            Pd=E/L/A
            ele=([item[0],model_idx], item[1], item[2], item[3])
            if filter:
                #Latency dynamic, Power density and area limit
                if L<(1+L_fac)*L_limit and L>(1-L_fac)*L_limit and Pd<PD_limit and A<A_limit_max:
                    #import pdb;pdb.set_trace() 
                    filtered_data.append(ele)
            else:
                filtered_data.append(ele)
            #print(model, model1, L_limit, L, Pd, A)
    #import pdb;pdb.set_trace() 

    if not filtered_data:
        print("No data is within limits. Update the number of bins or run DSE around different initialization point")
    return filtered_data


def split_models(model_list, m, file_name):
    train_set=[]
    test_set=[]

    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, file_name)
    data=read_json_file(relative_path)

    sorted_data = dict(sorted(data.items(), key=lambda item: item[1]))
    keys = list(sorted_data.keys())
    values = list(sorted_data.values())

    n=len(model_list)-m
    mid_start = (len(keys) - n) // 2
    mid_end = mid_start + n

    test_set_keys = keys[mid_start:mid_end]

    train_set_keys = [key for key in keys if key not in test_set_keys][:m]

    for key in test_set_keys[:]:  # Iterate over a copy
        if isinstance(sorted_data[key], int):  # If value is a whole number
            key_index = keys.index(key)

            # Search for the nearest valid replacement
            replacement_key = None
            left, right = key_index - 1, key_index + 1
            
            while replacement_key is None:
                if left >= 0 and keys[left] not in test_set_keys:
                    replacement_key = keys[left]
                elif right < len(keys) and keys[right] not in test_set_keys:
                    replacement_key = keys[right]
                
                # Expand search range
                left -= 1
                right += 1
            #print(replacement_key)
            # Swap the whole number key with the found replacement
            if replacement_key:
                test_set_keys.remove(key)  # Remove whole number from test set
                train_set_keys.append(key)  # Move whole number to train set
                test_set_keys.append(replacement_key)  # Add replacement key to test set
                train_set_keys.remove(replacement_key)
    for model in model_list:
        if model.split('.')[1] in train_set_keys:
            train_set.append(model)
        else:
            test_set.append(model)

    #train_set = {key: sorted_data[key] for key in train_set_keys}
    #test_set = {key: sorted_data[key] for key in test_set_keys}

    #import pdb;pdb.set_trace() 

    return train_set, test_set


def run_test(model_list, config, clusters, directory_name, dse_count,  Nocdse_BW, nopdse_BW, nopdse_idx):
    cluster_split=True
    mode ="jacc"
    model_len=len(model_list)
    cluster_len=len(clusters)
    PPA,max_cluster_idx=[[0]*model_len,[0]*model_len, [0]*model_len,[0]*model_len,[0]*model_len, [0]*model_len,[0]*model_len, [0]*model_len,[0]*model_len, [0]*model_len, [0]*model_len, [0]*model_len], [0]*model_len
    coverage=[]
    node_coverage, edge_coverage, table_cluster_coverage, node_utli_cluster, edge_util_cluster=[0]*cluster_len,[0]*cluster_len, [0]*cluster_len,[0]*cluster_len, [0]*cluster_len  
    #ind_graphs,_,_,_=construct_ind_graphs(model_list, False)
    #for cluster_idx, cluster in enumerate(clusters):
        #table_cluster_coverage[cluster_idx], node_coverage[cluster_idx], edge_coverage[cluster_idx],  node_utli_cluster[cluster_idx], edge_util_cluster[cluster_idx]=cluster_analyze(ind_graphs, cluster[0].nodes(), cluster[2], model_list)
    #print(config)
    jacc_similarity=[[0]*cluster_len for _ in range(model_len)]
    coverage=[[],[],[],[]]
    cluster_split_dict={item:False for item in model_list}
    for model_idx in range(model_len):
        similarity=[]
        if mode=="coverage":
            product_list = [(node_coverage[i][model_idx]**2) * edge_coverage[i][model_idx] for i in range(len(node_coverage))]
            cluster_idx = max(enumerate(product_list), key=lambda x: x[1])[0]
        elif mode=="jacc":
            for cluster_idx_check,cluster in enumerate(clusters):
                #import pdb;pdb.set_trace() 
                current_dir = os.path.dirname(__file__)
                relative_path = os.path.join(current_dir, 'params.json')
                write_json_file(config[0][cluster_idx_check], relative_path)
                #import pdb;pdb.set_trace() 
                graph,_,_,_=construct_ind_graphs([model_list[model_idx]], False)
                weighted_intersection, weighted_union=jacc_sim(cluster[0],graph[0])
                similarity.append(weighted_intersection / weighted_union if weighted_union else 0)
            cluster_idx=similarity.index(max(similarity)) if similarity else print("Error: Similarity list is empty")
        jacc_similarity[model_idx]=similarity
        max_cluster_idx[model_idx]=cluster_idx
        #import pdb;pdb.set_trace() 
        if nx.number_connected_components(clusters[cluster_idx][0])==1 and cluster_split:
            #To duplicate
            #1. add postfix to name of the nodes of clusters[0][0] with '1' and '2' and recompose
            #2. halve the area of each node of recomposed graph
            #3.update the config_cluster by halving the 'n_SA'
            cluster_nodes_updated_1=nx.relabel_nodes(clusters[cluster_idx][0], lambda x: str(x) + '1')
            #second graph with postfix '2'
            cluster_nodes_updated_2=nx.relabel_nodes(clusters[cluster_idx][0], lambda x: str(x) + '2')
            #combine both graphs
            cluster_nodes_updated=nx.compose(cluster_nodes_updated_1, cluster_nodes_updated_2)
            #import pdb;pdb.set_trace()
            for node in cluster_nodes_updated.nodes():
                original_value=float(cluster_nodes_updated.nodes[node]['third_comment'].split(':')[1])
                cluster_nodes_updated.nodes[node]['third_comment']='A:'+str(original_value/2)
            print("models:", model_list[model_idx])
            print("cluster updated nodes:", list(cluster_nodes_updated.nodes()))
            edge_dict={}
            for edge in cluster_nodes_updated.edges():
                edge_key = frozenset([edge[0], edge[1]])
                edge_dict[edge_key] = cluster_nodes_updated.edges[edge]['weight']
            #import pdb;pdb.set_trace()   
            cluster_spec_list=[cluster_nodes_updated, dict(cluster_nodes_updated.nodes()), edge_dict]
            config_spec=config[0][cluster_idx].copy()
            #config_spec['col_PE']=config_spec['col_PE']//2
            #config_spec['n_SA']=config_spec['n_SA']//2
            #config_spec['no_modules_act']=config_spec['no_modules_act']//2
            #config_spec['no_modules_pool']=config_spec['no_modules_pool']//2
            cluster_split_dict[model_list[model_idx]]=True
            modified_file=True
        else:
            cluster_spec_list=clusters[cluster_idx]
            config_spec=config[0][cluster_idx].copy()
            modified_file=False
            #import pdb;pdb.set_trace()
        #print("Running DSE for model ", model_list[model_idx])
        PPA_temp,_, coverage_temp=run_DSE_cluster([[model_list[model_idx]]], [cluster_spec_list , Nocdse_BW[cluster_idx], nopdse_BW[cluster_idx], nopdse_idx[cluster_idx]], config_spec, False, None, None, directory_name, dse_count, modified_file)
        #import pdb;pdb.set_trace()
        for idx,item in enumerate(PPA_temp):
            PPA[idx][model_idx]=item[0]
        for idx,item in enumerate(coverage_temp):
            coverage[idx].append(item[0])
        #import pdb;pdb.set_trace()
        #clusters[cluster_idx]=cluster_spec_list.copy()
        #config[0][cluster_idx]=config_spec.copy()

    if cluster_len>1:
        plot_jacc(jacc_similarity, ['C'+str(i+1) for i in range(cluster_len)], [item.split('.')[1].upper() for item in model_list], directory_name, 'test', [list(cluster[0].nodes()) for cluster in clusters])
    #import pdb;pdb.set_trace() 
    #coverage=[node_coverage[max_cluster_idx[model_idx]][model_idx] for model_idx in range(model_len)]
    #coverage_edge=[edge_coverage[max_cluster_idx[model_idx]][model_idx] for model_idx in range(model_len)]
    #utli_node=[node_utli_cluster[max_cluster_idx[model_idx]][model_idx] for model_idx in range(model_len)]
    #util_edge=[edge_util_cluster[max_cluster_idx[model_idx]][model_idx] for model_idx in range(model_len)]

    return PPA, coverage, max_cluster_idx, cluster_split_dict
