import re
import copy
import math
import numpy
from collections import defaultdict
from prettytable import PrettyTable
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Utils import cycles, compute_ddr_latency
import time
import os
import json

def extract_info(line, result_prev, image_size, line_no, layer_no, model, count_prev, end_loop):
    #import pdb;pdb.set_trace() 
    if line==')':
        return {'loop':end_loop+1}
    # Regular expression pattern to extract information
    pattern1 = r"\((\d+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\), padding=\((\d+), (\d+)\)\)"
    pattern2 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\), padding=\((\d+), (\d+)\), bias=(.*?)\)"
    pattern3 = r"\((\d+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\), bias=(.*?)\)"
    pattern4 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\)\)"
    pattern5 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\), padding=\((\d+), (\d+)\)\)"
    pattern6 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\), bias=(.*?)\)"
    pattern7 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\), padding=\((\d+), (\d+)\), dilation=\((\d+), (\d+)\), bias=(.*?)\)"
    pattern8 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\), padding=\((\d+), (\d+)\), bias=(.*?)"
    pattern9 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\), bias=(.*?)"
    pattern10 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+), (\d+)\), stride=\((\d+), (\d+)\), padding=\((\d+), (\d+)\), groups=(\d+), bias=(.*?)\)"
    pattern11 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+),\), stride=\((\d+),\), padding=\((\d+),\)\)"
    pattern12 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+),\), stride=\((\d+),\), padding=\((\d+),\), bias=(.*?)\)"
    pattern13 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+),\), stride=\((\d+),\)\)"
    pattern14 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+),\), stride=\((\d+),\), padding=\((\d+),\), groups=(\d+)\)"
    pattern15 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+),\), stride=\((\d+),\), padding=\((\d+),\), dilation=\((\d+),\), groups=(\d+)\)"
    pattern16 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+),\), stride=\((\d+),\), bias=(.*?)\)"
    pattern17 = r"\((\w+)\): (\w+)\((\d+), (\d+), kernel_size=\((\d+),\), stride=\((\d+),\), groups=(\d+), bias=(.*?)\)"
    
    # re.match function to search for the pattern in the line
    match1 = re.match(pattern1, line)
    match2 = re.match(pattern2, line)
    match3 = re.match(pattern3, line)
    match4 = re.match(pattern4, line)
    match5 = re.match(pattern5, line)
    match6 = re.match(pattern6, line)
    match7 = re.match(pattern7, line)
    match8 = re.match(pattern8, line)
    match9 = re.match(pattern9, line)
    match10 = re.match(pattern10, line)
    match11 = re.match(pattern11, line)
    match12 = re.match(pattern12, line)
    match13 = re.match(pattern13, line)
    match14 = re.match(pattern14, line)
    match15 = re.match(pattern15, line)
    match16 = re.match(pattern16, line)
    match17 = re.match(pattern17, line)
    #print(line)
    #print(match1, match2, match3, match4, match5, match6, match7, match8, match9, match10)
    #import pdb;pdb.set_trace() 
    if match1 or match2 or match3 or match4 or match5 or match6 or match7 or match8 or match9 or match10 or match11 or match12 or match13 or match14 or match15 or match16 or match17:
        groups=None
        if match1 or match5:
            match = match1 or match5
            #print(match)
            
            index,layer_type, in_channel, out_channel, kernel_size_1, kernel_size_2, stride_1, stride_2, padding_1, padding_2 = match.groups()
        elif match2:
            match = match2
            #print(match)
            
            index,layer_type, in_channel, out_channel, kernel_size_1, kernel_size_2, stride_1, stride_2, padding_1, padding_2, bias= match.groups()
        elif match3 or match6:
            match = match3 or match6
            #print(match)
            
            index,layer_type, in_channel, out_channel, kernel_size_1, kernel_size_2, stride_1, stride_2, bias = match.groups()
        elif match4:
            match = match4
            #print(match)
            
            index,layer_type, in_channel, out_channel, kernel_size_1, kernel_size_2, stride_1, stride_2 = match.groups()
        elif match7:
            match=match7
            index,layer_type, in_channel, out_channel, kernel_size_1, kernel_size_2, stride_1, stride_2, padding_1, padding_2, dilation_1,dilation_2, bias= match.groups()
        elif match8:
            match = match8
            #print(match.groups())
            
            index,layer_type, in_channel, out_channel, kernel_size_1, kernel_size_2, stride_1, stride_2, padding_1, padding_2, params = match.groups()
        elif match9 :
            match = match9
            #print(match.groups())
            
            index,layer_type, in_channel, out_channel, kernel_size_1, kernel_size_2, stride_1, stride_2, params = match.groups()
        elif match10:
            match = match10
            #print(match.groups())
            index,layer_type, in_channel, out_channel, kernel_size_1, kernel_size_2, stride_1, stride_2, padding_1, padding_2, groups, params = match.groups()
        elif match11:
            match = match11
            #print(match.groups())
            
            index,layer_type, in_channel, out_channel, kernel_size_1, stride_1,padding_1 = match.groups()
            kernel_size_2=stride_2=padding_2=1
        elif match12:
            match = match12
            #print(match.groups())
            
            index,layer_type, in_channel, out_channel, kernel_size_1, stride_1,padding_1, bias = match.groups()
            kernel_size_2=stride_2=padding_2=1
            #import pdb;pdb.set_trace() 
        elif match13:
            match = match13
            #print(match.groups())
            
            index,layer_type, in_channel, out_channel, kernel_size_1, stride_1= match.groups()
            kernel_size_2=stride_2=padding_2=1
            #import pdb;pdb.set_trace() 
        elif match14:
            match = match14
            #print(match.groups())
            
            index,layer_type, in_channel, out_channel, kernel_size_1, stride_1,padding_1, groups= match.groups()
            kernel_size_2=stride_2=padding_2=1
            #import pdb;pdb.set_trace() 
        elif match15:
            match = match15
            #print(match.groups())
            
            index,layer_type, in_channel, out_channel, kernel_size_1, stride_1,padding_1, dilation_1, groups= match.groups()
            kernel_size_2=stride_2=padding_2=1
            #import pdb;pdb.set_trace() 
        elif match16:
            match = match16
            #print(match.groups())
            index,layer_type, in_channel, out_channel, kernel_size_1, stride_1, bias= match.groups()
            kernel_size_2=stride_2=padding_2=1
            #import pdb;pdb.set_trace() 
        else:
            match = match17
            #print(match.groups())
            index,layer_type, in_channel, out_channel, kernel_size_1, stride_1, groups, bias= match.groups()
            kernel_size_2=stride_2=padding_2=1
            #import pdb;pdb.set_trace() 
        #print("line_no:", line_no)
        #print(match.groups(), image_size)
        if model.startswith('inception') and line_no==313:
            #import pdb;pdb.set_trace()
            image_size[0]=17
            image_size[1]=17
            #import pdb;pdb.set_trace()
        image_size_prev= copy.copy(image_size)
        if stride_1:
            pad_1=padding_1 if 'padding_1' in locals() else 0
            pad_2=padding_2 if 'padding_2' in locals() else 0
            dil_1=int(dilation_1) if 'dilation_1' in locals() else 1
            maxpool_off=2*int(pad_1)- dil_1*(int(kernel_size_1)-1)-1
            for item in range(len(image_size)):
                #print(maxpool_off)
                #print(image_size[item])
                image_size[item]+=maxpool_off
                image_size[item]=math.floor(image_size[item]/int(stride_1))
                image_size[item]+=1
        if 'projection' in index:
            #import pdb;pdb.set_trace() 
            return None

        return {
            "line": line_no,
            "layer_no": layer_no,
            "layer_type": layer_type.lower(),
            "in_size": (image_size_prev[0], image_size_prev[1]),
            "out_size":(image_size[0], image_size[1]),
            "in_channel/features": int(in_channel),
            "out_channel/features": int(out_channel),
            "kernel_size": (int(kernel_size_1), int(kernel_size_2)),
            "stride": (int(stride_1), int(stride_2)),
            "padding": (int(pad_1), int(pad_2)),
            "groups": int(groups) if groups is not None else 1,
            "count": count_prev
        }
    else: 
        pattern1 = r"\((\w+)\): (\w+)\(output_size=\((\d+), (\d+)\)\)"
        pattern2 = r"\((\d+)\): (\w+)\(output_size=\((\d+), (\d+)\),(.*?)\)"
        match1 = re.match(pattern1, line)
        match2 = re.match(pattern2, line)
        groups=None
        if match1 or match2:
            match=match1 or match2
            #print(match.groups())
            if match1:
                index, layer_type, output_size_1, output_size_2 = match.groups()
            else:
                index, layer_type, output_size_1, output_size_2, param = match.groups()
            image_size_prev= copy.copy(image_size)
            image_size[0]=int(output_size_1)
            image_size[1]=int(output_size_2)
             
            return {
                "line": line_no,
                "layer_no": layer_no,
                "layer_type": layer_type.lower(),
                "in_size": (image_size_prev[0], image_size_prev[1]),
                "out_size":(image_size[0], image_size[1]),
                "in_channel/features": int(result_prev['out_channel/features']),
                "out_channel/features": int(result_prev['out_channel/features']),
                "kernel_size": '-',
                "stride":  '-',
                "padding":'-',
                "groups": int(groups) if groups is not None else 1,
                "count": count_prev
            }
        else:
            #print(line)
            pattern1 = r"\((\w+)\): (\w+)\((\d+), (.*?)\)"
            pattern2 = r"\((\d+)\): (\w+)\((.*?)\)"
            pattern3 = r"\((\w+)\): (\w+)\((.*?)\)"
            pattern4 = r"\((\w+)\): (\w+)\((\d+), (\d+)(.*?)\)"
            pattern5 = r"\((\d+-\d+)\): (\d+) x (.*?)"
            match1 = re.match(pattern1, line)
            match2 = re.match(pattern2, line)
            match3 = re.match(pattern3, line)
            match4 = re.match(pattern4, line)
            match5 = re.match(pattern5, line)

            in_channels=None
            out_channels=None
            count=None
            #if match3:
                #import pdb;pdb.set_trace() 
            #print(match1, match2, match3, match4, match5)
            #import pdb;pdb.set_trace() 
            ref_dict=['layer_type','in_features', 'out_features','kernel_size','stride', 'padding', 'p']
            if match1 or match2 or match3 or match4 or match5:
                match=match4 if match4 else match1 if match1 else match2 if match2 else match3 if match3 else match5 
                if match4:
                    index, layer_type, in_channels, out_channels, params_str = match.groups()
                elif match5:
                    #print(match.groups())
                    index, count, params_str = match.groups()
                elif not match1:
                    #print(match.groups())
                    index, layer_type, params_str = match.groups()
                else:
                    #print(match.groups())
                    index, layer_type, in_channels, params_str = match.groups()
                params = [param.strip() for param in params_str.split(',')]
                params_dict = {}
                
                #print(match.groups())
                #print(model)
                image_size_prev= copy.copy(image_size)
                #print(params_str,in_channels,image_size_prev)
                if "=" in params_str and not params_str.startswith(','):
                    for param in params:
                        key, value = param.split('=')
                        if key in ref_dict: 
                            #import pdb;pdb.set_trace() 
                            params_dict[key.strip()] = eval(value.strip())
                    if 'stride' in params_dict:
                        maxpool_off=2*params_dict['padding'] - 1*(params_dict['kernel_size']-1)-1
                        for item in range(len(image_size)):
                            image_size[item]+=maxpool_off
                            image_size[item]=math.floor(image_size[item]/params_dict['stride'])
                            image_size[item]+=1
                    if layer_type.startswith('Linear'):
                        if model == 'swint':
                            if index == 'reduction':
                                #import pdb;pdb.set_trace() 
                                image_size[0]= int(image_size_prev[0]/2)
                                image_size[1]= int(image_size_prev[1]/2)
                            else:
                                #import pdb;pdb.set_trace() 
                                image_size[0]= image_size_prev[0]
                                image_size[1]= image_size_prev[1]
                        elif np.prod(image_size_prev)>=128:
                            #print(image_size_prev)
                            image_size[0]= image_size_prev[0]
                            image_size[1]= image_size_prev[1]
                        else:
                            #import pdb;pdb.set_trace() 
                            image_size[0]= image_size_prev[0]=1
                            image_size[1]= image_size_prev[1]=1                   
                    elif layer_type.startswith('Flatten'):
                        out_channels=result_prev['out_channel/features']*numpy.prod(image_size_prev)
                        image_size[0]= image_size_prev[0]
                        image_size[1]= image_size_prev[1]
                elif not params_str:
                    if count is not None:
                        #if model.startswith("moe") and index =='0-7':
                            #count=2
                            #import pdb;pdb.set_trace() 
                        return {"count":count_prev*int(count)}

                
                
                if 'Norm' in layer_type or layer_type.startswith('Dropout') or layer_type.startswith('DropPath') or layer_type.startswith('Identity') or layer_type.startswith('MyLoss') or layer_type.startswith('Stochastic') or layer_type.startswith('BufferList') or layer_type.startswith('Dinov2LayerScale') or layer_type.startswith('Sequential'):
                    return None                   
                elif 'Embed' in layer_type:
                    return out_channels
                #elif 'GELUActi'in layer_type or 'NewGELU' in layer_type:
                elif layer_type.startswith('GELU') or layer_type.startswith('NewGELU'):
                    if model.startswith('graph'):
                        #image_size[0]= image_size_prev[0]=1
                        #image_size[1]= image_size_prev[1]=1
                        in_channels=out_channels=768
                    layer_type='GELU'+layer_type.split('_')[1] if '_' in layer_type else 'GELU'
                elif layer_type.startswith('Conv1D'):
                    #import pdb;pdb.set_trace()                                   
                    image_size[0]= image_size_prev[0]
                    image_size[1]= image_size_prev[1]
                    params_dict['kernel_size']=(1)
                    params_dict['stride']=(1)
                    params_dict['padding']=(0)
                    #import pdb;pdb.set_trace()                                   
                return {
                    "line": line_no,
                    "layer_no": layer_no,
                    "layer_type": layer_type.lower(),
                    "in_size": (int(image_size_prev[0]), int(image_size_prev[1])),
                    "out_size":(image_size[0], image_size[1]),
                    "in_channel/features": int(in_channels) if in_channels is not None else int(params_dict['in_features']) if 'in_features' in params_dict else int(result_prev['out_channel/features']),
                    "out_channel/features": int(out_channels) if out_channels is not None else int(params_dict['out_features'])if 'out_features' in params_dict else int(result_prev['out_channel/features']),
                    "kernel_size": int(params_dict['kernel_size']) if 'kernel_size' in params_dict else '-',
                    "stride":  int(params_dict['stride']) if 'stride' in params_dict else '-',
                    "padding": int(params_dict['padding']) if 'padding' in params_dict else '-',
                    "groups": int(params_dict['groups']) if 'groups' in params_dict else 1,
                    "count": count_prev
                }

def update_edge_info(grouped_layers, key, layer_idx, layers, comm_latency, incr):
    edge_info=grouped_layers[key][3] if len(grouped_layers[key])>3 else []
    for idx_edge in range(len(edge_info)): 
        if edge_info[idx_edge][0]==str.upper(layers[layer_idx+incr]['layer_type']):
            edge_info[idx_edge][1]=edge_info[idx_edge][1]+layers[layer_idx]['count']
            edge_info[idx_edge][2]=edge_info[idx_edge][2]+comm_latency*layers[layer_idx]['count']
            break
        if idx_edge==len(edge_info)-1:       
            grouped_layers[key][3].append([str.upper(layers[layer_idx+incr]['layer_type']),layers[layer_idx+incr]['count'], comm_latency])
    return grouped_layers

def group_layers(layers):
    start_time = time.time()
    motivation_figure_plot=False
    grouped_layers = defaultdict(list)
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'params.json')
    with open(relative_path) as f:
        params = json.load(f)

    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
    #import pdb;pdb.set_trace() 
    for idx, layer in enumerate(layers):
        start_time_layer = time.time()
        #key = (str.upper(layer['layer_type']),layer['in_size'], layer['out_size'], layer['in_channel/features'], layer['out_channel/features'],
        #       layer['kernel_size'], layer['stride'], layer['padding'])
        key = (str.upper(layer['layer_type']))
        compute_cycles, comm_latency, node_ds, util=cycles(key, 0, layer, params, PPA_config)
        _, _, node_ds_module, _ =cycles(key, 1, layer, params, PPA_config)
        ddr_latency_ns, _, _=compute_ddr_latency(key, layer,params, PPA_config)
        #print(f"Layer {idx} compute cycles: {compute_cycles}")
        #print(f"Layer {idx} grouping time taken cycles:", time.time() - start_time_layer)
        #print(comm_latency)
        #if comm_latency==0:
            #import pdb;pdb.set_trace()
        if key not in grouped_layers:
            grouped_layers[key].append([[layer['layer_no']], [node_ds], [node_ds_module], [ddr_latency_ns]])
            if motivation_figure_plot:
                grouped_layers[key].append([layers[idx]['count']])
            else:
                grouped_layers[key].append([compute_cycles])
            grouped_layers[key].append([util])
            if idx<len(layers)-1:
                #import pdb;pdb.set_trace()
                if layers[idx]['layer_type'][-1]=='1' or layers[idx]['layer_type'][-1]=='2':
                    if idx< len(layers)-2:
                        #import pdb;pdb.set_trace()
                        if layers[idx]['count']!=layers[idx+2]['count'] or layers[idx]['layer_type'][-1]==layers[idx+2]['layer_type'][-1]:
                            grouped_layers[key].append([[str.upper(layers[idx+2]['layer_type']),layers[idx+2]['count'], comm_latency]])
                        if layers[idx]['layer_type'][-1]=='1':
                            #if layers[idx+2]['layer_type'].startswith('conv') or layers[idx+2]['layer_type'].startswith('linear'):
                            if layers[idx]['count']!=layers[idx+3]['count'] or layers[idx]['layer_type'][-1]==layers[idx+3]['layer_type'][-1]:
                                grouped_layers[key][3].append([str.upper(layers[idx+3]['layer_type']),layers[idx+3]['count'], comm_latency])
                        else:
                            #if layers[idx+1]['layer_type'].startswith('conv') or layers[idx+1]['layer_type'].startswith('linear'):
                            if layers[idx]['count']!=layers[idx+1]['count'] or layers[idx]['layer_type'][-1]==layers[idx+1]['layer_type'][-1]:
                                grouped_layers[key][3].append([str.upper(layers[idx+1]['layer_type']),layers[idx+1]['count'], comm_latency])
                else:
                    grouped_layers[key].append([[str.upper(layers[idx+1]['layer_type']),layers[idx+1]['count'], comm_latency]])
        else:
            grouped_layers[key][0][0].append(layer['layer_no'])
            grouped_layers[key][0][3].append(ddr_latency_ns)
            if motivation_figure_plot:
                grouped_layers[key][1].append(layers[idx]['count'])
            else:
                grouped_layers[key][1].append(compute_cycles)
            grouped_layers[key][2].append(util)      
            #print(f"Layer {idx} grouping time taken nodes:", time.time() - start_time_layer)
            if idx<len(layers)-1:
                #edge_info=grouped_layers[key][3]                    
                if layers[idx]['layer_type'][-1]=='1' or layers[idx]['layer_type'][-1]=='2':
                    if idx< len(layers)-2:
                        if layers[idx]['count']!=layers[idx+2]['count'] or layers[idx]['layer_type'][-1]==layers[idx+2]['layer_type'][-1]:
                            grouped_layers=update_edge_info(grouped_layers, key, idx, layers, comm_latency, 2)
                        if layers[idx]['layer_type'][-1]=='1':
                            #if layers[idx+2]['layer_type'].startswith('conv') or layers[idx+2]['layer_type'].startswith('linear'):
                            if layers[idx]['count']!=layers[idx+3]['count'] or layers[idx]['layer_type'][-1]==layers[idx+3]['layer_type'][-1]:
                                grouped_layers=update_edge_info(grouped_layers, key, idx, layers, comm_latency, 3)
                        else:
                            #if layers[idx+1]['layer_type'].startswith('conv') or layers[idx+1]['layer_type'].startswith('linear'):
                            if layers[idx]['count']!=layers[idx+1]['count'] or layers[idx]['layer_type'][-1]==layers[idx+1]['layer_type'][-1]:
                                grouped_layers=update_edge_info(grouped_layers, key, idx, layers, comm_latency, 1)
                else:
                    grouped_layers=update_edge_info(grouped_layers, key, idx, layers, comm_latency, 1)
                    
                   
    #print(grouped_layers)
    #import pdb;pdb.set_trace() 
    #print("Grouped layers time taken:", time.time() - start_time)
    return grouped_layers

# Function to create a graph from grouped layers
def create_graph(grouped_layers, graph_image_name, model, plot):
    node_index= defaultdict(list)
    start_time = time.time()
    G = nx.DiGraph()
    #import pdb;pdb.set_trace() 
   # Step 1: Assign nodes
    for idx, (key, values) in enumerate(grouped_layers.items(), start=1):
        #label = key[0]+'_'+str(idx) if len(key[0])<10 else key[0][0:10]+'_'+str(idx)
        indices=20
        label = key if len(key)<indices else key[0:indices]
        wrap_text = "\n".join(label[i:i + indices] for i in range(0, len(label), indices))  
        comment = "L:"+f"{values[0][1][0]['L']}"   
        second_comment = "E:"+f"{values[0][1][0]['E']}"
        third_comment = "A:"+f"{values[0][1][0]['A']}" 
        fourth_comment = "A_mod:"+f"{values[0][2][0]['A']}" 
        sixth_comment = "HW_MACs:"+f"{values[0][1][0]['HW_MACs']}"
        seventh_comment = "Q_out_max:"+f"{values[0][1][0]['Q_out_max']}" 
        eighth_comment = "Q_in_max:"+f"{values[0][1][0]['Q_in_max']}" 
        ninth_comment = "Q_wg_max:"+f"{values[0][1][0]['Q_wg_max']}" 


        """
        comment = "layers:"+f"{len(values[0])}"  
        second_comment = f"{key[3]}" +"_"+f"{key[4]}"
        kernel=key[5][0] if isinstance(key[5], tuple) else key[5]
        stride=key[6][0] if isinstance(key[6], tuple)else key[6]
        second_comment+="_"+ f"{kernel}"+"_"+ f"{stride}" if key[5]!='-'else ''
        third_comment="_"+f"{key[1][0]}" +"_"+f"{key[2][0]}"
        G.add_node(idx, label=wrap_text, wrap=True, comment=comment, second_comment=second_comment, third_comment=third_comment)
        """
        #import pdb;pdb.set_trace() 
        #G.add_node(idx, label=str.upper(wrap_text), wrap=True, weight=sum(values[1][0]))
        #print(values[0][3])
        G.add_node(key, label=str.upper(wrap_text), wrap=True, weight=sum(values[1]),comment=comment, second_comment=second_comment, third_comment=third_comment, fourth_comment=fourth_comment, fifth_comment=sum(values[2]), sixth_comment=sixth_comment, seventh_comment=seventh_comment, eighth_comment=eighth_comment, ninth_comment=ninth_comment, ddr_latency=sum(values[0][3]))
        node_index[key]=idx
        #G.add_node(idx, label=str(key[0])+'_'+str(idx))
        #print(str(key[0])+'_'+str(idx))
        #print(second_comment)
        #import pdb;pdb.set_trace() 
    
    # Step 2: Construct edges
    for node, values in grouped_layers.items():
        #import pdb;pdb.set_trace() 
        #print(node, values)
        if len(values)>3:
            neighbors=values[3]
            for idx in range(len(neighbors)):
                #print(node, neighbors[idx])
                #if (node[-1]=='1' and neighbors[idx][0][-1]=='2') or (node[-1]=='2' and neighbors[idx][0][-1]=='1'):
                    #import pdb;pdb.set_trace()
                    #continue
                #import pdb;pdb.set_trace() 
                #G.add_edge(node, neighbors[idx][0], weight=neighbors[idx][1]*neighbors[idx][2])
                #G.add_edge(node, neighbors[idx][0], weight=neighbors[idx][1])
                G.add_edge(node, neighbors[idx][0], weight=neighbors[idx][2])
    
    #import pdb;pdb.set_trace() 
    #print("Nodes:", G.nodes(), "Edges:", G.edges())

    num_nodes = len(G)
    num_edges = len(G.edges())
    min_fig_size = 15
    edge_scale_factor = 1
    node_sizes = [len(str(G.nodes[node]['label'])) * 100 for node in G.nodes]  

    fig_size = (num_nodes * 0.4 + num_edges * edge_scale_factor)
    # Seed for reproducibility
    seed_value = 40
    np.random.seed(seed_value)
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    L_node={}
    for node in G.nodes():
        node_name=node[:-1] if node.endswith('1') or node.endswith('2') else node
        L_node[node_name] = max(L_node.get(node_name, 0), G.nodes[node]['weight'])
    weights_compute=sum([L_node[node] for node in L_node])
    #print(model,"Total Node Weight:", weights_compute, "Total Edge Weight:", sum(weights))
    weights_lat=weights_compute+sum(weights)
    #weights_lat=sum([G.nodes[node]['weight'] for node in G.nodes()])+sum(weights)
    A=sum([grouped_layers[node][0][1][0]['A'] for node in G.nodes()])*1e+6
    L=weights_lat*1e-6
    #E=sum([grouped_layers[node][0][1][0]['E']*G.nodes[node]['weight']*1e-9/grouped_layers[node][0][1][0]['L'] for node in G.nodes()])*1e+6
    E=sum([grouped_layers[node][0][1][0]['E']*G.nodes[node]['fifth_comment'] for node in G.nodes()])*1e+6
    #import pdb;pdb.set_trace() 
    
    if plot:
        # Graph with improved visibility
        #pos = nx.spring_layout(G, scale=3,seed=42, iterations=500)  
        #Graph with Kamada-Kawai layout and adjusted node sizes
        pos = nx.kamada_kawai_layout(G, scale=0.2) 
        #pos = nx.planar_layout(G, scale=1)  
        labels = {n: str(G.nodes[n]['label']) + '\n' + str(G.nodes[n]['weight']) for n in G.nodes}
        #node_sizes = [len(str(G.nodes[node]['label'])) * 200 for node in G.nodes]  
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        plt.subplots_adjust(top=0.97,bottom=0)  #
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_color='skyblue',
            font_weight='bold',
            node_size=1400,
            font_size=8,
            edgelist=edges,
            edge_color=weights,
            width=2.0, 
            arrowsize=20,  
            connectionstyle='arc3,rad=0.1',  
        )

        #node_labels = {node: label for node, label in nx.get_node_attributes(G, 'label').items()}
        #nx.draw_networkx_labels(G, pos, labels=node_labels, font_weight='bold', font_size=8, verticalalignment='center')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9, font_weight='bold',label_pos=0.2)
        """
        for node, (x, y) in pos.items():
            comment = G.nodes[node]['comment']
            plt.text(x + 0.0003*fig_size, y-0.0003*fig_size, comment, fontsize=9, verticalalignment='center',weight='bold', bbox=dict(boxstyle='round,pad=0.01', edgecolor='none', facecolor='white', alpha=0.7))
            second_comment = G.nodes[node]['second_comment']
            plt.text(x + 0.005, y-0.002, second_comment, fontsize=9, verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.2', edgecolor='none', facecolor='white', alpha=0.7))
        """
        #import pdb;pdb.set_trace() 
        for node in G.nodes():
            if G.has_edge(node, node):
                x, y = pos[node]
                plt.text(x +0.0005*fig_size, y-0.004*fig_size, str(edge_labels[(node, node)]), fontsize=9, color='red', verticalalignment='center', weight='bold', bbox=dict(boxstyle='round,pad=0.1', edgecolor='none', facecolor='white', alpha=0.7))
        ax.set_title("Layer Connections Graph -"+str.upper(model), fontsize=24, fontweight='bold')
        #plt.figtext(0, 0, "Comments for each node represents the number of layers of the specific node", wrap=True, horizontalalignment='left', fontsize=18,  color='darkblue')
        plt.figtext(0, 0.01*fig_size, "Total Latency: "+str(L)+" ms", wrap=True, horizontalalignment='left', fontsize=18,  color='darkblue')
        plt.figtext(0, 0, "Total Area: "+str(A)+" mm2", wrap=True, horizontalalignment='left', fontsize=18,  color='darkblue')
        plt.figtext(0, 0.02*fig_size, "Total Energy: "+str(E)+" uJ", wrap=True, horizontalalignment='left', fontsize=18,  color='darkblue')
        plt.savefig(graph_image_name)
        plt.show()
        plt.close()
    
    #print("Graph plotting time taken:", time.time() - start_time, "seconds", plot)
    
    return G,L,E,A
    #G.clear()
    #pos.clear()
    #plt.close()
    #import pdb;pdb.set_trace() 