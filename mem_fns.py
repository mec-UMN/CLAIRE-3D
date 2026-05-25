import os
import json
import pandas as pd
import math
import itertools
from optimize_fns import construct_ind_graphs_sub_fn
from Utils import cycles
from json_utils import *
import numpy
import matplotlib.pyplot as plt

def run_mem_dse(directory_name, cluster_common, cluster_custom, cluster_opts,  config_common,config_custom, config_opts):
    os.system("rm -r "+directory_name)
    os.system("mkdir "+directory_name)
    #import pdb;pdb.set_trace() 

    #Optimal/Libray synthesized dse
    memsize_opts=[]
    cluster_opts_memdse=[]
    for cluster_idx, cluster_opt in enumerate(cluster_opts):
        #import pdb;pdb.set_trace() 
        result=mem_dse(cluster_opt, config_opts[0][cluster_idx], directory_name)
        memsize_opts.append(result[0])
        cluster_opts_memdse.append(result[1])

    import pdb;pdb.set_trace() 

    #generic dse
    memsize_common, cluster_common_memdse=mem_dse(cluster_common, config_common[0][0], directory_name)

    memsize_custom=[]
    cluster_custom_memdse=[]
    #Custom dse
    for cluster_idx, cluster in enumerate(cluster_custom):
        result= mem_dse(cluster, config_custom[0][cluster_idx], directory_name)
        memsize_custom.append(result[0])
        cluster_custom_memdse.append(result[1])

    return  cluster_common_memdse,cluster_custom_memdse,cluster_opts_memdse, memsize_custom, memsize_common, memsize_opts

def mem_dse(cluster, config_compute, directory_name):
    model_list=cluster[1]
    print("run_mem", model_list)
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        data = json.load(f)
    mem_variables = {key: value for key, value in data.items() if key.startswith("MEM")}
    tech_node=data['tech_node_mem']
    area_scale_factor=data['area_scale_'+str(tech_node)+'nm']
    num_act=data['num_bits_act']
    ddr_variables = {key: value for key, value in data.items() if key.startswith("DDR")}
    #import pdb;pdb.set_trace() 
    access_Q=[]
    #access_Q=[[[0,0,0,0],[0,0,0,0],[0,0,0,0]]]
    psum=False
    buffer_sizes= buffer_inital(cluster[0], 0, None)
    #buffer_sizes=[1e6, 1e6, 1e6, 1e6]
    merged_df=read_mem_file("./Mem_Files/")

    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'params.json')
    write_json_file(config_compute, relative_path)
    L_compute=[]
    for model in model_list:
        res=data_volume_mem(model, cluster[0], num_act, config_compute)
        print(model)
        #import pdb;pdb.set_trace() 
        print("SRAM rd volume (in, out, wg, psum) in MB", {key: [item * 1e-6 / 8 for item in res[0][key]] for key in res[0]})
        print("SRAM wr volume (in, out, wg, psum) in MB", {key: [item * 1e-6 / 8 for item in res[1][key]] for key in res[1]})
        print("DRAM WR/rd volume (in, out, wg, psum) in MB",{key: [item * 1e-6 / 8 for item in res[2][key]] for key in res[2]})
        print("compute latency in ms", res[4]*1e-6)
        L_compute.append(res[4]*1e-6)
        #res=(1e6, 1e6, 1e6, True)
        #Q_wr_sramaccess, Q_rd_sramaccess, Q_rd_wr_ddraccess, psum=res
        access_Q.append(res)
        if res[3]:
            psum=True

    #import pdb;pdb.set_trace() 

    #mem_aval_combinations = merged_df[['NB', 'NW', 'CM']].drop_duplicates()
    combined_tuples={}
    NB_start_point=32
    for node_idx, node in enumerate(cluster[0].nodes()):
        filtered_combinations_list=[]
        for buff_idx, size in enumerate(buffer_sizes[node]): 
            #if buff_idx==2:
                #import pdb;pdb.set_trace() 
            filtered_combinations = merged_df[
                (merged_df['NB'] > NB_start_point) &  # NB > specified NB
                ((merged_df['NB'] * merged_df['NW'] * merged_df['CM']) >= size)  # NB * NW * CM > specified size
            ]
            # Step 2: Sort by closeness to the specified NB value
            target_NB = NB_start_point
            filtered_combinations = filtered_combinations.copy()
            filtered_combinations["NB_diff"] = abs(filtered_combinations["NB"] - target_NB)
            filtered_combinations = filtered_combinations.sort_values("NB_diff")

            # Step 3: Take the top 10 nearest configurations
            top_10_configs = filtered_combinations.head(3)

            # Optional: drop the NB_diff column if not needed
            top_10_configs = top_10_configs.drop(columns="NB_diff")
            #import pdb;pdb.set_trace() 
            filtered_combinations_list.append(top_10_configs.values.tolist())
        combined_tuples[node] = [tuple(combination) for combination in list(itertools.product(*filtered_combinations_list))]
        
        #import pdb;pdb.set_trace() 
    
    import pdb;pdb.set_trace() 
    # Area obtained in um^2
    #area+=area_mem(N_bits_arr, CM_fac_arr, N_words_arr, mem_variables, area_scale_factor, psum, config_compute, layer_name)
    #print("area: ", area)
    #range of configs - source from filtered excel file
    L_rd_list=[]
    L_wr_list=[]
    L_ddr_list=[]
    for model_idx, model in enumerate(model_list):
        L_rd_list.append({})
        L_wr_list.append({})
        L_ddr_list.append({})
        for node_idx, node in enumerate(cluster[0].nodes()):
            layers=construct_ind_graphs_sub_fn(model)[0]
            layer_names = list(set(layer['layer_type'].upper() for layer in layers))
            #import pdb;pdb.set_trace() 
            if node in layer_names:
                L_rd_list[model_idx][node]=[]
                L_wr_list[model_idx][node]=[]
                L_ddr_list[model_idx][node]=[]
                for tuple_idx, config_tuple in enumerate(combined_tuples[node]):
                    N_words_arr, N_bits_arr, CM_fac_arr = zip(*[(row[0:3]) for row in config_tuple])
                    N_words_arr, N_bits_arr, CM_fac_arr = map(list, [N_words_arr, N_bits_arr, CM_fac_arr])
                    #import pdb;pdb.set_trace() 
                    # Latency obtained is ns, energy in J
                    L_ddr, E_ddr=latency_ddr(access_Q[model_idx][2][node],ddr_variables, config_compute, node)
                    L_rd, L_wr, E_mem =latency_mem(N_bits_arr, CM_fac_arr, N_words_arr ,access_Q[model_idx][0][node],access_Q[model_idx][1][node], config_tuple, config_compute, node)
                    print("Evaluating number of words: ", N_words_arr, "number of bits: ", N_bits_arr, "column mux: ", CM_fac_arr)
                    print("model: ", model, "Node: ", node, "Latency DDR in ms:" , L_ddr , "Latency SRAM Read in ms:", L_rd,"Latency SRAM Write in ms:", L_wr,"Energy DDR in uJ:", E_ddr,"Energy SRAM in uJ:",E_mem)
                    L_rd_list[model_idx][node].append(L_rd)
                    L_wr_list[model_idx][node].append(L_wr)
                    L_ddr_list[model_idx][node].append(L_ddr)

    L_rd_total, L_wr_total, L_ddr_total=[], [], []
    for model_idx, model in enumerate(model_list):
        L_rd_total.append([sum(values) for values in zip(*L_rd_list[model_idx].values())])
        L_wr_total.append([sum(values) for values in zip(*L_wr_list[model_idx].values())])
        L_ddr_total.append([sum(values) for values in zip(*L_ddr_list[model_idx].values())])
        x = list(range(len(L_rd_total[model_idx])))

        plt.scatter(x, L_rd_total[model_idx], label='Data Points for model '+model)
        plt.axhline(y=L_compute[model_idx], color='r', linestyle='--', label=f'Compute Latency: {L_compute[model_idx]}')

        plt.xlabel('Config Number')
        plt.ylabel('Latency in ms')
        plt.title('Latency of SRAM read vs mem configs for model' + model)
        plt.legend()
        plt.grid(True)
        plt.savefig(directory_name+"Lread_sram_"+model+".png")
        plt.close()

        plt.scatter(x, L_wr_total[model_idx], label='Data Points for model '+model)
        plt.axhline(y=L_compute[model_idx], color='r', linestyle='--', label=f'Compute Latency: {L_compute[model_idx]}')

        plt.xlabel('Config Number')
        plt.ylabel('Latency in ms')
        plt.title('Latency of SRAM write vs mem configs for model' + model)
        plt.legend()
        plt.grid(True)
        plt.savefig(directory_name+"Lwrite_sram_"+model+".png")
        plt.close()

        plt.scatter(x, L_ddr_total[model_idx], label='Data Points for model '+model)
        plt.axhline(y=L_compute[model_idx], color='r', linestyle='--', label=f'Compute Latency: {L_compute[model_idx]}')

        plt.xlabel('Config Number')
        plt.ylabel('Latency in ms')
        plt.title('Latency of DDR vs mem configs for model' + model)
        plt.legend()
        plt.grid(True)
        plt.savefig(directory_name+"L_ddr_"+model+".png")
        plt.close()
    import pdb;pdb.set_trace() 
        
    return buffer_sizes, cluster

def area_mem(N_bits_arr,CM_fac_arr,N_words_arr , mem_variables, tech_node_scaling, psum, config_compute, layer_name):
    globals().update(mem_variables)
    area=0
    fac_mul_buff=config_compute["n_SA"] if layer_name.startswith("CONV2D")  or layer_name.startswith('LINEAR') else 1
    for idx, N_bits in enumerate(N_bits_arr):
        CM_fac=CM_fac_arr[idx]
        N_words=N_words_arr[idx]
        mul =2 if psum else 1 # to account for partial sum buffer
        width=MEM_W_cell*CM_fac*N_bits+ math.floor(CM_fac*(N_bits-1)/MEM_Val_CM)*MEM_Col_dec*MEM_W_dec+MEM_W_other
        height=MEM_H_cell*N_words/CM_fac+MEM_H_other
        area+=mul*width*height*fac_mul_buff/tech_node_scaling**2
    return area

def latency_mem(N_bits_arr, CM_fac_arr, N_words_arr , Q_wr_arr, Q_rd_arr, config, config_compute, layer_name):
    L_rd, L_wr, E=0,0,0
    for idx, N_bits in enumerate(N_bits_arr):
        CM_fac=CM_fac_arr[idx]
        N_words=N_words_arr[idx]
        Tcc, Tcq, Tcax, Tcdx, Tcwx,Tcmx, Tcqx, uW_MHz_Wr, uW_MHz_Rd=tuple(config[idx][-9:])
        #import pdb;pdb.set_trace() 
        hold_time=max(Tcax, Tcdx, Tcwx,Tcmx, Tcqx)
        L_rd_s=Tcc/2+hold_time+Tcq
        L_wr_s=Tcc/2+hold_time
        #import pdb;pdb.set_trace() 
        pJ_Wr_s=uW_MHz_Wr/Tcc*L_rd_s
        pJ_Rd_s=uW_MHz_Rd/Tcc*L_wr_s
        L_wr+=L_wr_s*math.ceil(Q_wr_arr[idx]/N_bits) #Units in ns
        L_rd+=L_rd_s*math.ceil(Q_rd_arr[idx]/N_bits) #Units in ns
        E+=pJ_Wr_s*math.ceil(Q_wr_arr[idx]/N_bits)+pJ_Rd_s*math.ceil(Q_rd_arr[idx]/N_bits) #Units in pJ
    E*=config_compute["n_SA"] if layer_name.startswith("CONV2D")  or layer_name.startswith('LINEAR') else 1
    return L_rd*1e-6, L_wr*1e-6, E*1e-6

def latency_ddr(Q_ddr,ddr_variables, config_compute, layer_name):
    globals().update(ddr_variables)
    #import pdb;pdb.set_trace() 
    mul_fac=config_compute['n_SA'] if layer_name.startswith("CONV2D")  or layer_name.startswith('LINEAR') else 1
    L_ddr=sum(Q_ddr)*mul_fac/DDR3_BW # Units in seconds
    E_ddr=DDR3_J_b*sum(Q_ddr)*mul_fac # units in J
    return L_ddr*1e3, E_ddr*1e6

def buffer_inital(cluster, mode, layer_name):
    #Q_in_max, Q_out_max, Q_wg_max is same for compute cycle of SA
    if mode:
        node=layer_name
        in_buf=float(cluster.nodes[node]['eighth_comment'].split(':')[1])
        out_buf=float(cluster.nodes[node]['seventh_comment'].split(':')[1])
        wg_buf=float(cluster.nodes[node]['ninth_comment'].split(':')[1])
        psum_buf=float(cluster.nodes[node]['seventh_comment'].split(':')[1]) 
        return in_buf, out_buf, wg_buf, psum_buf
    else:
        # max because this is used for comparing the max buffer size for a node with the max available buffer size for a bank
        in_buf, out_buf, wg_buf, psum_buf={},{},{},{}
        for node in cluster.nodes():
            in_buf[node]=float(cluster.nodes[node]['eighth_comment'].split(':')[1])
            out_buf[node]=float(cluster.nodes[node]['seventh_comment'].split(':')[1])
            wg_buf[node]=float(cluster.nodes[node]['ninth_comment'].split(':')[1])if node.startswith('CONV') or node.startswith('LINEAR') else 0 
            psum_buf[node]=float(cluster.nodes[node]['seventh_comment'].split(':')[1])if node.startswith('CONV') or node.startswith('LINEAR') else 0 
        dicts = [in_buf, out_buf, wg_buf, psum_buf]
        keys = dicts[0].keys()
        #import pdb;pdb.set_trace() 
        return {key: tuple(d[key] for d in dicts) for key in keys}
def data_volume_mem(model, cluster, num_act, config_compute):
    psum={}
    layers=construct_ind_graphs_sub_fn(model)[0]
    #update params with config
    #create graphs
    Q_wr_s_arr, Q_rd_s_arr, Q_rd_d_arr,Q_wr_d_arr={}, {}, {},{}
    total_compute_latency=0
    layer_names = list(set(layer['layer_type'].upper() for layer in layers))
    for  layer in layers:
        layer_name = (str.upper(layer['layer_type']))
        hw_size=buffer_inital(cluster, 1,layer_name )
        mul_fac=config_compute['n_SA'] if layer_name.startswith("CONV2D")  or layer_name.startswith('LINEAR') else 1
        hw_size_single_SA= [size/mul_fac for size in hw_size] # Buffer for one SA
        compute_latency=cycles(layer_name, 0, layer)[0]
        compute_cycles=compute_latency*1e-9/float(cluster.nodes[layer_name]["comment"].split(':')[1])
        if layer_name.startswith("CONV") or layer_name.startswith("LINEAR"):
            out_size=layer['out_channel/features']*numpy.prod(layer['out_size'])*num_act
            wg_size=layer['in_channel/features']*numpy.prod(layer['kernel_size'])*num_act if layer_name.startswith("CONV")  else layer['in_channel/features']*num_act #kernel_size in y-direction when mapped to systolic array
            if wg_size<hw_size[2]:
                #CASE 1: SA_x, n_SA sufficient 
                case_reuse=1
            else:
                #CASE 2: SA_x, n_SA not sufficient
                case_reuse=2
                psum=1
            wg_size*=layer['out_channel/features']
        else:
            case_reuse=3

        #temp=[x*compute_cycles for x in hw_size]
        temp=[x*compute_cycles for x in hw_size_single_SA]
        #print(hw_size, compute_cycles, wg_size, layer_name, out_size, case_reuse)
        #import pdb;pdb.set_trace() 
        if layer_name not in Q_rd_d_arr:
            Q_rd_d_arr[layer_name]=[0]*4
        if layer_name not in Q_rd_s_arr:
            Q_rd_s_arr[layer_name]=[0]*4
        if layer_name not in Q_wr_d_arr:
            Q_wr_d_arr[layer_name]=[0]*4
        if layer_name not in Q_wr_s_arr:
            Q_wr_s_arr[layer_name]=[0]*4
        match case_reuse:
            case 1:  
                temp[-1]=0
                Q_rd_s_arr[layer_name]= [x + y for x, y in zip(Q_rd_s_arr[layer_name], temp)]      #reads from input, output (DRAM write), weight buffer  during the compute cycle
                temp[2]= wg_size/mul_fac                                            #Weights fetch from DRAM to weight buffer
                Q_wr_s_arr[layer_name]=[x + y for x, y in zip(Q_wr_s_arr[layer_name], temp)]         #writes to input and output buffer during the compute cycle
                temp[1] = 0
                Q_rd_d_arr[layer_name]= [x + y for x, y in zip(Q_rd_d_arr[layer_name], temp)] #inputs and weights read from dram, weight reuse
            case 2:
                #partial buffer is included here
                psum[layer_name]=True
                Q_wr_s_arr[layer_name]= [x + y for x, y in zip(Q_wr_s_arr[layer_name], temp)]
                temp[1]+=out_size/mul_fac  #additional output buffer read while stroing to DRAM due to partial buffer
                Q_rd_s_arr[layer_name]=[x + y for x, y in zip(Q_rd_s_arr[layer_name], temp)] 
                temp[1:3]=[0, temp[2], 0]
                Q_rd_d_arr[layer_name]= [x + y for x, y in zip(Q_rd_d_arr[layer_name], temp)] 
            case 3:  
                temp[-2:] = [0, 0]
                Q_wr_s_arr[layer_name]= [x + y for x, y in zip(Q_wr_s_arr[layer_name], temp)]
                Q_rd_s_arr[layer_name]=[x + y for x, y in zip(Q_rd_s_arr[layer_name], temp)]
                temp[1] = 0
                Q_rd_d_arr[layer_name]= [x + y for x, y in zip(Q_rd_d_arr[layer_name], temp)]
        #if layer_name =="CONV2D":
            #import pdb;pdb.set_trace() 
        #print(Q_wr_s_arr, Q_rd_s_arr, Q_rd_d_arr)
        Q_wr_d_arr[layer_name][1]+= out_size/mul_fac
        total_compute_latency+=compute_latency
    Q_rd_wr_d_arr={}
    for layer_name in layer_names:
        Q_rd_wr_d_arr[layer_name]=[x + y for x, y in zip(Q_wr_d_arr[layer_name], Q_rd_d_arr[layer_name])]
    return Q_wr_s_arr, Q_rd_s_arr, Q_rd_wr_d_arr, psum, total_compute_latency

def read_mem_file(directory_name):
    df1 = pd.read_csv(directory_name+'group1.csv')
    df2 = pd.read_csv(directory_name+'group2.csv')
    df3 = pd.read_csv(directory_name+'group3.csv')

    merged_df = pd.concat([df1, df2, df3], ignore_index=True)
    merged_df = merged_df.loc[:, ~merged_df.columns.str.contains('ssg|ffg', case=False)]
    columns_to_rename = {
        'tt0p8v25c_0p8v_0v_0v.Tcc.Min (ns)': 'Tcc',      
        'tt0p8v25c_0p8v_0v_0v.Tcq.Max (ns)': 'Tcq',  
        'tt0p8v25c_0p8v_0v_0v.Tcax.Min (ns)': 'Tcax',
        'tt0p8v25c_0p8v_0v_0v.Tcdx.Min (ns)': 'Tcdx',
        'tt0p8v25c_0p8v_0v_0v.Tcwx.Min (ns)': 'Tcwx',
        'tt0p8v25c_0p8v_0v_0v.Tcmx.Min (ns)': 'Tcmx',
        'tt0p8v25c_0p8v_0v_0v.Tcqx.Min (ns)': 'Tcqx',
        'worst.tt0p8v25c_0p8v_0v_0v.power_dissipation_RM0_mode.WR (uW/MHz)': 'uW_MHz_Wr',
        'worst.tt0p8v25c_0p8v_0v_0v.power_dissipation_RM0_mode.RD (uW/MHz)': 'uW_MHz_Rd'
    }

    merged_df = merged_df.rename(columns=columns_to_rename)
    #for col in merged_df.columns:
        #print(col)
    fields_to_keep = ['NW', 'NB' , 'CM', 'BK', 'Tcc', 'Tcq', 'Tcax', 'Tcdx', 'Tcwx','Tcmx', 'Tcqx', 'uW_MHz_Wr', 'uW_MHz_Rd']  
    merged_df = merged_df[fields_to_keep]
    #print("Filtered DataFrame Headers:")
    #for col in merged_df.columns:
        #print(col)
    return merged_df
