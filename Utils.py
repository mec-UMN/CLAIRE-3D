import sys
import numpy
import math
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import copy
def print_file(text_file_name, table):
    original_stdout = sys.stdout
    try:
        #print(text_file_name)
        with open(text_file_name, 'w') as file:
            sys.stdout = file
            print(table)

    finally:
        sys.stdout = original_stdout
        
def round_up(v, decimals):
    factor = 10 ** decimals
    return math.ceil(v * factor) / factor

def cycles_old(layer_name, config, layer_info):
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'params.json')

    with open(relative_path) as f:
        params = json.load(f)

    relative_path = os.path.join(current_dir, 'PPA_config_old.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)

    if config==0:
        n_mod=params["no_modules"]
        n_SA=params["n_SA"]
        BA_outch=params["no_adders"]
    elif config==1:
        n_SA=1
        BA_outch=1
        n_mod=1

    #systolic array 256x64 for convolution producing 256x64 partial outputs in one entire operation of SA.
    SA_x=params["row_PE"]
    SA_y=params["col_PE"]
    Qreq_SA=SA_y*PPA_config["num_bits_PE"]*n_SA
    L_SA=PPA_config["L_PE"]*(SA_x-1+SA_y)
    Q_SA=SA_x*SA_y*PPA_config["num_bits_PE"]*n_SA
    T_SA=Q_SA/L_SA
    A_SA=PPA_config["A_PE"]*SA_x*SA_y*n_SA
    E_SA=PPA_config["E_PE"]*SA_x*SA_y*n_SA

    #For Linear layers-64 Butterfly structures of MACs giving 256 partial outputs in one entire operation of butterfly adder
    BA_pout=params["no_outs_adder"]
    Qreq_BA=PPA_config["num_bits_Adder"]*BA_outch*BA_pout
    L_BA=PPA_config["L_Adder"]*math.log(BA_pout)
    Q_BA=PPA_config["num_bits_Adder"]*BA_outch*BA_pout
    T_BA=Q_BA/L_BA 
    A_BA=PPA_config["A_Adder"]*BA_outch*BA_pout  
    E_BA=PPA_config["E_Adder"]*BA_outch*BA_pout   

    #Common variables for remaining modules 
    Qreq_mod=PPA_config["num_bits_act"]*n_mod
    Q_mod=PPA_config["num_bits_act"]*n_mod
    
    BW_noc=PPA_config["BW_noc"] #40 links, 8 bit per link in 1ns

    #BW_nop=params["BW_nop"] #assuming AIB2.0 -7680 Gbps
    #import pdb;pdb.set_trace() 
    
    bitwidth=8
    Qout=layer_info['out_channel/features']*numpy.prod(layer_info['out_size'])*PPA_config["num_bits_act"]
    node_ds={}
    if layer_name == "CONV2D" or layer_name == "CONVTRANSPOSE2D" or layer_name=="CONV1D":
        #import pdb;pdb.set_trace() 
        fac_outch=math.ceil(layer_info['out_channel/features']/SA_y)
        fac_inch=math.ceil(layer_info['in_channel/features']*numpy.prod(layer_info['kernel_size'])/SA_x)
        fac_inch*=math.ceil(numpy.prod(layer_info['out_size'])/(SA_x*n_SA))
        compute_cycles=fac_outch*fac_inch*L_SA
        node_ds["L"]=L_SA
        node_ds["T"]=T_SA
        node_ds["Q"]=Q_SA
        node_ds["Qreq"]=Qreq_SA
        node_ds["A"]=A_SA
        node_ds["E"]=E_SA
      

    elif layer_name == "LINEAR":
        fac_outch=math.ceil(layer_info['out_channel/features']/BA_outch)
        fac_inch=math.ceil(numpy.prod(layer_info['in_size'])*layer_info['in_channel/features']/BA_pout)
        compute_cycles=fac_outch*fac_inch*L_BA
        node_ds["L"]=L_BA
        node_ds["T"]=T_BA
        node_ds["Q"]=Q_BA
        node_ds["Qreq"]=Qreq_BA
        node_ds["A"]=A_BA
        node_ds["E"]=E_BA

    elif layer_name =="FLATTEN" or layer_name == "PERMUTE":
        compute_cycles=PPA_config["clk_period"]
        node_ds["L"]=PPA_config["clk_period"]
        node_ds["T"]=1/PPA_config["clk_period"]
        node_ds["Q"]="Qin"
        node_ds["Qreq"]="Qin"
        node_ds["A"]=1e-6
        node_ds["E"]=1e-16

    else: # Note: Formula for activation and pooling modules: RELU, RELU6, GELU, TANH, SILU, MAXPOOL2D, AVGPOOLS, ROIALIGN, LASTLEVELMAXPOOL
        compute_cycles=math.ceil(Qout/Q_mod)*PPA_config["L_"+layer_name]
        node_ds["L"]=PPA_config["L_"+layer_name]
        node_ds["T"]=Q_mod/PPA_config["L_"+layer_name]
        node_ds["Q"]=Q_mod
        node_ds["Qreq"]=Qreq_mod
        node_ds["A"]=PPA_config["A_"+layer_name]*n_mod
        node_ds["E"]=PPA_config["E_"+layer_name]*n_mod



    #print(compute_cycles-1)
    return int(compute_cycles*1e+9), math.ceil(Qout*bitwidth/BW_noc*1e+9), node_ds

def cycles_old_v2(layer_name, config, layer_info, params, PPA_config):

    if config==0:
        n_mod=params["no_modules"]
        #if layer_name.startswith("CONV") or layer_name=="LINEAR": 
            #n_SA=params["n_SA_"+layer_name]
        n_SA=params["n_SA"]
        #BA_outch=params["no_adders"]
    elif config==1:
        n_SA=1
        #BA_outch=1
        n_mod=1

    #if layer_name ends with 1 or 2, remove it
    if layer_name[-1]=='1' or layer_name[-1]=='2':
        layer_name=layer_name[:-1]
    #systolic array 256x64 for convolution producing 256x64 partial outputs in one entire operation of SA.
    #if layer_name.startswith("CONV") or layer_name=="LINEAR":
    SA_x=params["row_PE"]
    SA_y=params["col_PE"] if "col_PE" in params else params["row_PE"]
    L_SA=PPA_config["L_PE"]*(SA_x-1+SA_y)
    A_SA=(PPA_config["A_PE"]*(SA_x*SA_y-SA_y)+PPA_config["A_PE_toprow"]*SA_y)*n_SA
    P_SA=(PPA_config["P_PE"]*(SA_x*SA_y-SA_y)+PPA_config["P_PE_toprow"]*SA_y)*n_SA
    #E_SA=P_SA*L_SA
    E_PE=PPA_config["L_PE"]*PPA_config["P_PE"]

    """
    #For Linear layers-64 Butterfly structures of MACs giving 256 partial outputs in one entire operation of butterfly adder
    BA_pout=params["no_outs_adder"]
    L_BA=PPA_config["L_Adder"]*math.log(BA_pout)
    A_BA=PPA_config["A_Adder"]*BA_outch*BA_pout*math.log(BA_pout)  
    E_BA=PPA_config["E_Adder"]*BA_outch*BA_pout*math.log(BA_pout)   
    """
    #Common variables for remaining modules 
    Q_mod=PPA_config["num_bits_act"]*n_mod
    
    BW_noc=PPA_config["BW_noc"] #40 links, 8 bit per link in 1ns

    #BW_nop=params["BW_nop"] #assuming AIB2.0 -7680 Gbps
    #import pdb;pdb.set_trace() 
    
    #bitwidth=8
    Qout=layer_info['out_channel/features']*numpy.prod(layer_info['out_size'])*PPA_config["num_bits_act"]
    node_ds={}
    if layer_name == "CONV2D" or layer_name == "CONVTRANSPOSE2D" or layer_name=="CONV1D"  or layer_name== "CONVTRANSPOSE1D":
        #import pdb;pdb.set_trace() 
        fac_outch=math.ceil(layer_info['out_channel/features']/SA_y)
        fac_inch=math.ceil(layer_info['in_channel/features']*numpy.prod(layer_info['kernel_size'])*numpy.prod(layer_info['out_size'])/layer_info['groups']/(SA_x*n_SA))
        E_util=layer_info['in_channel/features']*numpy.prod(layer_info['kernel_size'])*layer_info['out_channel/features']*numpy.prod(layer_info['out_size'])*layer_info['count']/layer_info['groups']
        compute_cycles=fac_outch*fac_inch*L_SA*layer_info['count']
        node_ds["L"]=L_SA
        node_ds["A"]=A_SA
        node_ds["E"]=E_PE
        node_ds["HW_MACs"]=n_SA*SA_x*SA_y
        node_ds["Q_out_max"]=n_SA*SA_y*PPA_config["num_bits_act"]
        node_ds["Q_in_max"]=n_SA*SA_x*PPA_config["num_bits_act"]
        node_ds["Q_wg_max"]=n_SA*SA_x*SA_y*PPA_config["num_bits_act"]


    elif layer_name == "LINEAR":
        #import pdb;pdb.set_trace() 
        n_SA_dup=math.ceil(layer_info['count']/n_SA)    #Number of col of SAs the experts needs to be duplicated onto 
        fac_outch=math.ceil(layer_info['out_channel/features']/(SA_y))*n_SA_dup #Number of times SAs are computed due to number of output channels and number of experts
        max_n_SA_kernpart=math.floor(n_SA/layer_info['count']) if n_SA_dup<=1 else 1 #Max Number of row of SAs the kernel can be split
        if numpy.prod(layer_info['in_size'])>=128:
            fac_inch=math.ceil(layer_info['in_channel/features']/SA_x)*numpy.prod(layer_info['in_size'])
        else:
            fac_inch=math.ceil(numpy.prod(layer_info['in_size'])*layer_info['in_channel/features']/SA_x)
        fac_inch=math.ceil(fac_inch/max_n_SA_kernpart)
        E_util=layer_info['in_channel/features']*numpy.prod(layer_info['in_size'])*layer_info['out_channel/features']*layer_info['count']
        #print(layer_info)
        #print("fac_outch, fac_inch, L_SA:", fac_outch, fac_inch, L_SA)
        compute_cycles=fac_outch*fac_inch*L_SA
        node_ds["L"]=L_SA
        node_ds["A"]=A_SA
        node_ds["E"]=E_PE
        node_ds["HW_MACs"]=n_SA*SA_x*SA_y
        node_ds["Q_out_max"]=n_SA*SA_y*PPA_config["num_bits_act"]
        node_ds["Q_in_max"]=n_SA*SA_x*PPA_config["num_bits_act"]
        node_ds["Q_wg_max"]=n_SA*SA_x*SA_y*PPA_config["num_bits_act"]



    elif layer_name =="FLATTEN" or layer_name == "PERMUTE"  or layer_name=="UPSAMPLER" :
        compute_cycles=PPA_config["clk_period"]*layer_info['count']
        node_ds["L"]=PPA_config["clk_period"]
        node_ds["A"]=1e-10
        node_ds["E"]=1e-15
        E_util=compute_cycles
        node_ds["HW_MACs"]=0
        node_ds["Q_out_max"]=0
        node_ds["Q_in_max"]=0
        node_ds["Q_wg_max"]=0

    else: # Note: Formula for activation and pooling modules: RELU, RELU6, GELU, TANH, SILU, MAXPOOL2D, AVGPOOLS, ROIALIGN, LASTLEVELMAXPOOL
        compute_cycles=math.ceil(Qout/Q_mod)*PPA_config["L_"+layer_name]*layer_info['count']
        node_ds["L"]=PPA_config["L_"+layer_name]
        node_ds["A"]=PPA_config["A_"+layer_name]*n_mod
        node_ds["E"]=PPA_config["E_"+layer_name]*n_mod
        E_util=compute_cycles
        node_ds["HW_MACs"]=0
        node_ds["Q_out_max"]=Q_mod
        node_ds["Q_in_max"]=Q_mod
        node_ds["Q_wg_max"]=0

    #print(compute_cycles-1)
    #return int(compute_cycles*1e+9), math.ceil(Qout*bitwidth/BW_noc*1e+9), node_ds, E_util
    return int(compute_cycles*1e+9), Qout/BW_noc*1e+9, node_ds, E_util

def cycles(layer_name, config, layer_info, params, PPA_config):  # noqa: C901
    if config == 0:
        n_mod_act = params["no_modules_act"]
        n_mod_pool = params["no_modules_pool"]
        n_SA  = params["n_SA"]
    elif config == 1:
        n_SA  = 1
        n_mod_act, n_mod_pool = 1, 1
 
    if layer_name[-1] == '1' or layer_name[-1] == '2':
        layer_name = layer_name[:-1]
 
    SA_x = params["row_PE"]
    SA_y = params["col_PE"] if "col_PE" in params else params["row_PE"]
    L_SA = PPA_config["L_PE"] * (SA_x - 1 + SA_y)
    A_SA = (PPA_config["A_PE"] * (SA_x * SA_y - SA_y) +
            PPA_config["A_PE_toprow"] * SA_y) * n_SA
    E_PE = PPA_config["L_PE"] * PPA_config["P_PE"]
 
    Q_mod_act  = PPA_config["num_bits_act"] * n_mod_act
    Q_mod_pool = PPA_config["num_bits_act"] * n_mod_pool
    BW_noc = PPA_config["BW_noc"]
 
    Qout = (layer_info['out_channel/features']
            * numpy.prod(layer_info['out_size'])*layer_info['count']
            * PPA_config["num_bits_act"])
    node_ds = {}
 
    def _single_sa_latency(M, K, N):
        tiles_M   = math.ceil(M / SA_x)
        tiles_K   = math.ceil(K / SA_x)
        tiles_N   = math.ceil(N / SA_y)
        #num_tiles = tiles_M * tiles_K * tiles_N
        U         = K / (SA_x * tiles_K)
        Fm, Fn    = tiles_M, tiles_N
        rm = ((M - 1) % SA_x) + 1
        rn = ((N - 1) % SA_y) + 1
        nzfr_num = (M + N - 1
                    + (Fm - 1) * (Fn - 1) * (SA_x + SA_y - 1)
                    + (Fm - 1) * (rn - 1)
                    + (Fn - 1) * (rm - 1))
        nzfr_den = Fm * Fn * (SA_x + SA_y - 1)
        NZFR = nzfr_num / nzfr_den
        #print(f"M: {M}, K: {K}, N: {N}, tiles_M: {tiles_M}, tiles_K: {tiles_K}, tiles_N: {tiles_N}, U: {U}, Fm: {Fm}, Fn: {Fn}, rm: {rm}, rn: {rn}, nzfr_num: {nzfr_num}, nzfr_den: {nzfr_den}, NZFR: {NZFR}")
        #print(f"SA_x: {SA_x}, SA_y: {SA_y}, Latency: {(SA_x * U + (SA_x + SA_y + 1) * NZFR + SA_x + 1)}")
        return (SA_x * U + (SA_x + SA_y + 1) * NZFR + SA_x + 1)
 
    def _multi_sa_latency(M, K, N):
        N_tilde  = min(N, SA_y * math.ceil(N / (SA_y * n_SA)))
        n_N      = math.ceil(N / N_tilde)
        n_M      = n_SA - n_N if n_SA >= 2 * n_N else 1
        M_tilde  = min(M, SA_x * math.ceil(M / (SA_x * n_M)))
        tiles_np = (math.ceil(M / (SA_x * n_M))
                    * math.ceil(K / SA_x)
                    * math.ceil(N / (SA_y * n_N)))
        #import pdb; pdb.set_trace()
        #print(f"M_tilde: {M_tilde}, K: {K}, N_tilde: {N_tilde}, tiles_np: {tiles_np}")
        return round(tiles_np * _single_sa_latency(M_tilde, K, N_tilde))
 
    def _compute_latency(M, K, N):
        return (_multi_sa_latency(M, K, N))
 
    if layer_name in ("CONV2D", "CONVTRANSPOSE2D", "CONV1D", "CONVTRANSPOSE1D"):
        M = layer_info['out_channel/features']
        K = int(layer_info['in_channel/features']
                * numpy.prod(layer_info['kernel_size'])
                / layer_info['groups'])
        N = numpy.prod(layer_info['out_size'])
        #Assert error if any of M, K, N is 0
        if M == 0 or K == 0 or N == 0:
            print(f"Layer info: {layer_info}")
            raise ValueError("Invalid layer dimensions")
        compute_cycles  = _compute_latency(M, K, N) * layer_info['count']*PPA_config["L_PE"]
        E_util = (layer_info['in_channel/features']
                  * numpy.prod(layer_info['kernel_size'])
                  * layer_info['out_channel/features']
                  * numpy.prod(layer_info['out_size'])
                  * layer_info['count']
                  / layer_info['groups'])
        node_ds.update({
            "L"         : PPA_config["L_PE"],
            "A"         : A_SA,
            "E"         : E_PE,
            "HW_MACs"   : n_SA * SA_x * SA_y,
            "Q_out_max" : n_SA * SA_y * PPA_config["num_bits_act"],
            "Q_in_max"  : n_SA * SA_x * PPA_config["num_bits_act"],
            "Q_wg_max"  : n_SA * SA_x * SA_y * PPA_config["num_bits_act"],
        })
    elif layer_name == "LINEAR":
        M = layer_info['out_channel/features']
        K = layer_info['in_channel/features']
        N = int(numpy.prod(layer_info['in_size']))
        compute_cycles = _compute_latency(M, K, N) * layer_info['count']*PPA_config["L_PE"]
        E_util = (layer_info['in_channel/features']
                  * numpy.prod(layer_info['in_size'])
                  * layer_info['out_channel/features']
                  * layer_info['count'])
        node_ds.update({
            "L"         : PPA_config["L_PE"],
            "A"         : A_SA,
            "E"         : E_PE,
            "HW_MACs"   : n_SA * SA_x * SA_y,
            "Q_out_max" : n_SA * SA_y * PPA_config["num_bits_act"],
            "Q_in_max"  : n_SA * SA_x * PPA_config["num_bits_act"],
            "Q_wg_max"  : n_SA * SA_x * SA_y * PPA_config["num_bits_act"],
        })
    elif layer_name in ("FLATTEN", "PERMUTE", "UPSAMPLER"):
        compute_cycles = layer_info['count'] * PPA_config["clk_period"]
        E_util = layer_info['count']
        node_ds.update({
            "L"         : PPA_config["clk_period"],
            "A"         : 1e-10,
            "E"         : 1e-15,
            "HW_MACs"   : 0,
            "Q_out_max" : 0,
            "Q_in_max"  : 0,
            "Q_wg_max"  : 0,
        })
    else:
        n_mod = n_mod_pool if layer_name in ("MAXPOOL2D", "AVGPOOL2D", "ROIALIGN", "LASTLEVELMAXPOOL", "ADAPTIVEAVGPOOL2D", "MOBILEBERTPOOLER") else n_mod_act
        Q_mod = Q_mod_pool if layer_name in ("MAXPOOL2D", "AVGPOOL2D", "ROIALIGN", "LASTLEVELMAXPOOL", "ADAPTIVEAVGPOOL2D", "MOBILEBERTPOOLER") else Q_mod_act
        compute_cycles = (math.ceil(Qout / Q_mod)
                          * PPA_config["L_" + layer_name]
                          * layer_info['count'])
        E_util = compute_cycles
        node_ds.update({
            "L"         : PPA_config["L_" + layer_name],
            "A"         : PPA_config["A_" + layer_name] * n_mod,
            "E"         : PPA_config["E_" + layer_name] * n_mod,
            "HW_MACs"   : 0,
            "Q_out_max" : Q_mod,
            "Q_in_max"  : Q_mod,
            "Q_wg_max"  : 0,
        })
    #import pdb; pdb.set_trace()
    #print(f"Layer: {layer_name},Compute Cycles: {compute_cycles}")
    return int(compute_cycles * 1e9), Qout / BW_noc * 1e9, node_ds, E_util

def compute_ddr_latency(layer_name, layer_info, params, PPA_config):
    ddr_wrap_size_r_in = layer_info['in_channel/features']*numpy.prod(layer_info['in_size'])*PPA_config["num_bits_act"]*layer_info['count']
    ddr_reuse_mul_r = 1
    #print(layer_name, layer_info)
    if layer_name.startswith("CONV"):
        ddr_wrap_size_r_wg = numpy.prod(layer_info['kernel_size'])*layer_info['in_channel/features']*layer_info['out_channel/features'] * PPA_config["num_bits_act"]*layer_info['count']/layer_info['groups']
        ddr_reuse_mul_r *= numpy.prod(layer_info['out_size']) if ddr_wrap_size_r_wg > params["col_PE"]*params["n_SA"]*PPA_config["num_bits_act"] else 1
        #print(layer_name, "ddr_wrap_size_r_wg", ddr_wrap_size_r_wg, "ddr_reuse_mul_r", ddr_reuse_mul_r, "params", params)
    elif layer_name.startswith("LINEAR"):
        ddr_wrap_size_r_wg = layer_info['out_channel/features']*layer_info['in_channel/features']*PPA_config["num_bits_act"]*layer_info['count']
    N_raccess_rd= math.ceil(ddr_wrap_size_r_in/PPA_config["ddr_bus_width_burst"])
    if layer_name.startswith("CONV") or layer_name.startswith("LINEAR"):
        N_raccess_rd+= math.ceil(ddr_wrap_size_r_wg/PPA_config["ddr_bus_width_burst"])*ddr_reuse_mul_r
    latency_rd= N_raccess_rd*PPA_config["ddr_row_latency_burst"]
    latency_rd = latency_rd + math.ceil(latency_rd/PPA_config["ddr_row_latency_burst"]/PPA_config["ddr_no_cols_per_row"])* PPA_config["row activation_latency"]
    latency_rd = latency_rd + math.floor(latency_rd/PPA_config["ddr_refresh_window"])*(PPA_config["refresh_latency_read"]-2*PPA_config["row activation_latency"])
    latency_rd = latency_rd/ PPA_config["ddr_clock_frequency"]
    #write
    ddr_wrap_size_w = layer_info['out_channel/features']*numpy.prod(layer_info['out_size'])*PPA_config["num_bits_act"]*layer_info['count']
    ddr_reuse_mul_w = 1
    N_raccess_wr=math.ceil(ddr_wrap_size_w/PPA_config["ddr_bus_width_burst"])*ddr_reuse_mul_w
    latency_wr=N_raccess_wr*PPA_config["ddr_row_latency_burst"]
    latency_wr = latency_wr + math.ceil(latency_wr/PPA_config["ddr_row_latency_burst"]/PPA_config["ddr_no_cols_per_row"])* PPA_config["row activation_latency"]
    latency_wr = latency_wr + math.floor(latency_wr/PPA_config["ddr_refresh_window"])*(PPA_config["refresh_latency_write"]-2*PPA_config["row activation_latency"])
    latency_wr = latency_wr/ PPA_config["ddr_clock_frequency"]
    #print(layer_name, "latency_rd+latency_wr", latency_rd+latency_wr)
    return (latency_rd+latency_wr)*1e9, N_raccess_rd+N_raccess_wr, PPA_config["ddr_bus_width_burst"]

def update_nop_width(weight, mode, noc_channel_BW, nop_width, PPA_config):
    
    tech_node=PPA_config['tech_node_'+mode]
    #deepscale tool - https://sourceforge.net/projects/deepscaletool/
    #nrz-8 lanes - https://ieeexplore.ieee.org/document/9366048/
    #aib-1 channel
    #grs - 16 links - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6487789
    #Assuming chiplets are placed at a distance of 1 mm to calculate Cload
    # factor 2 is considered as two IOs needs to be passed to communicate between two chiplets
    type_connec='3d' if mode=='ucie3d' else '2d'
    #print("type_connec", type_connec)
    L_IO=2*(PPA_config['Lcons_IO_'+type_connec]+PPA_config['Lfac_IO_'+type_connec]*PPA_config['Cload_IO_'+type_connec]*1e12)
    BW_IO=nop_width*PPA_config['N_IOs_signal_'+mode]/L_IO if L_IO>0 else float('inf')
    E_IO=(PPA_config['Cpin_IO_'+type_connec]+PPA_config['Cload_IO_'+type_connec])*(PPA_config['VddIO_'+mode])**2
    BW_nop=PPA_config['BW_'+mode]*nop_width
    J_b_nop=PPA_config['J_b_'+mode]/PPA_config['energy_scale_'+str(tech_node)+'nm']

    #Nop_width=PPA_config['BW_noc']
    Nop_width=BW_nop
    if noc_channel_BW:
        BW_noc=noc_channel_BW
    else:
        BW_noc=PPA_config['BW_noc']
    #E_comm=weight*PPA_config['BW_noc']*(PPA_config['J_b_router'])*1e-9
    
    if weight is not None:
        E_comm=weight*PPA_config['BW_noc']*(J_b_nop+PPA_config['J_b_router'])*1e-9
        E_IO*=weight*PPA_config['BW_noc']*1e-9
    else:
        E_comm=None
    #print("nop width in Gbps", Nop_width*1e-9)
    #print("noc width in Gbps", PPA_config['BW_noc']*1e-9)
    #print("E_comm", E_comm)
    return BW_nop, PPA_config['BW_noc'], E_comm, BW_IO, E_IO

def beachhead(mode, nop_width, PPA_config):
    type_connec='3d' if mode=='ucie3d' else '2d'
    BW_nop=PPA_config['BW_'+mode]*nop_width
    if type_connec=='3d':
        BW_m2_nop=PPA_config['BW_m2_'+mode]
        area=BW_nop/BW_m2_nop
        length=math.sqrt(area)
    else:
        BW_m_nop=PPA_config['BW_m_'+mode]
        length=BW_nop/BW_m_nop
    return BW_nop, length


def width_nop(mode, BW_nop):
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
    nop_width=BW_nop/PPA_config['BW_'+mode]
    return nop_width

def width_noc(BW_noc):
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
    noc_width=BW_noc/(PPA_config['num_bits_per_link_noc']*PPA_config["Throughput_noc_router_max"])
    return noc_width

def area_io(mode, nop_width):
    A_IO=0
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)

    type_connec='3d' if mode=='ucie3d' else '2d'
    A_IO=(PPA_config['area_io_'+type_connec]+PPA_config['area_esd_'+type_connec])*nop_width*PPA_config['N_IOs_'+mode]
    A_IO_mod=PPA_config['area_io_'+type_connec]+PPA_config['area_esd_'+type_connec]
    n_pins=PPA_config['N_IOs_'+mode]*nop_width

    return A_IO, A_IO_mod, n_pins

def area_nop(mode,nop_width):
    A_nop=0
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
    
    A_nop=PPA_config['area_'+mode]*nop_width
    A_nop_mod=PPA_config['area_'+mode]
    return A_nop, A_nop_mod

def update_noc_width(graph, edge, channel_BW):
    current_dir = os.path.dirname(__file__)
    relative_path = os.path.join(current_dir, 'PPA_config.json')
    with open(relative_path) as f:
        PPA_config = json.load(f)
    if channel_BW:
        BW_noc=channel_BW
    else:
        BW_noc=PPA_config['BW_noc']
    E_comm=graph.edges[edge]['weight']*PPA_config['BW_noc']*(PPA_config['J_b_router'])*1e-9
    #print("E_comm", E_comm)
    return BW_noc, PPA_config['BW_noc'], E_comm

def closest_rectangle(n):
    closest_x, closest_y = 1, n
    min_diff = abs(closest_x - closest_y)

    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:  
            y = n // x
            diff = abs(x - y)
            if diff < min_diff:
                closest_x, closest_y = x, y
                min_diff = diff

    return closest_x, closest_y

def grid_lego(A):
    #A is in mm2
    edge_length=math.sqrt(A)
    A=(math.ceil(edge_length*2)/2)**2 #Design Chiplets with edge size of 0.5mm per grid
    return A

def plot_jacc(jacc_similarity, xticks, yticks, directory_name, file, cluster_nodes):
    #import pdb;pdb.set_trace() 
    jacc_similarity_up=[copy.copy(row) for row in jacc_similarity]
    if file=='train':
        for i in range(len(jacc_similarity_up)):
            for j in range(i+1,len(jacc_similarity_up)):
                jacc_similarity_up[i][j] = None  # Set upper triangle to None

    # Format the annotations to show up to 2 decimal points
    annot_matrix = [[None if value is None else round(value, 2) for value in row] for row in jacc_similarity_up]

    plt.figure(figsize=(8, 8))
    sns.heatmap(annot_matrix, annot=True, cmap='Blues', square=True, cbar_kws={'label': 'Jaccard Similarity'},  xticklabels=xticks,yticklabels=yticks,  )
    word='Training'if file=='train' else'Test'
    plt.title(word +' Phase Jaccard Similarity Matrix')
    plt.xlabel('Training set AI Models' if file=='train' else 'Config')
    plt.ylabel( word +' set AI Models')
    plt.savefig(directory_name+f'jac_'+file+f'.png')
    plt.show()
    plt.close()

    if file=='test':
        for row in cluster_nodes:
            with open(directory_name+"./jacc_comment.txt", 'a') as file_1:
                file_1.write(f"{row}\n")