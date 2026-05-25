from optimize_fns import *
from table_utils import *
from mem_fns import *
from noc_fns import *
import time
import os 
import pickle

os.system("rm -rf __pycache__")
#input_mode=0 #0 similar few models
#input_mode=1 #1 diverse few models
input_mode=2 #2 diverse many models

dse_arr=[1,1,1,1,1]
dse_idx=0
dse_arr_test=[1,1,1,1,1]
dse_idx_test=0
opt=True
nop=True
plot_stats=True
test=True
pre_trained=False
dse_count=3
#param_opt_mode="EDAP"
param_opt_mode="high_performance"
#param_opt_mode="compact"
#param_opt_mode="cost"
split = True
#num_opt_clusters=3 # Static: User-defined
num_opt_clusters=None #dynamic
fac_split=0.8
if input_mode ==0: #Training set - 4 similar models
    model_list = ['vision.vgg16', 'vision.resnet18', 'vision.alexnet', 'vision.densenet121']
elif  input_mode ==1: #Training set - 4 diverse models
    model_list =['hugging.bert', 'vision.vgg16', 'model.peanutRCNN', 'vision.swint']
elif input_mode ==2: #Training set - large set
    model_list = ['huggingllm.bert','huggingllm.moe', 'hugging.whisper', 'huggingllm.gpt','huggingllm.graph','hugging.dpt', 'hugging.vit', 
                'huggingllm.llama', 'hugging.ast', 'hugging.detr', 'hugging.dino','vision.vgg16', 'vision.resnet18', 
                'vision.alexnet', 'vision.densenet121', 'model.peanutRCNN', 'vision.resnet50', 'vision.mobilenetv2', 'vision.swint']
    model_list += ['vision.squeezenet', 'sr.edsr',  'hugging.dinosmall', 'hugging.chronos', 'hugging.mobilevit', 'audio.convtasnet', 'vision.effnet', 'hugging.t5', 'huggingllm.mobilellm', 'huggingllm.opt', 'huggingllm.gptneo', 'huggingllm.distilbert', 'huggingllm.distilroberta', 'vision.vgg11', 'vision.inception']
    fac_split=0.7
file_name='Params/params_model_' + str(input_mode) +'.json'

no_model=len(model_list)
if not split:
    training_set=model_list
else:
    train_len=int(fac_split*no_model)
    training_set, test_set=split_models(model_list,train_len, file_name)
    test_len=len(test_set)
    train_len=len(training_set)

print("Training set", [item.split('.')[1]for item in training_set])
if split:
    print("Test set", [item.split('.')[1]for item in test_set])
#Construct Individual graph on NoC params on a default config in .json
#import pdb;pdb.set_trace() 
start=time.time()
reset_config()

if not pre_trained:
    directory_name = os.path.dirname(__file__)+'/Stats_Clusters_training_'+param_opt_mode+'_'+str(dse_count)+'/'
    #os.system("rm -r "+directory_name)
    [os.remove(os.path.join(directory_name, f)) for f in os.listdir(directory_name) if os.path.isfile(os.path.join(directory_name, f))]
    os.system("mkdir "+directory_name)
    #import pdb;pdb.set_trace() 

    print("--------------------------------Finding the configuration based on training set--------------------------------------")
    #Step 1: Individual graphs
    print("Step 1: Obtain configs")
    #Optimal configuration loop - Run DSE on different configs-identify top 5
    #optimal configuration across all graphs and Construct universal cluster based on optimal config 
    if (dse_arr[dse_idx]):
        ind_graphs,_,_,_=construct_ind_graphs(training_set, False)
        outs=run_DSE_cluster([training_set,ind_graphs], None, None, opt, param_opt_mode, num_opt_clusters, directory_name+'0_compute_dse/', dse_count)
        PPA_common, config_common, universal_cluster_computedse,PPA_custom, config_custom,cluster_custom_computedse, PPA_opts, config_opts,cluster_opts_computedse, ind_graphs_common,  ind_graphs_list_custom, ind_graphs_list_opt=outs
        with open('./Intermediate_Outputs/Training_data_computedse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
            pickle.dump(outs, file) 
        del outs
    else:
        with open('./Intermediate_Outputs/Training_data_computedse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
            ins = pickle.load(file)
        
        PPA_common, config_common, universal_cluster_computedse,PPA_custom, config_custom,cluster_custom_computedse, PPA_opts, config_opts,cluster_opts_computedse, ind_graphs_common,  ind_graphs_list_custom, ind_graphs_list_opt= ins
        del ins
    #print("time_to_execute_computedse", time.time()-start)
    end_computedse=time.time()
    #import pdb;pdb.set_trace()
    if plot_stats:
        if opt:
            PPA_curr_opt=[[0]*train_len,[0]*train_len,[0]*train_len]

            #import pdb;pdb.set_trace()
            
            for bin_idx in range(len(cluster_opts_computedse)):
                for config_idx, model in enumerate(cluster_opts_computedse[bin_idx][1]):
                    for model_idx, model_1 in enumerate(training_set):
                        if model_1==model:
                            break
                    #print(model_idx)
                    #import pdb;pdb.set_trace()        
                    PPA_curr_opt[0][model_idx]=PPA_opts[bin_idx][0][config_idx]
                    PPA_curr_opt[1][model_idx]=PPA_opts[bin_idx][1][config_idx]
                    PPA_curr_opt[2][model_idx]=PPA_opts[bin_idx][2][0] 
        #import pdb;pdb.set_trace() 
        plot_stat_PPA(PPA_common, PPA_custom, PPA_curr_opt, config_custom, opt,  directory_name+'0_compute_dse/', train_len, training_set)
        del PPA_curr_opt
    dse_idx+=1
    """
    if (dse_arr[dse_idx]):
        outs= run_mem_dse(directory_name+'1_mem_dse/', universal_cluster_computedse, cluster_custom_computedse, cluster_opts_computedse,  config_common,config_custom, config_opts)
        universal_cluster_memdse,cluster_custom_memdse,cluster_opts_memdse, memsize_custom, memsize_common, memsize_opts=outs
        with open('./Intermediate_Outputs/Training_data_memdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
            pickle.dump(outs, file) 
        del outs
    else:
        with open('./Intermediate_Outputs/Training_data_memdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
            ins = pickle.load(file)
        
        universal_cluster_memdse,cluster_custom_memdse,cluster_opts_memdse, memsize_custom, memsize_common, memsize_opts= ins
        del ins"
    """
    universal_cluster_memdse,cluster_custom_memdse, cluster_opts_memdse=universal_cluster_computedse, cluster_custom_computedse, cluster_opts_computedse
    del universal_cluster_computedse, cluster_custom_computedse, cluster_opts_computedse
    dse_idx+=1

    if (dse_arr[dse_idx]):
        outs=run_noc_dse(directory_name+'2_noc_dse/',  universal_cluster_memdse,config_common, cluster_custom_memdse, config_custom, cluster_opts_memdse, config_opts,  ind_graphs_common,  ind_graphs_list_custom, ind_graphs_list_opt, PPA_common, PPA_custom, PPA_opts)
        Nocdse_BW_common, universal_cluster, Nocdse_BW_custom, cluster_custom,  Nocdse_BW_opts, cluster_opts, PPA_common_nocdse, PPA_custom_nocdse, PPA_opt_nocdse=outs
        with open('./Intermediate_Outputs/Training_data_nocdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
            pickle.dump(outs, file) 
        del outs
    else:
        with open('./Intermediate_Outputs/Training_data_nocdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
            ins = pickle.load(file)
        
        Nocdse_BW_common, universal_cluster, Nocdse_BW_custom, cluster_custom,  Nocdse_BW_opts, cluster_opts, PPA_common_nocdse, PPA_custom_nocdse, PPA_opt_nocdse= ins
        del ins
    #print("time_to_execute_nocdse", time.time()-end_computedse)
    end_nocdse=time.time()
    if plot_stats:
        if opt:
            PPA_curr_opt_nocdse=[[0]*train_len,[0]*train_len,[0]*train_len]

            #import pdb;pdb.set_trace()
            
            for bin_idx in range(len(cluster_opts_memdse)):
                for config_idx, model in enumerate(cluster_opts_memdse[bin_idx][1]):
                    for model_idx, model_1 in enumerate(training_set):
                        if model_1==model:
                            break
                    #print(model_idx)
                    #import pdb;pdb.set_trace()        
                    PPA_curr_opt_nocdse[0][model_idx]=PPA_opt_nocdse[bin_idx][0][config_idx]
                    PPA_curr_opt_nocdse[1][model_idx]=PPA_opt_nocdse[bin_idx][1][config_idx]
                    PPA_curr_opt_nocdse[2][model_idx]=PPA_opt_nocdse[bin_idx][2][0] 
        #import pdb;pdb.set_trace() 
        #print(PPA_common, PPA_custom, PPA_opts)
        #print(PPA_common_nocdse, PPA_custom_nocdse, PPA_curr_opt_nocdse)
        plot_stat_PPA(PPA_common_nocdse, PPA_custom_nocdse, PPA_curr_opt_nocdse, config_custom, opt,  directory_name+'2_noc_dse/', train_len, training_set)
        del PPA_curr_opt_nocdse
    del  universal_cluster_memdse,cluster_custom_memdse, cluster_opts_memdse
    del PPA_common, PPA_custom, PPA_opts
    #del PPA_common_nocdse, PPA_custom_nocdse, PPA_opt_nocdse
    dse_idx+=1
    
    #import pdb;pdb.set_trace() 
    if nop:
        #Step 2: Clustering
        print("Step 2: Perform clustering")
        if (dse_arr[dse_idx]):
            os.system("rm -r "+directory_name+'3_clusters_opt/')
            os.system("mkdir "+directory_name+'3_clusters_opt/')
            os.system("rm -r "+directory_name+'3_clusters_common/')
            os.system("mkdir "+directory_name+'3_clusters_common/')
            os.system("rm -r "+directory_name+'3_clusters_custom/')
            os.system("mkdir "+directory_name+'3_clusters_custom/')
            #Apply clustering algorithm and split it into clusters
            cluster_common=find_optimal_cluster(universal_cluster, directory_name+'3_clusters_common/')

            #import pdb;pdb.set_trace() 
            del universal_cluster
            if opt:
                cluster_opt_bfcombine=[]
                for cluster_grp in cluster_opts:
                    #print("Optimal Cluster", cluster_grp[1])
                    cluster_opt_bfcombine.append(find_optimal_cluster(cluster_grp, directory_name+'3_clusters_opt/'))
                    #import pdb;pdb.set_trace() 
                #combine clusters that are similar
                cluster_opt=combine_similar_clusters(cluster_opt_bfcombine, directory_name+'3_clusters_opt/')             
            cluster_custom_gl=[]
            for item in cluster_custom:
                cluster_custom_gl.append(find_optimal_cluster(item, directory_name+'3_clusters_custom/'))
            del cluster_custom
            outs=(cluster_common, cluster_opt,cluster_custom_gl)
            with open('./Intermediate_Outputs/Training_data_cluster_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
                pickle.dump(outs, file) 
            
            del outs
            #import pdb;pdb.set_trace() 
        else:
            with open('./Intermediate_Outputs/Training_data_cluster_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
                ins = pickle.load(file)
            
            cluster_common, cluster_opt,cluster_custom_gl= ins
            del ins
        dse_idx+=1
        #print("time_to_execute_cluster", time.time()-end_nocdse)
        end_cluster=time.time()
        
        #import pdb;pdb.set_trace() 
        
        #Step 3: NoP graphs
        print("Step 3:Add NoP params")
        if (dse_arr[dse_idx]):
            os.system("rm -r "+directory_name+'4_nop_io_custom/')
            os.system("mkdir "+directory_name+'4_nop_io_custom/')
            os.system("rm -r "+directory_name+'4_nop_io_opt/')
            os.system("mkdir "+directory_name+'4_nop_io_opt/')
            os.system("rm -r "+directory_name+'4_nop_io_common/')
            os.system("mkdir "+directory_name+'4_nop_io_common/')
            cost_benefit=[]
            #import pdb;pdb.set_trace()
            #update appropriate edges with NoP Params -calc NoP BW based on chiplet area
            #Run DSE on different configs and find optimal config
            print("Finding config with NoP for common cluster")
            PPA_curr, config_opt_up_common,coverage_common=run_DSE_cluster([training_set], [cluster_common, Nocdse_BW_common, None, None], config_common[0], False, None, None, directory_name+'4_nop_io_common/', dse_count)
            nop_dse_BW_common=PPA_curr[8]
            nop_idx_common=PPA_curr[9]
            #update appropriate edges with NoP Params -calc NoP BW based on chiplet area
            #Run DSE on different configs and find optimal config
            #import pdb;pdb.set_trace()
            print("Finding config with NoP for custom clusters")
            PPA_curr_custom,config_opt_up_custom,coverage_custom=[[],[],[],[], [],[], [], [], [],[], [], []],[[],[]],[[],[],[],[]]
            nop_dse_BW_custom, nop_idx_custom=[], []
            for idx_cluster,item_cluster in enumerate(cluster_custom_gl):
                PPA_curr_temp, config_opt_up_custom_temp,coverage_custom_temp=run_DSE_cluster([[training_set[idx_cluster]]], [item_cluster, Nocdse_BW_custom[idx_cluster],  None, None], config_custom[0][idx_cluster], False, None, None, directory_name+'4_nop_io_custom/', dse_count)
                for idx,item in enumerate(PPA_curr_temp):
                    #import pdb;pdb.set_trace() 
                    #print(idx)
                    PPA_curr_custom[idx].append(item[0])
                nop_dse_BW_custom.append(PPA_curr_temp[8][0])
                nop_idx_custom.append(PPA_curr_temp[9][0])
                config_opt_up_custom[0].append(config_opt_up_custom_temp)
                #import pdb;pdb.set_trace() 
                config_opt_up_custom[1].append(item_cluster[0].nodes())
                for idx,item in enumerate(coverage_custom_temp):
                    coverage_custom[idx].append(item[0])
            #PPA.append([sum(item) for item in PPA_curr_custom])
            #del cluster_custom_gl
            del config_opt_up_custom_temp
            del PPA_curr_temp
            print("Finding config with NoP for optimal clusters")
            if opt:
                config_opt_up_opt, PPA_curr_opt,coverage_opt=[[0]*train_len,[0]*train_len],[[0]*train_len,[0]*train_len, [0]*train_len, [0]*train_len,[0]*train_len, [0]*train_len, [0]*train_len, [0]*train_len,[0]*train_len, [0]*train_len, [0]*train_len, [0]*train_len], [[0]*train_len,[0]*train_len,[0]*train_len,[0]*train_len]
                nop_dse_BW_opt, nop_idx_opt=[], []
                #import pdb;pdb.set_trace()
                for bin_idx, config_opt in enumerate(config_opts[0]):
                    cost_custom=0
                    #print(config_opt)
                    #import pdb;pdb.set_trace()
                    PPA_curr_opt_temp, config_opt_up_opt_temp, coverage_opt_temp=run_DSE_cluster([cluster_opts[bin_idx][1]], [cluster_opt[bin_idx],Nocdse_BW_opts[bin_idx], None, None], config_opt, False,None, None, directory_name+'4_nop_io_opt/', dse_count)
                    nop_dse_BW_opt.append(PPA_curr_opt_temp[8][0])
                    nop_idx_opt.append(PPA_curr_opt_temp[9][0])
                    #import pdb;pdb.set_trace()
                    for config_idx, model in enumerate(cluster_opts[bin_idx][1]):
                        for model_idx, model_1 in enumerate(training_set):
                            if model_1==model:
                                break
                        cost_custom+=PPA_curr_custom[3][model_idx]
                        #print(model_idx)
                        #import pdb;pdb.set_trace()        
                        for idx,item in enumerate([0,0]+PPA_curr_opt_temp[2:4]+[0,0,0]+PPA_curr_opt_temp[7:11]):
                            #print(idx, item, config_idx, model_idx)
                            if item!=0:
                                PPA_curr_opt[idx][model_idx]=item[0] 
                        for idx,item in enumerate(PPA_curr_opt_temp[0:2]+[0,0]+PPA_curr_opt_temp[4:7]+[0,0,0, 0]+PPA_curr_opt_temp[11:]):
                            #print(idx, item, config_idx, model_idx)
                            if item!=0:
                                PPA_curr_opt[idx][model_idx]=item[config_idx]
                        config_opt_up_opt[0][model_idx]=config_opt_up_opt_temp
                        #import pdb;pdb.set_trace()
                        config_opt_up_opt[1][model_idx]=cluster_opt[bin_idx]
                        for coverage_idx,_ in enumerate(coverage_opt_temp):
                            coverage_opt[coverage_idx][model_idx]=coverage_opt_temp[coverage_idx][config_idx]
                    #print(config_opt,config_opt_up_opt_temp, config_opt_up_opt)
                    norm_cost_opt=PPA_curr_opt_temp[3][0]/PPA_curr[3][0]
                    norm_cost_cstm=cost_custom/PPA_curr[3][0]
                    cost_benefit.append([set([item.split('.')[1] for item in cluster_opts[bin_idx][1]]),f"{float(norm_cost_cstm):.3f}",f"{float(norm_cost_opt):.3f}",f"{float(norm_cost_cstm/norm_cost_opt):.3f}"])
                    
                #PPA.append([sum(item) for item in PPA_curr_opt])
            #print(cost_benefit)
            #import pdb;pdb.set_trace()
            outs=PPA_curr, config_opt_up_common,coverage_common, PPA_curr_custom, config_opt_up_custom, coverage_custom, PPA_curr_opt, config_opt_up_opt, coverage_opt, norm_cost_opt, norm_cost_cstm, cost_benefit
            with open('./Intermediate_Outputs/Training_data_nopdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
                pickle.dump(outs, file) 
            del outs
        else:
            with open('./Intermediate_Outputs/Training_data_nopdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
                ins = pickle.load(file)
            
            PPA_curr, config_opt_up_common,coverage_common, PPA_curr_custom, config_opt_up_custom, coverage_custom, PPA_curr_opt, config_opt_up_opt, coverage_opt, norm_cost_opt, norm_cost_cstm, cost_benefit= ins
            del ins
        #print("time_to_execute_nopdse", time.time()-end_cluster)
        end_nopdse=time.time()
        #import pdb;pdb.set_trace()

    else:
        print("code to be written")
        import pdb;pdb.set_trace()

        PPA_curr=PPA_common
        PPA_curr_custom=PPA_custom
        config_opt_up_custom=config_custom
        config_common=config_common
        if opt:
            PPA_curr_opt=[[0]*train_len,[0]*train_len,[0]*train_len]
            config_opt_up_opt=[[0]*train_len, [0]*train_len]

            #import pdb;pdb.set_trace()
            
            for bin_idx in range(len(cluster_opts)):
                for config_idx, model in enumerate(cluster_opts[bin_idx][1]):
                    for model_idx, model_1 in enumerate(training_set):
                        if model_1==model:
                            break
                    #print(model_idx)
                    #import pdb;pdb.set_trace()        
                    PPA_curr_opt[0][model_idx]=PPA_opts[bin_idx][0][config_idx]
                    PPA_curr_opt[1][model_idx]=PPA_opts[bin_idx][1][config_idx]
                    PPA_curr_opt[2][model_idx]=PPA_opts[bin_idx][2][0] 
                    config_opt_up_opt[0][model_idx]=config_opts[0][bin_idx]
                    config_opt_up_opt[1][model_idx]=config_opts[1][bin_idx]
                    #import pdb;pdb.set_trace()
                #print(config_opt,config_opt_up_opt_temp, config_opt_up_opt)
    #import pdb;pdb.set_trace()
        
    #print(PPA)
    print("time_to_execute",time.time()-start)

    #import pdb;pdb.set_trace() 
    if plot_stats:
        plot_stat_PPA(PPA_curr, PPA_curr_custom, PPA_curr_opt, config_custom, opt, directory_name, train_len, training_set)
    #import pdb;pdb.set_trace() 

    desired_width=30
    table_group_cost = PrettyTable()
    #table_group.field_names = ['Node Name','In_Channels','Out_Channels','Kernel','Stride','Inter Node Frequency','Total Nodes','Degree of Connectivity ']
    if opt:
        table_group=PPA_table_info(None, training_set,  PPA_curr, PPA_curr_custom, PPA_curr_opt)
        table_group_config=config_table_info(None, training_set, coverage_common, coverage_custom,coverage_opt , config_common, config_opt_up_custom, config_opt_up_opt, [idx for idx,_ in enumerate(training_set) ])
        table_group_cluster=cluster_table_info(None, training_set, cluster_common, cluster_custom_gl, config_opt_up_opt[1],[idx for idx,_ in enumerate(training_set) ] )
        table_group_cost.field_names=['Algorithms', 'NormCost-Cstm','NormCost-Opt', 'Benefit of Optimal solution (~times)']
        for column in table_group_cost.field_names[1:]:
            table_group_cost.align[column] = "r" 
        table_group_cost.add_row([
            "Training set", "", "", "",  
        ])
        for row in cost_benefit:
            table_group_cost.add_row(row)
    else:
        #import pdb;pdb.set_trace()
        table_group.field_names = ['Algorithm','L-Gen','L-Cstm','L-%',  'E-Gen','E-Cstm','E-%', 'A-Gen', 'A-Cstm','A-%', 'pD-Gen', 'pD-Cstm','pD-%', 'Config-Custom','Config-Generic']
        
        for idx in range(len(PPA_curr_custom[0])):
            #import pdb;pdb.set_trace()
            table_group.add_row([
            training_set[idx].split('.')[1],
            f"{float(PPA_curr[0][idx]):.3f}",
            f"{float(PPA_curr_custom[0][idx]):.3f}",
            str(f"{(float(PPA_curr[0][idx])-float(PPA_curr_custom[0][idx]))/float(PPA_curr_custom[0][idx])*100:.3f}")+"%",
            f"{float(PPA_curr[1][idx]*1e-3):.3f}",
            f"{float(PPA_curr_custom[1][idx]*1e-3):.3f}",
            str(f"{(float(PPA_curr[1][idx])-float(PPA_curr_custom[1][idx]))/float(PPA_curr_custom[1][idx])*100:.3f}")+"%",
            f"{float(PPA_curr[2][0]):.3f}",
            f"{float(PPA_curr_custom[2][idx]):.3f}",
            str(f"{(float(PPA_curr[2][0])-float(PPA_curr_custom[2][idx]))/float(PPA_curr_custom[2][idx])*100:.3f}")+"%",
            f"{float(PPA_curr[1][idx]/PPA_curr[0][idx]/PPA_curr[2][0]):.3f}",
            f"{float(PPA_curr_custom[1][idx]/PPA_curr_custom[0][idx]/PPA_curr_custom[2][idx]):.3f}",
            str(f"{(float(PPA_curr[1][idx]/PPA_curr[0][idx]/PPA_curr[2][0])-float(PPA_curr_custom[1][idx]/PPA_curr_custom[0][idx]/PPA_curr_custom[2][idx]))/float(PPA_curr_custom[1][idx]/PPA_curr_custom[0][idx]/PPA_curr_custom[2][idx])*100:.3f}")+"%",
            [config_opt_up_custom[0][idx],config_opt_up_custom[1][idx]],
            [config_common[0], config_common[1]],
            ])
    
    outs=(test_set, config_opts, cluster_opt, config_common, cluster_common, param_opt_mode,config_opt_up_opt, Nocdse_BW_opts, nop_dse_BW_opt, nop_idx_opt, Nocdse_BW_common, nop_dse_BW_common, nop_idx_common)
    with open('Training_data_dse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
        pickle.dump(outs, file)   
    text_file_name = directory_name+"Output.txt"
    print_file(text_file_name,table_group)
    text_file_name = directory_name+"Output_Config.txt"
    print_file(text_file_name,table_group_config)
    text_file_name = directory_name+"Output_Cost.txt"
    print_file(text_file_name,table_group_cost)
    text_file_name = directory_name+"Output_Cluster.txt"
    print_file(text_file_name,table_group_cluster)
else:
    with open('Training_data_dse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
        ins = pickle.load(file)
        
    test_set, config_opts, cluster_opt, config_common, cluster_common, param_opt_mode, _, Nocdse_BW_opts, nop_dse_BW_opt, nop_idx_opt, Nocdse_BW_common, nop_dse_BW_common, nop_idx_common = ins
#import pdb;pdb.set_trace()


if test:
    directory_name = os.path.dirname(__file__)+'/Stats_Clusters_test_'+param_opt_mode+'_'+str(dse_count)+'/'
    os.system("rm -r "+directory_name)
    os.system("mkdir "+directory_name)
    os.system("rm -r "+directory_name+'4_nop_io_opt/')
    os.system("mkdir "+directory_name+'4_nop_io_opt/')
    os.system("rm -r "+directory_name+'4_nop_io_common/')
    os.system("mkdir "+directory_name+'4_nop_io_common/')
    print("--------------------------------Testing coverage of test set--------------------------------------")
    print("Optimal Config")
    #import pdb;pdb.set_trace()
    if opt:
        #Test on optimal configuration
        PPA_test_opt,coverage_opt_test, max_cluster_idx, cluster_split_dict=run_test(test_set, config_opts, cluster_opt, directory_name+'4_nop_io_opt/', dse_count, Nocdse_BW_opts, nop_dse_BW_opt, nop_idx_opt)
        #import pdb;pdb.set_trace()
    
    #Test on generic configuration
    print("Common Config")
    PPA_test_common, coverage_common_test,_, _=run_test(test_set, [[config_common[0]],[config_common[1]]], [cluster_common], directory_name+'4_nop_io_common/', dse_count, [Nocdse_BW_common], nop_dse_BW_common, nop_idx_common)
    cost_common_test=PPA_test_common[3][0]
    reset_config()
    #import pdb;pdb.set_trace()
    #Identify Custom Configuration 
    print("Custom Config")
    if (dse_arr_test[dse_idx_test]):
        #import pdb;pdb.set_trace()
        ind_graphs,L,E,A=construct_ind_graphs(test_set, False, cluster_split_dict)
        #import pdb;pdb.set_trace()
        outs=run_DSE_cluster([test_set,ind_graphs], None, None, False, param_opt_mode, None, directory_name+'0_compute_dse/', dse_count, cluster_split_dict)
        _,_,_, PPA_custom_computedse_test, config_custom_computedse_test, cluster_custom_computedse_test,_,_,_, _,ind_graphs_list_custom_test,_=outs
        with open('./Intermediate_Outputs/Test_data_computedse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
            pickle.dump((PPA_custom_computedse_test, config_custom_computedse_test, cluster_custom_computedse_test,ind_graphs_list_custom_test) , file) 
        del outs
    else:
        with open('./Intermediate_Outputs/Test_data_computedse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
            ins = pickle.load(file)
        
        PPA_custom_computedse_test, config_custom_computedse_test, cluster_custom_computedse_test,ind_graphs_list_custom_test= ins
        del ins
    dse_idx_test+=1
    #TO DO: Memory code
    dse_idx_test+=1
    if (dse_arr_test[dse_idx_test]):
        outs=run_noc_dse(directory_name+'2_noc_dse/',  None,None, cluster_custom_computedse_test, config_custom_computedse_test, None, None,  None,  ind_graphs_list_custom_test, None, None, PPA_custom_computedse_test, None)
        _, _, Nocdse_BW_custom_test, cluster_nocdse_custom_test,  _, _, _, PPA_custom_nocdse_test, _=outs
        with open('./Intermediate_Outputs/Test_data_nocdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
            pickle.dump((Nocdse_BW_custom_test, cluster_nocdse_custom_test, PPA_custom_nocdse_test), file) 
        del outs
    else:
        with open('./Intermediate_Outputs/Test_data_nocdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
            ins = pickle.load(file)
        
        Nocdse_BW_custom_test, cluster_nocdse_custom_test, PPA_custom_nocdse_test= ins
        del ins
    dse_idx_test+=1

    #import pdb;pdb.set_trace()
    
    if (dse_arr_test[dse_idx_test]):
        os.system("rm -r "+directory_name+'3_clusters/')
        os.system("mkdir "+directory_name+'3_clusters/')
        cluster_custom_gl_test=[]
        for item in cluster_nocdse_custom_test:
            cluster_custom_gl_test.append(find_optimal_cluster(item,  directory_name+'3_clusters/'))
        del cluster_nocdse_custom_test
        with open('./Intermediate_Outputs/Test_data_cluster_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
            pickle.dump(cluster_custom_gl_test, file) 
        #import pdb;pdb.set_trace() 
    else:
        with open('./Intermediate_Outputs/Test_data_cluster_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
            ins = pickle.load(file)
        
        cluster_custom_gl_test= ins
        del ins
    dse_idx_test+=1

    #import pdb;pdb.set_trace()

    if (dse_arr[dse_idx_test]):
        os.system("rm -r "+directory_name+'4_nop_io_custom/')
        os.system("mkdir "+directory_name+'4_nop_io_custom/')
        PPA_curr_custom_test,config_opt_up_custom_test,coverage_custom_test=[[],[],[],[], [],[], [], [], [],[], [], []],[[],[]],[[],[],[],[]]
        nop_dse_BW_custom_test, nop_idx_custom_test=[], []
        for idx_cluster,item_cluster in enumerate(cluster_custom_gl_test):
            PPA_curr_temp, config_opt_up_custom_temp,coverage_custom_test_temp=run_DSE_cluster([[test_set[idx_cluster]]], [item_cluster, Nocdse_BW_custom_test[idx_cluster], None, None], config_custom_computedse_test[0][idx_cluster], False, None, None, directory_name+'4_nop_io_custom/', dse_count)
            for idx,item in enumerate(PPA_curr_temp):
                PPA_curr_custom_test[idx].append(item[0])
            nop_dse_BW_custom_test.append(PPA_curr_temp[8][0])
            nop_idx_custom_test.append(PPA_curr_temp[9][0])
            config_opt_up_custom_test[0].append(config_opt_up_custom_temp)
            config_opt_up_custom_test[1].append(item_cluster[0].nodes())
            for idx,item in enumerate(coverage_custom_test_temp):
                coverage_custom_test[idx].append(item[0])
        #import pdb;pdb.set_trace()
        
        outs=PPA_curr_custom_test, config_opt_up_custom_test, coverage_custom_test
        with open('./Intermediate_Outputs/Test_data_nopdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'wb') as file:
            pickle.dump(outs, file) 
        del outs
    else:
        with open('./Intermediate_Outputs/Test_data_nopdse_'+param_opt_mode+'_'+str(dse_count)+'.pkl', 'rb') as file:
            ins = pickle.load(file)
        
        PPA_curr_custom_test, config_opt_up_custom_test, coverage_custom_test= ins
        del ins
        #import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()
    cost_benefit_test=[]
    num_opts=len(config_opts[0])
    cost_opt_test, cost_cstm_test =[0]*num_opts, [0]*num_opts
    test_opt_list=[[] for _ in range(num_opts)]
    for model_idx, cluster_idx in enumerate(max_cluster_idx):
        cost_opt_test[cluster_idx]=PPA_test_opt[3][model_idx]
        cost_cstm_test[cluster_idx]+=PPA_curr_custom_test[3][model_idx]
        test_opt_list[cluster_idx].append(test_set[model_idx].split('.')[1])
    if opt:
        for cluster_idx, item in enumerate(cost_opt_test):
            if item!=0:
                cost_benefit_test.append([set(test_opt_list[cluster_idx]), f"{float(cost_cstm_test[cluster_idx]/cost_common_test):.3f}", f"{float(item/cost_common_test):.3f}",f"{float(cost_cstm_test[cluster_idx]/item):.3f}" ])
        #print(cost_benefit_test)

    cluster_coverage_file_name = directory_name+"coverage_test_optimal_config.txt"
    print_file(cluster_coverage_file_name, coverage_opt_test)

    cluster_coverage_file_name = directory_name+"coverage_test_generic_config.txt"
    print_file(cluster_coverage_file_name, coverage_common_test)
    table_group_cost = PrettyTable()
    table_group=PrettyTable()
    table_group_config=PrettyTable()
    table_group_cluster=PrettyTable()
    if opt:
        table_group=PPA_table_info(table_group, test_set,  PPA_test_common, PPA_curr_custom_test, PPA_test_opt)
        table_group_config=config_table_info(table_group_config, test_set, coverage_common_test, coverage_custom_test,coverage_opt_test , config_common, config_opt_up_custom_test, config_opts, max_cluster_idx)
        table_group_cluster=cluster_table_info(table_group_cluster, test_set, cluster_common, cluster_custom_gl_test, cluster_opt,max_cluster_idx)
        table_group_cost.field_names=['Algorithms', 'NormCost-Cstm','NormCost-Opt', 'Benefit of Optimal solution (~times)'] 
        table_group_cost.add_row([
            "Test set", "", "", "",  
        ])
        for row in cost_benefit_test:
            table_group_cost.add_row(row)
    
    del coverage_custom_test, coverage_opt_test, coverage_common_test
    del cluster_common, cluster_custom_gl_test, cluster_opt

    text_file_name = directory_name+"Output.txt"
    print_file(text_file_name,table_group)
    text_file_name = directory_name+"Output_Config.txt"
    print_file(text_file_name,table_group_config)
    text_file_name = directory_name+"Output_Cost.txt"
    print_file(text_file_name,table_group_cost)
    text_file_name = directory_name+"Output_Cluster.txt"
    print_file(text_file_name,table_group_cluster)

    del table_group_cost, table_group_config, table_group

    if plot_stats:
        bar_width = 0.4
        r1 = np.arange(len(PPA_curr_custom_test[0]))*(bar_width + 0.1)*4
        r2 = [x + bar_width+0.1 for x in r1]
        r3 = [x + bar_width+0.1 for x in r2]
        #import pdb;pdb.set_trace() 

        fig, ax = plt.subplots(figsize=(24, 16))
        ax.set_yscale('log')

        bars1 = ax.bar(r1, PPA_curr_custom_test[0], color='blue', width=bar_width, edgecolor='grey', label='Custom')
        if opt:
            bars2 = ax.bar(r2, PPA_test_opt[0], color='green', width=bar_width, edgecolor='grey', label='Optimal')
        bars3 = ax.bar(r3, PPA_test_common[0], color='red', width=bar_width, edgecolor='grey', label='Generic')
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
        ax.set_title('Latency Comparison between Generic, Optimal and Custom configurations- Test set')
        ax.set_xticks([r*(bar_width + 0.1)*4 + bar_width for r in range(test_len)])
        #import pdb;pdb.set_trace() 
        ax.set_xticklabels(test_set, rotation=45)

        ax.legend()
        plt.tight_layout()
        plt.savefig(directory_name+f'cluster_opt_generic_design_latency_test.png')
        plt.close()


        r1 = np.arange(len(PPA_curr_custom_test[1]))*(bar_width + 0.1)*4
        r2 = [x + bar_width+0.1 for x in r1]
        r3 = [x + bar_width+0.1 for x in r2]
        #import pdb;pdb.set_trace() 

        fig, ax = plt.subplots(figsize=(24, 16))
        ax.set_yscale('log')

        bars1 = ax.bar(r1, PPA_curr_custom_test[1], color='blue', width=bar_width, edgecolor='grey', label='Custom')
        if opt:
            bars2 = ax.bar(r2, PPA_test_opt[1], color='green', width=bar_width, edgecolor='grey', label='Optimal')
        bars3 = ax.bar(r3, PPA_test_common[1], color='red', width=bar_width, edgecolor='grey', label='Generic')
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
        ax.set_title('Energy Comparison between Generic, Optimal and Custom configurations- Test set')
        ax.set_xticks([r*(bar_width + 0.1)*4 + bar_width for r in range(test_len)])
        ax.set_xticklabels(test_set, rotation=45)

        ax.legend()
        plt.tight_layout()
        plt.savefig(directory_name+f'cluster_opt_generic_design_energy_test.png')
        plt.close()

        r1 = np.arange(len(PPA_curr_custom_test[2]))*(bar_width + 0.1)*4
        r2 = [x + bar_width+0.1 for x in r1]
        r3 = [x + bar_width+0.1 for x in r2]
        #import pdb;pdb.set_trace() 

        fig, ax = plt.subplots(figsize=(24, 16))
        ax.set_yscale('log')

        bars1 = ax.bar(r1, PPA_curr_custom_test[2], color='blue', width=bar_width, edgecolor='grey', label='Custom')
        if opt:
            bars2 = ax.bar(r2, PPA_test_opt[2], color='green', width=bar_width, edgecolor='grey', label='Optimal')
        bars3 = ax.bar(r3, PPA_test_common[2], color='red', width=bar_width, edgecolor='grey', label='Generic')
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
        ax.set_title('Area Comparison between Generic, Optimal and Custom configurations- Test set')
        ax.set_xticks([r*(bar_width + 0.1)*4 + bar_width for r in range(test_len)])
        ax.set_xticklabels(test_set, rotation=45)

        ax.legend()
        plt.tight_layout()
        plt.savefig(directory_name+f'cluster_opt_generic_design_area_test.png')
        plt.close()

    #import pdb;pdb.set_trace()
