from Postprocess import stats, graphs_analyze, graphs_analyze_1, cluster_generate, edges_analyze
from Utils import print_file
from optimize_fns import *
import time
#model_list = ['vision.vgg16', 'vision.resnet18', 'vision.vit16','vision.mobilenetv3', 'model.peanutRCNN', 'model.peanutEncoder']
model_list = ['hugging.bert','hugging.moe', 'hugging.whisper', 'hugging.gpt','hugging.graph','hugging.dpt', 'hugging.vit', 
                'hugging.llama', 'hugging.ast', 'hugging.detr', 'hugging.dino','vision.vgg16', 'vision.resnet18', 
                'vision.alexnet', 'vision.densenet121', 'model.peanutRCNN', 'model.peanutEncoder', 'vision.mobilenetv2', 'vision.swint']
"""

model_list = ['hugging.bert','hugging.moe', 'hugging.whisper', 'hugging.gpt','hugging.graph','hugging.dpt', 'hugging.vit', 
                'hugging.llama', 'hugging.ast', 'hugging.detr', 'hugging.dino','vision.swint']

model_list = ['vision.vgg16', 'vision.resnet18', 
                'vision.alexnet', 'vision.densenet121', 'model.peanutRCNN', 'model.peanutEncoder', 'vision.mobilenetv2']
"""
start=time.time()
os.system("rm -r ./Stats_Clusters")
os.system("mkdir Stats_Clusters")
reset_config()
#model_list = ['hugging.vit']
num_bits=8

G=construct_ind_graphs(model_list, True)[0]
#table=graphs_analyze(G)
folder_path="./Stats_Chiplet_Size/"
os.makedirs(folder_path, exist_ok=True)
#text_file_name = folder_path+"Stats_1.txt"
#print_file(text_file_name, table)

table, common_nodes,high_frequency_nodes, block_frequency=graphs_analyze_1(G)
text_file_name = folder_path+"Stats_2.txt"
print_file(text_file_name,table)
#import pdb;pdb.set_trace() 

table, edge_frequency, node_frequency=edges_analyze(block_frequency)
text_file_name = folder_path+"Stats_3.txt"
print_file(text_file_name,table)

folder_path="./Stats_Clusters/"
os.makedirs(folder_path, exist_ok=True)
cluster_graph_G, _=cluster_generate(G, edge_frequency, node_frequency, folder_path, model_list, True)
#cluster_nodes, cluster_edges = edges_prob(G, common_nodes, high_frequency_nodes, block_frequency, directory_name, model_list)
#import pdb;pdb.set_trace() 

"""
table_loops, table_loops_unique=loops_analyze(G)
folder_path="./Stats_Dataflow/"
directory_name=folder_path
os.makedirs(directory_name, exist_ok=True)
text_file_name = directory_name+"Loops.txt"
print_file(text_file_name,table_loops)
text_file_name = directory_name+"Loops_Unique.txt"
print_file(text_file_name,table_loops_unique)

table_subgraph=subgraph_analyze(G, common_nodes, model_list)
folder_path="./Stats_Clusters/"
directory_name=folder_path
os.makedirs(directory_name, exist_ok=True)
text_file_name = directory_name+"Subgraphs.txt"
print_file(text_file_name, table_subgraph)
"""




