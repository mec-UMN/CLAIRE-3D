def edges_prob(G, common_nodes,high_frequency_nodes, block_frequency, directory_name, model_list):
    nodes_matrix=[]
    for idx, node in enumerate(block_frequency):
        nodes_matrix.append([node])
        nodes_matrix[idx].append((block_frequency[node][4]))
###update
        nodes_matrix[idx].append((block_frequency[node][5])) 
    
    universal_cluster_graph_image_name = directory_name+"universal_graph.png"
    unique_nodes, unique_edges = universal_cluster_graph(nodes_matrix, universal_cluster_graph_image_name)
    table_cluster_coverage=cluster_analyze(G, unique_nodes, unique_edges, model_list)
    cluster_coverage_file_name = directory_name+"universal_cluster_coverage.txt"
    print_file(cluster_coverage_file_name, table_cluster_coverage)
    
    cluster_graph_image_name = directory_name+"cluster_graph_prob.png"
    prob=prob_edges(unique_nodes,unique_edges, common_nodes, model_list)
    cluster_nodes, cluster_edges, high_frequency_nodes=prob_cluster_graph(prob, unique_nodes, high_frequency_nodes, common_nodes, cluster_graph_image_name)
    table_cluster_coverage=cluster_analyze(G, cluster_nodes, cluster_edges, model_list)
    cluster_coverage_file_name = directory_name+"Cluster_Coverage_prob.txt"
    print_file(cluster_coverage_file_name, table_cluster_coverage)
    
    cluster_graph_image_name = directory_name+"cluster_graph.png"
    cluster_nodes, cluster_edges=cluster_graph(high_frequency_nodes,cluster_nodes, cluster_edges, cluster_graph_image_name)
    table_cluster_coverage=cluster_analyze(G, cluster_nodes, cluster_edges, model_list)
    cluster_coverage_file_name = directory_name+"Cluster_Coverage.txt"
    print_file(cluster_coverage_file_name, table_cluster_coverage)

    #import pdb;pdb.set_trace() 
    return cluster_nodes, cluster_edges

def prob_edges(unique_nodes,unique_edges, common_nodes, model_list):
    
    prob=[[0 for _ in range(len(unique_nodes))] for _ in range(len(unique_nodes))]
    for edge in unique_edges:
        node1 = edge[0]
        node2 = edge[1]
        idx1=-1
        idx2=-1
        for idx, node in enumerate(unique_nodes):
            if node==node1:
                idx1=idx
            if node==node2:
                idx2=idx
        if idx==-1 or idx2==-1:
            print("ERROR")
            import pdb;pdb.set_trace() 

        if node1 in common_nodes:
            list_check=list(itertools.chain.from_iterable(common_nodes[node1][4]))
            for item in list_check:
                if item == node2:
                    #import pdb;pdb.set_trace() 
                    prob[idx1][idx2]+=1
        elif node2 in common_nodes:
            list_check=list(itertools.chain.from_iterable(common_nodes[node2][4]))
            for item in list_check:
                if item == node1:
                    prob[idx2][idx1]+=1
        #print(idx1, node1, idx2, node2)
        #print(prob)
    max_times=len(model_list)
    prob=[[x / max_times for x in row] for row in prob]
    return prob

def prob_cluster_graph(prob, unique_nodes, high_frequency_nodes, common_nodes, cluster_graph_image_name):
    unique_nodes_list=list(unique_nodes)
    cluster_nodes=set()
    cluster_edges=set()
    for idx_row, row in enumerate(prob):
        node_updated=False
        for idx, x in enumerate(row):
            node1=unique_nodes_list[idx_row]
            node2=unique_nodes_list[idx]
            if x>0.4:
                #print(x, idx_row, node1, idx, node2)
                node_updated=True
                cluster_nodes.add(node1)
                cluster_nodes.add(node2)
                cluster_edges.add((node1, node2))
        if not node_updated and node1 in common_nodes:
            if common_nodes[node1][1]>5 and node1 not in high_frequency_nodes:
                high_frequency_nodes[node1]=common_nodes[node1]
    #import pdb;pdb.set_trace() 

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
    
    ax.set_title("Cluster Graph based on Probabilities", fontsize=24, fontweight='bold')
    plt.savefig(cluster_graph_image_name)
    plt.show()

    return cluster_nodes, cluster_edges, high_frequency_nodes

def loops_analyze(G):
    loop_dict = {}  

    for graph_idx, graph in enumerate(G):
        loops = list(nx.simple_cycles(graph))
        #print(f"Graph {graph_idx+1} has {len(loops)} loops:")
        #print(loops)
        for loop_idx, nodes in enumerate(loops):
            loop_dict[f'loop_{graph_idx}_{loop_idx}'] = {}  
            for node_idx, node in enumerate(nodes):
                loop_dict[f'loop_{graph_idx}_{loop_idx}'][f'node_{node_idx}'] = {
                    'config': graph.nodes[node],
                    'loop_frequency': int(graph.nodes[node]['comment'].split(':')[1])
                }

    desired_width=30
    table_loop= PrettyTable()
    table_loop.field_names = ['Loop Name', 'Type','In_Channels','Out_Channels','Kernel','Stride','In Size','Out Size','Loop Frequency', 'Node Names',]
    for idx, (key, value) in enumerate(loop_dict.items(), start=1):
        val_type = '_'.join([value[key]['config']['label'].split('_')[0] for key in value])
        val_inch = '_'.join([value[key]['config']['label'].split('_')[1] for key in value])
        val_outch = '_'.join([value[key]['config']['label'].split('_')[2] for key in value])
        val_insize = '_'.join([value[key]['config']['third_comment'].split('_')[1] for key in value])
        val_outsize = '_'.join([value[key]['config']['third_comment'].split('_')[2] for key in value])
        val_freq=min([value[key]['loop_frequency'] for key in value])
        val_kernel='_'.join([value[key]['config']['label'].split('_')[3] if len(value[key]['config']['label'].split('_'))>3 else '-'for key in value])
        val_str='_'.join([value[key]['config']['label'].split('_')[4] if len(value[key]['config']['label'].split('_'))>4 else '-' for key in value])
        #node_names='_'.join([value[key]['config']['label'] + value[key]['config']['third_comment']  for key in value])
        node_names='_'.join([value[key]['config']['label'] for key in value])
        table_loop.add_row([
           key,
           val_type,
           val_inch,
           val_outch,
           val_kernel,
           val_str,
           val_insize,
           val_outsize,
           val_freq,
           node_names,
        ])
    
    unique_node_names = {}

    unique_table = PrettyTable()
    unique_table.field_names = table_loop.field_names+['Inter graph frequency']

    for i in range(len(table_loop._rows)):
        node_name = table_loop._rows[i][-1]  
        if node_name not in unique_node_names:
            unique_table.add_row(table_loop._rows[i]+[1])
            unique_node_names[node_name]=i
        else:
            #import pdb;pdb.set_trace() 
            unique_table._rows[unique_node_names[node_name]][-1]+=1
            unique_table._rows[unique_node_names[node_name]][8]+=table_loop._rows[i][8]

    #import pdb;pdb.set_trace() 

    return table_loop, unique_table

def subgraph_analyze(G, common_nodes, model_list):
    
    isomorphism={}
    reorder_common_nodes={}
    for node_start in common_nodes:
        for i in range(len(common_nodes[node_start][2])):
            graph_idx=common_nodes[node_start][2][i]
            if graph_idx in reorder_common_nodes:
                reorder_common_nodes[graph_idx][0].append(common_nodes[node_start][3][i])
                reorder_common_nodes[graph_idx][1].append(node_start)
            else:
                reorder_common_nodes[graph_idx]=[[common_nodes[node_start][3][i]]]
                reorder_common_nodes[graph_idx].append([node_start])
                #import pdb;pdb.set_trace() 
    sorted_common_nodes= {k: list(zip(*sorted(zip(v[0], v[1])))) for k, v in reorder_common_nodes.items()}
    import pdb;pdb.set_trace() 

    clustered_data = {}

    for key, value in sorted_common_nodes.items():
        points = value[0]
        labels = value[1]
        clusters = []
        start = points[0]
        start_label = labels[0]
        cluster_start = start
        cluster_labels = [start_label]
        for i in range(1, len(points)):
            if points[i] == points[i-1] + 1:
                cluster_labels.append(labels[i])
            else:
                clusters.append([(cluster_start, points[i-1]), tuple(cluster_labels)])
                cluster_start = points[i]
                cluster_labels = [labels[i]]
        clusters.append([(cluster_start, points[-1]), tuple(cluster_labels)])
        clustered_data[key] = clusters

    similar_clusters = {}
    compared_set = set()

    for key1, clusters1 in clustered_data.items():
        for cluster1 in clusters1:
            label1 = cluster1[1]
            for key2 in range(key1 + 1, len(clustered_data) + 1):
                for cluster2 in clustered_data[key2]:
                    label2 = cluster2[1]
                    if label1 == label2:
                        if  label1 in compared_set and  key1 in compared_set and  key2 in compared_set :
                            continue
                        else:
                            similar_clusters.setdefault(tuple(label1),[model_list[key1-1],tuple(cluster1[0])])
                            similar_clusters.setdefault(tuple(label1)).append(model_list[key2-1])
                            similar_clusters.setdefault(tuple(label1)).append(tuple(cluster2[0]))
                            compared_set.update((label1, key1))
                            compared_set.update((label2, key2))
                        

    #import pdb;pdb.set_trace() 

    keys_frequency = Counter(similar_clusters.keys())        

    desired_width=30
    table_subgraph = PrettyTable()
    table_subgraph.field_names = ['Cluster combination(Block_Inch_OutCh_Kx_stride)','Algorithms containing the cluster','Corresponding nodes']
    for idx, (key, value) in enumerate(similar_clusters.items(), start=1):
        table_subgraph.add_row([
           key,
           value[::2],
           value[1::2]
        ])

    import pdb;pdb.set_trace() 

    return table_subgraph
