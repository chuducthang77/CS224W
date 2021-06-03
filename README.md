# CS224W
Notes and tutorial obtained from courses CS224W 

## 1. Introduction:

- **Tools**: Pytorch Geometric (PyG), DeepSNAP, GraphGym, NetworkX, SNAP.PY
- Graph: entites with relations/interactions
- Types: Networks (social, communication, biomedicine, ...) vs representation (information, software, relational structures, similarity networks)
- Complex: Arbitrary size, complex topological order, dynamic, no ordering or reference 
- Representation: nodes $\rightarrow$ d-dimensional embeddings
- Traditional: Graphlets, Graph Kernels
- Node embeddings: DeepWalk, Node2Vec
- GNN: GCN, GraphSAGE, GAT
- Knowledge graphs and reasoning: TransE, BetaE
- Tasks/Level: Node, graph prediction/generation, community/sub-graph, edge

- Node classification, Link prediction, clustering, graph classification, graph generation, graph evolution
    * Node-level: Alphafold (protein folding, spatial graph)
    * Edge-level: Recommender Systems, node relation prediction, drug side effects
    * Subgraph-level: Traffic prediction 
    * Graph-level: Drug discovery (generate novel ones or optimize existing ones), Physical simulation (evolution of graph)

- Representation: Unique vs Non-unique, Link = nature of problem
- Links:
    * Directed: Arcs (Phone calls, followers)
    * Undirected: symmetrical, reciprocal (collaborations, friendship)
- Nodes:
    * Directed: in-degree vs out-degree, source vs sink, avg($\frac{E}{N}$)
    * Undirected: #Adjacent link, Avg ($\frac{2E}{N}$)
- Bipartite Graph: subset U and V such that eve (Author-Papers, Actors-Movies, Users-Movies)
- Folded networks: (collaboration, co-rating)
- Adjacency matrix: non-symmetric directed graph, sparse 
- Edge list
- Attributes: weight, ranking, type, sign,...
- Types of graph: self-edges, multigraph, unweighted, weighted
- Connected vs disconnected graph (giant component + isolated node)
- Strongly connected vs weakly connected (connected if disregard the edge directions)
- Strongly connected components (SCC): not every node is part of strongly connected components 
    * In-component vs Out-component

## 2. Traditional Methods for Machine Learning in Graphs
- Features: d-dimensional vectors:
- Objects: nodes, edges, subgraph, graph
- Objective function

    ### 2.1 Node-level features:
    - Node degree ($k_v$)
    - Node centrality ($c_v$): node degree + node importance taken account
        * Eigenvector centrality: Important as if surrounded by important nodes, recursive manner (alpha * x = A * x), largest eigenvalue is always unique and positive, leading = centrality
        * Betweenness centrality: Lie on many shortest paths between other nodes
        * Closeness centrality: small shortest path lengths to other nodes
    - Clustering coefficient ($e_v$): between 0 and 1, #Triangles in ego-network
    - Graphlets: Rooted connected non-isomorphic subgraphs
        * Degree: #Edges node touches
        * Clustering coefficient: #Triangle
        * GDV: #Graphlet, local topological similarity 
    - Important-based (influential nodes in the graph) vs structure-based (role of node based on local neighbor)

    ### 2.2 Link-level prediction:
    - Predicted new link: rank top K node without a link  
    - Features for pair of nodes
    - Methods:
        - Randomly missing links (static network)
        - Links over time: Evolving network, based on time $t_0$ to $t_0'$, predict in the future $t_1$ to $t_1'$ (citation, social, transaction)
    - Score (c): Number of common neighbors $\rightarrow$ non-increasing order $\rightarrow$ first n links are new links 
    - Features:
        - Distance-based: shortest-path distance between 2 nodes, not capture the degree of neighborhood overlap
        - Local neighborhood overlap: Common neighbors (intersection), Jaccard's coefficient (intersection/union), Adamic-Adar index ($\frac{1}{log(k_U)}$, social network). Drawback: return 0 if no common neighbor, but still potential
        - Global neighborhood overlap: Katz index (number of paths of all lengths between 2 pairs of nodes) - $P^{(K)}$ = $A^K$
        $$S = (I - \beta A)^{-1} - I$$
    

    ### 2.3 Graph-level features:
    - Kernel methods: 
        * Similarity between 2 graph data points
        * Kernel matrix: for all data points, positive semidefinite
        * Representation $\phi$ such that
        $$\phi(G) \dot \phi(G')$$
    - Types (Bag of * )
        * Graphlet: * = graphlet, 2 differences (no roots and not connected ), expensive even with the help of NP-hard, $G_k = (g_1, g_2, ..., g_{n_k})$
        $$(f_G)_i = \#(g_i \in G) \text{ for } i = 1, 2, ... n_k$$
        $$K(G, G') = h_G^T h_{G'}$$
        * Weisfeiler-Lehman: generalized version bag-of-word, color refinement, hash function, counter number of occurrence for different colors, computationally efficient (only colors appeared in 2 graphs needs to track)