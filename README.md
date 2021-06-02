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