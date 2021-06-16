# CS224W
Notes and tutorial obtained from courses CS224W 

## 1. Introduction:

- **Tools**: Pytorch Geometric (PyG), DeepSNAP, GraphGym, NetworkX, SNAP.PY
- Graph: entites with relations/interactions
- Types: Networks (social, communication, biomedicine, ...) vs representation (information, software, relational structures, similarity networks)
- Complex: Arbitrary size, complex topological order, dynamic, no ordering or reference 
- Representation: nodes <img src="http://www.sciweavers.org/tex2img.php?eq=%20%5Crightarrow%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt=" \rightarrow " width="19" height="8" /> d-dimensional embeddings
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
    * Directed: in-degree vs out-degree, source vs sink, avg(<!-- $\frac{E}{N}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7BE%7D%7BN%7D">)
    * Undirected: #Adjacent link, Avg (<!-- $\frac{2E}{N}$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=%5Cfrac%7B2E%7D%7BN%7D">)
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
    - Node degree (<!-- $k_v$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=k_v">)
    - Node centrality (<!-- $c_v$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=c_v">): node degree + node importance taken account
        * Eigenvector centrality: Important as if surrounded by important nodes, recursive manner (alpha * x = A * x), largest eigenvalue is always unique and positive, leading = centrality
        * Betweenness centrality: Lie on many shortest paths between other nodes
        * Closeness centrality: small shortest path lengths to other nodes
    - Clustering coefficient (<!-- $e_v$ --> <img style="transform: translateY(0.1em); background: white;" src="https://render.githubusercontent.com/render/math?math=e_v">): between 0 and 1, #Triangles in ego-network
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

## 3. Node Embedding
- Representation learning: no need to do feature engineering every time 
    ### 3.1 Node

    - Feature representation/Embedding: learnt the function to predict automatically
        - Similarity of embeddings = similarity in network
        - Encoder network information
        - Many types of prediction
    - Embedding: Similarity in embedding space = similarity in nodes
    - Decoder: embedding $\rightarrow$ similarity (dot product)
    - Encoder:
        - "Shallow" - Embedding-lookup : Z (embedding matrix $\mathbb{R}^{d\times|V|}$) and v (indicator vector $\mathbb{I}^{|V|}$)

    - Unsupervised/self-supervised: no node labels, no node features
    - Task independency: Used for any tasks

    ### 3.2 Random Walk
    - Embedding vector $z_u$ and probability P($v|z_u$)
    - Probability: softmax or sigmoid function
    - $z_u^T z_v \approx$ probability u and v co-occur on a random walk 
    - Expressivity: Incorporate local and higher-order neighborhood information 
        - High probability = more paths connecting 2 nodes
    - Efficiency: Consider only pairs that co-occurrence
    - $N_R(u): $ neighborhood of u obtained by some random walk
    - Log-likelihood objective: $max_f \sum_{u \in V} log P(N_R(u)| z_u)$
        - Feature representation that are predictive of the nodes in its random walk
    - Optimization: short fixed-length random walks (for each node in u) $\rightarrow$ multiset of nodes visited from u 
    - Loss: $\mathbb{L} = \sum_{u \in V} \sum_{v \in N_R(u)}$ -log(P($v|z_u$))
    - Problem: O($V^2$)
        - Negative sampling: log($\sigma(z_u^T z_v$)) - $\sum_{i = 1}^k log(\sigma(z_u^T z_{n_i}))$
        - Noise Contrastive Estimation 
        - Sample k proportional to its degree
        - Larger k = more robust estimate = higher bias (5-20)
    - Node2vec:
        - Key: flexible $N_R(u)$ = richer node embeddings
        - Biased $2^{nd}$ order random walk 
        - Global (DFS) vs Local (BFS)
        - Parameters: p (return to previous node) and q ("ratio" of BFS vs DFS) 
            - Probability $\frac{1}{p}$ and $\frac{1}{q}$
            - Smaller p = likely to return back to the previous node
        - Idea: Remember where you came from
        - Benefit: Linear complexity, individually parallelizable
        - Drawback: Embedding for each node separately $\rightarrow$ grows with the size of the graph
        - Node classification

    ### 3.3 Embedding entire graphs
    - Tasks: Classification toxic vs non-toxic, identifying anomalous graph
    - Simple idea: Node embedding $\rightarrow$ sum(avg) node embeddings
    - "Virtual node": Representation of sub(graph) $\rightarrow$ node2vec
    - Anonymous walk: instead of labels of nodes, index of first time we visited the node
        - Grows exponentially
        - Vector of n possible walk, record number of visits, probability distribution over walks 
        - Sampling: graph as a probability distribution, m random walks 
        $$m = (\frac{2}{\epsilon^2}(log(2^\eta - 2) - log(\delta)))$$
    - Walk embedding: $z_G$ - embedding of entire graph
        - Predict walks that co-occur in $\triangle$ = 1
        - Objective: $max_{Z, d} \frac{1}{T }\sum_{t=\triangle}^{T - \triangle} log P(w_t| w_{t - \triangle},...,w_{t + \triangle}, z_G)$
        - $N_R(u)$ set of random walk from u
        - $z_G$ will feed into neural network or use dot product 
    - Usage of node embedding:
        - Clustering
        - Node classification
        - Link prediction
        - Graph classification

## 4. PageRank
- Node embedding, random walk and matrix factorization are closely related
    ### 4.1. PageRank
    - Link: navigational vs transactional link
    - Web: directed graph
    - Not equally important
    - Link:
        - As vote: in-link and out-link (easier to fake)
        - Not all in-links are equal (recursive)
        - Out-link = $\frac{r_i}{d_i}$, importance = summation of in-link
    - Stochastic adjacency matrix M 
        - Page j, $d_j$ out-links, column stochastic matrix, sum to 1
    - Rank vector r 
        - Importance score of page i
        - Sum up to 1
        - r = M . r
    - Stationary distribution: p(t+1) = M p(t) = p(t)
    - Note: rank vector = principal eigenvector (eigenvalue of 1)
    - Long-term distribution that satisfies M(M(...M(Mu)))

    ### 4.2 Solving for rank vector
    - Problem: Pages are dead ends (no out-links) and spider trap
    - Spider trap: repeat within the same subgroup
        - Solution: $\beta$ (0.8-0.9) follows the same path but 1 - $\beta$ teleport 
        - Not what we want: only one page is important, the rest is useless
    - Dead end: Converge to 0, "leak out"
        - Fix the matrix with the column full of 0 with equal distribution
        - Mathematically problem: make the column stochastic 
    - PageRank equation: 
    $$r_j = \sum_{i\rightarrow j} \beta\frac{r_i}{d_i} + (1 - \beta)\frac{1}{N}$$

    ### 4.3 Random walk with restarts
    - Bipartite User-Item Graph: Proximity on graphs
    - Node proximity: shortest path or common neighbor
    - Personalized PageRank: Not teleport uniformly, but to subset S
    - Random walks with restarts: teleport back to the starting node
    - Algorithm: random neighbor and record the visit, probability $\alpha$ to go back the set of query nodes, after a few iterations, highest visit counts will have highest proximity
    - Benefits: multiple connections, paths, direct and indirect connections, degree of the node

## 5. Message Passing and Node classification
- Label for some nodes, but not for others $\rightarrow$ semi-supervised node classification
    ### 5.1 Message Passing and node classification
    - Collective classification: assign all nodes together
    - Correlations exist: nearby nodes
        - Homophily (individual characteristic $\rightarrow$ social connections) vs influence (vice versa)
        - Guilt-by-association
    -  Classification label: features, labels of neighbor, features of neighbor
    - Guilt-by-association: Malicious/benign web page
    - Examples: Document classification, part of speech, link prediction, optical character recognition, image/3D data segmentation, entity resolution in sensor networks, spam and fraud detection
    - Markov Assumption: label of one node depends on label of its neighbor (degree 1 of neighbor)
        - Local classifier: initial labels
        - Relational classifier: correlations between nodes
        - Collective inference: propagate correlations through networks 
    - Local classifier:
        - No network information, standard classification 
    - Relational classifier: label based on labels/attributes of neighbors
    - Collective inference: apply relational classifier iteratively until inconsistency between neighboring labels is minimized
    - Semi-supervised

    ### 5.2 Relational Classifiers and Iterative Classifiers
    - Relational classifiers:
        - No node attributes
        - Class probability $Y_v$: weighted average of class probabilities of neighbors
            - Labeled nodes: ground-truth label $Y_v^*$
            - Unlabeled nodes: $Y_v$ = 0.5
        - Challenges: no node feature information and not guaranteed converge   
    - Iterative classifiers:
        - 2 classifiers:
            - $\phi_1(f_v)$: Predict node based on node feature vectors
            - $\phi_2(f_v, z_v)$: node feature vector + summary of labels of neighbors
        - Vector $z_v$: histogram of number of each label, most common label, num of different labels
            - I/O: Incoming/Outgoing neighbor label information vector
            - $I_0$ = 1: 1 incoming pages is labelled 0 
        - Use $\phi_1$ to predict the initial label $\rightarrow$ Update $z_v$ and update label $Y_v$ again
        - Challenges: Not guaranteed convergence

    