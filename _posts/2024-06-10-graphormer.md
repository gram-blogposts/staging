---
layout: distill
title: Do Transformers Really Perform Bad for Graph Representation?
description: A first-principles blog post to understanding the Graphormer.
tags: distill formatting
giscus_comments: true
date: 2022-06-10
featured: true


authors:
  - name: Anonymized
    url: "https://en.wikipedia.org/wiki/Anonymized"
    affiliations:
      name: Anonymized

bibliography: distill-template/2018-12-22-distill.bib

toc:
  # - name: Introduction & Motivation
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  # - name: Preliminaries
  - name: Graphormer
    subsections:
        - name: Introduction
        - name: Preliminaries
        - name: Centrality Encoding
        - name: Spatial Encoding
        - name: Edge Encoding
        - name: VNode
  - name: Theoretical aspects on expressivity
    subsections:
        - name: Fact 1 and 2
        - name: Fact 3
  - name: Results
  - name: Extra?

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---



## Graphormer
<!-- abcd -->



### Introduction
The Transformer architecture has revolutionized sequence modeling. Its versatility is demonstrated by its application in various domains, from natural language processing to computer vision to even reinforcement learning. With its strong ability to learn rich representations across domains, it seems natural that the power of the transformer can be adapted to graphs. 

The main challenge with applying a transformer to graph data is that there is no obvious sequence-based representation of graphs. Graphs are commonly represented by adjacency matrices or lists, which lack inherent order and are thus unsuitable for transformers.

The primary reason for finding a sequence-based representation of a graph is to combine the advantages of a transformer (such as its high scalability) with the ability of graphs to capture non-sequential and multidimensional relationships. Graph Neural Networks (GNNs) employ various constraints during training, for example, GNNs apply various constraints during training, such as enforcing valency limits when generating molecules. However, choosing such constraints may not be as straightforward for other problems. With transformers, we can apply these very constraints in a simpler manner, analogous to applying a causal mask. A transformer-based generalization of graphs can also aid in discovering newer ways to apply constraints in GNNs by presenting existing concepts in an intuitive manner.

This is where Graphormer and its various novelties come in. Graphormer introduces Centrality Encoding to capture the node importance, Spatial Encoding to capture the structural relations, and Edge Encoding to capture the nonlinear relationships between nodes. In addition to this, Graphormer makes other architecture more explainable and easier to implement by making various existing architecture special cases of Graphormer. 

---

### Preliminaries

- **Graph Neural Networks (GNNs)**: Consider a graph $$G = \{V, E\}$$ where $$V = \{v_1, v_2, \cdots, v_n\}$$ and $$n = |V|$$ is the number of nodes. Each node $$v_i$$ has a feature vector $$x_i$$. Modern GNNs update node representations iteratively by aggregating information from neighbors. The representation of node $$v_i$$ at layer $$l$$ is $$h^{(l)}_i$$, with $$h_i^{(0)} = x_i$$. The aggregation and combination at layer $$l$$ are defined as: 
  $$a_{i}^{(l)}=\text{AGGREGATE}^{(l)}\left(\left\{h_{j}^{(l-1)}: j \in \mathcal{N}(v_i)\right\}\right)$$ 
  $$h_{i}^{(l)}=\text{COMBINE}^{(l)}\left(h_{i}^{(l-1)}, a_{i}^{(l)}\right)$$ 
  where $$\mathcal{N}(v_i)$$ is the set of neighbors of $$v_i$$. Common aggregation functions include MEAN, MAX, and SUM. The COMBINE function fuses neighbor information into the node representation. For graph-level tasks, a READOUT function aggregates node features $$h_i^{(L)}$$ from the final iteration into a graph representation $$h_G$$:
  $$h_{G}=\operatorname{READOUT}\left(\left\{h_{i}^{(L)} \mid v_i \in G \right\}\right)$$
  READOUT can be a simple summation or a more complex pooling function.
 
- **Transformer**: The Transformer architecture comprises layers with two main components: a self-attention module and a position-wise feed-forward network (FFN). Let $$H = [h_1^\top, \cdots, h_n^\top]^\top\in ℝ^{n\times d}$$ be the input to the self-attention module, where $$d$$ is the hidden dimension and $$h_i\in ℝ^{1\times d}$$ is the hidden representation at position $$i$$. The input $$H$$ is projected using matrices $$W_Q\inℝ^{d\times d_K}, W_K\inℝ^{d\times d_K}$$, and $$W_V\inℝ^{d\times d_V}$$ to obtain representations $$Q, K, V$$. Self-attention is computed as:
  $$Q = HW_Q,\ K = HW_K,\ V = HW_V,\ A = \frac{QK^\top}{\sqrt{d_K}},\ Attn(H) = \text{softmax}(A)V$$
  where $$A$$ captures the similarity between queries and keys. This self-attention mechanism allows the model to understand relevant information in the sequence comprehensively.

<!-- For simplicity of illustration, we consider the single-head self-attention and assume $$d_K = d_V = d$$. The extension to the multi-head attention is standard and straightforward, and we omit bias terms for simplicity. xxxx One of the main properties of the Transformer that makes it so effective in processing sequences is its ability to model long-range dependencies and contextual information with its receptive field. Specifically, each token in the input sequence can interact with (or pay “attention” to) every other token in the sequence when transforming its representation xxxx. -->

<!-- ![ [Source](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)](Spatial%20Encoding%20d515dd50b6354ab19b8310fab3005464/Untitled.png) -->
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-10-graphormer/Untitled.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An illustration of the attention mechanism.
</div>

<!-- An illustration of attention mechanism at play for a translation task. Notice how each word(or token) can attend to different parts of the sequence, forward or backward. [Source](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html) -->

---

### Centrality Encoding

In a sequence modeling task, Attention captures the semantic correlations between the nodes (tokens).
The goal of this encoding is to capture the most important nodes in the graph.
Let's take an example.
Say we want to compare airports and find which one is the largest.
We need a common metric to compare them, so we take the sum of the total daily incoming and outgoing flights, giving us the busiest airports. This is what the algorithm is doing logically to identify the 'busiest' nodes.
Additionally, the learnable vectors allow the Graphormer to 'map' out the nodes. All this culminates in better performance for graph-based tasks such as molecule generation.

To understand how this works, let's cover a few terms. 

<!-- <convert to bullet list> -->
- Indegree - Number of incoming edges incident on a vertex in a directed graph.
- Outdegree - Number of outgoing edges from a vertex in a directed graph.

Now we can understand the Centrality Encoding, which is given as: 

$$h_{i}^{(0)} = x_{i} + z^{-}_{deg^{-}(v_{i})} + z^{+}_{deg^{+}(v_{i})}$$

Let's analyse this term by term:

- $$h_{i}^{(0)}$$ - Representation ($$h$$) of vertice i ($$v_{i}$$) at the 0th layer (first input)
- $$x_{i}$$ - Feature vector of vertice i ($$v_{i}$$)
- $$z^{-}_{deg^{-}(v_{i})}$$ - Learnable embedding vector ($$z$$) of the indegree ($$deg^{-}$$) of vertice i ($$v_{i}$$)
- $$z^{+}_{deg^{+}(v_{i})}$$ - Learnable embedding vector ($$z$$) of the outdegree ($$deg^{+}$$) of vertice i ($$v_{i}$$)

This is an excerpt of the code used to compute the Centrality Encoding

```py
self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0) 
self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

node_feature = (node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree))
```

<!-- num_in_degree is the indegree and hidden_dim is the size of the embedding vector - the Embedding function call converts this number (indegree) to a learnable vector of size hidden_dim, which is then added to the node_feature. A similar procedure is done with num_out_degree, resulting in the implementation of Equation 5. -->


<!-- <put simple explanation first then equations and code> - talk about graph based example -->

---


### Spatial Encoding


There are several methods for encoding the position information of the tokens in a sequence.
In a graph, however, there is a problem. Graphs consist of nodes (analogous to tokens) connected with edges in a non-linear, multi-dimensional space. There’s no inherent notion of an “ordering” or a “sequence” in its structure, but as with positional information, it’ll be helpful if we inject some sort of structural information when we process the graph. 

<!-- A naive solution would be to learn the encodings themselves. Another would be to perform some operation on the graph structure, such as a random walk, or components from the feature matrix. The intuition is to perform an operation on the graph to extract some “structural” information.  -->

The authors propose a novel encoding called *Spatial Encoding.* The idea is a simple combination of learnable encodings and walk-based methods mentioned earlier: take as input a pair of nodes (analogous to tokens) and output a scalar value as a function of the shortest-path-distance (SPD) between the nodes. This scalar value is then added to the element corresponding to the operation between the two nodes in the Query-Key product matrix. 

$$
A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + b_{\phi(v_i, v_j)}
$$


The above equation shows the modified computation of the Query-Key Product matrix. Notice that the additional term $$b_{\phi(v_i, v_j)}$$  is a learnable scalar value and acts like a bias term. Since this structural information is independent of which layer of our model is using it, we share this value across all layers. 

The benefits of using such an encoding are: 
1. Our receptive field has effectively increased, as we are no longer limited to the information from our neighbours, as is what happens in conventional message-passing networks.
2. The model determines the best way to adaptively attend to the structural information. For example - if the scalar valued function is a decreasing function for a given node, we know that the nodes closer to our node are more important than the farther ones.


---

### Edge Encoding

Graphormer's edge encoding method significantly enhances the way the model incorporates structural features from graph edges into its attention mechanism. The prior approaches either add edge features to node features or use them during aggregation, propagating the edge information only to associated nodes. Graphormer's approach ensures that edges play a vital role in the overall node correlation. We consider the shortest path and the specific features of edges along that path, and this way, the model can better capture spatial relationships within the graph.

Initially, node features $$(h_i, h_j)$$ and edge features $$(x_{e_n})$$ from the shortest path between nodes are processed. For each pair of nodes $$(v_i, v_j)$$, the edge features on the shortest path $$SP_{ij}$$ are averaged after being weighted by learnable embeddings $$(w^E_n)$$, this results in the edge encoding $$c_{ij}$$:

$$ c_{ij} = \frac{1}{N} \sum_{n=1}^{N} x_{e_n} (w^E_n)^T $$

This is then incorporated as the edge features into the attention score between nodes via a bias-like term. After incorporating the edge and spatial encodings, the value of $$A_{ij}$$ is now:

$$ A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + b_{\phi(v_i,v_j)} + c_{ij} $$

This ensures that edge features directly contribute to the attention score between any two nodes, allowing for a more nuanced and comprehensive utilization of edge information. The impact is significant, and it greatly improves the performance, as proven empirically in the Experiments section. 

---
### VNode

The \[VNode\] (or a Virtual Node) is arguably one of the most important contributions from the work. It is an artificial node that is connected to <b>all</b> other nodes. The authors cite this paper<d-cite key="gilmer2017neuralmessagepassingquantum"></d-cite> as an empirical motivation, but a better intuition behind the concept is as a generalization of the \[CLS\] token widely used in NLP and Vision. 
<!-- The sharp reader will notice that this has an important implication on $b$ and $\phi$, because the \[VNode\] is connected to every node, -->

$$
\phi([VNode], v) = 1, \forall v \in G
$$

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0"> <!--Dummy divs to take up space, need to do this because height, width tags don't work with the given image class-->
    </div>
    <div class="col-sm-6 mt-3 mt-md-0"> <!-- Note  this is a trick to make the image small keep it center but also not too small (using -6)-->
        {% include figure.liquid loading="eager" path="assets/img/2024-06-10-graphormer/vnode3.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    </div>
</div>

But as this is not a <b>physical connection</b>, and to provide the model with this important geometric information, $$ b_{\phi([VNode], v)} $$ is set to be a <b>distinct</b> learnable vector (for all $$v$$).



\[CLS\] tokens are often employed as "summary" tokens for text and provide a global context to the model. With graphs and text being different modalities, the \[VNode\] also helps in <b>relaying</b> global information to distant or non-connected clusters in a graph. This is significantly important to the model's expressivity, as this information might otherwise never propagate. (This is the intuition behind the upcoming proofs and has been verified empirically). In fact, the \[VNode\] becomes a learnable and dataset-specific READOUT function!

<!-- As we pointed out, \[CLS\] tokens are used for varied downstream tasks, in a similar way, \[VNode\] can be (and is) used as the final representation of the Graph, i.e., this becomes a learnable and dataset-specfic READOUT function! -->

This is can be implemented as follows:
<!-- <d-code block language="python"> -->
```python
    # Initialize the VNode
    self.v_node = nn.Embedding(1, num_heads) # one per head (different from CLS)
    ...
    # During forward pass (suppose VNode is the first node)
    ...
    headed_emb = self.v_node.weight.view(1, self.num_heads, 1)
    graph_attn[:, :, 1:, 0] = graph_attn[:, :, 1:, 0] + headed_emb
        #(n_graph, n_heads, n_nodes + 1, n_nodes + 1)
    graph_attn[:, :, 0, :] = graph_attn[:, :, 0, :] + headed_emb
    ...
```
<!-- </d-code> -->

We again emphasize that the information-relay point of view is much more important to the model than the summary-token view. The design choice of one \[VNode\] per head reflects that.

---

### Theoretical aspects on expressivity

These are the three main facts from the paper,

1. With appropriate weights and $$ \phi $$, GCN, GraphSAGE, GIN are all <b>special cases</b> of a Graphormer.
2. Graphormer is better than architectures that are limited by the 1-WL test. (so <b>all</b> traditional GNNs!)
3. With appropriate weights, <b>every node</b> representation in the output can be MEAN-READOUT.

The [spatial-encoding](link_to_spatial_eqn) provides the model with important geometric information. Observe that with an appropriate $$b_{\phi(v_i, v_j)}$$ the model can <b>find (learn)</b> neighbours for any $$v_i$$ and thus easily implement <b>mean-statistics (GCN!)</b>. By knowing the degree (some form of [centrality-encoding](link_to_centrality_eqn)), mean-statistics can be transformed to sum-statistics; it (indirectly) follows that various statistics can be learned by different heads, which leads to varied representations, and allow GraphSAGE, GIN or GCN to be modeled as a Graphormer.

Fact 2 follows from Fact 1, with GIN being the most powerful traditional GNN, which can theoretically identify all graphs distinguishable by the 1-WL test, as it is now a special case of Graphormer. The latter can do the same (& more!).

More importantly, Fact 3 implies that Graphormer allows the flow of <i>Global</i> (and Local) information within the network. This truly sets the network apart from traditional GNNs, which can only aggregate local information up to a fixed radius (or depth).

Traditional GNNs are <i>designed</i> to prevent this type of flow, as with their architecture, this would lead to over-smoothening. However, the clever design around $$[VNode]$$ prevents this from happening in Graphormer. The addition of a supernode along with Attention and the learnable $$b_{\phi(v_i, v_j)}$$ facilitate this, the $$[VNode]$$ can relay global information, and the attention mechanism can selectively choose from there.

---
### Experiments

The researchers conducted comprehensive experiments to evaluate Graphormer's performance against state-of-the-art models like GCN<d-cite key="kipf2017semisupervisedclassificationgraphconvolutional"></d-cite>, GIN<d-cite key="xu2019powerfulgraphneuralnetworks"></d-cite>, DeeperGCN<d-cite key="li2020deepergcnneedtraindeeper"></d-cite>, and the Transformer-based GT<d-cite key="dwivedi2021generalizationtransformernetworksgraphs"></d-cite>.

Two variants of Graphormer, *Graphormer* (L=12, d=768) and a smaller *GraphormerSMALL* (L=6, d=512), were evaluated on the [OGB-LSC](https://ogb.stanford.edu/docs/lsc/) quantum chemistry regression challenge (PCQM4M-LSC), one of the largest graph-level prediction dataset with over 3.8 million graphs.

The results, as shown in Table 1, demonstrate Graphormer's significant performance improvements over previous top-performing models such as GIN-VN, DeeperGCN-VN, and GT.

Table 1: Results on PCQM4M-LSC

| Model | Parameters | Train MAE | Validate MAE |
|-------|------------|-----------|--------------|
| GIN-VN | 6.7M | 0.1150 | 0.1395 |
| DeeperGCN-VN | 25.5M | 0.1059 | 0.1398 |
| GT | 0.6M | 0.0944 | 0.1400 |
| GT-Wide | 83.2M | 0.0955 | 0.1408 |
| GraphormerSMALL | 12.5M | 0.0778 | 0.1264 |
| Graphormer | 47.1M | 0.0582 | 0.1234 |

Notably, Graphormers did not encounter over-smoothing issues, with both training and validation errors continuing to decrease as model depth and width increased, thereby going beyond the *1-WL* test. In contrast, the Graph Transformer (GT) model showed no performance gain despite a significant increase in parameters from GT to GT-Wide, highlighting Graphormer's scaling capabilities.

Further experiments for graph-level prediction tasks were performed on datasets from popular leaderboards like [OGBG](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) (MolPCBA, MolHIV) and [benchmarking-GNNs](https://paperswithcode.com/paper/benchmarking-graph-neural-networks) (ZINC) which also showed Graphormers consistently outperforming top-performing GNNs.

By using the ensemble with ExpC<d-cite key="yang2020breakingexpressivebottlenecksgraph"></d-cite>, Graphormer was able to reach a 0.1200 MAE and win the graph-level track in the OGB Large-Scale Challenge.

### Comparison against State-of-the-Art Molecular Representation Models

Let's first take a look at GROVER<d-cite key="rong2020selfsupervisedgraphtransformerlargescale"></d-cite>, a transformer-based GNN boasting 100 million parameters and pre-trained on a massive dataset of 10 million unlabeled molecules.

The authors further fine-tune GROVER on MolHIV and MolPCBA to achieve competitive performance along with supplying additional molecular features such as morgan fingerprints and other 2D features. Note that the Random Forest model fitted on these features alone outperforms the GNN model, showing the huge boost in performance granted by the same.

Table 2: Comparison between Graphormer and GROVER on MolHIV

| method | # param. | AUC (%) |
|--------|---------|-----------|
| Morgan Finger Prints + Random Forest | 230K | 80.60±0.10 |
| GROVER | 48.8M | 79.33±0.09 |
| GROVER (LARGE)| 107.7M | 80.32±0.14 |
| Graphormer-FLAG | 47.0M | 80.51±0.53 |

However, as evident in Table 2, Graphormer manages to outperform it consistently on the benchmarks without even using the additional features (known to boost performance), which showcases it increases the expressiveness of complex information.

In conclusion, the Graphormer presents a novel way of applying Transformers to graph representation using the three structural encodings. While it has demonstrated strong performance across various benchmark datasets, significant progress has been made since the original paper. Structure-Aware Transformer <d-cite key="chen2022structureawaretransformergraphrepresentation"></d-cite> improves on the initial Transformer by incorporating structural information by extracting subgraph representations. DeepGraph <d-cite key="zhao2023layersbeneficialgraphtransformers"></d-cite> explores the benefits of deeper graph transformers by enhancing global attention with substructure tokens and local attention. Despite the success of these architectures, some challenges still remain; for example, the quadratic complexity of the self-attention module limits its use on large graphs. Therefore, future development of efficient Graphormer is necessary.



