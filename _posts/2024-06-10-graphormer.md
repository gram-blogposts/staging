---
layout: distill
title: Do Transformers Really Perform Bad for Graph Representation?
description: A first-principles blog post to understanding the Graphormer.
tags: distill formatting
giscus_comments: true
date: 2024-06-10
featured: true

authors:
  - name: Anonymized
    url: "https://en.wikipedia.org/wiki/Anonymized"
    affiliations:
      name: Anonymized

bibliography: distill-template/2018-12-22-distill.bib

toc:
  - name: Introduction & Motivation
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Preliminaries
  - name: Graphormer
    subsections:
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

## Introduction & Motivation

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) that brought a significant improvement to the loading and rendering speed, which is now [on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).

---

## Preliminaries

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

---

## Graphormer

### Introduction
The Transformer archtechture has revolutionised the field of sequence modeling. The versatility of the architechture is demonstrated from its application from various domains, from natural language processing, to computer vision, to even reinforcement learning.
With its strong ability to learn strong representations across domains, it seems only natural that the power of the transformer can be extended to the graph domain. This paper tackles the challenge of learning graphs using a transformer. The aforementioned domains rely on the conversion of the input to a sequence be it a sequence of tokens, patches or actions. However, a graph has no such direct analogue. Taking concepts used in the transformer, one is able to encode a graph using Graphormer.

Challenges with learning a graph - there is no sequence based representation of graphs, the one representation that places the relative position of a node to another is the adjacency matrix (similar to how a sentence places the order of the tokens) , however since a node is 'placed' in 2 or more dimensions - it is incompatible to be used in a transformer 

Uses for learning graph reprensetation with transformer - 1) Scale, compute, etc.
2) setup intuition for more current stuff, constraints ...

need smooth transition to start of the paper here.

The paper introduces a Centrality Encoding in Graphormer to capture the node importance in the graph, a Spatial Encoding in Graphormer to capture the structural relation between nodes.

### Preliminaries
GNN - formulas
Transformer - formula

### Centrality Encoding

Attention in a sequence modeling task that captures the semantic correlations between nodes (tokens).

Now the goal of this encoding is to capture the most important nodes in the graph
To understand this section let's cover a few terms. 

<convert to bullet list>
i. Indegree - Number of incoming edges incident on a vertex in a directed graph. The vertex has an indegree of 2 (2 red arrows)   
ii. Outdegree - Number of outgoing edges from a vertex in a directed graph. The vertex has an outdegree of 1 (1 green arrow)

img here

Now we can understand Equation 5 which is given as: $h_{i}^{(0)} = x_{i} + z^{-}_{deg^{-}(v_{i})} + z^{+}_{deg^{+}(v_{i})}$ 

lets analyse this term by term:
- $h_{i}^{(0)}$ -> representation ($h$) of vertice i ($v_{i}$) at the 0th layer (first input)
- $x_{i}$ -> feature vector of vertice i ($v_{i}$)
- $z^{-}_{deg^{-}(v_{i})}$ -> learnable embedding vector ($z$) of the indegree ($deg^{-}$) of vertice i ($v_{i}$)
- $z^{+}_{deg^{+}(v_{i})}$ -> learnable embedding vector ($z$) of the outdegree ($deg^{+}$) of vertice i ($v_{i}$)


This is an excerpt of the the code used to to compute the Centrality Encoding
```py
self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0) 
self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)

node_feature = (node_feature + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree))
```
num_in_degree is the indegree and hidden_dim is the size of the embedding vector - the Embedding function call converts this number (indegree) to a learnable vector of size hidden_dim, which is then added to the node_feature. A similar procedure is done with num_out_degree, resulting in the implementation of Equation 5.


With the 'how' being understood, one must understand why such a system works.
To explain this lets take an example.
Say I want to compare airports around the world, and find which one is the largest.
I need a common metric across all to compare them, so I simply take the sum of the total daily incoming and outgoing flights, giving me the world's busiest airports. This is what the algorithm is doing on a logical level, to identify the 'busiest' nodes.
Additionally, the learnable vectors allow the Graphormer to 'map' out the nodes of the graph.
The softmax function allows the capturing of this information, called the node importance signal in the paper.

<put simple explanation first then equations and code> - talk about graph based example

### Spatial Encoding
shift some passages, to preliminary
Before trying to understand about what this encoding does and why it’s needed, let’s take a small detour about how positional encodings work in transformers. 

One of the main properties of the Transformer architecture that makes it so effective in processing sequences is its ability to model long-range dependencies and contextual information with its receptive field. In more specific terms, each token in the input sequence can interact with (or pay “attention” to) every other token in the sequence when transforming its representation. The mechanism, called *self-attention,* allows the model to gain a more comprehensive understanding of the relevant information encoded in the sequence. 

<!-- ![ [Source](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)](Spatial%20Encoding%20d515dd50b6354ab19b8310fab3005464/Untitled.png) -->
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-10-graphormer/Untitled.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An illustration of attention mechanism at play for a translation task. Notice how each word(or token) can attend to different parts of the sequence, forward or backward.
</div>

An illustration of attention mechanism at play for a translation task. Notice how each word(or token) can attend to different parts of the sequence, forward or backward. [Source](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html)

trim below, and keep here, let all above be shift to preliminary.
However, the information about the position of the token in the sequence is lost in this mechanism. This is because each token is interacting with all other tokens in the sequence, and thus making it difficult for the transformer to understand the positioning of each token. This doesn’t happen in traditional networks such as RNNs or  LSTMs, as they process each token at a time and thus the auto-regressive relationship enables it to learn about the occurence of a token in the sequence.

(If you are thinking about why positional information is important in the first place, take a translation task for example. You wouldn’t want your translated sentence to be all jumbled up and consisting of keywords all around the place! [The original paper](https://arxiv.org/abs/1706.03762) has an ablation study which empirically confirms this)


Literature uses several methods for encoding this position information.  In one such method, each position in the input sequence is given a unique embedding vector, which is added to the token embeddings. This explicitly tells the transformer the position of each token. Other methods rely on using “relative” positional information by encoding the relative distances between them. This means the model knows how far apart any two tokens are in the sequence. All these methods are shown to give good results. We won’t go into the details here, as there are several amazing articles and resources on the web which explain this.

Let’s get back to graphs.

You probably already have noticed a problem with our setup. Graphs consist of nodes(analogous to tokens), connected with edges in a non-linear, multi-dimensional space. There’s no inherent notion of an “ordering” or a “sequence” in its structure, but as with positional information it’s gonna be helpful if we inject some sort of structural information when we process the graph-structured data. 

How can we go about doing this? A naive way would be to just learn the encodings themselves. Another way would be to perform some sort of an operation on the graph structure. Examples include random-walk methods and Laplacian eigenvectors of the node feature matrix. The intuition would be to perform an operation on the structure to extract some “structural” information. 

The authors propose a novel a novel encoding, which they call *Spatial Encoding.* The idea is a simple combination of learnable encodings and walk-based methods mentioned earlier:  take as input a pair of nodes(reminder: analogous to tokens) and output a scalar value as a function of the shortest-path-distance between the nodes. This scalar value is then added to the element corresponding to the operation between the two nodes in the Query-Key product matrix. 

$$
A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + b_{\phi(v_i, v_j)}
$$

The above equation shows the modified computation of the Query-Key Product matrix. Notice that the additional term $b_{\phi(v_i, v_j)}$  is a learnable scalar value and acts like a bias term. Since this strucutral information is independent of which layer of our model is using it, we let this value be shared across all the layers. 

The benefits of using such a formulation of the encoding is several fold - 1. Our receptive field is effectively increased, as we are no longer limited to the information from our neighbours, as what happens in conventional message-passing networks, and 2. we let the model figure out the best way to adaptively attend to the structural information. For example - if the scalar valued function is a decreasing function for a given node, we know that the nodes closer to our node are more important(in some sense) compared to the ones far away.


### Edge Encoding
Graphormer's edge encoding method significantly enhances the way the model incorporates structural features from graph edges into its attention mechanism. The prior approaches either add edge features to node features or use them during aggregation, propagating the edge information only to associated nodes. Graphormer's approach ensures that edges play a vital role in the overall node correlation. We consider the shortest path and the specific features of edges along that path, and this way, the model can better capture spatial relationships within the graph.

Initially, node features $(h_i, h_j)$ and edge features $(x_{e_n})$ from the shortest path between nodes are processed. For each pair of nodes $(v_i, v_j)$, the edge features on the shortest path $SP_{ij}$ are averaged after being weighted by learnable embeddings $(w^E_n)$ (however, the authors have not explained the rationale behind using the mean value rather than any other aggregation), this results in the edge encoding $c_{ij}$:

$$ c_{ij} = \frac{1}{N} \sum_{n=1}^{N} x_{e_n} (w^E_n)^T $$

This edge encoding is then incorporated as the edge features into the attention score between nodes via a bias term. However, you may question why are these features (edge and spatial encodings) being added to the attention scores as such. When considering where to incorporate these features into the attention calculation, it’s essential to ensure that the chosen approach carries over to all layers rather than being limited to just the start or end. After we incorporate the edge and spatial encodings as bias, the value of $A_{ij}$ is modified to be:

$$ A_{ij} = \frac{(h_i W_Q)(h_j W_K)^T}{\sqrt{d}} + b_{\phi(v_i,v_j)} + c_{ij} $$

This process ensures that edge features directly contribute to the attention score between any two nodes, allowing for a more nuanced and comprehensive utilization of edge information. The impact of this method is significant, and it greatly improves the model's performance, as proven empirically in the Experiments section. 

### VNode

The \[VNode\] (or a Virtual Node) is arguably one of the most important contributions from the work. It is an artificial node which is connected to <b>all</b> other nodes. Although the paper cites another <a href="https://arxiv.org/abs/1704.01212">paper</a> as an empirical motivation, a better intuition behind the concept is as a generalization of the \[CLS\] token widely used in NLP and Vision for downstream tasks. The sharp reader will notice that this has an important implication on $b$ and $\phi$, because the \[VNode\] is connected to every node,

$$ 
\phi([VNode], v) = 1, \forall v \in G 
$$

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0"> <!--Dummy divs to take up space, need to do this because height, width tags don't work with the given image class-->
    </div>
    <div class="col-sm-6 mt-3 mt-md-0"> <!-- Note  this is a trick to make the image small keep it center but also not too small (using -6)-->
        {% include figure.liquid loading="eager" path="assets/img/2024-06-10-graphormer/VNode.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    </div>
</div>

But as this is not a <b>physical connection</b>, and to provide the model with this important geometric information, $b_{\phi([VNode], v)}$ is set to be a <b>distinct</b> learnable vector (for all $v$).


\[CLS\] tokens are often employed as "summary" tokens for text and provide a global context to the model; In addition to this default advantage, as graphs and text are different modalities, the \[VNode\] also helps in <b>relaying</b> global information to distant or non-connected clusters in a Graph, this is significantly important to the model's expressivity, as this information might otherwise never propagate. (This is the intuition behind proofs in the next section, and has also been verified empirically)


As we pointed out, \[CLS\] tokens are used for varied downstream tasks, in a similar way, \[VNode\] can be (and is) used as the final representation of the Graph, i.e., this becomes a learnable and dataset-specfic READOUT function!

This is <i>extremely</i> simple in code and can be implemented as follows (in PyTorch):
<d-code block language="python">
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
</d-code>

We would again emphasize that the information-relay point of view is much more important to the model than the summary-token view, the design choice of one \[VNode\] per head reflects that.

### Experiments

The Graphormer was benchmarked against state-of-the-art GNNs like GCN, GIN, their VN variants, as well as other leading models such as multi-hop GIN, [DeeperGCN](https://arxiv.org/abs/2006.07739), and the Transformer-based [GT](https://arxiv.org/abs/2012.09699) model.

Two model sizes, *Graphormer* (L=12, d=768) and a smaller *GraphormerSMALL* (L=6, d=512), were evaluated on the [OGB-LSC](https://ogb.stanford.edu/docs/lsc/) quantum chemistry regression challenge (PCQM4M-LSC), one of the largest graph-level prediction dataset with over 3.8 million graphs where it significantly outperformed previous state-of-the-art models like GIN-VN and GT as seen in Table 1. Notably, Graphormers did not encounter over-smoothing issues, with both training and validation errors continuing to decrease as model depth and width increased, thereby going beyond the *1-WL* test.

Table 1: Results on PCQM4M-LSC

| method | # param. | train MAE | validate MAE |
|--------|---------|-----------|--------------|
| GCN | 2.0M | 0.1318 | 0.1691 |
| GIN | 3.8M | 0.1203 | 0.1537 |
| GCN-VN | 4.9M | 0.1225 | 0.1485 |
| GIN-VN | 6.7M | 0.1150 | 0.1395 |
| GINE-VN | 13.2M | 0.1248 | 0.1430 |
| DeeperGCN-VN | 25.5M | 0.1059 | 0.1398 |
| GT | 0.6M | 0.0944 | 0.1400 |
| GT-Wide | 83.2M | 0.0955 | 0.1408 |
| GraphormerSMALL | 12.5M | 0.0778 | 0.1264 |
| Graphormer | 47.1M | 0.0582 | 0.1234 |

Further experiments for graph-level prediction tasks were performed on datasets from popular leaderboards like [OGBG](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol) (MolPCBA, MolHIV) and [benchmarking-GNNs](https://paperswithcode.com/paper/benchmarking-graph-neural-networks) (ZINC) which also showed Graphormers consistently outperforming top-performing GNNs.

Table 2: Results on MolPCBA

| method | #param. | AP (%) |
|--------|---------|--------|
| DeeperGCN-VN+FLAG | 5.6M | 28.42±0.43 |
| DGN | 6.7M | 28.85±0.30 |
| GINE-VN | 6.1M | 29.17±0.15 |
| PHC-GNN | 1.7M | 29.47±0.26 |
| GINE-APPNP | 6.1M | 29.79±0.30 |
| GIN-VN (fine-tune) | 3.4M | 29.02±0.17 |
| Graphormer-FLAG | 119.5M | 31.39±0.32 |

Table 3: Results on MolHIV.

| method | #param. | AUC (%) |
|--------|---------|---------|
| GCN-GraphNorm | 526K | 78.83±1.00 |
| PNA | 326K | 79.05±1.32 |
| PHC-GNN | 111K | 79.34±1.16 |
| DeeperGCN-FLAG | 532K | 79.42±1.20 |
| DGN | 114K | 79.70±0.97 |
| GIN-VN (fine-tune) | 3.3M | 77.80±1.82 |
| Graphormer-FLAG | 47.0M | 80.51±0.53 |

Table 4: Results on ZINC.

| method | #param. | test MAE |
|--------|---------|----------|
| GIN | 509,549 | 0.526±0.051 |
| GraphSage | 505,341 | 0.398±0.002 |
| GAT | 531,345 | 0.384±0.007 |
| GCN | 505,079 | 0.367±0.011 |
| GatedGCN-PE | 505,011 | 0.214±0.006 |
| MPNN (sum) | 480,805 | 0.145±0.007 |
| PNA | 387,155 | 0.142±0.010 |
| GT | 588,929 | 0.226±0.014 |
| SAN | 508,577 | 0.139±0.006 |
| GraphormerSLIM | 489,321 | 0.122±0.006 |

The paper also dives into a series of ablation studies to assess the effects of the encodings proposed by the authors, whose results can be summed up as follows:

- Node Relation Encoding: Spatial encoding significantly outperformed traditional positional encodings like Laplacian PE, demonstrating its superior ability to capture node relationships.

- Centrality Encoding: Incorporating degree-based centrality encoding resulted in a substantial performance boost, underscoring its critical role in graph data modeling.

- Edge Encoding: The attention bias based edge encoding outperformed conventional methods, highlighting its effectiveness in capturing spatial information on edges.



## Theoretical aspects on expressivity

We first list down the three important facts from the paper and then discuss them in detail,

1. With appropriate weights and $ \phi $, GCN, GraphSAGE, GIN are all <b>special cases</b> of a Graphormer.
2. Graphormer is better than architectures that are limited by the 1-WL test. (so <b>all</b> traditional GNNs!)
3. With appropriate weights, <b>every node</b> representation in the output can be MEAN-READOUT.

### Fact 1 and 2

This is intuitive for the most part, it is not tough to absorb the fact that the [spatial-encoding](link_to_spatial_eqn) provides the model with important geometric information (clearly, the node-degree is one of them). First off, observe that with an appropriate $b_{\phi(v_i, v_j)}$ the model can <b>find (learn)</b> neighbours for any $v_i$ and thus easily implement <b>mean statistics (GCN!)</b>. Secondly, knowing the degree (some form of [centrality-encoding](link_to_centrality_eqn)), mean-statistics can be transformed to sum-statistics; and although it does not follow so directly, different and complicated statistics can be learned by different heads, which lead to varied representations, and allow GraphSAGE, GIN or GCN to be modelled as a Graphormer.

Fact 2 follows from Fact 1, as GIN is anyways the most powerful traditional GNN, which can theoretically distinguish all graphs distinguishable by the 1-WL test, now as it is just a special case of Graphormer, the latter can do the same (& more!).

The Proofs for Fact 1 are really easy to follow, but feel free to skip them.
{% details Proof(s) for Fact 1 %}
For each type of aggregation, we provide simple function and weight definitions that achieve it, 
* <b>Mean Aggregate</b> :
    - Set $ b_{\phi(v_i, v_j)} = 0 $ when $\phi(v_i, v_j) = 1$ and $-\infty$ otherwise,
    - Set $ W_Q = 0, W_K = 0$ and let $ W_V = I$ (Identity matrix), using these,
    - $$ h^{(l)}_{v_i} = \sum_{v_j \in N(v_i)} softmax(A_{ij}) * (W_v * h^{(l-1)}_{v_j}) \implies h^{(l)}_{v_i} = \frac{1}{|N(v_i)|}*\sum_{v_j \in N(v_i)} h^{(l-1)}_{v_j} $$
* <b>Sum Aggregate</b> :
    - For this, we just need to get the mean aggregate and then multiply by $ \|N(v_i)\| $,
    - Loosely, the degree can be extracted from a [centrality-encoding](link_to_centrality_eqn) by an attention head, and then the FFN can multiply this to the learned mean aggregate, the latter part is not so loose, because it is a direct consequence of the universal approximation theorem.
* <b>Max Aggregate</b> :
    - For this one we assume that if we have $t$ dimensions in our hidden state, we <i>also</i> have t heads.
    - The proof is such that each Head will extract the maximum from neighbours, clearly, to only keep immediate neighbours around, we can use the same formulation for $b$ and $\phi$ as in the mean aggregate.
    - Using $W_K = e_t$ (t-th unit vector), $W_K = e_t$ and $W_Q = 0$ (Identity matrix), we can get a pretty good approximation to the max aggregate. To get the full deal however, we need a <i>hard-max</i> instead of the <i>soft-max</i> being used; to accomplish this we finally consider the bias in the query layer (i.e., something like `nn.Linear(in_dim, out_dim, use_bias=True)`), set it to $T \cdot I$ with a high enough $T$ (temperature), this will make the soft-max behave like a hard-max.
{% enddetails %}

For Fact 2, we explain the example from the paper, along with explicitly providing the final WL representation. Again, feel free to skip this part.
{% details Example for Fact 2 %}
First we need to fix some notation for the WL test, briefly, it can be expressed as -

$$ c^{(k+1)}(v) = HASH(c^{(k)}(v), \{c^{(k)}(u)\}_{u \in N(v)} )$$

where $c^{(k)}(v)$ is the $k^{th}$ iteration representation (color for convinience) of node $v$ and importantly $HASH$ is an <i>injective</i> hash function. Additionally, all nodes with the same color have the same feature vector

Given this, consider the following graphs -
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
    </div>
    <div class="col-sm-6 mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2024-06-10-graphormer/wl-test.gif" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
    </div>
</div>

The hashing process converges in one iteration itself, now the 1-WL test would count number of colors and that vector would act as the final graph representation, which for both of the graphs will be $ [0, 0, 4, 2] $ (i.e., $ [count(a), count(b), count(x), count(y)] $), even though they are different, the 1-WL test fails to distinguish them. There are several such cases and so traditional GNNs are fairly limited in their expressivity.

However for the graphormer, Shortest Path Distances (SPD) directly affects attention weights (because the paper uses SPD as $\phi(v_i, v_j)$), and if we look at the SPD sets for the two types of nodes (red and blue) in both the graphs, (we have ordered according to the BFS traversal by top left red node, though any ordering would suffice)

* Left graph -
    - Red nodes - $$ \{ 0, 1, 1, 2, 2, 3 \} $$
    - Blue nodes - $$ \{1, 0, 2, 1, 1, 2\} $$
* Right graph -
    - Red nodes - $$ \{0, 1, 1, 2, 3, 3\} $$
    - Blue nodes - $$ \{1, 0, 1, 1, 2, 2\} $$

What is important is not that red and blue nodes have a different SPD set, <u><i>but that these two types of nodes have different SPD sets across the two graphs</i></u>, this signal can help the model distinguish the two graphs and is the reason why Graphormer is better than 1-WL test limited architectures.
{% enddetails %}

### Fact 3

The proof behind Fact 3 is really easy to follow if you have checked the proof of [Fact1][link to facts 1 and 2]. Nevertheless, what is more important is the power it lends to the model, this fact implies that Graphormer allows the flow of <i>Global</i> information within the network (in addition to Local). This truly sets the network apart from traditional GNNs which can only aggregate local information upto a fixed radius (or depth).

Importantly, traditional GNNs are <i>designed</i> to prevent this type of a flow as with their architecture this would lead to over smoothening, however, the clever design around $[VNode]$ prevents this from happening in Graphormer. This is verified empirically and proved ahead, but intuitively the addition of a supernode along with Attention and the learnable $b_{\phi(v_i, v_j)}$ already facilitate for this, the $[VNode]$ can relay global information and the attention mechanism can selectively choose from there. If this explanation is not enough a concrete proof od the fact follows,

{% details Proof for Fact 3 %}
Setting $W_Q = W_K = 0$, and the bias terms in both to be $T \cdot 1$ (where T is temperature), as well as, setting $W_V = I$ (Identity matrix), with a large enough $T$ (much larger than the scale of $b_{\phi(v_i, v_j)}$, so that $T^2 1 1^T$ can dominate), we can get MEAN-READOUT on all nodes. Note that while this proof doesn't require $[VNode]$, it should be noted that, the $[Vnode]$ is very important to establish a <b>balance</b> between this completely global flow and the local flow. As in a normal setting, with the $T$ not being too large, the only way for global information is through the $[VNode]$, as the $b_{\phi(v_i, v_j)}$ would most likely limit information from nodes that are very far.
{% enddetails %}

[link to facts 1 and 2]: #fact-1-and-2


## Interactive Plots

You can add interative plots using plotly + iframes :framed_picture:

<div class="l-page">
  <iframe src="{{ '/assets/plotly/distill-template/demo.html' | relative_url }}" frameborder='0' scrolling='no' height="500px" width="100%" style="border: 1px dashed grey;"></iframe>
</div>

The plot must be generated separately and saved into an HTML file.
To generate the plot that you see above, you can use the following code snippet:

{% highlight python %}
import pandas as pd
import plotly.express as px
df = pd.read_csv(
'<https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv>'
)
fig = px.density_mapbox(
df,
lat='Latitude',
lon='Longitude',
z='Magnitude',
radius=10,
center=dict(lat=0, lon=180),
zoom=0,
mapbox_style="stamen-terrain",
)
fig.show()
fig.write_html('assets/distill-template/plotly/demo.html')
{% endhighlight %} -->

---

## Details boxes

Details boxes are collapsible boxes which hide additional information from the user. They can be added with the `details` liquid tag:

{% details Click here to know more %}
Additional details, where math $$ 2x - 1 $$ and `code` is rendered correctly.
{% enddetails %}

---

## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body` sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

---

## Images

This is an example post with image galleries.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/distill-template/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/distill-template/7.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>

Images can be made zoomable.
Simply add `data-zoomable` to `<img>` tags that you want to make zoomable.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/distill-template/8.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/distill-template/10.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

The rest of the images in this post are all zoomable, arranged into different mini-galleries.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/distill-template/11.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/distill-template/12.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/distill-template/7.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

---

## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or *underscores* (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or **underscores**.

Combined emphasis with **asterisks and *underscores***.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
   ⋅⋅\* Unordered sub-list.
3. Actual numbers don't matter, just that it's a number
   ⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

- Unordered list can use asterisks

- Or minuses

- Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links.
<http://www.example.com> or <http://www.example.com> and sometimes
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style:
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style:
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```

```python
s = "Python syntax highlighting"
print s
```

```
No language indicated, so no syntax highlighting.
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        |      Are      |  Cool |
| ------------- | :-----------: | ----: |
| col 3 is      | right-aligned | $1600 |
| col 2 is      |   centered    |   $12 |
| zebra stripes |   are neat    |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the
raw Markdown line up prettily. You can also use inline Markdown.

| Markdown | Less      | Pretty     |
| -------- | --------- | ---------- |
| *Still*  | `renders` | **nicely** |
| 1        | 2         | 3          |

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote.

Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*.
