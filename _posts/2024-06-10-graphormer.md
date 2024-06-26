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

### Centrality Encoding

### Spatial Encoding

### Edge Encoding

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

---

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

<!-- ## Interactive Plots

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
'https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv'
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

Emphasis, aka italics, with _asterisks_ (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or **underscores**.

Combined emphasis with **asterisks and _underscores_**.

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

* Or minuses

- Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links.
http://www.example.com or <http://www.example.com> and sometimes
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
| _Still_  | `renders` | **nicely** |
| 1        | 2         | 3          |

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can _put_ **Markdown** into a blockquote.

Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a _separate paragraph_.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the _same paragraph_.
