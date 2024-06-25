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

The \[VNode\] (or a Virtual Node) is arguably one of the most important contributions from the work. It is an artificial node which is connected to <b>all</b> other nodes. Although the paper cites another paper as an empirical motivation, a better intuition behind the concept is as a generalization of the \[CLS\] token widely used in NLP and Vision for downstream tasks. The sharp reader will notice that this has an important implication on $b$ and $\phi$, because the \[VNode\] is connected to every node,

$$ \phi([VNode], v) = 1, \forall v \in G $$

But as this is not a <b>physical connection</b>, and to provide the model with this important geometric information, $b_{\phi([VNode], v)}$ is set to be a <b>distinct</b> learnable vector (for all $v$).<br>

\[CLS\] tokens are often employed as "summary" tokens for text and provide a global context to the model; In addition this default advantage, as graphs and text are different modalities, the \[VNode\] also helps in <b>relaying</b> global information to distant or non-connected clusters in a Graph, this is significantly important to the model's expressivity, as this information might otherwise never propagate. (This is the intuition behind proofs in the next section, and has also been verified empirically)<br>

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
{% endhighlight %}

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
