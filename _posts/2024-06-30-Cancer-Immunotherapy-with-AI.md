---
layout: distill
title: Cancer Immunotherapy Design with Geometric Deep Learning
description: Cancer Immunotherapy Design with Geometric Deep Learning
tags: distill formatting
giscus_comments: false
date: 2024-06-13
featured: true

authors:
  - name: Anonymous
    url: "https://Anonymous"
    affiliations:
      name: Anonymous

bibliography: 2024-06-30-Cancer-Immunotherapy-with-AI.bib
---

# Cancer Immunotherapy Design with Geometric Deep Learning

Cancer remains one of the most formidable challenges in medicine, but recent advancements in immunotherapy and artificial intelligence (AI) are opening new frontiers in treatment. This blog post explores the intersection of cancer immunotherapy and AI.

## What is Cancer?

Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells in the body. It begins when genetic mutations in a cell's DNA disrupt the normal process of cell growth, division, and death. This leads to the formation of tumors, which can be benign (non-cancerous) or malignant (cancerous). Malignant tumors can invade nearby tissues and metastasize, spreading to other parts of the body. There are various types of cancer, classified by the type of cell initially affected <d-cite key="hanahan2011hallmarks"></d-cite>. Treatment options vary and may include surgery, radiation therapy, chemotherapy, targeted therapy, and immunotherapy.

## Why is Cancer So Difficult to Cure?

Cancer is challenging to cure due to its genetic diversity and constant mutation, leading to treatment resistance <d-cite key="siegel2020colorectal"></d-cite>. Tumors consist of heterogeneous cells, making it difficult for a single therapy to target all cancer cells effectively. Additionally, cancer cells can invade nearby tissues and metastasize, forming secondary tumors that are hard to detect and treat. The ability of cancer cells to evade the immune system and the protective nature of the tumor microenvironment further complicate treatment efforts. Moreover, many cancer treatments, such as chemotherapy and radiation, can harm healthy cells, causing severe side effects and limiting safe dosage levels. Late detection often means the disease is more advanced and harder to treat successfully. Personalized treatment plans are necessary but complex to develop, as each patient's cancer is unique. These factors underscore the need for ongoing research to develop more effective and targeted therapies.

## What is Cancer Immunotherapy and What Makes It Different?

Cancer immunotherapy is a treatment that enhances the body's immune response to fight cancer. It involves mechanisms such as the Major Histocompatibility Complex (MHC) and T-cell response. MHC is a protein complex found on almost all nucleated cells in the body that presents fragments of proteins (antigens) on their surface. These antigens can be normal self-antigens or, in the case of cancer cells, abnormal or mutated antigens. T-cells, a type of white blood cell, patrol the body searching for these antigens. When a T-cell recognizes an abnormal antigen presented by the MHC on a cancer cell, it becomes activated and can kill cancer cells directly or recruit other immune cells to help eliminate the cancer.

Cancer immunotherapy is relatively new compared to traditional treatments like surgery, chemotherapy, and radiation. The concept dates back over a century, but significant advancements have been made in the past few decades. The first immune checkpoint inhibitor, ipilimumab (Yervoy), was approved by the FDA in 2011 <d-cite key="pardoll2012blockade"></d-cite>.

The main types of cancer immunotherapy include immune checkpoint inhibitors, CAR-T cell therapy, and cancer vaccines. Immune checkpoint inhibitors block proteins that prevent the immune system from attacking cancer cells more effectively. CAR-T cell therapy involves modifying a patient's T cells to target and kill cancer cells <d-cite key="june2018car"></d-cite>. Cancer vaccines stimulate the immune system to attack cancer cells by presenting them with specific antigens found on cancer cells <d-cite key="melief2015therapeutic"></d-cite>.

Immunotherapy differs from traditional treatments in several ways. It enhances the body's immune response to identify and attack cancer cells, potentially leading to long-lasting protection against cancer recurrence. Immunotherapy can be highly specific, targeting only cancer cells and sparing normal cells, reducing side effects compared to traditional therapies. Some patients experience long-term remission with immunotherapy, as the immune system can continue to recognize and attack cancer cells even after treatment ends. Additionally, immunotherapy can be combined with other cancer treatments to enhance overall effectiveness.

These distinctive features make cancer immunotherapy a promising and rapidly evolving area in cancer treatment. Understanding the interaction between the immune system and cancer is key to overcoming the challenges and improving the effectiveness of these therapies. Therefore, it is essential to explore how AI can enhance the development and effectiveness of cancer immunotherapy.

## AI for Cancer Immunotherapy

The two most important interactions are peptide-MHC interaction, to predict which peptides will be shown at the cell surface, and MHC-T-Cell interaction. In this blog post, I will focus on peptide-MHC structure modeling. Accurate modeling of the 3D structure and properties of p-MHC can lead to better-targeted therapies and improved immune responses against cancer cells. Additionally, speed is extremely important due to the massive number of peptide-MHC pairs for each patient and the diversity of MHCs. While there is a very good physical simulation model called Pandora <d-cite key="marzella2022pandora"></d-cite>, it is much slower than potential neural network-based models. One major challenge for using neural networks here is that there is very limited data available, making it challenging to generalize from.

## Peptide-MHC Structure Modeling with Diffusion Models

We need to choose a framework for modeling p-MHC. AI for structural biology has developed quickly in the last few years. The first major breakthroughs in using AI for biology came with non-generative models like AlphaFold 2 <d-cite key="jumper2021highly"></d-cite> and RoseTTAFold <d-cite key="baek2021accurate"></d-cite>. AlphaFold 2 and RoseTTAFold revolutionized protein structure prediction by accurately modeling the three-dimensional shapes of proteins from their amino acid sequences. Both models have significantly advanced our understanding of protein folding, paving the way for new therapeutic discoveries. However, generative models represent the next frontier in AI for biology. Unlike non-generative models that predict static outcomes, generative models can create new data instances, offering more dynamic and flexible solutions. Diffusion models <d-cite key="yang2023diffusion"></d-cite>, flow matching <d-cite key="lipman2022flow"></d-cite>, and GFlowNets <d-cite key="bengio2023gflownet"></d-cite> are prominent examples of generative models. Diffusion models, in particular, have gained popularity for their ability to generate high-quality molecular structures by simulating the gradual process of denoising from a random state.

Work using Diffusion Models such as DiffSBDD <d-cite key="schneuing2022structure"></d-cite> has shown that it is possible to create small molecules for protein pockets using diffusion models. RF-Diffusion has demonstrated the ability to design the backbone structure of proteins with these models. Very recently, AlphaFold 3 <d-cite key="abramson2024accurate"></d-cite> has put diffusion models in the spotlight again by replacing AlphaFold 2's structure prediction model, extending the structure prediction capabilities from proteins to any multi-component complex in biology. Given their flexibility and power, we choose a diffusion model for our task.

### Choosing an Input Representation

For p-MHC we are dealing with a large number of degrees of freedom. For this reason, our first aim is to reduce the degrees of freedom with prior knowledge as much as possible. It has been shown that modeling peptides and proteins at their amino acid level is sufficient to model the full structure. This means that instead of modelling every single atom of the peptide and the MHC pocket we only model a single node for each amino acid. We can later add back the remaining atoms using a simple non-generative regression model <d-cite key="dauparas2022robust"></d-cite>. For the amino acid, we can use several representations:

- **Frame-based representation:** Captures the structure of the entire amino acid.
- **C-alpha only representation:** Uses only the alpha carbon of the amino acid, simplifying the structure.
- **C-alpha + side chain orientation representation:** Includes both the alpha carbon and the side chain orientation, providing more detailed structural information.

In this blog post, we will discuss C-alpha only representation and C-alpha + side chain orientation representation. Since only the protein pocket is relevant for the binding process and its structure is largely fixed, we can use AlphaFold2 to predict the structure of the MHC and its pocket. There are a lot more peptides than there are MHCs for each person, which makes this approach computationally feasible.

First, we use the 3D position of the alpha carbon (C-alpha only representation) and a one hot encoding of the amino acid type as node feature.

### Concept of Equivariance and Equivariant Neural Networks

For the neural network, we can use our knowledge of geometric representations. The joint translation and rotation of the peptide-MHC complex does not matter for its function. But if we use a normal graph neural network, then it would have to learn to treat a translated or rotated complex the same as the original complex. This usually requires more data and is generally harder. For our task, we want our neural network to be equivariant to rotation and translation but not to reflection, hence SE(3) normally. An equivariant function with respect to a group transforms the input in a way that applying a group action to the input is equivalent to applying the same group action to the output. There are different ways to encode equivariance into a network's architecture. The most straightforward way is to only let the network use inherently equivariant information.

For a graph in \(\mathbb{R}^3\), the weighted sum of pairwise differences between nodes is inherently equivariant with respect to rotation and translation. This is the key idea behind the construction of the EGNN network <d-cite key="satorras2021n"></d-cite>, which is the default choice for equivariant diffusion models.

Using amino acids as nodes is very efficient but it does lose a lot of potential information, for instance, the amino acid sidechain orientation (C-alpha + side chain orientation representation). While the neural network might be able to learn this implicitly from enough data, we can also give the orientation as further input without increasing the number of nodes. This, however, changes the space that the SE(3) group acts on to \(\mathbb{R}^3 \times S^2\) or \(\mathbb{R}^3 \times S^3\). The advantage of using \(\mathbb{R}^3 \times S^2\) is that it is a lot more computationally efficient than indexing on \(\mathbb{R}^3 \times S^3\). Further, \(\mathbb{R}^3 \times S^2\) is sufficient to fully represent information on \(\mathbb{R}^3 \times S^3\).

Ponita <d-cite key="bekkers2023fast"></d-cite> is a new, very lean and fast architecture. It achieves equivariance on \(\mathbb{R}^3 \times S^2\) by defining a bijective mapping from any point pair to an invariant attribute such that any point pair in the equivalence class of the group G is mapped to the same attribute and any attribute maps only to one such equivalence class.

$$
\mathbb{R}^3 \times S^2 : \quad [(\mathbf{p}_i, \mathbf{o}_i), (\mathbf{p}_j, \mathbf{o}_j)] \quad \mapsto \quad a_{ij} = \left( \begin{array}{c} 
\| (\mathbf{p}_j - \mathbf{p}_i) - \mathbf{o}_i^\top (\mathbf{p}_j - \mathbf{p}_i) \mathbf{o}_i \| \\
\mathbf{o}_i^\top (\mathbf{p}_j - \mathbf{o}_i) \\
\arccos (\mathbf{o}_i^\top \mathbf{o}_j)
\end{array} \right)
$$

Ponita doesn’t have the high computational overhead of using specialized Clebsch-Gordan tensor products and is very efficient due to the ability to separate group convolutions over different group parts. It also achieves state-of-the-art performance on QM9.

Both using the C-alpha atom only representation with an EGNN and using the C-alpha atom + sidechain orientation representation with Ponita have shown great promise for modeling molecules and for conditional generation in my experiments. Peptide-MHC is difficult as the data we are working with is often not very diverse, which makes it hard to learn for generative models. There are numerous more implementation details involved in building a model like this, which exceed the scope of this blog post. The two most significant, which I want to mention, are encoding the location of the amino acids within the peptide chain as well as using energy guidance to guide the denoising process during sampling. When constructed correctly, a diffusion model can effectively predict the joint structure of peptide-MHCs.

![Diffusion Model generates peptide-MHC structure from random noise: Round points represent C-alpha atoms, while the rest forms the MHC protein pocket. Peptide nodes start as random noise and are progressively denoised by the trained diffusion model into the correct structure.](pMHC-Diffusion.png)

## Open Challenges

Beyond peptide-MHC interaction, many open problems in cancer immunotherapy can potentially be solved with AI. Modeling the structure of T-cells interacting with p-MHCs is one such problem. The regions of the T-cell receptors are highly variable, making predicting their structure even more challenging than that of p-MHC. Additionally, designing new TCRs (T-cell receptors) with feature diffusion is highly relevant for enhanced T-cell therapy, where TCRs are modified to target a patient's specific cancer. Adding time dynamics to the structure modeling is also an intriguing direction, as biological complexes are always dynamic and change their structure to fulfill different functions at different times. Other promising methods for molecule generation include Diffusion Models, Flow Matching, and GFlow Networks. Finally, there is a significant need for better evaluation metrics. Current metrics often result in molecules that look good theoretically but fail in biological experiments.

## Conclusion

AI has immense potential to revolutionize cancer immunotherapy by providing advanced tools to model and predict complex biological interactions. While significant progress has been made, many challenges remain. Continued advancements in AI and machine learning will be critical in overcoming these challenges and developing more effective, personalized cancer treatments.

## References

[1] Josh Abramson et al. “Accurate structure prediction of biomolecular interactions with AlphaFold 3”. In: Nature (2024), pp. 1–3.  
[2] Minkyung Baek et al. “Accurate prediction of protein structures and interactions using a three-track neural network”. In: Science 373.6557 (2021), pp. 871–876.  
[3] Erik J Bekkers et al. “Fast, Expressive SE (n) Equivariant Networks through Weight-Sharing in Position-Orientation Space”. In: arXiv preprint arXiv:2310.02970 (2023).  
[4] Yoshua Bengio et al. “Gflownet foundations”. In: The Journal of Machine Learning Research 24.1 (2023), pp. 10006–10060.  
[5] Justas Dauparas et al. “Robust deep learning–based protein sequence design using ProteinMPNN”. In: Science 378.6615 (2022), pp. 49–56.  
[6] Douglas Hanahan and Robert A Weinberg. “Hallmarks of cancer: the next generation”. In: cell 144.5 (2011), pp. 646–674.  
[7] John Jumper et al. “Highly accurate protein structure prediction with AlphaFold”. In: nature 596.7873 (2021), pp. 583–589.  
[8] Carl H June et al. “CAR T cell immunotherapy for human cancer”. In: Science 359.6382 (2018), pp. 1361–1365.  
[9] Yaron Lipman et al. “Flow matching for generative modeling”. In: arXiv preprint arXiv:2210.02747 (2022).  
[10] Dario F Marzella et al. “PANDORA: a fast, anchor-restrained modelling protocol for peptide: MHC complexes”. In: Frontiers in Immunology 13 (2022), p. 878762.  
[11] Cornelis JM Melief et al. “Therapeutic cancer vaccines”. In: The Journal of clinical investigation 125.9 (2015), pp. 3401–3412.  
[12] Drew M Pardoll. “The blockade of immune checkpoints in cancer immunotherapy”. In: Nature reviews cancer 12.4 (2012), pp. 252–264.  
[13] Vıctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. “E (n) equivariant graph neural networks”. In: International conference on machine learning. PMLR. 2021, pp. 9323–9332.  
[14] Arne Schneuing et al. “Structure-based drug design with equivariant diffusion models”. In: arXiv preprint arXiv:2210.13695 (2022).  
[15] Rebecca L Siegel et al. “Colorectal cancer statistics, 2020”. In: CA: a cancer journal for clinicians 70.3 (2020), pp. 145–164.  
[16] Ling Yang et al. “Diffusion models: A comprehensive survey of methods and applications”. In: ACM Computing Surveys 56.4 (2023), pp. 1–39.
