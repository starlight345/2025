---

layout: distill
title: "AI810 Blog Post (20253793)" 
description: "We explore a full inverse design pipeline using TabNet-based property embedding prediction, GRU-based SMILES decoding, and reinforcement learning for structural refinement."
date: 2025-05-25
permalink: /blog/inverse-polymer/ 
future: true
htmlwidgets: true

authors:
  - name: Anonymous

bibliography: 2025-05-25-inverse-polymer.bib

toc:
  - name: Introduction
  - name: Prior Work and Motivation
  - name: TabNet-based Retrieval
  - name: GRU Decoder for SMILES Generation
  - name: Reinforcement Learning with MCTS
  - name: Challenges and Future Work
  - name: Conclusion
---


# Introduction

Polymer informatics — the data-driven understanding and design of polymers — has undergone a major transformation with the advent of machine learning. Traditional approaches that relied on costly simulations or manual heuristics are increasingly being replaced by models capable of predicting polymer properties with impressive accuracy. Recent developments in multitask graph neural networks (GNNs) have enabled joint learning across diverse property datasets, improving prediction generalizability and scalability <d-cite key="Gurnani_2023"></d-cite>. Similarly, attention-based deep tabular models such as TabNet have shown promising results in feature-aware regression tasks, especially for high-dimensional property vectors <d-cite key="DBLP:journals/corr/abs-1908-07442"></d-cite>.

Language models designed specifically for polymer sequences, like **PolyBERT**, further push the frontier by encoding pSMILES (polymer SMILES) strings into meaningful embeddings that capture structural and functional nuances <d-cite key="Kuenneth_2023"></d-cite>. These tools collectively allow efficient forward prediction: given a candidate polymer, one can quickly estimate its properties — an essential capability in materials screening pipelines.

However, the *inverse problem* — generating a valid polymer structure that matches a desired set of properties — remains largely underexplored. Solving this problem is critical for high-throughput inverse design workflows, where new candidates must be proposed to meet target functional criteria. While some recent methods like **MMPolymer** <d-cite key="wang2024mmpolymermultimodalmultitaskpretraining"></d-cite> and **PolyGET** <d-cite key="feng2023polygetacceleratingpolymersimulations"></d-cite> begin to address generalization and physics-aware modeling, they stop short of delivering full structure generation pipelines.

This blog explores a complete inverse design framework that bridges forward predictors and generative models. By combining TabNet-based embedding regressors, GRU-based SMILES decoders, and reinforcement learning-based structure refinement, we aim to create a pipeline that can both infer and construct polymers tailored to specific property targets. Ultimately, our goal is to replace the GRU decoder with a Transformer-based decoder and train a TabNet encoder that aligns with this Transformer decoder, thereby enabling efficient, gradient-guided inverse design. This architecture also avoids the need for large language models (LLMs), instead focusing on structured decoding with interpretable and task-specific modules. The long-term motivation is to align TabNet’s property-derived latent space with the decoder’s expectations, using a combination of reconstruction loss and gradient field coupling to improve consistency.

--- 

# Prior Work and Motivation

A growing body of work has advanced the capabilities of polymer property prediction. In particular, multitask GNNs trained across large-scale datasets have demonstrated strong performance by learning shared representations that transfer across different property domains <d-cite key="Gurnani_2023"></d-cite>. These models benefit from message-passing architectures that naturally align with molecular graph structures and are well-suited to high-throughput settings.

TabNet, an attention-based model for tabular data, offers interpretable yet expressive modeling of polymer properties by selecting relevant input features per decision step <d-cite key="DBLP:journals/corr/abs-1908-07442"></d-cite>. Its ability to handle heterogeneous and high-dimensional descriptors has made it attractive for regression tasks in materials science.

Simultaneously, chemical language models have emerged as powerful tools for sequence representation. **PolyBERT** <d-cite key="Kuenneth_2023"></d-cite>, for instance, uses transformer encoders pretrained on millions of pSMILES strings to generate embeddings that are structurally and semantically rich. These embeddings serve as the backbone for both property prediction and generation tasks.

Despite these advances, the majority of research has focused on *forward prediction* — estimating properties from structures — leaving the inverse task relatively underdeveloped. Recent attempts to improve generalization and robustness include **MMPolymer** <d-cite key="wang2024mmpolymermultimodalmultitaskpretraining"></d-cite>, which unifies visual, structural, and property data in a multimodal pretraining scheme, and **PolyGET** <d-cite key="feng2023polygetacceleratingpolymersimulations"></d-cite>, which incorporates equivariant transformers and forcefield supervision for high-fidelity polymer simulations. However, neither method fully addresses inverse structure generation from target properties.

Additionally, recent studies have raised concerns about the ability of large language models to interpret SMILES strings accurately. For example, Jang et al.<d-cite key="jang2025improvingchemicalunderstandingllms"></d-cite>   show that even advanced LLMs struggle with basic tasks like ring counting in SMILES. Their CLEANMOL framework emphasizes the importance of structured supervision and explicit parsing strategies — reinforcing our design decision to pursue modular, domain-aligned models instead of monolithic LLMs.

Our work fills this gap by designing a full inverse pipeline. We begin with TabNet to map properties into the PolyBERT latent space, then decode those embeddings into SMILES sequences using a GRU-based model, and finally refine the sequences using reinforcement learning to enforce validity and property satisfaction. This end-to-end approach merges retrieval, generation, and optimization — enabling data-driven polymer design beyond database lookups.

Ultimately, our goal is to replace the GRU decoder with a Transformer-based decoder and train a TabNet encoder that aligns with this Transformer decoder, thereby enabling efficient, gradient-guided inverse design. This architecture also avoids the need for large language models (LLMs), instead focusing on structured decoding with interpretable and task-specific modules.

---

# TabNet-based Retrieval

Our initial inverse task leverages TabNet to regress from a target property vector to a latent polymer embedding.

TabNet structurally incorporates Transformer-style attention, making it well-suited for sequential and feature-selection tasks. However, in the PolyOne dataset (introduced in the PolyBERT paper), the 29 properties are not sequential but rather independent columns. This raised concerns about whether TabNet, typically strong in sequential or hierarchical data, would be effective in this context.

To better understand the relationship between individual properties and the PolyBERT embedding, we applied structured masking during training — both for the property-to-embedding direction and the reverse embedding-to-property prediction. This bidirectional setup allowed us to assess how each property influences the latent space and vice versa. We hoped that by forcing the model to reason with only partial information at each step, TabNet would learn to identify key feature interactions.

We re-implemented TabNet in PyTorch to go beyond the constraints of the default scikit-learn style API. Our version supports flexible masking, introspection of learned feature importance, and compatibility with large-scale batch training. Each property was treated as an input candidate, and TabNet learned to extract signal from structured subsets of features, even in the absence of inherent temporal order.

The key insight is that while the 29 properties are nominally independent, they still exhibit correlations and latent structure that TabNet’s attention-based mechanism can capture. Our results demonstrate that TabNet can learn to associate individual and joint property patterns with meaningful embedding regions.


# Architectural Overview

TabNet operates in multiple sequential decision steps, where each step performs two main operations:

Feature Transformation: The input is passed through stacked GLU blocks, which act like fully connected layers with learnable gates. These layers extract rich nonlinear representations of the selected input subset.

Feature Selection via Attentive Transformers: Before each step, TabNet computes a sparse probability distribution over input features (a feature mask), telling the model which features to focus on. These masks are generated using a learned attention mechanism, followed by a sparsemax activation.

Unlike softmax, which assigns nonzero weights to all features, sparsemax produces sparse masks where many features are exactly zero. This aligns well with the physical intuition that only a few properties dominate each structure-property relationship in polymers.

# Why Masking Matters

We apply structured masking during both directions of training — from embedding to property and vice versa — to probe how different subsets of properties influence the latent space. This setup forces TabNet to reason with partial information, encouraging it to learn conditional dependencies and build robust internal representations.

The learned masks dynamically change across decision steps and samples, allowing the model to focus on different feature subsets depending on the context — a behavior we later use for model explanation. While we did not fully analyze mask sparsity patterns, preliminary inspection showed that certain dimensions (e.g., thermal vs. mechanical properties) were consistently co-selected, hinting at latent grouping structures that TabNet discovered from data alone.


### Performance Comparison on PolyOne Dataset

Although our ultimate goal is to generate embeddings from properties, we first tested whether TabNet could perform the reverse task: predicting properties from embeddings. Our reasoning was simple — if TabNet cannot reconstruct properties from the dense 600-dimensional PolyBERT embeddings, it is unlikely to succeed in generating those embeddings in the first place.

These results were obtained using 50M training samples, 10M validation, and 10M test — a subset of the full PolyOne dataset due to memory constraints. All experiments fixed the random seed, and models were trained with identical optimizers, learning rates, and batch sizes to ensure fair comparison.

| Model                       | Test Loss | Test R² Score |
|----------------------------|-----------|----------------|
| PolyBERT (linear classifier) | 982.18    | 0.92           |
| TabNet (default)           | 1611.53   | 0.54           |
| TabNet (tuned)             | **5.90**  | **0.97**       |


This confirmed TabNet’s expressive capacity, especially after tuning decision step count, hidden dimensions, and regularization strength.

In this setup, TabNet was trained to predict the original 29 properties from the 600-dimensional PolyBERT embeddings. The strong performance, particularly after hyperparameter tuning, demonstrated that TabNet could indeed model the nuanced dependencies between structural features (as encoded in the embeddings) and property outputs.

The tuned TabNet significantly outperforms the baseline and even matches or exceeds the performance of the PolyBERT encoder in this prediction setup. This reinforces our choice to adopt TabNet as the inverse design encoder.

Despite initial concerns about TabNet’s suitability — given that it was originally designed for sequential or hierarchical data — our experiments revealed that it can still perform strongly on non-sequential, independently distributed tabular inputs like polymer properties. This finding is supported by two key observations:

1. **Masking-Based Training**: We applied structured masking during both embedding-to-property and property-to-embedding training. This encouraged TabNet to selectively attend to informative subsets of properties, learning how different combinations influence the PolyBERT latent space. While we did not perform a detailed analysis of the learned masks, their preliminary behavior suggested the emergence of property groupings (e.g., thermal vs. mechanical), hinting that TabNet can extract meaningful latent structure even from non-sequential tabular inputs.

2. **Performance Gains After Tuning**: The substantial jump in R² score — from 0.54 (default) to 0.97 (tuned) — in the property prediction task underscores TabNet’s capacity to model complex relationships between polymer properties and their embeddings. This dramatic improvement reassured us that TabNet is not inherently limited by the independence of input features and can, in fact, serve as an effective encoder in the inverse design setting.

Thus, treating the 29 property dimensions as structured inputs — empowered by TabNet’s attention and feature selection mechanisms — proved not only feasible but effective. The ability to dynamically weight features per decision step turned out to be a strength, enabling robust modeling of property interactions that might otherwise be lost in simpler feedforward architectures.

This architectural flexibility, coupled with strong empirical results, motivated our final decision to adopt TabNet as the encoder for inverse design. It enables both targeted feature attribution (e.g., identifying which properties drive specific embeddings) and gradient-based refinement strategies for guiding generative pipelines.

The pipeline is visualized below:

{% include figure.html path="assets/img/2025-05-25-inverse-polymer/TabNet_knn_pipeline.png" class="img-fluid" %}

We trained TabNet on a dataset with up to 25M samples and evaluated performance using a KNN-based retrieval in the 600-dim embedding space. Evaluation was accelerated with FAISS but showed a severe drop in performance as the candidate pool grew:

- **10M training → 1.5M eval**: Top-1 accuracy = 46.3%  
- **25M training → 2.5M eval**: Top-1 accuracy = 19.7%

While fast, this retrieval method suffers from three main limitations:

1. Requires a large embedding database at inference.
2. Retrieves existing polymers, not novel ones.
3. Cannot guarantee structural validity or property match.

These issues motivate the shift toward generative modeling.


---

# GRU Decoder for SMILES Generation

To enable direct generation of polymer structures, we design a GRU-based decoder that maps 600-dimensional embeddings to pSMILES sequences.

**Why GRU?**

* SMILES are sequential: token order is critical.
* GRU models are efficient and less prone to overfitting on limited data compared to Transformers.
* Proven success in molecular generation tasks (e.g., ChemTS).

We train the decoder using teacher forcing on a dataset of (embedding, SMILES) pairs tokenized via a PolyBERT-trained tokenizer.

* **Model**: 2-layer GRU (hidden=512, dropout=0.1)
* **Loss**: Cross-entropy with ignore_index=0 for padding
* **Validation**: Token-level accuracy ≈ 0.976

**Reconstruction Examples**

{% include figure.html path="assets/img/2025-04-28-inverse-polymer/reconstruction\_examples.png" class="img-fluid" %}

However, decoding embeddings predicted by TabNet (rather than real embeddings) leads to poor results. This is likely due to a mismatch between the latent spaces: the decoder only sees valid PolyBERT embeddings during training, not the out-of-distribution outputs from TabNet.

---

# Reinforcement Learning with MCTS

To refine the raw SMILES outputs and enforce structural validity + property satisfaction, we employ a reinforcement learning module based on Monte Carlo Tree Search (MCTS).

### MDP Setup

* **State**: pSMILES string at time step t
* **Action**: Add/remove atoms/bonds (e.g., replace [*]CC[*] with [*]COC[*])
* **Reward**:

  * Validity via RDKit
  * Structural similarity to reference (Tanimoto)
  * Property accuracy via a pretrained TabNet

$$
J(S) = \alpha \cdot R_{\text{sim}} + \beta \cdot R_{\text{prop}}, \quad r(S) =
\begin{cases}
\frac{J(S)}{1 + |J(S)|} & \text{if valid} \\
-1 & \text{otherwise}
\end{cases}
$$

### Diagram

{% include figure.html path="assets/img/2025-04-28-inverse-polymer/mcts\_diagram.jpg" class="img-fluid" %}

This allows the model to iteratively refine noisy SMILES samples toward chemically valid, property-matching outputs.

---

# Diffusion and RL for Polymer Generation

To overcome the limitations of autoregressive decoding and tree-based refinement, we explore **latent diffusion models (LDMs)** combined with **reinforcement learning (RL)** for inverse polymer design.

### Key Ideas:

* TabNet generates a latent z from input properties.
* We add a noise vector \epsilon to z and decode it using a pretrained diffusion model.
* An RL agent learns to predict \epsilon such that the generated pSMILES maximally satisfies the target properties.

### RL Setup

* **State**: (TabNet embedding z, target properties)
* **Action**: noise vector \epsilon (sampled from learned distribution or predicted deterministically)
* **Reward**: weighted score of validity + property match (via RDKit + PolyBERT-based predictor)

### Experiment Plan

1. Pretrain diffusion decoder on clean PolyBERT embeddings.
2. Use multi-sample generation to estimate best-of-N performance (baseline).
3. Train an RL agent to optimize \epsilon for better generations.
4. Compare RL-guided sampling vs. classifier-free guidance.

This approach connects retrieval-based inverse design with controllable, expressive generative models.

---

# Experiment Results

**TabNet Retrieval**

* Top-1 retrieval accuracy dropped from 46.3% to 19.7% as training set increased from 10M to 25M samples, showing database scalability issues.

**GRU Decoder**

* Token-level reconstruction accuracy: 97.6%
* Generation from TabNet embeddings yielded invalid SMILES in over 60% of samples (highlighting latent mismatch).

**Diffusion Pretraining**

* \[TO BE INSERTED after LDM training completion: validity %, novelty %, property match score.]

**RL-based Noise Tuning**

* \[TO BE INSERTED after policy training: reward progression curve, sample generations, success rate vs. baseline.]

---

# Challenges and Future Work

* **Latent mismatch**: Decoder cannot generalize to TabNet-predicted embeddings without joint training.
* **Decoder drift**: GRU loses initial embedding signal across long sequences. Potential fixes:

  * Feed embedding at every timestep.
  * Use Transformer-based decoder.
* **Reward modeling**: Property-based rewards are difficult to optimize due to noise and non-differentiability.
* **Diffusion tuning**: RL-guided noise tuning introduces a large action space; reward shaping and offline RL are under consideration.
* **Guidance tradeoffs**: RL offers exploration and sample-specific adaptation, unlike static classifier-free guidance.

---

# Conclusion

We presented a hybrid inverse design pipeline for polymers that bridges predictive modeling and generative chemistry. Starting from TabNet-based embedding regression, we trained a GRU decoder to reconstruct polymer structures, refined them via MCTS, and now move toward diffusion-based generative modeling with reinforcement learning.

Our proposed Diffusion+RL approach enables one-to-many generation aligned with desired properties and sets the foundation for robust, property-controlled molecular design.

---

*This blog post was submitted as part of the AI810 course at KAIST. Student ID: 20253793.*