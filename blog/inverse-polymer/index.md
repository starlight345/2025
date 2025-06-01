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

To better understand the significance of our KNN-based retrieval performance, we compared it against a **random retrieval baseline**, which provides a reference for interpreting Top-1 accuracy in large candidate spaces.

- For **1.5M candidates**, the expected Top-1 accuracy from random selection is:
  
  $$
  \text{Random Top-1 Accuracy} = \frac{1}{1{,}500{,}000} \approx 0.000067\%
  $$

- For **2.5M candidates**, the expected Top-1 accuracy from random selection is:
  
  $$
  \frac{1}{2{,}500{,}000} \approx 0.00004\%
  $$

In contrast, our TabNet-based retrieval achieves:

- **46.3%** accuracy on 1.5M candidates → over **690,000× better** than random
- **19.7%** accuracy on 2.5M candidates → over **490,000× better** than random

These comparisons clearly demonstrate that the model has learned a **highly structured and informative embedding space**, even when the candidate pool becomes extremely large.

> While the absolute Top-1 accuracy drops with larger candidate sets, the performance remains **orders of magnitude better** than random guessing.

However, to more rigorously evaluate the retrieval quality, it would have been helpful to analyze the **distributional structure** of the polymer embeddings in the latent space — for example, through **density plots, clustering analysis, or distance histograms**. These could have revealed whether the drop in performance is due to increased overlap between clusters, embedding crowding, or simply the combinatorial explosion of candidates.

Unfortunately, such analysis was beyond the scope of our current experiments. Nevertheless, these results provide strong evidence that learned embeddings capture meaningful property-driven structure — justifying our transition from retrieval to **generative modeling**, which offers greater flexibility and novelty.

Despite its strong performance relative to random baselines, the retrieval-based approach has inherent limitations that restrict its utility in real-world inverse design tasks.

Most notably, it is constrained to **selecting from a fixed database** — meaning it cannot generate entirely new polymer structures beyond what was seen during training. This restricts novelty and limits the method's ability to explore the vast chemical design space. Additionally, even high-ranking candidates in the embedding space may **lack structural validity** or **fail to meet property constraints** due to the imperfect alignment between embeddings and molecular feasibility.

These limitations highlight a crucial need for **generative modeling**, which allows us to go beyond retrieval and **construct novel polymer structures** directly from target properties. By training a decoder that maps embeddings into pSMILES sequences, we unlock the ability to explore the design space in a controlled and flexible manner — a capability retrieval alone cannot provide.


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



# GRU Decoder for SMILES Generation

To enable direct generation of polymer structures, we designed a GRU-based decoder that maps 600-dimensional embeddings to pSMILES sequences. SMILES are inherently sequential — token order directly affects molecular meaning — and GRUs are memory-efficient RNNs that perform well on limited data. Compared to LSTMs, GRUs train faster and generalize better under data scarcity. GRU-based generation has seen prior success in models like ChemTS and MolGPT. While ChemTS explores SMILES generation via random mutation and reward-guided exploration, our model decodes structured embeddings into full SMILES strings using learned associations. Our approach differs in that it performs direct decoding from polymer property-guided latent vectors and aims to reconstruct meaningful structures deterministically.

We implemented a 2-layer GRU with hidden size 512 and dropout 0.1, trained via teacher forcing on (embedding, SMILES) pairs tokenized using a pretrained PolyBERT tokenizer. We used cross-entropy loss with padding tokens ignored and achieved ≈97.6% token-level accuracy on held-out validation sets. To enhance robustness, Gaussian noise was optionally added to the latent embedding during training, enabling more stable generation from perturbed embeddings. During inference, greedy decoding reconstructs SMILES sequences from embeddings, using the 600-dim input as the initial hidden state.

While generation from real PolyBERT embeddings produces valid outputs, decoding TabNet-generated embeddings often fails. This is likely due to a mismatch between latent distributions — the decoder was only trained on real embeddings, not the broader, noisier space produced by TabNet. To address this, we plan to fine-tune the decoder using noisy or perturbed embeddings and pursue joint encoder-decoder training strategies. A longer-term solution involves training the decoder to robustly map a local neighborhood around the embedding — aligning closely with diffusion-based models where multiple generations are possible from a single latent z.

## Seq2Seq Comparison and EOS Prediction Challenges

We also benchmarked our GRU decoder against a Transformer-based encoder-decoder (e.g., T5-style) trained to reconstruct SMILES, optionally using PolyBERT embeddings as input. Despite achieving similar top-1 matches, seq2seq decoders exhibited significant issues with predicting EOS tokens correctly. Even after doubling the weight of EOS tokens during training, validity remained low. This was tested under multiple configurations, and the results are summarized below. Here, Tanimoto similarity refers to a fingerprint-based structural similarity score ranging from 0 to 1.


| Model Variant             | Validity (%) | Mean Tanimoto |
| ------------------------- | ------------ | ------------- |
| GRU Decoder               | 23.0         | 0.9775        |
| Seq2Seq (index=40)        | 16.0         | 0.9425        |
| Seq2Seq (EOS×2, index=40) | 17.0         | 0.9643        |

These findings highlight that both GRU and seq2seq decoding approaches still require significant improvement in terms of validity. GRU decoding currently appears more robust for latent-to-sequence generation, especially when working with a fixed embedding distribution. Seq2seq models may benefit from architectural innovations or stronger EOS supervision but underperformed in our setting. Moreover, we observed that the decoder’s performance improves when the latent embedding is repeatedly concatenated to the token inputs at each step — mitigating the vanishing context issue over long sequences.

## Toward Robustness and Validity: Planned Improvements

To further address generation variability and increase model robustness, we plan to extend this setup into a diffusion-style training regime. In this framework, Gaussian noise is repeatedly added to the latent z, and the decoder is trained to map each noisy version back to the same SMILES output. This trains the decoder to produce consistent generations even under perturbations, laying the groundwork for controlled diversity and one-to-many mapping via policy-guided sampling.

In addition to sequential decoding challenges, our experiments revealed a strong tendency to mispredict EOS tokens, even under weighted loss training. This observation suggests the decoder has difficulty learning global termination constraints from local token sequences. As a remedy, we plan to enhance decoder supervision through context-aware mechanisms that track decoding progress and apply structural constraints.

To mitigate invalid SMILES generation, we also consider incorporating structural priors. Inspired by <d-cite key="jin2019junctiontreevariationalautoencoder"></d-cite>, we plan to investigate generation strategies that first construct coarse-grained tree-like motifs (e.g., thiophene rings, alkyl chains) before assembling complete molecular graphs. While we do not adopt JT-VAE directly, we aim to emulate the concept of hierarchical structure by generating valid fragments first, followed by fine-grained atom-level decoding. This hybrid approach could improve syntactic correctness by preventing impossible ring closures or malformed branches.

Recognizing that sequential decoders trained under teacher forcing often lack awareness of chemical constraints, we are also exploring rule-based masking and memory mechanisms. These would allow the decoder to track open and closed rings, and enforce one-time closure constraints during generation. Rather than relying solely on attention-based decoders to implicitly learn such rules, we propose to incorporate explicit tracking of generation context — for instance, remembering ring-opening events and ensuring that each ring is closed exactly once. While implementing such chemically informed generation constraints is challenging, it may offer a principled solution to reduce invalid outputs in autoregressive decoders.

## Joint Training Setup and Latent Alignment

In parallel, we are training a TabNet encoder jointly with a frozen Transformer decoder to explore inverse generation from properties to SMILES. This setup allows the encoder to align with the latent space expected by the decoder. Although the decoder is not updated, the encoder learns to map property vectors to meaningful latent representations compatible with the decoder’s generation logic. Given the strong performance of TabNet in the property prediction task (as discussed in a prior section), we anticipate that its embeddings will effectively align with the decoder’s latent space to produce valid molecular structures. Ultimately, we aim to align the gradient fields of a PolyBERT encoder and TabNet encoder, as explored in our coupling loss experiments. This joint training framework is expected to bridge the semantic gap between property-driven and structure-driven representations, laying the foundation for a unified inverse design model.

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