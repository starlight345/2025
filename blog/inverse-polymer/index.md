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

TabNet structurally incorporates Transformer-style attention, making it well-suited for sequential and feature-selection tasks. However, in the PolyOne dataset (introduced in the PolyBERT paper), the 29 properties are not sequential but rather independent columns — raising concerns about whether TabNet, typically strong in sequential or hierarchical data, would still be effective in this context.

To investigate this, we applied unsupervised pre-training on structured tables, as shown below:

{% include figure.html path="assets/img/2025-05-25-inverse-polymer/tabnet_structure.png" class="img-fluid" %}

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

- 10M training → 1.5M eval: Top-1 accuracy = 46.3%  
- 25M training → 2.5M eval: Top-1 accuracy = 19.7%

To better understand the significance of our KNN-based retrieval performance, we compared it against a random retrieval baseline, which provides a reference for interpreting Top-1 accuracy in large candidate spaces.

- For 1.5M candidates, the expected Top-1 accuracy from random selection is:
  
  $$
  \text{Random Top-1 Accuracy} = \frac{1}{1{,}500{,}000} \approx 0.000067\%
  $$

- For 2.5M candidates, the expected Top-1 accuracy from random selection is:
  
  $$
  \frac{1}{2{,}500{,}000} \approx 0.00004\%
  $$

In contrast, our TabNet-based retrieval achieves:

- **46.3%** accuracy on 1.5M candidates → over 690,000× better than random
- **19.7%** accuracy on 2.5M candidates → over 490,000× better than random

These comparisons clearly demonstrate that the model has learned a highly structured and informative embedding space, even when the candidate pool becomes extremely large.

> While the absolute Top-1 accuracy drops with larger candidate sets, the performance remains orders of magnitude better than random guessing.

However, to more rigorously evaluate the retrieval quality, it would have been helpful to analyze the distributional structure of the polymer embeddings in the latent space — for example, through density plots, clustering analysis, or distance histograms. These could have revealed whether the drop in performance is due to increased overlap between clusters, embedding crowding, or simply the combinatorial explosion of candidates.

Unfortunately, such analysis was beyond the scope of our current experiments. Nevertheless, these results provide strong evidence that learned embeddings capture meaningful property-driven structure — justifying our transition from retrieval to generative modeling, which offers greater flexibility and novelty.

Despite its strong performance relative to random baselines, the retrieval-based approach has inherent limitations that restrict its utility in real-world inverse design tasks.

Most notably, it is constrained to selecting from a fixed database — meaning it cannot generate entirely new polymer structures beyond what was seen during training. This restricts novelty and limits the method's ability to explore the vast chemical design space. Additionally, even high-ranking candidates in the embedding space may lack structural validity or fail to meet property constraints due to the imperfect alignment between embeddings and molecular feasibility.

These limitations highlight a crucial need for generative modeling, which allows us to go beyond retrieval and construct novel polymer structures directly from target properties. By training a decoder that maps embeddings into pSMILES sequences, we unlock the ability to explore the design space in a controlled and flexible manner — a capability retrieval alone cannot provide.


---

# GRU Decoder for SMILES Generation

To enable direct generation of polymer structures, we designed a GRU-based decoder that maps 600-dimensional embeddings to pSMILES sequences. SMILES are inherently sequential — token order directly affects molecular meaning — and GRUs are memory-efficient RNNs that perform well on limited data. Compared to LSTMs, GRUs train faster and generalize better under data scarcity. GRU-based generation has seen prior success in models like ChemTS<d-cite key="Yang_2017"></d-cite> and MolGPT<d-cite key="doi:10.1021/acs.jcim.1c00600"></d-cite>. While ChemTS explores SMILES generation via random mutation and reward-guided exploration, our model decodes structured embeddings into full SMILES strings using learned associations. Our approach differs in that it performs direct decoding from polymer property-guided latent vectors and aims to reconstruct meaningful structures deterministically.

We implemented a 2-layer GRU with hidden size 512 and dropout 0.1, trained via teacher forcing on (embedding, SMILES) pairs tokenized using a pretrained PolyBERT tokenizer. We used cross-entropy loss with padding tokens ignored and achieved ≈97.6% token-level accuracy on held-out validation sets. To enhance robustness, Gaussian noise was optionally added to the latent embedding during training, enabling more stable generation from perturbed embeddings. During inference, greedy decoding reconstructs SMILES sequences from embeddings, using the 600-dim input as the initial hidden state.

While generation from real PolyBERT embeddings produces valid outputs, decoding TabNet-generated embeddings often fails. This is likely due to a mismatch between latent distributions — the decoder was only trained on real embeddings, not the broader, noisier space produced by TabNet. To address this, we plan to fine-tune the decoder using noisy or perturbed embeddings and pursue joint encoder-decoder training strategies. A longer-term solution involves training the decoder to robustly map a local neighborhood around the embedding — aligning closely with diffusion-based models where multiple generations are possible from a single latent z.

**Decoder Benchmark: GRU vs. Seq2Seq**

We also benchmarked our GRU decoder against a Transformer-based encoder-decoder (e.g., T5-style) trained to reconstruct SMILES, optionally using PolyBERT embeddings as input. Despite achieving similar top-1 matches, seq2seq decoders exhibited significant issues with predicting EOS tokens correctly. Even after doubling the weight of EOS tokens during training, validity remained low. This was tested under multiple configurations, and the results are summarized below. Here, Tanimoto similarity refers to a fingerprint-based structural similarity score ranging from 0 to 1.

| Model Variant             | Validity (%) | Mean Tanimoto |
| ------------------------- | ------------ | ------------- |
| GRU Decoder               | 23.0         | 0.9775        |
| Seq2Seq                   | 16.0         | 0.9425        |
| Seq2Seq (EOS×2)           | 17.0         | 0.9643        |


These findings highlight that both GRU and seq2seq decoding approaches still require significant improvement in terms of validity. GRU decoding currently appears more robust for latent-to-sequence generation, especially when working with a fixed embedding distribution. Seq2seq models may benefit from architectural innovations or stronger EOS supervision but underperformed in our setting. Moreover, we observed that the decoder’s performance improves when the latent embedding is repeatedly concatenated to the token inputs at each step — mitigating the vanishing context issue over long sequences.

# Improving Robustness and Validity

To further address generation variability and increase model robustness, we plan to extend this setup into a diffusion-style training regime. In this framework, Gaussian noise is repeatedly added to the latent z, and the decoder is trained to map each noisy version back to the same SMILES output. This trains the decoder to produce consistent generations even under perturbations, laying the groundwork for controlled diversity and one-to-many mapping via policy-guided sampling.

In addition to sequential decoding challenges, our experiments revealed a strong tendency to mispredict EOS tokens, even under weighted loss training. This observation suggests the decoder has difficulty learning global termination constraints from local token sequences. As a remedy, we plan to enhance decoder supervision through context-aware mechanisms that track decoding progress and apply structural constraints.

To mitigate invalid SMILES generation, we also consider incorporating structural priors. Inspired by JT-VAE<d-cite key="jin2019junctiontreevariationalautoencoder"></d-cite>, we plan to investigate generation strategies that first construct coarse-grained tree-like motifs (e.g., thiophene rings, alkyl chains) before assembling complete molecular graphs. While we do not adopt JT-VAE directly, we aim to emulate the concept of hierarchical structure by generating valid fragments first, followed by fine-grained atom-level decoding. This hybrid approach could improve syntactic correctness by preventing impossible ring closures or malformed branches.

Recognizing that sequential decoders trained under teacher forcing often lack awareness of chemical constraints, we are also exploring rule-based masking and memory mechanisms. These would allow the decoder to track open and closed rings, and enforce one-time closure constraints during generation. Rather than relying solely on attention-based decoders to implicitly learn such rules, we propose to incorporate explicit tracking of generation context — for instance, remembering ring-opening events and ensuring that each ring is closed exactly once. While implementing such chemically informed generation constraints is challenging, it may offer a principled solution to reduce invalid outputs in autoregressive decoders.

# Joint Training and Latent Alignment

To enable inverse generation of polymer structures from desired properties, we propose a joint training framework where a TabNet encoder is trained alongside a frozen Transformer-based decoder. In this setup, the decoder remains fixed — preserving its learned generation dynamics — while the encoder is trained to produce latent representations that fall within the decoder’s expected embedding space. This alignment allows the TabNet encoder, conditioned only on property vectors, to generate molecular structures by implicitly matching the decoder’s internal distribution.

This strategy leverages the proven strength of TabNet in the property prediction task (as detailed in the retrieval section, where we achieved an R² of 0.97), and extends it to the generative domain. While the decoder is not updated during training, the encoder is progressively guided to encode property vectors into latent representations that are semantically meaningful and structurally decodable. This form of one-sided adaptation ensures that the generation process remains stable and grounded in the decoder’s learned grammar of SMILES construction.

Crucially, our long-term goal is to further refine this mapping through gradient field alignment between two distinct encoders: PolyBERT (trained on molecular structure) and TabNet (trained on properties). By minimizing the directional discrepancy between their gradient fields — essentially encouraging their latent outputs to guide decoding in a coherent manner — we aim to couple the property-driven and structure-driven pathways into a unified inverse design system. This semantic alignment ensures that both encoders induce consistent trajectories in the decoder’s latent space, thereby improving the plausibility and accuracy of generated structures.

---

**Appendix: Reconstruction Examples**
```text
1. Perfect Match: Aromatic Core
GT  : [*]c1cccc(-c2cccc3c(-c4ccc(NSC(=O)C(C)Cl)=C(O)C6ccc7c(c6)=O)NN(C6OC8C([*])OC8C7=C(O)C5=O)c4)cccc23)c1  
GEN : [*]c1cccc(-c2cccc3c(-c4ccc(NSC(=O)C(C)Cl)=C(O)C6ccc7c(c6)=O)NN(C6OC8C([*])OC8C7=C(O)C5=O)c4)cccc23)c1

2. Structural Mismatch: Functional Branch Shift
GT  : [*]N1C(=O)c2cccc(-c3ccc(-c4cc(C)C(NSC(=O)c6ccc(-c7c(C)cc(C([*])(C(F)(F)F)C(F)(F)F)c7C)cc6C5=O)c3)cc2)C1=O  
GEN : [*]N1C(=O)c2cccc(-c3ccc(-c4cc(C)C(NSC(=O)c6ccc(-c7c(C)cc(C([*])(C(F)(F)F)C(F)(F)F)c7C)cc6C5=O)c5C)c4)c

3. Core Replaced: Heterocycle Divergence
GT  : [*]c1cccc(C2CCC(n3c(C4ccc(C)cc4)c3)n4cccc43)C2)c1  
GEN : [*]c1cccc(C2CCC(n3c(Cc3cccc3=n3C([*])n4cccc43)C2)c1 
```

---

# Reinforcement Learning with MCTS

To improve the quality of generated SMILES and ensure both chemical validity and property alignment, we introduce a reinforcement learning framework based on Monte Carlo Tree Search (MCTS). This method serves as a postprocessing step that can refine imperfect outputs from a decoder—such as syntactically invalid or property-mismatched SMILES—without requiring retraining of the decoder itself.

The MCTS framework operates within a Markov Decision Process (MDP), where each state corresponds to a partial or complete SMILES string at a specific time step. Starting from an initial noisy generation, the agent iteratively modifies the string by adding, removing, or replacing tokens. Actions are drawn from a predefined vocabulary of atoms (e.g., C, O, N, Cl) and bond symbols (e.g., =, #, :, ring indices). Through these token-level edits, MCTS incrementally refines the structure while maintaining syntactic validity.

At terminal states, the agent receives a reward based on three key criteria: (1) chemical validity, verified using RDKit; (2) structural similarity to a reference, computed via Tanimoto similarity; and (3) property alignment, assessed using a pretrained TabNet predictor. These components are combined into a weighted objective:

$$
J(S) = \alpha \cdot R_{\text{sim}} + \beta \cdot R_{\text{prop}}, \quad r(S) =
\begin{cases}
\frac{J(S)}{1 + |J(S)|} & \text{if valid} \\
-1 & \text{otherwise}
\end{cases}
$$

This reward function favors chemically valid structures that are both similar to a target molecule and aligned with the desired properties. The tree search process uses this signal to prioritize promising edits and prune less useful branches, ultimately selecting high-reward candidates after multiple rollouts.

By applying MCTS after decoding, we gain robustness against errors from the base model. For example, even when the decoder generates an invalid or unrealistic structure, MCTS can recover a plausible and property-matching molecule through localized edits. This makes the method particularly useful when decoding from noisy latent embeddings, such as those predicted by TabNet in our pipeline.

Furthermore, the modularity of this approach allows it to be combined with any generative model, including GRU- or Transformer-based decoders. While our current method manipulates SMILES at the string level, future extensions may incorporate graph-based edits or substructure-aware rollouts, inspired by works like VGAE-MCTS<d-cite key="doi:10.1021/acs.jcim.3c01220"></d-cite>. Such enhancements could provide even finer control over chemical structure during the search process.

Ultimately, MCTS acts as a chemically grounded corrector that refines generated candidates into valid, property-aligned polymers. It bridges the gap between raw generation and task-specific optimization, improving both the plausibility and functional relevance of the final outputs.


---

# Diffusion and RL for Polymer Generation

While MCTS offers a powerful way to refine outputs from sequence-based decoders, it is inherently limited to local, symbolic edits of a given candidate. To explore more expressive and flexible modes of generation, we now turn to latent generative models—specifically, latent diffusion models (LDMs)—and propose combining them with reinforcement learning for property-conditioned polymer design. Unlike MCTS, which operates after decoding, our Diffusion+RL module is designed as a replacement for the decoder, decoding from perturbed latent vectors using a latent diffusion model instead of an autoregressive GRU. This module runs independently of the GRU decoder, and is intended to provide a more robust and diverse alternative during inference.

In our current pipeline, we use a GRU-based decoder to reconstruct SMILES from latent embeddings generated by TabNet. However, this setup suffers from a mismatch between the embedding distributions: TabNet is trained to regress into the PolyBERT latent space, but the decoder was trained separately and struggles to generalize to these new inputs. As a result, many generated SMILES are either invalid or chemically implausible.

To address this limitation, we propose replacing the decoder with a diffusion-based generator trained directly in the PolyBERT latent space. The idea is as follows: TabNet first maps a property vector to a latent embedding $z$, as before. Instead of decoding $z$ directly, we add a noise vector $\epsilon$ and decode the perturbed latent using a pretrained diffusion model. The decoding process thus becomes:

$$
z' = z + \epsilon \quad \Rightarrow \quad \text{pSMILES} = \text{Decoder}(z')
$$

Here, reinforcement learning enters as a mechanism for learning or selecting the optimal $\epsilon$ for a given $z$, such that the resulting pSMILES maximally satisfies the target properties. The RL agent observes the property vector and base embedding $z$, and outputs an $\epsilon$ vector as its action. The reward is computed based on the generated molecule's chemical validity and its alignment with the desired properties, using the same RDKit and PolyBERT-based predictors employed elsewhere in our pipeline.

This formulation effectively decouples the problem of embedding prediction from that of decoding: TabNet focuses on producing a meaningful latent representation, and the diffusion+RL module explores structured variations in the neighborhood of this embedding to find optimal molecules.

While this part of the project is still under development and we do not yet have experimental results to report, we believe this approach offers several conceptual advantages. First, it enables one-to-many generation: by sampling or optimizing different noise vectors $\epsilon$, we can obtain diverse candidate molecules consistent with the same target properties. Second, it allows for fine-grained control over the generation process—either through direct supervision or through policy learning in an RL setting. Lastly, since the diffusion decoder operates in a continuous, learned space, it may offer better robustness than token-level decoders when dealing with imperfect latent inputs from models like TabNet.

In summary, the Diffusion+RL approach builds naturally on our existing inverse design pipeline. Whereas MCTS acts as a symbolic corrector to patch individual outputs from the decoder, diffusion and RL operate at the embedding level, offering a new axis of controllability and expressivity. Together, these techniques aim to create a robust and versatile inverse design framework—capable of producing not just valid polymers, but polymers aligned with arbitrary target property profiles.

Experiments are underway and will be updated in a future version of this blog.

---

# Experiment Results

We conducted a series of experiments to validate each component in our proposed inverse design pipeline, from property-to-embedding regression to structure generation and refinement.

**TabNet Embedding Regression**

To evaluate the encoder’s ability to map properties into the latent PolyBERT space, we first trained TabNet in a property-to-embedding regression task. The resulting embeddings were tested via KNN retrieval from a large database of known polymers. Our best model achieved:

Top-1 retrieval accuracy:

- 46.3% on 1.5M candidates
- 19.7% on 2.5M candidates

These results are over 490,000× better than random, confirming that TabNet effectively learns a structured mapping from properties to meaningful embeddings. However, performance degraded as the candidate pool grew, reflecting the limitations of retrieval-based methods in scaling and novelty.

**SMILES Decoding: GRU vs. Seq2Seq**

We implemented two types of decoders to generate pSMILES from latent embeddings:

1. GRU Decoder

- 2-layer GRU with hidden size 512
- Token-level reconstruction accuracy: 97.6%
- Mean Tanimoto similarity: 0.9775
- Validity (syntactic): 23.0%

Generation from true PolyBERT embeddings produced accurate sequences. However, decoding TabNet-generated embeddings resulted in over 60% invalid SMILES, highlighting a latent distribution mismatch.

2. Transformer-based Seq2Seq Decoder

We also trained a T5-style decoder to reconstruct SMILES. Despite similar top-1 token accuracy, EOS token misprediction significantly lowered validity.

| Model Variant             | Validity (%) | Mean Tanimoto |
| ------------------------- | ------------ | ------------- |
| GRU Decoder               | 23.0         | 0.9775        |
| Seq2Seq                   | 16.0         | 0.9425        |
| Seq2Seq (EOS×2)           | 17.0         | 0.9643        |


This comparison motivated our exploration of joint encoder-decoder training, MCTS postprocessing, and diffusion+RL generation, aiming to improve structural validity and alignment with desired properties.


---

# Challenges and Future Work

Despite promising initial results, several challenges remain before our inverse design pipeline can be considered robust and general-purpose.

One key issue is the latent mismatch between the embeddings produced by TabNet and those expected by the decoder. While the GRU decoder performs well on true PolyBERT embeddings, its performance drops dramatically when applied to TabNet-generated ones. This is likely due to distributional shifts not accounted for during decoder training. To mitigate this, we plan to inject Gaussian noise into true embeddings during decoder training, thereby simulating the variability observed in TabNet outputs. This approach encourages the decoder to generalize to a broader latent distribution and improves its robustness against slight deviations in embedding space.

Another challenge involves decoder drift — the loss of embedding signal across long decoding sequences. In autoregressive models like GRU and Transformer, the initial latent vector is quickly forgotten unless explicitly reintroduced. Future work will experiment with latent fusion at each decoding step, or explore architectures that condition on global context more robustly, such as memory-augmented Transformers or latent injection strategies.

In terms of structural validity, both GRU and Transformer-based decoders struggle to predict correct termination tokens, leading to malformed SMILES. This points to a broader issue of syntactic constraint modeling. To address this, we plan to incorporate chemical rules directly into the decoding process, either through masked token prediction, constraint-based beam search, or fragment-based priors. Additionally, hybrid approaches like motif-first decoding — inspired by JT-VAE — may help enforce coarse-to-fine structural coherence.

From the optimization side, our reward functions remain brittle. Property-based rewards, especially when derived from pretrained predictors, are often noisy or poorly calibrated. This makes reinforcement learning difficult to stabilize. We are investigating smoother reward landscapes via continuous-valued targets and ensemble-based uncertainty modeling, as well as exploring offline RL setups where guidance can be learned from high-reward demonstrations.

Finally, while diffusion-based models with RL noise tuning offer flexibility, they pose their own difficulties: the action space becomes large and exploration-heavy, and naive policies may collapse to trivial solutions. To handle this, we are considering curriculum learning (starting from small perturbations), hierarchical noise decoders, and reward shaping techniques that encourage valid intermediate states.

In the long term, we envision a unified training framework where TabNet, decoder, and RL modules co-evolve in a modular yet coupled fashion. This would allow the system to learn end-to-end mappings from properties to valid, diverse, and functionally aligned polymers — not only solving the inverse problem, but doing so in a controllable and chemically meaningful way.

---

# Conclusion

This work presents an iterative approach to the inverse design of polymers, integrating structured predictors and sequence-based generative models into a unified framework.

We began by demonstrating that TabNet can effectively regress from property vectors into the latent space of PolyBERT, achieving high retrieval accuracy even in large candidate pools. However, retrieval alone is limited in novelty and flexibility.

To overcome these limits, we introduced a GRU-based decoder capable of reconstructing SMILES sequences from latent embeddings. While accurate on clean inputs, it struggled with TabNet-generated embeddings due to distributional mismatch — motivating the need for decoder-aware training and latent regularization.

To refine imperfect generations, we implemented Monte Carlo Tree Search (MCTS), enabling local, symbolic edits that improve both chemical validity and property alignment. This allowed us to salvage otherwise invalid generations and enforce task-specific constraints without retraining the decoder.

Looking forward, we aim to replace the decoder with a diffusion model trained in the latent space, and introduce reinforcement learning as a policy to guide stochastic decoding. This would enable one-to-many generation of diverse yet property-consistent polymers — a crucial step for real-world molecular design.

In essence, this pipeline evolves from retrieval → generation → optimization, gradually increasing generative capacity while preserving interpretability. By combining attention-based encoders, sequential decoders, and policy-driven refinement, we take a concrete step toward fully modular, property-driven polymer generation.

> Our journey reflects the philosophy of inverse design: to reason backward from properties to structure, not just approximately — but reliably, validly, and creatively.

In the future, we aim to unify all modules—including TabNet, decoder, and reinforcement components—into a single end-to-end trainable system that can continuously adapt to property constraints while generating novel structures beyond existing chemical libraries.

---

*This blog post was submitted as part of the AI810 course at KAIST. Student ID: 20253793.*