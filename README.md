# SupMarginCon
# Angular Margin Based Contrastive Learning

Although the training objective attempts to pull the representations of similar speakers closer together and push the representations of different speakers apart, these representations may not be sufficiently discriminative or robust against noise.

## Cosine Similarity

We define the cosine similarity as:

$$
\cos{\theta_{i,p}} = \frac{z_{i}^{\top} z_{p}}{\|z_{i}\| \|z_{p}\|},
$$

where $\theta_{i,p}$ is the angle between the embeddings $z_i$ and $z_p$. A similar formula applies to $z_i$ and $z_a$. The decision boundary for $z_{i}$, given specific $p$ (positive) and $a$ (negative) samples, is defined as:

$$
\theta_{i, p} = \theta_{i, a}.
$$

### Problem of Small Perturbations

Without a decision margin, small perturbations of the embedding vectors around the decision boundary may result in incorrect decisions (see Figs. 1(b) and 1(c)).

## SupMarginCon Loss

To address this issue, we introduce an additive angular margin $m$ to the decision boundary. The resulting loss, called **Supervised Margin Contrastive (SupMarginCon)** loss, is defined as:

$$
\mathcal{L}_{\text{SupMarginCon}} = 
\sum_{i=1}^{N} \frac{-1}{|P(i)|} \sum_{p \in P(i)} 
\log \frac{\exp \left(\cos \left(\theta_{i, p} + m\right) / \tau\right)}
{\sum_{a \in A(i)} \exp \left(\cos \left(\theta_{i, a}\right) / \tau\right)}.
$$

### Effect of Margin $m$

With this loss, the decision boundary for $z_{i}$ is shifted to:

$$
\theta_{i, p} + m = \theta_{i, a}.
$$

Minimizing this loss encourages:
- **Compactness**: Reducing $\theta_{i, p}$ for positive samples.
- **Divergence**: Increasing $\theta_{i, a}$ for negative samples.

This improves **alignment** (closeness of positive-pair embeddings) and **uniformity** (distribution of embeddings), both of which are critical to contrastive learning.

## Comparison with SupCon Loss

The SupMarginCon loss provides more discriminative properties than conventional loss functions, such as the Supervised Contrastive (SupCon) loss, as it incorporates the margin $m$ to enhance robustness and decision-making accuracy.

---

## Figures

### Decision Boundary and Perturbations

![Decision Boundary without Margin](assets/SupMarginConA.pdf)  
_Fig. 1(a): Decision boundary $\theta_{i,p} = \theta_{i,a}$ without a margin._

![Perturbation in Positive Sample](assets/SupMarginConB.pdf)  
_Fig. 1(b): Incorrect decision due to perturbation on $\mathbf{z}_p$._

![Perturbation in Negative Sample](assets/SupMarginConC.pdf)  
_Fig. 1(c): Incorrect decision due to perturbation on $\mathbf{z}_a$._

![Decision Boundary with Margin](assets/SupMarginConD.pdf)  
_Fig. 1(d): SupMarginCon with margin $m$ ensures robustness against perturbations._

With the margin $m$, embeddings $z_p$ and $\z_a$ can tolerate larger perturbations without causing incorrect decisions.

# Citation

```bibtex
@inproceedings{li2023discriminative,
  title={Discriminative speaker representation via contrastive learning with class-aware attention in angular space},
  author={Li, Zhe and Mak, Man-Wai and Meng, Helen Mei-Ling},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
