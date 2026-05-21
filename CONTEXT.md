# Research Context: Intersectional Algorithmic Harm in Platform Labour

**AlgoFairness Pornometrics — MSc Dissertation (Distinction), University of Essex, 2025**  
**Author:** Louise Silva Ferreira  

---

## 1. The Problem This Research Is Trying to Solve

Algorithmic fairness research has an empirical gap. The majority of documented interventions — reweighing, adversarial debiasing, threshold calibration — are evaluated on standard benchmark datasets that are small, balanced, and drawn from contexts where the protected attribute is cleanly observable. The conditions that make real-world fairness failures *hard* — extreme class imbalance, inferred rather than self-reported attributes, compound intersectional identities, and economic consequences that play out over time — are largely absent from the evaluation literature.

This dissertation was designed to close part of that gap by working in exactly those conditions: a corpus of 535,236 videos with heavily imbalanced group representation, demographic attributes inferred from user-submitted metadata, and economic outcomes (engagement, platform visibility, income proxy) rather than administrative decisions as the dependent variable.

The choice of platform is not incidental. Adult content platforms occupy a structurally specific position in the platform labour economy: they are one of the few digital labour markets where the product, the platform, and the demographic profile of the workforce are inseparable. Category and tag assignments are not neutral organisational tools. They are the mechanism through which a platform determines which content is surfaced, to whom, under which search terms, and with what algorithmic support. Classification bias in this context is, directly, income bias.

---

## 2. Why Intersectionality as an Analytical Framework

The concept of intersectionality — developed by Kimberlé Crenshaw (1989, 1991) and extended through Patricia Hill Collins's matrix of domination — holds that systems of oppression do not operate independently: race, gender, class, and sexuality interact to produce qualitatively distinct forms of disadvantage that cannot be derived from any single axis alone. A Black woman's experience of discrimination is not the sum of anti-Black racism plus sexism; it is a third category of harm that neither framework alone can capture.

In machine learning, this has a precise technical implication: a model that achieves demographic parity for race (as a binary) and demographic parity for gender (as a binary) may still produce severe disparities for Black women as an intersectional group. Standard fairness audits — which tend to evaluate single protected attributes — will miss this. The evaluation framework in this research is therefore structured around four intersectional groups (Black Women, Asian Women, Latina Women, White Women) from the outset, not derived from single-axis breakdowns.

The empirical results confirm why this matters. Adversarial debiasing, trained with a demographic parity constraint, was the only intervention that *worsened* outcomes for two groups simultaneously (Asian Women EOD: 0.105 → 0.283; Latina Women EOD: 0.195 → 0.360). Understanding why requires intersectional analysis: Asian-coded content does not just have a different demographic distribution — it has a categorically distinct semantic structure. The KL divergence of Asian Women's category distribution from the global distribution is 0.555, the highest of any group, compared to 0.129 for Latina Women and 0.173 for White Women. Asian content vocabulary *is* the classification signal. An adversarial network that penalises the model for using group-predictive features is not removing bias — it is removing the information the model needs to correctly classify this content at all. This is not a tuning failure. It is a structural feature of intersectional data that a single-axis fairness constraint cannot address.

---

## 3. Platform Labour and the Stakes of Classification

The platform under analysis — Redtube — operates within a broader political economy of digital sex work that remains poorly theorised in algorithmic accountability literature. Several features of this economy are analytically important.

**Occupational pipeline segregation by race.** The "Verified Amateurs" tag — a credential assigned by the platform to distinguish independent creators from professional studio content — is distributed in a racially structured pattern. Latina-coded content carries this tag at a rate of 76.1%; Asian-coded content at 56.1%; Black-coded content at 54.3%. White-coded content carries it at 19.3% — the lowest rate of any group. This is not evidence that White Women are less likely to be amateur creators; it reflects that Caucasian-coded content in the corpus is concentrated in professional studio production, while content from racinalised groups is concentrated in the amateur-independent pipeline. Race and professional tier are structurally correlated in the dataset in a way that directly affects how classification models learn.

**The compounding structure of harm.** The three harm types documented in this research are not parallel and independent — they compound. The *classification harm* (recall gap of 19.1 pp for Black Women vs White Women) reduces visibility, which reduces engagement. The *temporal economic harm* (rating slope declining 3.38 rating-points/year faster for Black Women, 95% bootstrap CI [−3.75, −2.98]) is not explained by the classification gap alone — it persists in the raw data, before any model is applied. The *linguistic harm* — documented through PMI analysis showing that racial identity is encoded primarily through user-submitted tags ("black girl": PMI 5.07; "ebony female": 5.07; "ghetto hd": 5.07) and through HurtLex mapping (Black x Female: personal stigmatisation terms present in 74.1% of videos; aggressive sexual material in 81.4%) — is structural to the platform's metadata architecture. It is not produced by the model; it is the infrastructure that the model learns from.

Any intervention that operates only at the model level leaves the linguistic and temporal harms entirely unaddressed. This is the core policy finding: model-level fairness interventions are necessary but insufficient.

---

## 4. The Adversarial Debiasing Failure as a Theoretical Contribution

The adversarial debiasing result merits extended treatment because it is the most theoretically significant finding in the dissertation, and the least predictable from the existing literature.

Adversarial debiasing as implemented in this research follows the architecture of Zhang et al. (2018): a predictor network learns the classification task while an adversary network attempts to predict the protected attribute from the predictor's intermediate representations. The predictor is trained to simultaneously maximise task accuracy and minimise the adversary's success. Under this architecture, the training objective is equivalent to a demographic parity constraint on the learned representation.

The architecture fails here for a reason that is fundamental rather than incidental. Asian Women's content is the most semantically distinct subset in the corpus (KL divergence: 0.555). The vocabulary that predicts "Asian Women" and the vocabulary that predicts the category labels these videos carry are largely identical — there is no separate "fair" latent space from which category can be predicted without group membership also being predictable. Forcing the model to unlearn group-predictive features is, in this context, forcing it to unlearn task-predictive features. The result is a model that cannot categorise Asian Women's content correctly (recall drops) while also failing to remove the group signal (the adversary still wins, just on noisier features). The EOD for Asian Women more than doubles, from 0.105 to 0.283.

This failure has a precise implication: in domains where protected-group membership and task-relevant features are semantically entangled — which is likely whenever the protected attribute is a property of the *content* rather than the *producer* — adversarial debiasing will not work as theorised. The intervention required is upstream of the model: different feature representations (not tag-based), in-domain data augmentation for underrepresented groups, or decomposition of the label space so that group-associated categories are treated as a distinct prediction problem. None of these are standard in the fairness toolkit.

---

## 5. What the Pareto Analysis Shows

The dissertation evaluates six strategies against two objectives: classification accuracy (on the full test set) and fairness (measured as 1 − max|EOD| across all groups). The Pareto frontier contains exactly two non-dominated strategies:

**Pre-processing reweighing** (accuracy: 0.953, fairness: 0.895): The best accuracy of any strategy, including the BERT baseline, and a 60% reduction in Black Women's EOD (0.191 → 0.076). The reweighing approach assigns higher sample weights to underrepresented group-label combinations during training, correcting for the corpus imbalance without requiring the model to unlearn group-predictive features. Critically, it improves accuracy for all groups simultaneously — White Women recall rises from 0.755 to 0.930; Black Women from 0.564 to 0.854 — because the baseline model was undertrained on the minority groups, and correcting the weight distribution improves the overall decision boundary. Reweighing is Pareto-dominant for any deployment context where accuracy matters.

**Post-processing ThresholdOptimizer** (accuracy: 0.837, fairness: 0.935): Near-exact parity across all groups (EOD range: −0.018 to 0.000) at a 4.1 percentage-point accuracy cost. Negative EOD values indicate marginal over-correction, within the range of measurement noise. This strategy is deployment-optimal when the policy requirement is parity — for example, under a regulatory mandate that recall rates for different demographic groups must fall within a defined tolerance. The 4.1 pp cost is the price of that guarantee.

The strategies outside the Pareto frontier are dominated on at least one dimension and should not be selected in a deployment context with explicit accuracy and fairness objectives. The adversarial strategy is dominated on *both* dimensions — it achieves neither good accuracy nor good fairness — making it the clearest case of a theoretically motivated intervention producing empirically negative results.

---

## 6. The Regulatory Gap

No current algorithmic accountability framework — not the EU AI Act (2024), not the Digital Services Act (2022), not the UK Online Safety Act (2023) — mandates intersectional fairness auditing for adult content platforms. The EU AI Act's high-risk classification covers automated hiring, credit scoring, and education systems, but not content recommendation or classification systems for worker-creators on adult platforms. The DSA imposes transparency and audit obligations on Very Large Online Platforms but does not specify fairness metrics or require publication of group-level performance data. The UK Online Safety Act addresses illegal content and child safety but does not engage with economic discrimination against adult creators.

This creates a structural accountability gap. The harm documented in this research is real, measurable, and ongoing. Latina Women's content is recalled correctly at 56.0% — a rate that would not be tolerated in a hiring algorithm subject to disparate impact analysis. Black Women's ratings are declining at a rate that, if sustained, will widen the engagement gap year-on-year with no corrective mechanism. But because no regulatory framework requires the platform to measure, report, or correct these disparities, they are invisible to policy.

The methodological framework developed in this dissertation — corpus construction, intersectional group definition, fairness metric suite, Pareto analysis across intervention types — is designed to be portable across platforms. The audit can be replicated with any platform that exposes sufficient metadata, and the results can be compared across platforms and time. Developing that audit infrastructure into a generalised framework for platform accountability is the next step.

---

## 7. Methodological Notes for Future Researchers

Several decisions in this project carry forward implications for anyone extending this work.

**On inferred attributes:** Race and gender are inferred from user-submitted tags and categories, not self-reported. This is ethically necessary (no self-report mechanism exists at scale) and analytically limiting (the inference is imperfect, culturally specific, and potentially circular). All findings should be interpreted as findings about *how the platform categorises content* rather than about the *creators themselves*. The analysis documents what the system does, not what the creators are.

**On the label definition:** The positive class is defined as videos assigned to at least one of the platform's four protected demographic categories (Black Women, Asian Women, Latina Women, White Women). Videos in the "Other" category — 86.3% of the corpus — are the negative class. This framing means the classification task is "does this video feature racially categorised content?" — not "which racial category does this video belong to?" The fairness analysis then asks whether the model's accuracy on the positive class is equal across groups. This is not the only possible framing, and future work should consider multi-class formulations.

**On the temporal harm:** The rating slope gap (−3.38/year) is estimated from OLS regression on yearly means, with bootstrap confidence intervals on the slope difference. This is a descriptive finding about trend divergence; it does not establish causality. Platform algorithmic changes, changes in user demographics, shifts in content supply, and external market factors could all contribute to the trend. The finding establishes the existence of a statistically significant divergence that demands explanation, not the explanation itself.

**On BERT:** The DistilBERT baseline (test accuracy: 0.926, F1: 0.871) was run on the corrected corpus. BERT reduces fairness gaps (Black Women EOD: 0.066 vs RF baseline 0.191; Latina Women EOD: 0.176) but is dominated in Pareto space by RF reweighing on accuracy and RF post-processing on fairness. Its interpretability limitations and computational cost make it less suitable for deployment-oriented auditing contexts. It is included as an upper bound on text-semantic classification performance.

---

## References

Crenshaw, K. (1989). Demarginalizing the intersection of race and sex: A Black feminist critique of antidiscrimination doctrine, feminist theory and antiracist politics. *University of Chicago Legal Forum*, 1989(1), 139–167.

Crenshaw, K. (1991). Mapping the margins: Intersectionality, identity politics, and violence against women of color. *Stanford Law Review*, 43(6), 1241–1299.

Collins, P. H. (2000). *Black feminist thought: Knowledge, consciousness, and the politics of empowerment* (2nd ed.). Routledge.

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating unwanted biases with adversarial learning. *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society*, 335–340.

Barocas, S., Hardt, M., & Narayanan, A. (2023). *Fairness and machine learning: Limitations and opportunities*. MIT Press. [https://fairmlbook.org](https://fairmlbook.org)

Bird, S., Dudík, M., Edgar, R., Horn, B., Lutz, R., Milan, V., Sameki, M., Wallach, H., & Walker, K. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. *Microsoft Research Technical Report*.

Berk, R., Heidari, H., Jabbari, S., Kearns, M., & Roth, A. (2021). Fairness in criminal justice risk assessments: The state of the art. *Sociological Methods & Research*, 50(1), 3–44.

Noble, S. U. (2018). *Algorithms of oppression: How search engines reinforce racism*. NYU Press.

Benjamin, R. (2019). *Race after technology: Abolitionist tools for the new Jim Code*. Polity Press.

Buolamwini, J., & Gebru, T. (2018). Gender shades: Intersectional accuracy disparities in commercial gender classification. *Proceedings of the 1st Conference on Fairness, Accountability, and Transparency (FAccT)*, 77–91.
