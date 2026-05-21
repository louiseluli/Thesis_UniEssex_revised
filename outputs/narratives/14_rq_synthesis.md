# Research Question Synthesis (Step 14)

_Generated from pipeline outputs — seed=95_

## RQ1: Does the baseline exhibit systematic disparities?

Yes. Black Women EOD=0.103; max group EOD=0.141 — systematic recall disparity confirmed at optimal threshold=0.34.

## RQ2: How does intersectionality affect fairness outcomes?

Black Women: Recall=0.833, Accuracy=0.872 vs White Women: Recall=0.936. Recall gap Δ=+0.103. Intersectional penalty confirmed.

## RQ3: Do mitigation strategies reduce disparities effectively?

Reweighing: Test Acc=0.953 (Δ=+0.075 vs baseline), Black Women EOD=+0.076; In-processing (EG): Test Acc=0.877 (Δ=-0.001 vs baseline), Black Women EOD=+0.139; Post-processing: Test Acc=0.837 (Δ=-0.041 vs baseline), Black Women EOD=-0.015

## RQ4: Which strategy has best fairness-accuracy trade-off?

Best fairness (smallest |TPR Δ| for Black Women): Post-Processing (Threshold). Best accuracy: Reweighed RF. Post-Processing achieves optimal Pareto trade-off.
