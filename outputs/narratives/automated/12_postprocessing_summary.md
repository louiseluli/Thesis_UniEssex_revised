# Automated Summary: Post-Processing (ThresholdOptimizer, EO)

## Overall Metrics

Split  Accuracy  Precision  Recall    F1
  Val     0.840      0.758   0.639 0.693
 Test     0.839      0.755   0.640 0.693

## Group Metrics

       Group     N  Accuracy  Precision  Recall    F1
 Asian Women  3039     0.841      0.752   0.623 0.681
 Black Women  3193     0.832      0.766   0.642 0.698
Latina Women  2814     0.822      0.802   0.625 0.702
       Other 92248     0.839      0.752   0.641 0.692
 White Women  5754     0.842      0.770   0.650 0.705

## Disparities vs. White Women

Comparison Group  Accuracy Disparity  Equal Opportunity Difference  Precision Disparity
     Asian Women               0.001                         0.027                0.018
     Black Women               0.010                         0.008                0.004
    Latina Women               0.020                         0.025               -0.032
           Other               0.003                         0.009                0.018

## Top 5 Outliers (by confident mistakes)

 video_id                                                              title Group  y_true  y_pred  prob  margin_abs
193962521                      It seems the thunderstorm is already passing. Other       0       1   0.0         0.5
103975511                                             【無】洗練された大人のいやし亭 日南りん 1 Other       0       1   0.0         0.5
193965281                                            I liked it. So unusual. Other       0       1   0.0         0.5
193963631                                                      Feeling great Other       0       1   0.0         0.5
193962651 Somehow it started uncertainly, but in general it turned out great Other       0       1   0.0         0.5

*Note:* Titles in other languages may be harder to interpret; tags/categories help anchor semantics.