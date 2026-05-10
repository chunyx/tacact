# Regularization-only LeNet_LSTM 10-seed Report

## A. Purpose
This is a post-hoc targeted improvement validation for regularization only, not part of the original fair multi-model comparison.

## B. Compared models
- Original LeNet_LSTM
- LeNet_LSTM + label_smoothing=0.05 + weight_decay=3e-4

## C. Seed protocol
- Seeds used: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
- Same subject-independent split seeds were used for both models.

## D. Overall mean ± std comparison
- Accuracy: original 0.899505 ± 0.009733, regularized 0.913958 ± 0.014895
- Macro-F1: original 0.899533 ± 0.009596, regularized 0.913515 ± 0.015565
- Inference time (ms): original 0.175817 ± 0.006851, regularized 0.423634 ± 0.006879
- Params (M): original 0.331708 ± 0.000000, regularized 0.803548 ± 0.000000

## E. Hold / Static Drag comparison
- class_3 F1 (Hold): original 0.705204, regularized 0.724155, delta 0.018952
- class_6 F1 (Static Drag): original 0.699199, regularized 0.697992, delta -0.001208
- class_3->6 errors: original 75.600000, regularized 66.800000, delta -8.800000
- class_6->3 errors: original 86.700000, regularized 96.000000, delta 9.300000

## F. Per-class changes
- Most improved classes:
  - class_8: delta_mean_f1=0.058754
  - class_11: delta_mean_f1=0.036174
  - class_2: delta_mean_f1=0.022011
  - class_3: delta_mean_f1=0.018952
  - class_10: delta_mean_f1=0.011884
- Most degraded classes:
  - class_1: delta_mean_f1=-0.004189
  - class_6: delta_mean_f1=-0.001208
  - class_5: delta_mean_f1=-0.000948
  - class_0: delta_mean_f1=0.004818
  - class_4: delta_mean_f1=0.005698

## G. Subject-wise comparison
- subject 26 macro-F1: original 0.743375, regularized 0.759093

## H. Interpretation
This comparison tests whether moderate regularization alone improves subject-independent tactile recognition without changing the original LeNet_LSTM architecture.

## I. Recommendation
The regularized model shows a higher mean Macro-F1 with negligible parameter overhead, so it is a useful ablation and a viable lightweight improvement path.
