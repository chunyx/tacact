# MotionInput + Regularization 10-seed Report

## A. Purpose
This is a post-hoc targeted improvement validation, not part of the original fair multi-model comparison.

## B. Compared models
- Original LeNet_LSTM
- LeNet_LSTM_MotionInput + LS0.05 + WD3e-4

## C. Seed protocol
- Seeds used: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
- Same subject-independent split seeds were used for both models.

## D. Overall mean ± std comparison
- Accuracy: original 0.899505 ± 0.009733, improved 0.934948 ± 0.012971
- Macro-F1: original 0.899533 ± 0.009596, improved 0.934305 ± 0.013559
- Inference time (ms): original 0.175817 ± 0.006851, improved 0.468207 ± 0.012987
- Params (M): original 0.331708 ± 0.000000, improved 0.803948 ± 0.000000

## E. Hold / Static Drag comparison
- class_3 F1 (Hold): original 0.705204, improved 0.767844, delta 0.062641
- class_6 F1 (Static Drag): original 0.699199, improved 0.741500, delta 0.042301
- class_3->6 errors: original 75.600000, improved 61.800000, delta -13.800000
- class_6->3 errors: original 86.700000, improved 82.600000, delta -4.100000

## F. Per-class changes
- Most improved classes:
  - class_8: delta_mean_f1=0.100119
  - class_11: delta_mean_f1=0.067712
  - class_3: delta_mean_f1=0.062641
  - class_6: delta_mean_f1=0.042301
  - class_2: delta_mean_f1=0.035323
- Most degraded classes:
  - class_5: delta_mean_f1=0.001730
  - class_1: delta_mean_f1=0.004670
  - class_10: delta_mean_f1=0.010326
  - class_7: delta_mean_f1=0.015677
  - class_0: delta_mean_f1=0.020075

## G. Subject-wise comparison
- subject 26 macro-F1: original 0.743375, improved 0.790306

## H. Interpretation
This comparison tests whether explicit frame-difference motion input plus moderate regularization consistently improves dynamic tactile recognition under the same subject-independent split seeds.

## I. Recommendation
The improved model shows a higher mean Macro-F1 with only a small overhead in inference time and parameter count, so it is the current recommended improved LeNet_LSTM variant.
