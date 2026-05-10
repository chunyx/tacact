# LeNet_LSTM Diagnosis Report

## A. Selected run
- Selected run: `/home/yaxin/tacact/outputs_main_experiment_parallel_20260429_150413/parallel_workers/LeNet_LSTM_seed43/subject_seed43`
- Selection rule: highest test macro-F1, then highest test accuracy, then latest modified time.
- Selected test macro-F1: 0.918365
- Selected test accuracy: 0.918229

## B. Overall result
- Test accuracy: 0.918229
- Test macro-F1: 0.918365
- Test macro-precision: 0.919435
- Test macro-recall: 0.918229
- Inference time (ms): 0.417689
- Parameter count (M): 0.803548
- Best validation epoch: 19.0

## C. Weakest classes
- class_6 (class_id=6): F1=0.7050, precision=0.7006, recall=0.7094
- class_3 (class_id=3): F1=0.7284, precision=0.7195, recall=0.7375
- class_7 (class_id=7): F1=0.9190, precision=0.9754, recall=0.8687
- class_0 (class_id=0): F1=0.9423, precision=0.9408, recall=0.9437
- class_11 (class_id=11): F1=0.9490, precision=0.9675, recall=0.9312

Possible interpretation: the weakest classes likely correspond to tactile patterns that the current frame encoder or temporal pooling separates less reliably.

## D. Most confused class pairs
- true class_6 -> pred class_3: 89 samples
- true class_3 -> pred class_6: 72 samples
- true class_7 -> pred class_8: 18 samples
- true class_1 -> pred class_5: 17 samples
- true class_11 -> pred class_6: 13 samples

Interpretation note: class names are generic (`class_i`), so confusion semantics cannot be fully inferred from labels alone. The saved representative examples should be used to judge whether the confusion looks more spatial or more temporal.

## E. Correct vs wrong sample statistics
- Statistical comparison files saved in `correct_vs_wrong_stats.csv`.

## F. Subject-wise generalization
- subject 26: accuracy=0.7875, macro_f1=0.7590
- subject 5: accuracy=0.8958, macro_f1=0.8946
- subject 10: accuracy=0.9229, macro_f1=0.9210
- subject 21: accuracy=0.9354, macro_f1=0.9306
- subject 4: accuracy=0.9396, macro_f1=0.9404

## G. Training stability
- best_val_f1_epoch: 19
- best_val_loss_epoch: 10
- val loss increases while train loss decreases: True
- val F1 oscillation std after best epoch: 0.001544
- clear overfitting pattern: True
- optimization looks unstable: False

## H. Diagnosis conclusion
- 5. optimization instability / overfitting
- 1. spatial feature extraction limitation
- 4. subject-independent generalization difficulty
- 2. temporal fusion limitation

### Evidence
- Validation loss rises relative to its minimum while training loss continues to stay very low, which is a classic overfitting pattern.
- A small subset of classes is much weaker than the rest, suggesting the frame encoder is not equally discriminative for all tactile patterns.
- Performance varies strongly across held-out subjects, indicating subject-specific transfer difficulty.
- The most common confusions are concentrated in a few class pairs rather than uniformly spread, which can indicate insufficient temporal discrimination for similar action patterns.

## I. Suggested next-step modifications
- Because validation loss rises after the best epoch while train loss becomes extremely small, stronger regularization such as label smoothing, SWA, or checkpoint averaging is justified.
- Because only a few classes are consistently weak, a lightweight channel-attention module such as ECA/SE in the frame encoder is a targeted next step to strengthen spatial discrimination without redesigning the whole model.
- Because errors concentrate on some held-out subjects, subject-level normalization or subject-robust augmentation is a better-matched next step than only tuning the classifier head.
- Because the main confusions are concentrated in a few class pairs, lightweight temporal attention pooling is a targeted next step if the representative examples appear temporally similar.

## Notes on feature definitions
- `active_length` = number of preprocessed frames with any non-zero absolute tactile delta (> 1e-6).
- `active_area_mean/max` = mean/max number of active taxels per frame using the same > 1e-6 rule on preprocessed cached frames.
- `motion_energy` = mean absolute difference between consecutive preprocessed frames.
- All sequence statistics were computed from the cached preprocessed tactile sequences when available, before dataset standardization.
