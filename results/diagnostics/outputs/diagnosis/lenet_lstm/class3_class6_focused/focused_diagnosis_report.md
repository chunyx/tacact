# Focused class_3 / class_6 Diagnosis Report

## A. Purpose
The goal is to diagnose why the current LeNet_LSTM strongly confuses class_3 and class_6 before changing the model.

## B. Selected run
- Run: `/home/yaxin/tacact/outputs_main_experiment_parallel_20260429_150413/parallel_workers/LeNet_LSTM_seed43/subject_seed43`
- test_accuracy = 0.918229166666667
- test_macro_f1 = 0.918365439263423
- test_macro_precision = 0.919435408522443
- test_macro_recall = 0.918229166666664
- inference_ms = 0.417688768357038
- params_m = 0.803548
- best_epoch = 19
- best_val_loss = 0.454684924043936
- best_val_acc = 0.922321428571429
- best_val_f1 = 0.922768108958380
- training_seconds = 1920.899531

## C. Label mapping
- class_3 -> gesture 4 -> Hold
- class_6 -> gesture 7 -> Static Drag
- Note: the gesture names were found in the repository-side auxiliary visualization script, not in the experiment result files themselves.

## D. Confusion summary
- class_3 -> class_6 errors: 72
- class_6 -> class_3 errors: 89

## E. Confidence diagnosis
- mean confidence for correct class_3: 0.989197
- mean confidence for correct class_6: 0.987952
- mean confidence for class_3 -> class_6 errors: 0.957099
- mean confidence for class_6 -> class_3 errors: 0.979548
- Interpretation: high-confidence systematic errors.

## F. Weak / short signal diagnosis
- correct class_3 active_length mean: 78.2754
- class_3 -> class_6 active_length mean: 72.5278
- correct class_6 active_length mean: 77.5991
- class_6 -> class_3 active_length mean: 75.9213
- all correct(3/6) motion_energy mean: 0.4324
- all confused(3/6) motion_energy mean: 0.3957
- Interpretation: if confused samples are clearly shorter or lower-energy, weak/short tactile signal difficulty is a plausible factor.

## G. Subject diagnosis
- subject 26: class_3->6 errors=22, class_6->3 errors=13, class_3_accuracy=0.3500, class_6_accuracy=0.6500
- subject 26: class_3->6=22, class_6->3=13
- subject 5: class_3->6=13, class_6->3=4
- subject 4: class_3->6=13, class_6->3=0
- subject 26 contributes disproportionately to this confusion relative to a uniform subject split.

## H. Temporal diagnosis
- See the four temporal plots and `temporal_curves_class3_class6.csv`.
- If the correct class_3 and correct class_6 curves are distinguishable but the confused curves collapse toward each other, this supports a temporal fusion limitation.

## I. Spatial diagnosis
- See `average_pressure_maps_class3_class6.png`, `difference_pressure_maps_class3_class6.png`, and `confused_vs_correct_pressure_maps.png`.
- If correct class_3 and class_6 maps already look very similar, the confusion may reflect intrinsic action similarity. If subtle but localized differences exist, the current frame encoder may be missing them.

## J. Overfitting diagnosis
- best_val_loss_epoch = 10
- best_val_f1_epoch = 19
- train loss keeps decreasing while val loss rises: True
- This supports trying regularization-focused improvements before architectural changes if the main issue is late-epoch memorization.

## K. Main bottleneck conclusion
- ambiguous label/action similarity or systematic feature overlap
- subject-independent generalization difficulty
- overfitting / insufficient regularization
- temporal fusion limitation and/or spatial feature extraction limitation

## L. Evidence-based next-step recommendations
- Because train loss keeps decreasing while validation loss rises after epoch 10, stronger regularization such as label smoothing, stronger weight decay/dropout, SWA, or checkpoint averaging should be tried before structural changes.
- Because the wrong class_3/class_6 predictions are highly confident, the issue is not just uncertain classification; it likely reflects consistent feature overlap. Structural changes should only be justified after checking whether temporal curves or spatial maps show systematic differences.
- If the temporal curves show class_3 and class_6 differ mainly in how the signal evolves over time rather than frame-level intensity, lightweight temporal attention pooling is justified.
- If the average pressure maps show subtle but localized spatial differences between class_3 and class_6, lightweight ECA/SE in the frame encoder is justified.
- If confused samples are shorter/weaker, action-region refinement, time weighting, or weak-signal augmentation is a better-matched next step than simply increasing model size.
- If subject 26 or a few held-out subjects contribute disproportionately, subject-robust augmentation such as pressure scaling, temporal speed perturbation, or spatial shift augmentation is better supported than changing only the classifier head.
- If the spatial maps and temporal curves for class_3 and class_6 are genuinely very similar even in correct cases, then part of the confusion may reflect intrinsic action similarity rather than a simple model defect.
