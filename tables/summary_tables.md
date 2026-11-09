# EVSA v2.0.2 Verified Publication Tables (Recalculated from Raw Seed Data)

## `table_exp1_generalization.csv`

| agent | task_success_rate_mean | unseen_instance_sr_mean | unseen_compositional_sr_mean | exact_plan_accuracy_mean | step_alignment_accuracy_mean | average_reward_mean | eval_time_ms_per_task_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline A: Static | 0.2648 | 0.3075 | 0.1794 | 0.1237 | 0.6118 | 0.3862 | 0.1311 |
| Baseline B: Conventional Memory | 0.2963 | 0.3781 | 0.1325 | 0.196 | 0.6206 | 0.4098 | 0.4044 |
| Baseline C: Reflection | 0.2565 | 0.2981 | 0.1731 | 0.1356 | 0.6 | 0.3767 | 0.219 |
| Baseline D: Experience Replay | 0.2758 | 0.3531 | 0.1212 | 0.1569 | 0.6005 | 0.3895 | 0.2498 |
| Proposed Agent E: EVSA | 0.545 | 0.6619 | 0.3113 | 0.454 | 0.8344 | 0.6463 | 0.6206 |

## `table_exp2_continual_learning.csv`

| agent | initial_sr_mean | block1_sr_mean | final_sr_mean | total_improvement_abs_mean | total_improvement_pct_mean | auc_sr_mean | strategies_retained_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline A: Static | 0.2927 | 0.2922 | 0.2823 | -0.01042 | -2.794 | 0.2825 | 0 |
| Baseline B: Conventional Memory | 0.2927 | 0.2401 | 0.3568 | 0.06406 | 23.63 | 0.3163 | 125.2 |
| Baseline C: Reflection | 0.2927 | 0.2651 | 0.2755 | -0.01719 | -4.864 | 0.2684 | 155.2 |
| Baseline D: Experience Replay | 0.2927 | 0.25 | 0.3229 | 0.03021 | 12.27 | 0.2945 | 143.7 |
| Proposed Agent E: EVSA | 0.2927 | 0.3286 | 0.6521 | 0.3594 | 126.9 | 0.5431 | 21.5 |

## `table_exp3_distribution_shift.csv`

| agent | pre_shift_sr_mean | immediate_post_shift_sr_mean | performance_degradation_mean | recovered_post_shift_sr_mean | adaptation_gain_mean | recovery_rate_pct_mean | adaptation_speed_episodes_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline A: Static | 0.5 | 0.232 | 0.268 | 0.254 | 0.022 | 52.3 | 87.3 |
| Baseline B: Conventional Memory | 0.5462 | 0.282 | 0.2642 | 0.329 | 0.047 | 61.03 | 88.25 |
| Baseline C: Reflection | 0.5112 | 0.29 | 0.2213 | 0.271 | -0.019 | 53.93 | 78.7 |
| Baseline D: Experience Replay | 0.5425 | 0.274 | 0.2685 | 0.308 | 0.034 | 58.55 | 84.6 |
| Proposed Agent E: EVSA | 0.8175 | 0.36 | 0.4575 | 0.568 | 0.208 | 70.22 | 121.6 |

## `table_exp4_feedback_corruption.csv`

| corruption_rate_eta | agent | subopt_rollouts_mean | false_acceptance_rate_mean | candidate_false_promotion_rate_mean | false_rejection_rate_mean | memory_contamination_rate_mean | recovery_after_misleading_sr_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | Baseline A: Static | 279.6 | 0 | 0 | 0 | 0 | 0.497 |
| 0.1 | Baseline B: Conventional Memory | 286.2 | 0.2199 | 0.2199 | 0.1635 | 0.587 | 0.512 |
| 0.1 | Baseline C: Reflection | 289.6 | 0.2244 | 0.2244 | 0.1552 | 0.7308 | 0.475 |
| 0.1 | Baseline D: Experience Replay | 285.9 | 0.2193 | 0.2193 | 0.1665 | 0.6127 | 0.523 |
| 0.1 | Proposed Agent E: EVSA | 208.2 | 0.01341 | 0.0808 | 0.4411 | 0.3546 | 0.841 |
| 0.25 | Baseline A: Static | 270.6 | 0 | 0 | 0 | 0 | 0.479 |
| 0.25 | Baseline B: Conventional Memory | 285.4 | 0.3281 | 0.3281 | 0.3082 | 0.7635 | 0.499 |
| 0.25 | Baseline C: Reflection | 291.5 | 0.3256 | 0.3256 | 0.3075 | 0.8294 | 0.486 |
| 0.25 | Baseline D: Experience Replay | 287.1 | 0.3134 | 0.3134 | 0.3112 | 0.7765 | 0.471 |
| 0.25 | Proposed Agent E: EVSA | 233.2 | 0.0176 | 0.08639 | 0.5901 | 0.4636 | 0.773 |
| 0.35 | Baseline A: Static | 261.6 | 0 | 0 | 0 | 0 | 0.483 |
| 0.35 | Baseline B: Conventional Memory | 283.2 | 0.3871 | 0.3871 | 0.3553 | 0.8312 | 0.476 |
| 0.35 | Baseline C: Reflection | 287.3 | 0.3787 | 0.3787 | 0.3893 | 0.855 | 0.494 |
| 0.35 | Baseline D: Experience Replay | 283.9 | 0.37 | 0.37 | 0.3894 | 0.8349 | 0.452 |
| 0.35 | Proposed Agent E: EVSA | 248.3 | 0.02532 | 0.1132 | 0.6544 | 0.6829 | 0.749 |

## `table_exp5_ablation.csv`

| variant | overall_stream_sr_mean | noisy_pre_shift_sr_mean | post_shift_recovered_sr_mean | heldout_unseen_sr_mean | pre_shift_contamination_mean | post_shift_contamination_mean | validated_strategies_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EVSA (Full) | 0.5381 | 0.6858 | 0.5858 | 0.3786 | 0.3318 | 0.4189 | 8.95 |
| w/o Evidence Validation | 0.5551 | 0.6883 | 0.625 | 0.4328 | 0.7056 | 0.6922 | 42.55 |
| w/o Confidence Estimation | 0.5306 | 0.6692 | 0.5933 | 0.4094 | 0.3311 | 0.2916 | 6.25 |
| w/o Applicability Conditions | 0.2906 | 0.3058 | 0.2717 | 0.1505 | 0.9 | 0.6 | 0.95 |
| w/o Shift Detection | 0.5156 | 0.7033 | 0.4967 | 0.3302 | 0.3253 | 0.5355 | 12.95 |
| w/o Memory Invalidation | 0.5308 | 0.6958 | 0.565 | 0.3833 | 0.3419 | 0.6329 | 18.2 |

## `table_exp6_efficiency.csv`

| agent | task_success_rate_mean | total_wall_time_ms_mean | eval_time_ms_per_task_mean | memory_kb_mean | active_memory_items_mean | total_rollouts_mean | total_tokens_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline A: Static | 0.2648 | 91.72 | 0.1311 | 0 | 0 | 647.8 | 4.112e+04 |
| Baseline B: Conventional Memory | 0.2963 | 226.2 | 0.4044 | 106.1 | 399.4 | 645.3 | 5.286e+04 |
| Baseline C: Reflection | 0.2565 | 171.7 | 0.219 | 10.62 | 40 | 651.4 | 1.037e+05 |
| Baseline D: Experience Replay | 0.2758 | 165.5 | 0.2498 | 29.87 | 112.5 | 648.3 | 5.771e+04 |
| Proposed Agent E: EVSA | 0.545 | 403.4 | 0.6206 | 13.14 | 48.55 | 590.5 | 4.676e+04 |

## `table_sensitivity_30seeds.csv`

| agent | task_success_rate_mean | unseen_instance_sr_mean | unseen_compositional_sr_mean |
| --- | --- | --- | --- |
| Baseline A: Static | 0.2646 | 0.3063 | 0.1812 |
| Baseline B: Conventional Memory | 0.2912 | 0.3744 | 0.125 |
| Baseline C: Reflection | 0.2518 | 0.2933 | 0.1688 |
| Baseline D: Experience Replay | 0.2736 | 0.3483 | 0.1242 |
| Proposed Agent E: EVSA | 0.5517 | 0.6681 | 0.3187 |

## `table_sensitivity_fairness.csv`

| agent | overall_sr_mean | unseen_instance_sr_mean | unseen_compositional_sr_mean |
| --- | --- | --- | --- |
| EVSA (Direct Macro Exec) | 0.545 | 0.6619 | 0.3113 |
| EVSA (Stochastic Prompt Exec) | 0.4962 | 0.5878 | 0.3131 |
| Replay (Stochastic Prompt Exec) | 0.2758 | 0.3531 | 0.1212 |
| Replay (Direct Macro Exec) | 0.3148 | 0.4197 | 0.105 |

## `table_statistical_tests_all.csv`

| proposed | comparator | cohens_dz | family | metric | primary_test | primary_pvalue_raw | primary_pvalue_holm | primary_pvalue_holm_global74 | sensitivity_t_pvalue_holm_global74 | significant_at_05 | significant_global74_at_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Proposed Agent E: EVSA | Baseline A: Static | 4.512 | RQ1_Common_Benchmark_SR | task_success_rate | Paired t-test | 2.715e-14 | 8.146e-14 | 1.928e-12 | 1.874e-12 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 4.468 | RQ1_Common_Benchmark_SR | task_success_rate | Paired t-test | 3.239e-14 | 8.146e-14 | 2.242e-12 | 2.178e-12 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 4.31 | RQ1_Common_Benchmark_SR | task_success_rate | Paired t-test | 6.234e-14 | 8.146e-14 | 4.177e-12 | 4.052e-12 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 6.349 | RQ1_Common_Benchmark_SR | task_success_rate | Paired t-test | 5.033e-17 | 2.013e-16 | 3.725e-15 | 3.725e-15 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 3.988 | RQ1_Unseen_Instance_SR | unseen_instance_sr | Paired t-test | 2.526e-13 | 5.051e-13 | 1.616e-11 | 1.566e-11 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 3.806 | RQ1_Unseen_Instance_SR | unseen_instance_sr | Paired t-test | 5.83e-13 | 5.83e-13 | 3.673e-11 | 3.556e-11 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 4.292 | RQ1_Unseen_Instance_SR | unseen_instance_sr | Paired t-test | 6.708e-14 | 2.012e-13 | 4.427e-12 | 4.293e-12 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 4.654 | RQ1_Unseen_Instance_SR | unseen_instance_sr | Paired t-test | 1.542e-14 | 6.167e-14 | 1.11e-12 | 1.079e-12 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 1.573 | RQ1_Unseen_Compositional_SR | unseen_compositional_sr | Paired t-test | 1.073e-06 | 1.073e-06 | 3.756e-05 | 3.219e-05 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 2.064 | RQ1_Unseen_Compositional_SR | unseen_compositional_sr | Paired t-test | 1.884e-08 | 5.652e-08 | 7.347e-07 | 6.782e-07 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 1.679 | RQ1_Unseen_Compositional_SR | unseen_compositional_sr | Paired t-test | 4.24e-07 | 8.479e-07 | 1.526e-05 | 1.357e-05 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 2.122 | RQ1_Unseen_Compositional_SR | unseen_compositional_sr | Paired t-test | 1.218e-08 | 4.873e-08 | 4.995e-07 | 4.629e-07 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 4.454 | RQ2_Final_Episode_SR | final_sr | Paired t-test | 3.429e-14 | 1.029e-13 | 2.332e-12 | 2.263e-12 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 3 | RQ2_Final_Episode_SR | final_sr | Paired t-test | 3.847e-11 | 3.847e-11 | 2.154e-09 | 2.077e-09 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 5.422 | RQ2_Final_Episode_SR | final_sr | Paired t-test | 9.354e-16 | 3.741e-15 | 6.828e-14 | 6.828e-14 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 3.207 | RQ2_Final_Episode_SR | final_sr | Paired t-test | 1.209e-11 | 2.417e-11 | 7.01e-10 | 6.768e-10 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 0.8764 | RQ2_Early_Block1_SR | block1_sr | Paired t-test | 0.0009207 | 0.0009207 | 0.01749 | 0.01749 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 1.869 | RQ2_Early_Block1_SR | block1_sr | Paired t-test | 8.691e-08 | 3.476e-07 | 3.216e-06 | 2.868e-06 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 1.487 | RQ2_Early_Block1_SR | block1_sr | Wilcoxon signed-rank | 8.807e-05 | 0.0001761 | 0.002189 | 6.277e-05 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 1.426 | RQ2_Early_Block1_SR | block1_sr | Paired t-test | 4.085e-06 | 1.226e-05 | 0.0001348 | 0.0001061 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 3.789 | RQ4_Post_Shift_Recovered_SR | recovered_post_shift_sr | Paired t-test | 6.328e-13 | 2.531e-12 | 3.923e-11 | 3.797e-11 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 2.107 | RQ4_Post_Shift_Recovered_SR | recovered_post_shift_sr | Paired t-test | 1.356e-08 | 1.356e-08 | 5.422e-07 | 5.015e-07 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 2.435 | RQ4_Post_Shift_Recovered_SR | recovered_post_shift_sr | Paired t-test | 1.309e-09 | 3.928e-09 | 5.892e-08 | 5.499e-08 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 2.233 | RQ4_Post_Shift_Recovered_SR | recovered_post_shift_sr | Paired t-test | 5.383e-09 | 1.077e-08 | 2.315e-07 | 2.153e-07 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 1.31 | RQ4_Adaptation_Gain | adaptation_gain | Paired t-test | 1.219e-05 | 3.657e-05 | 0.0003657 | 0.0002804 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 0.9561 | RQ4_Adaptation_Gain | adaptation_gain | Paired t-test | 0.0004083 | 0.0004083 | 0.008575 | 0.008575 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 1.359 | RQ4_Adaptation_Gain | adaptation_gain | Paired t-test | 7.591e-06 | 3.036e-05 | 0.0002353 | 0.0001822 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 1.291 | RQ4_Adaptation_Gain | adaptation_gain | Paired t-test | 1.45e-05 | 3.657e-05 | 0.0004205 | 0.000319 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 0.9541 | RQ4_Recovery_Rate_Pct | recovery_rate_pct | Paired t-test | 0.0004167 | 0.001667 | 0.008575 | 0.008575 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 0.4114 | RQ4_Recovery_Rate_Pct | recovery_rate_pct | Paired t-test | 0.08149 | 0.09881 | 0.8149 | 0.8149 | False | False |
| Proposed Agent E: EVSA | Baseline C: Reflection | 0.7769 | RQ4_Recovery_Rate_Pct | recovery_rate_pct | Paired t-test | 0.002538 | 0.007613 | 0.0406 | 0.0406 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 0.4694 | RQ4_Recovery_Rate_Pct | recovery_rate_pct | Paired t-test | 0.0494 | 0.09881 | 0.5434 | 0.5434 | False | False |
| Proposed Agent E: EVSA | Baseline A: Static | 2.475 | RQ4_Unshifted_Domain_Retention | unshifted_domain_sr | Paired t-test | 1.001e-09 | 4.002e-09 | 4.703e-08 | 4.403e-08 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 1.927 | RQ4_Unshifted_Domain_Retention | unshifted_domain_sr | Paired t-test | 5.437e-08 | 1.631e-07 | 2.066e-06 | 1.903e-06 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 1.915 | RQ4_Unshifted_Domain_Retention | unshifted_domain_sr | Wilcoxon signed-rank | 5.722e-06 | 1.144e-05 | 0.0001831 | 2.048e-06 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 1.426 | RQ4_Unshifted_Domain_Retention | unshifted_domain_sr | Wilcoxon signed-rank | 4.768e-05 | 4.768e-05 | 0.00124 | 0.0001061 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | -2.15 | RQ6_Contamination_eta_0.10 | memory_contamination_rate | Paired t-test | 9.852e-09 | 9.852e-09 | 4.138e-07 | 3.842e-07 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | -2.409 | RQ6_Contamination_eta_0.10 | memory_contamination_rate | Paired t-test | 1.568e-09 | 3.137e-09 | 6.901e-08 | 6.43e-08 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | -2.707 | RQ6_Contamination_eta_0.10 | memory_contamination_rate | Paired t-test | 2.234e-10 | 6.702e-10 | 1.229e-08 | 1.162e-08 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 3.499 | RQ6_PostCleanRecovery_eta_0.10 | recovery_after_misleading_sr | Paired t-test | 2.614e-12 | 7.842e-12 | 1.542e-10 | 1.49e-10 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 3.664 | RQ6_PostCleanRecovery_eta_0.10 | recovery_after_misleading_sr | Paired t-test | 1.151e-12 | 4.604e-12 | 7.021e-11 | 6.791e-11 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 4.853 | RQ6_PostCleanRecovery_eta_0.10 | recovery_after_misleading_sr | Wilcoxon signed-rank | 8.782e-05 | 8.782e-05 | 0.002189 | 5.169e-13 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 3.138 | RQ6_PostCleanRecovery_eta_0.10 | recovery_after_misleading_sr | Paired t-test | 1.765e-11 | 3.531e-11 | 1.006e-09 | 9.71e-10 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | -1.566 | RQ6_Contamination_eta_0.25 | memory_contamination_rate | Wilcoxon signed-rank | 3.624e-05 | 3.815e-05 | 0.0009785 | 3.316e-05 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | -1.64 | RQ6_Contamination_eta_0.25 | memory_contamination_rate | Wilcoxon signed-rank | 1.907e-05 | 3.815e-05 | 0.0005341 | 1.835e-05 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | -1.545 | RQ6_Contamination_eta_0.25 | memory_contamination_rate | Paired t-test | 1.375e-06 | 4.124e-06 | 4.674e-05 | 3.849e-05 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 2.569 | RQ6_PostCleanRecovery_eta_0.25 | recovery_after_misleading_sr | Paired t-test | 5.392e-10 | 1.649e-09 | 2.696e-08 | 2.534e-08 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 2.446 | RQ6_PostCleanRecovery_eta_0.25 | recovery_after_misleading_sr | Paired t-test | 1.22e-09 | 1.947e-09 | 5.61e-08 | 5.244e-08 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 2.479 | RQ6_PostCleanRecovery_eta_0.25 | recovery_after_misleading_sr | Paired t-test | 9.736e-10 | 1.947e-09 | 4.673e-08 | 4.381e-08 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 2.61 | RQ6_PostCleanRecovery_eta_0.25 | recovery_after_misleading_sr | Paired t-test | 4.121e-10 | 1.649e-09 | 2.184e-08 | 2.061e-08 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | -0.7412 | RQ6_Contamination_eta_0.35 | memory_contamination_rate | Paired t-test | 0.003641 | 0.006272 | 0.05097 | 0.05097 | True | False |
| Proposed Agent E: EVSA | Baseline C: Reflection | -0.796 | RQ6_Contamination_eta_0.35 | memory_contamination_rate | Paired t-test | 0.002091 | 0.006272 | 0.03554 | 0.03554 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | -0.7562 | RQ6_Contamination_eta_0.35 | memory_contamination_rate | Paired t-test | 0.00313 | 0.006272 | 0.04695 | 0.04695 | True | True |
| Proposed Agent E: EVSA | Baseline A: Static | 2.608 | RQ6_PostCleanRecovery_eta_0.35 | recovery_after_misleading_sr | Paired t-test | 4.188e-10 | 1.675e-09 | 2.184e-08 | 2.061e-08 | True | True |
| Proposed Agent E: EVSA | Baseline B: Conventional Memory | 2.578 | RQ6_PostCleanRecovery_eta_0.35 | recovery_after_misleading_sr | Paired t-test | 5.069e-10 | 1.675e-09 | 2.585e-08 | 2.433e-08 | True | True |
| Proposed Agent E: EVSA | Baseline C: Reflection | 2.548 | RQ6_PostCleanRecovery_eta_0.35 | recovery_after_misleading_sr | Paired t-test | 6.176e-10 | 1.675e-09 | 3.026e-08 | 2.841e-08 | True | True |
| Proposed Agent E: EVSA | Baseline D: Experience Replay | 2.971 | RQ6_PostCleanRecovery_eta_0.35 | recovery_after_misleading_sr | Wilcoxon signed-rank | 8.757e-05 | 8.757e-05 | 0.002189 | 2.408e-09 | True | True |
| EVSA (Full) | w/o Evidence Validation | -0.2786 | RQ3_Ablation_Overall_SR | overall_stream_sr | Paired t-test | 0.2279 | 0.6838 | 1 | 1 | False | False |
| EVSA (Full) | w/o Confidence Estimation | 0.2266 | RQ3_Ablation_Overall_SR | overall_stream_sr | Paired t-test | 0.3237 | 0.6838 | 1 | 1 | False | False |
| EVSA (Full) | w/o Applicability Conditions | 4.802 | RQ3_Ablation_Overall_SR | overall_stream_sr | Wilcoxon signed-rank | 8.845e-05 | 0.0004422 | 0.002189 | 6.182e-13 | True | True |
| EVSA (Full) | w/o Shift Detection | 0.6246 | RQ3_Ablation_Overall_SR | overall_stream_sr | Paired t-test | 0.0116 | 0.04639 | 0.1508 | 0.1508 | True | False |
| EVSA (Full) | w/o Memory Invalidation | 0.1938 | RQ3_Ablation_Overall_SR | overall_stream_sr | Paired t-test | 0.397 | 0.6838 | 1 | 1 | False | False |
| EVSA (Full) | w/o Evidence Validation | -0.3222 | RQ3_Ablation_PostShift_SR | post_shift_recovered_sr | Paired t-test | 0.1658 | 0.4975 | 1 | 1 | False | False |
| EVSA (Full) | w/o Confidence Estimation | -0.09493 | RQ3_Ablation_PostShift_SR | post_shift_recovered_sr | Paired t-test | 0.6759 | 0.8196 | 1 | 1 | False | False |
| EVSA (Full) | w/o Applicability Conditions | 3.554 | RQ3_Ablation_PostShift_SR | post_shift_recovered_sr | Paired t-test | 1.982e-12 | 9.908e-12 | 1.189e-10 | 1.149e-10 | True | True |
| EVSA (Full) | w/o Shift Detection | 0.8761 | RQ3_Ablation_PostShift_SR | post_shift_recovered_sr | Paired t-test | 0.0009241 | 0.003696 | 0.01749 | 0.01749 | True | True |
| EVSA (Full) | w/o Memory Invalidation | 0.1885 | RQ3_Ablation_PostShift_SR | post_shift_recovered_sr | Paired t-test | 0.4098 | 0.8196 | 1 | 1 | False | False |
| EVSA (Full) | w/o Evidence Validation | -0.3996 | RQ3_Ablation_Unseen_SR | heldout_unseen_sr | Paired t-test | 0.08988 | 0.2696 | 0.8149 | 0.8149 | False | False |
| EVSA (Full) | w/o Confidence Estimation | -0.3146 | RQ3_Ablation_Unseen_SR | heldout_unseen_sr | Paired t-test | 0.1756 | 0.3513 | 1 | 1 | False | False |
| EVSA (Full) | w/o Applicability Conditions | 2.707 | RQ3_Ablation_Unseen_SR | heldout_unseen_sr | Paired t-test | 2.236e-10 | 1.118e-09 | 1.229e-08 | 1.162e-08 | True | True |
| EVSA (Full) | w/o Shift Detection | 0.5326 | RQ3_Ablation_Unseen_SR | heldout_unseen_sr | Paired t-test | 0.02785 | 0.1114 | 0.3342 | 0.3342 | False | False |
| EVSA (Full) | w/o Memory Invalidation | -0.04902 | RQ3_Ablation_Unseen_SR | heldout_unseen_sr | Paired t-test | 0.8288 | 0.8288 | 1 | 1 | False | False |
| EVSA (Stochastic Prompt Exec) | Replay (Stochastic Prompt Exec) | 4.094 | Fairness_Execution_Parity | overall_sr | Paired t-test | 1.575e-13 | 1.575e-13 | 1.024e-11 | 9.92e-12 | True | True |
| EVSA (Direct Macro Exec) | Replay (Direct Macro Exec) | 4.471 | Fairness_Execution_Parity | overall_sr | Paired t-test | 3.203e-14 | 6.406e-14 | 2.242e-12 | 2.178e-12 | True | True |
