# CreditScope Week 11 Final Deck Narrative Plan

## Audience

- Course instructors and project evaluators
- Technical audience that expects a clear ML-to-product story

## Objective

Present CreditScope as a finalized, recall-oriented hybrid decision support system, not only as a model comparison exercise.

## Narrative Arc

1. Define the credit default risk problem and recall-first objective.
2. Show the dataset shape and Week 1-4 preparation logic.
3. Explain the final training pipeline and why XGBoost remained the deployed model.
4. Show how business rules convert the model into a review-oriented decision support tool.
5. Prove final validation with SHAP, confusion matrix, API checks, and demo scenarios.
6. Close with future work and project maturity.

## Slide List

1. Cover and final framing
2. Problem, dataset, and EDA summary
3. Preprocessing and feature engineering
4. Model comparison and final choice
5. Final model and threshold logic
6. Business rules and hybrid decision support
7. SHAP and FP/FN analysis
8. Final validation and demo scenarios
9. Future work

## Source Plan

- `outputs/training/training_metrics.json`
- `outputs/training/confusion_matrix.csv`
- `outputs/week10/model_comparison_metrics.csv`
- `outputs/week10/demo_predictions.json`
- `outputs/week10/api_smoke_test.json`
- `outputs/shap/shap_error_analysis_summary.json`
- `docs/week10/figures/ui_cockpit_week10.png`
- `docs/week10/figures/shap_false_positive_week10.png`
- `docs/week10/figures/shap_false_negative_week10.png`

## Visual System

- Warm neutral background with deep green and amber accents
- Strong editorial title style
- Native editable chart objects for metrics
- Existing project screenshots and SHAP figures used as proof visuals

## Editability Plan

- All titles, subtitles, labels, metric cards, and tables remain editable PowerPoint objects.
- Data-backed comparison visuals use native chart objects.
- Project figures are inserted only where they communicate evidence more clearly than recreated shapes.
