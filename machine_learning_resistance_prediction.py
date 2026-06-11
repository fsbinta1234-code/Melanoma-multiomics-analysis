"""
Phase 10 — Resistance prediction with Machine Learning.

Trains a Random Forest to classify resistant (ARoe) vs control (LacZ) samples
from the phosphoproteomic profile and ranks phosphosites by importance
(biomarker prioritisation). There are 48 samples (24 vs 24).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

import pipeline_config as cfg
from cleanDatas import CleanDatas


class MachineLearningResistancePrediction:

    @staticmethod
    def run_prediction() -> RandomForestClassifier:
        cfg.apply_style()
        phospho = CleanDatas.clean_phospho_sty_sites()

        X = phospho.T                     # samples × phosphosites
        y = np.array(cfg.maxquant_labels(phospho))   # 1=resistant, 0=control
        print(f"Samples: {X.shape[0]}  |  features: {X.shape[1]}  |  "
              f"resistant: {int(y.sum())}  control: {int((y == 0).sum())}")

        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)

        # Stratified 5-fold cross-validation (more robust estimate)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"Accuracy (5-fold CV) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Final model evaluated on a held-out test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f"Accuracy (held-out test) : {accuracy_score(y_test, pred):.3f}")
        print(classification_report(y_test, pred, target_names=["LacZ", "ARoe"], zero_division=0))

        MachineLearningResistancePrediction._plot_confusion(model, X_test, y_test)
        MachineLearningResistancePrediction._plot_importance(model, phospho.index)
        return model

    @staticmethod
    def _plot_confusion(model, X_test, y_test) -> None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_estimator(
            model, X_test, y_test, display_labels=["LacZ", "ARoe"],
            cmap="Blues", colorbar=False, ax=ax)
        ax.set_title("Confusion Matrix (test set)")
        cfg.save_figure(fig, "10_ml_confusion_matrix")

    @staticmethod
    def _plot_importance(model, feature_names, top_n: int = 20) -> None:
        importances = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(importances.index[::-1], importances.values[::-1],
                color=cfg.COLOR_CONTROL, edgecolor="#333333", alpha=0.9)
        ax.set_xlabel("Importance (Gini)")
        ax.set_title(f"Top {top_n} phosphosite predictors of resistance")
        ax.tick_params(axis="y", labelsize=8)
        cfg.save_figure(fig, "10_ml_feature_importance")


def main():
    MachineLearningResistancePrediction.run_prediction()


if __name__ == "__main__":
    main()
