import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from cleanDatas import CleanDatas


class MachineLearningResistancePrediction:
    """
    Predicts BRAFi/MEKi drug resistance using machine learning models trained
    on phosphoproteomic intensity profiles.

    A Random Forest classifier distinguishes resistant versus sensitive
    melanoma conditions and ranks phosphosites by feature importance for
    biomarker prioritisation.

    Expected Output: predictive resistance model, drug-response classifier,
    and biomarker prioritisation ranking.
    """

    # 6 binary labels: 3 control (0) followed by 3 resistant (1)
    SAMPLE_LABELS = [0, 0, 0, 1, 1, 1]

    @staticmethod
    def run_prediction(phospho_log2: pd.DataFrame = None) -> RandomForestClassifier:
        """
        Trains and evaluates a Random Forest resistance classifier.

        The phospho intensity matrix is transposed so that each sample
        (column) becomes a row and each phosphosite (row) becomes a feature.
        The classifier is trained on a stratified 70/30 split and evaluated
        with accuracy and a full classification report.

        Parameters
        ----------
        phospho_log2 : pd.DataFrame, optional
            Log2-normalised phospho intensity matrix (sites × samples).
            When None, CleanDatas.clean_phospho_sty_sites() is called.

        Returns
        -------
        RandomForestClassifier
            The fitted classifier.
        """
        if phospho_log2 is None:
            phospho_log2 = CleanDatas.clean_phospho_sty_sites()

        # Transpose: rows = samples, columns = phosphosite features
        X = phospho_log2.T
        y = MachineLearningResistancePrediction.SAMPLE_LABELS

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy    = accuracy_score(y_test, predictions)

        print(f"Accuracy : {accuracy:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))

        MachineLearningResistancePrediction._plot_feature_importance(
            model, phospho_log2.index
        )

        return model

    @staticmethod
    def _plot_feature_importance(
        model: RandomForestClassifier,
        feature_names: pd.Index,
        top_n: int = 20,
    ) -> None:
        """
        Plots the top-N most important phosphosite features.

        Parameters
        ----------
        model : RandomForestClassifier
            Fitted Random Forest model.
        feature_names : pd.Index
            Names of all phosphosite features (rows of the original matrix).
        top_n : int
            Number of top features to display.
        """
        importances = pd.Series(
            model.feature_importances_,
            index=feature_names,
        ).nlargest(top_n)

        fig, ax = plt.subplots(figsize=(9, 6))
        importances[::-1].plot(kind='barh', color='steelblue', ax=ax)

        ax.set_xlabel('Feature Importance (Gini)', fontsize=12)
        ax.set_title(f'Top {top_n} Phosphosite Predictors of Resistance', fontsize=13)
        ax.tick_params(axis='y', labelsize=8)

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    MachineLearningResistancePrediction.run_prediction()


if __name__ == "__main__":
    main()
