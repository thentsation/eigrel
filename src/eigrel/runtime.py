"""Helpers that generated programs, and the models they register, import at run time.

They need scikit-learn, which the generated code requires anyway; the compiler never imports this
module.
"""

from typing import Any

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


class EncodedLabelClassifier(ClassifierMixin, BaseEstimator):
    """A classifier for estimators that only accept labels 0..n-1, such as XGBoost.

    It encodes the labels before training and decodes the predictions, so the trained model (and
    any version registered from it) predicts the original classes, not their indices.
    """

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> 'EncodedLabelClassifier':
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.estimator_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        return self

    def predict(self, X: Any) -> Any:
        return self.encoder_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> Any:
        return self.estimator_.predict_proba(X)
