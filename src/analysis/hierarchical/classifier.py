import numpy as np
import pandas as pd
import stan


def test_model_signatures(betas: np.ndarray, e_factor=1.7, th_factor=0) -> np.ndarray:
    """
    Test percentage of samples from the posterior over betas that
    pass the test for all models in customtype.ModelName.
    """

    stdev = np.median(np.std(betas, axis=1))
    th = stdev * th_factor
    e = stdev * e_factor

    dra = (
        (betas[0] - betas[1] > th)
        * (betas[0] - betas[2] > th)
        * (betas[0] - betas[3] > th)
        * (betas[1] - betas[3] > th)
        * (betas[2] - betas[3] > th)
    )
    freq = (
        (abs(betas[0] - betas[1]) < e)
        * (abs(betas[2] - betas[3]) < e)
        * ((betas[0] - betas[2]) > th)
        * ((betas[0] - betas[3]) > th)
        * ((betas[1] - betas[2]) > th)
        * ((betas[1] - betas[3]) > th)
    )
    stakes = (
        (abs(betas[0] - betas[2]) < e)
        * (abs(betas[1] - betas[3]) < e)
        * ((betas[0] - betas[1]) > th)
        * ((betas[0] - betas[3]) > th)
        * ((betas[2] - betas[1]) > th)
        * ((betas[2] - betas[3]) > th)
    )
    ep = (
        (abs(betas[0] - betas[1]) < e)
        * (abs(betas[0] - betas[2]) < e)
        * (abs(betas[0] - betas[3]) < e)
        * (abs(betas[1] - betas[2]) < e)
        * (abs(betas[1] - betas[3]) < e)
        * (abs(betas[2] - betas[3]) < e)
    )
    dra2 = (
        (betas[0] - betas[3] > th)
        * (betas[0] - betas[1] > -e)
        * (betas[0] - betas[2] > -e)
        * (betas[1] - betas[3] > -e)
        * (betas[2] - betas[3] > -e)
        * np.logical_not(freq + stakes + ep)
    )

    return [100 * np.sum(x) / len(x) for x in [dra, dra2, freq, stakes, ep]]


class Classifier:
    def __init__(
        self, fit: stan.fit.Fit, equality_thresh: float = 1.7, class_thresh: float = 20
    ) -> None:
        self.samples = fit["beta"]
        self.equality_thresh = equality_thresh
        self.class_thresh = class_thresh
        self.perf_metrics = pd.DataFrame(
            columns=["DRAx", "DRA+", "Freq", "Stakes", "EP", "class"]
        )
        self.class_map = {0: "DRA", 1: "DRA", 2: "Freq", 3: "Stakes", 4: "EP", 5: None}

    def classify(self):
        for i, beta in enumerate(self.samples):
            model_signatures = test_model_signatures(beta, self.equality_thresh) + [
                self.class_thresh
            ]
            model_class = self.class_map[np.argmax(model_signatures)]
            self.perf_metrics.loc[i] = model_signatures[:-1] + [model_class]
        return self.perf_metrics

    def merge_with_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.perf_metrics.merge(df, left_index=True, right_index=True)

    def get_ids(self, model: str) -> list:
        return list(
            self.perf_metrics.loc[self.perf_metrics["class"] == model].index + 1
        )
