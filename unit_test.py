
import unittest
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class IrrigationModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(
            BASE_DIR / "model" / "irrigation_model.pkl"
        )

        df = pd.read_excel(
            BASE_DIR / "irrigation_prediction.csv.xlsx"
        )

        df = df.drop_duplicates()
        df = df.dropna(subset=["Irrigation_Need"])

        cls.X = df.drop(columns=["Irrigation_Need"])
        cls.X = cls.X[cls.model.feature_names_in_]

    def test_model_loading(self):
        self.assertIsNotNone(self.model)

    def test_prediction(self):
        sample = self.X.iloc[:5]
        predictions = self.model.predict(sample)

        self.assertEqual(len(predictions), 5)

    def test_prediction_classes(self):
        sample = self.X.iloc[:5]
        predictions = self.model.predict(sample)

        valid_classes = ["Low", "Medium", "High"]

        for prediction in predictions:
            self.assertIn(str(prediction).strip().title(), valid_classes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
