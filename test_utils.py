"""Unit tests for utils.py. Run:  python -m unittest test_utils -v"""
import unittest

import utils


class UtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = utils.load_data()
        cls.model = utils.load_model()
        cls.features = list(cls.model.feature_names_in_)

    def test_data_loaded(self):
        self.assertEqual(self.df.shape[1], 20)
        self.assertEqual(set(self.df[utils.TARGET].unique()), set(utils.CLASS_ORDER))

    def test_validation_accepts_median_record(self):
        rec = utils.default_record(self.df, self.features)
        self.assertEqual(utils.validate_inputs(rec), [])

    def test_validation_rejects_bad_values(self):
        rec = utils.default_record(self.df, self.features)
        rec.update({"Soil_pH": 15, "Soil_Moisture": -5, "Humidity": 120,
                    "Field_Area_hectare": 0, "Rainfall_mm": -1})
        self.assertEqual(len(utils.validate_inputs(rec)), 5)

    def test_prediction_and_probabilities(self):
        for cls in utils.CLASS_ORDER:
            rec = utils.sample_record(self.df, self.features, cls)
            label, probs = utils.predict(self.model, rec)
            self.assertIn(label, utils.CLASS_ORDER)
            self.assertAlmostEqual(sum(probs.values()), 1.0, places=6)

    def test_samples_are_predicted_correctly(self):
        for cls in utils.CLASS_ORDER:
            rec = utils.sample_record(self.df, self.features, cls)
            self.assertEqual(utils.predict(self.model, rec)[0], cls)

    def test_importance_sums_to_one(self):
        self.assertAlmostEqual(float(utils.grouped_importance(self.model).sum()), 1.0, places=6)

    def test_whatif_shape(self):
        rec = utils.default_record(self.df, self.features)
        curve = utils.whatif_curve(self.model, rec, "Soil_Moisture", self.df, points=10)
        self.assertEqual(curve.shape, (10, 3))

    def test_evaluation_matches_training_report(self):
        res = utils.evaluate_model(self.model, self.df)
        self.assertEqual(res["n_test"], 2000)
        self.assertGreater(res["accuracy"], 97.0)


if __name__ == "__main__":
    unittest.main()
