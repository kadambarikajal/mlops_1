import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris
import pickle
import os
import mlflow
import mlflow.sklearn


class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup any necessary objects or variables for tests."""
        print("Setting up the tests...")
        cls.data = load_iris()
        cls.X, cls.y = cls.data.data, cls.data.target
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)
        cls.model = RandomForestClassifier(random_state=42)

    def test_data_loading(self):
        """Test if the data is loading correctly."""
        self.assertEqual(self.X.shape, (150, 4))  # 150 samples, 4 features
        self.assertEqual(self.y.shape, (150,))    # 150 target labels

    def test_model_training(self):
        """Test if the model can be trained without errors."""
        self.model.fit(self.X_train, self.y_train)
        self.assertGreater(self.model.score(self.X_test, self.y_test), 0.8)

    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning with GridSearchCV."""
        param_grid = {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        self.assertIsNotNone(grid_search.best_estimator_)

    def test_model_save_and_load(self):
        """Test saving and loading the model using pickle."""
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

        # Ensure the model file was saved
        self.assertTrue(os.path.exists('model.pkl'))

        with open('model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        # Check if the loaded model is the same
        self.assertEqual(type(self.model), type(loaded_model))

    def test_model_mlflow(self):
        """Test MLflow tracking of the model."""
        with mlflow.start_run():
            mlflow.sklearn.log_model(self.model, "model")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", None)

            # Assert that the model is logged
            self.assertTrue(os.path.exists("mlruns"))

    @classmethod
    def tearDownClass(cls):
        """Cleanup any resources if necessary."""
        print("Cleaning up after tests...")
        if os.path.exists('model.pkl'):
            os.remove('model.pkl')


if __name__ == "__main__":
    unittest.main()
