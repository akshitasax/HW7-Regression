"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression.logreg import LogisticRegressor
from regression.utils import loadDataset

def test_prediction_with_nsclc_dataset():
    """
    Test the prediction of LogisticRegressor on real data from data/nsclc.csv
    and verify it by manually calculating the prediction "by hand" (outside the model).
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Load a subset of the real dataset for testing
    X, y = loadDataset()

    # For testing: use only 5 items to keep it simple
    X = X[:5]
    num_feats = X.shape[1]
    model = LogisticRegressor(num_feats)

    # Manually set weights to ensure reproducibility, including bias
    model.W = np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.2, 0.5])  # 6 features (default from loadDataset funct) + 1 bias

    # Add bias column to X (as in the actual code)
    X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

    # Calculate logits and probabilities 'by hand'
    logits = np.dot(X_bias, model.W)
    expected_probs = sigmoid(logits)
    expected_preds = (expected_probs >= 0.5).astype(int)

    # Use model's make_prediction (use X_bias for consistency with training method)
    preds = model.make_prediction(X_bias)

    # The predictions should exactly match the manual calculation
    assert preds == expected_preds

def test_loss_function():

	num_feats = 6  # Number of features in NSCLC dataset
	model = LogisticRegressor(num_feats)
	
	# Load a small batch of data from nsclc.csv for testing
	X_train, y_train = loadDataset()

	X = X_train[:5]
	y_true = y_train[:5]
	
	# Pick some weights and add bias term to X
	model.W = np.array([0.2, -0.1, 0.05, 0.09, -0.2, 0.11, 0.4])
	X_bias = np.hstack((X, np.ones((X.shape[0], 1))))

	# Compute raw model probabilities
	logits = np.dot(X_bias, model.W)
	y_pred_probs = 1 / (1 + np.exp(-logits))

	# Calculate expected loss using binary cross entropy (as implemented in model)
	eps = 1e-15  # Numerical stability
	y_pred_probs_clipped = np.clip(y_pred_probs, eps, 1 - eps)
	expected_loss = -np.mean(
		y_true * np.log(y_pred_probs_clipped) + (1 - y_true) * np.log(1 - y_pred_probs_clipped)
	)

	model_loss = model.loss_function(y_true, y_pred_probs)
	
	np.testing.assert_allclose(model_loss, expected_loss, rtol=1e-7)

def test_gradient():

	num_feats = 2  # Use two features for a manageable test
	model = LogisticRegressor(num_feats)

	# Select a small batch using two features from the nsclc dataset
	X_train, y_train = loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'AGE_DIAGNOSIS',
		],
	)
	# Use a small batch for test
	X = X_train[:5]
	y_true = y_train[:5]

	# Set weights manually (including bias term)
	model.W = np.array([0.25, -0.3, 0.5])

	# Add bias column
	Xb_bias = np.hstack((X, np.ones((X.shape[0], 1))))

	# Compute y_pred (probabilities)
	logits = np.dot(Xb_bias, model.W)
	y_pred = 1 / (1 + np.exp(-logits))

	# Hand-calculate gradient
	grad_expected = np.dot(Xb_bias.T, (y_pred - y_true)) / y_true.shape[0]

	# Compare to model's gradient calculation
	grad_model = model.calculate_gradient(y_true, Xb_bias)

	np.testing.assert_allclose(grad_model, grad_expected, rtol=1e-7)

def test_training():
	
	num_feats = 2
	model = LogisticRegressor(num_feats)

	# Use NSCLC dataset with two features

	X_train, X_val, y_train, y_val = loadDataset(
		features=[
			'Penicillin V Potassium 500 MG',
			'AGE_DIAGNOSIS',
		],
		split_percent=0.8,
		split_seed=42
	)

	num_feats = X_train.shape[1]
	model = LogisticRegressor(num_feats=num_feats, learning_rate=0.1, max_iter=20, tol=1e-6, batch_size=8)

	# Track initial weights for gradient update check
	initial_W = model.W.copy()

	# Run training
	model.train_model(X_train, y_train, X_val, y_val)

	# Test that training and validation loss goes down at some point
	assert len(model.loss_hist_train) > 3, "Not enough training steps to test loss decrease."

	# Loss should decrease at least once
	assert any(
		earlier > later

		# zipping loss in first epoch and last epoch
		for earlier, later in zip(model.loss_hist_train[:-1], model.loss_hist_train[1:])
	), "Training loss did not decrease during training."

	# Test that weights have been updated
	assert not np.allclose(model.W, initial_W), "Weights did not update during training."


