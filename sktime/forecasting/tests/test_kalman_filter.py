"""Test Kalman Filter forecaster."""

__author__ = ["oseiskar"]

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_almost_equal
from sktime.datasets import load_airline
from sktime.forecasting.kalman_filter import KalmanFilterForecasterSIMD
from sktime.split import temporal_train_test_split
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.panel import make_transformer_problem


@pytest.mark.skipif(
    not run_test_for_class(KalmanFilterForecasterSIMD),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_airline_prediction():
    """Test Kalman Filter forecaster on airline dataset, compare to mean of last 12 months."""
    y_all = load_airline()

    PERIOD = 12  # assuming yearly periodicity

    # level, trend, seasonal
    state_dim = 1 + 1 + PERIOD

    state_transition = np.zeros((state_dim, state_dim))
    state_transition[:2, :2] = np.array([[1, 1], [0, 1]])
    state_transition[2:-1, 3:] = np.eye(PERIOD - 1)
    state_transition[-1, 2] = 1

    LEVEL_DRIFT = 1e-6
    TREND_DRIFT = 1e-3
    SEASON_DRIFT = 10
    RANDOM_NOISE = 20

    process_noise = np.diag([LEVEL_DRIFT, TREND_DRIFT] + [SEASON_DRIFT] * PERIOD) ** 2
    measurement_function = np.array([[1, 0, 1] + [0] * (PERIOD - 1)])

    predictor = KalmanFilterForecasterSIMD(
        state_dim=state_dim,
        state_transition=state_transition,
        process_noise=process_noise,
        measurement_function=measurement_function,
        measurement_noise=RANDOM_NOISE**2,
        initial_state=np.zeros(state_dim),
        initial_state_covariance=np.eye(state_dim) * (RANDOM_NOISE * 10) ** 2,
    )

    y_train, y_test = temporal_train_test_split(y_all)

    y_predicted = predictor.fit(y_train).predict(fh=y_test.index)

    dummy_level = y_train[-PERIOD:].mean()
    y_predicted_dummy = np.ones_like(y_predicted) * dummy_level

    dummy_error_sum = np.abs(y_predicted_dummy - y_test).sum()
    predicted_error_sum = np.abs(y_predicted - y_test).sum()

    assert predicted_error_sum < dummy_error_sum * 10


@pytest.mark.skipif(
    not run_test_for_class(KalmanFilterForecasterSIMD),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_panel_prediction():
    """Check that panel smoothing gives the same results as the series version"""
    X = make_transformer_problem(n_instances=5, n_columns=1, n_timepoints=15)

    X_train, X_test = temporal_train_test_split(X)
    fh = list(range(X_test.shape[2]))

    predictor = KalmanFilterForecasterSIMD(
        state_transition=np.array([[1, 1], [0, 1]]),
        process_noise=np.diag([1e-6, 0.01]) ** 2,
        measurement_function=np.array([[1, 0]]),
        measurement_noise=50.0**2,
        initial_state=np.array([0, 0]),
        initial_state_covariance=np.eye(2) * 100**2,
    )

    panel_X_pred = predictor.fit(X_train).predict(fh=fh)

    nd_results = []
    for i in range(X.shape[0]):
        as_nd = X_train[i, ...].transpose()
        nd_out = predictor.fit(as_nd).predict(fh=fh)
        nd_results.append(nd_out.transpose())

    nd_X_pred = np.stack(nd_results, axis=0)
    assert_array_almost_equal(panel_X_pred, nd_X_pred)

    series_results = []
    for i in range(X.shape[0]):
        as_series = pd.Series(X_train[i, 0, :])
        series_out = predictor.fit(as_series).predict(fh=fh)
        series_results.append(series_out.to_numpy()[np.newaxis, ...])

    series_X_pred = np.stack(series_results, axis=0)
    assert_array_almost_equal(panel_X_pred, series_X_pred)

    dataframe_results = []
    for i in range(X.shape[0]):
        as_dataframe = pd.DataFrame(X_train[i, ...].transpose())
        dataframe_out = predictor.fit(as_dataframe).predict(fh=fh)
        dataframe_results.append(dataframe_out.to_numpy().transpose())

    dataframe_X_pred = np.stack(dataframe_results, axis=0)
    assert_array_almost_equal(panel_X_pred, dataframe_X_pred)
