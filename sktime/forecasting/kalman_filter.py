# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Vectorized Kalman Filter forecaster implemented using the simdkalman library."""

__author__ = ["oseiskar"]

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster

from sktime.transformations.series.kalman_filter._simdkalman import _SIMDKalmanAdapter


class KalmanFilterForecasterSIMD(BaseForecaster):
    """
    Vectorized Kalman Filter forecaster using the simdkalman package.

    The Kalman Filter is an unsupervised algorithm, consisting of
    several mathematical equations which are used to create
    an estimate of the state of a process. The Kalman Filter is typically
    used for denoising data,  or inferring the hidden state of data.

    This class is the adapter for the ``simdkalman`` package into ``sktime``.
    ``KalmanForecasterSIMD`` Kalman Filter based forecasting.

    The ``simdkalman`` package is ideal for Panels where similar
    Kalman Filters are applied in to multiple time series. The package
    applies multi-dimensional matrix operations, which can be an order
    of magnitude faster than the non-vectorized implementations.

    As long as the shapes of the parameters match reasonably according
    to the rules of matrix multiplication, this class is flexible in their
    exact nature accepting

     * scalars: ``process_noise = 0.1``
     * (2d) numpy matrices: ``process_noise = numpy.eye(2)``
     * 2d arrays: ``observation_model = [[1,2]]``

    See https://simdkalman.readthedocs.io/ for the mathematical definitions
    of the parameters.

    Parameters
    ----------
    state_transition : np.ndarray
        of shape (state_dim, state_dim).
        State transition matrix, also referred to as ``F``, is a matrix
        which describes the way the underlying series moves
        through successive time periods. Called ``A`` in ``simdkalman``.
    process_noise : np.ndarray
        of shape (state_dim, state_dim).
        Process noise matrix, also referred to as ``Q``,
        the uncertainty of the dynamic model.
    measurement_noise : np.ndarray
        of shape (measurement_dim, measurement_dim).
        Measurement noise matrix, also referred to as ``R``,
        represents the uncertainty of the measurements.
    measurement_function : np.ndarray
        of shape (measurement_dim, state_dim).
        Measurement equation matrix, also referred to as ``H``, adjusts
        dimensions of measurements to match dimensions of state.
    initial_state : np.ndarray, optional (default=None)
        of shape (state_dim,).
        Initial estimated system state, also referred to as ``X0``.
    initial_state_covariance : np.ndarray, optional (default=None)
        of shape (state_dim, state_dim).
        Initial estimated system state covariance, also referred to as ``P0``.
    hidden : bool, optional (default=False).
        This parameter affects ``transform``. If True, then ``transform`` will be
        inferring hidden state. If False, returns smoothed/filtered observations
        (see also ``denoising``), which always has the same dimensions as the
        input data, independent of the hidden state dimension.
    state_dim : int, optional (default=None).
        Ignored parameter for interface compatibility with other Kalman Filters
        in sktime.

    See Also
    --------
    sktime.transformations.series.KalmanFilterTransformerSIMD :
        ``simdkalman``-based Kalman Filter transformer for Series data.

    Notes
    -----
    ``simdkalman`` documentation :
        https://simdkalman.readthedocs.io/

    Examples
    --------
        Advanced example:

    >>> import numpy as np
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.kalman_filter import KalmanFilterForecasterSIMD
    >>> from sktime.split import temporal_train_test_split
    >>>
    >>> PERIOD = 12 # assuming yearly periodicity
    >>>
    >>> # level, trend, seasonal
    >>> state_dim = 1 + 1 + PERIOD
    >>>
    >>> state_transition = np.zeros((state_dim, state_dim))
    >>> state_transition[:2, :2] = np.array([[1, 1], [0, 1]])
    >>> state_transition[2:-1, 3:] = np.eye(PERIOD-1)
    >>> state_transition[-1, 2] = 1
    >>>
    >>> LEVEL_DRIFT = 1e-6
    >>> TREND_DRIFT = 1e-3
    >>> SEASON_DRIFT = 10
    >>> RANDOM_NOISE = 20
    >>>
    >>> process_noise = np.diag([LEVEL_DRIFT, TREND_DRIFT] + [SEASON_DRIFT]*PERIOD)**2
    >>> measurement_function = np.array([[1, 0, 1] + [0]*(PERIOD-1)])
    >>>
    >>> predictor = KalmanFilterForecasterSIMD(
    ...     state_dim = state_dim,
    ...     state_transition = state_transition,
    ...     process_noise = process_noise,
    ...     measurement_function = measurement_function,
    ...     measurement_noise = RANDOM_NOISE**2,
    ...     initial_state = np.zeros(state_dim),
    ...     initial_state_covariance = np.eye(state_dim) * (RANDOM_NOISE * 10)**2,
    ... )
    >>>
    >>> y_all = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y_all)
    >>> y_predicted = predictor.fit(y_train).predict(fh=y_test.index)
    >>>
    """

    _tags = {
        "y_inner_mtype": ["pd.Series", "pd.DataFrame", "np.ndarray", "numpy3D"],
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        # "capability:pred_var": True,
        "authors": ["oseiskar"],
        "python_dependencies": ["simdkalman"],
        "maintainers": ["oseiskar"],
    }

    def __init__(
        self,
        state_transition,
        process_noise,
        measurement_noise,
        measurement_function,
        initial_state=None,
        initial_state_covariance=None,
        hidden=False,
        state_dim=None,
    ):
        self.state_transition = state_transition
        self.process_noise = process_noise
        self.measurement_function = measurement_function
        self.measurement_noise = measurement_noise
        self.initial_state = initial_state
        self.initial_state_covariance = initial_state_covariance
        self.hidden = hidden
        self.state_dim = state_dim  # ignored

        super().__init__()

        # check that the parameters are OK (optional)
        self._build_adapter()

    def _build_adapter(self):
        return _SIMDKalmanAdapter(
            state_transition=self.state_transition,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            measurement_function=self.measurement_function,
            initial_state=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            hidden=self.hidden,
            denoising=False,
        )

    def _fit(self, y, X=None, fh=None):
        if len(y.shape) == 3:
            y_np = y.transpose(0, 2, 1)
        elif isinstance(y, pd.DataFrame):
            y_np = y.to_numpy()[np.newaxis, ...]
        elif len(y.shape) == 2:
            y_np = y[np.newaxis, ...]
        else:
            y_np = y.to_numpy()[np.newaxis, ..., np.newaxis]

        # NOTE: fails in check_estimator tests due to wrong dimensions in test data
        if self.hidden:
            assert y_np.shape[-1] == np.atleast_2d(self.measurement_function).shape[1]
        else:
            assert y_np.shape[-1] == np.atleast_2d(self.measurement_function).shape[0]

        # TODO: EM algorithm

        self._m, self._P, self._y_pred = self._build_adapter().fit_predict(y_np)

    def _predict(self, fh, X=None):
        index = fh.to_absolute(self.cutoff).to_pandas()

        pred_all_shape = (self._y_pred.shape[0], len(index)) + self._y_pred.shape[1:]
        if len(pred_all_shape) == 2:
            # undo simdkalman auto flattening
            pred_all_shape = pred_all_shape + (1,)

        predicted_all = np.empty(pred_all_shape)

        m = self._m
        P = self._P
        y_pred_cur = self._y_pred
        kalman_filter = self._build_adapter().build_kalman_filter()

        for i in range(len(index)):
            predicted_all[:, i, ...] = y_pred_cur.reshape(
                predicted_all[:, 0, ...].shape
            )

            m, P = kalman_filter.predict_next(m, P)
            y_pred_cur, _ = kalman_filter.predict_observation(m, P)

        if len(self._y.shape) == 3:
            return predicted_all.transpose(0, 2, 1)
        elif hasattr(self._y, "name"):
            return pd.Series(predicted_all[0, :, 0], index=index, name=self._y.name)
        elif isinstance(self._y, pd.Series):
            return pd.Series(predicted_all[0, :, 0], index=index)
        elif hasattr(self._y, "columns"):
            return pd.DataFrame(
                predicted_all[0, ...], index=index, columns=self._y.columns
            )
        elif isinstance(self._y, pd.DataFrame):
            return pd.DataFrame(predicted_all[0, ...], index=index)

        # nd.array
        return predicted_all[0, ...]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        params = [
            {
                # two hidden states, one observed variable
                "state_transition": np.array([[1, 1], [0, 1]]),
                "process_noise": np.diag([0.1, 0.01]),
                "measurement_function": np.array([[1, 0]]),
                "measurement_noise": 1.0,
            },
            {
                # 1d case
                "state_transition": 1,
                "process_noise": 0.1,
                "measurement_function": 1,
                "measurement_noise": 1.0,
                "initial_state": 10,
                "initial_state_covariance": 1,
            },
            # TODO: true multi-variate case
        ]
        return params
