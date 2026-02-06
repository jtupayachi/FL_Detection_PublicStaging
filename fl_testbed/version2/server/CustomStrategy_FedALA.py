"""
Custom FedALA (Federated Adaptive Local Aggregation) Strategy Implementation

FedALA adaptively aggregates local models using learnable weights based on model importance.
It learns to weight different layers differently during aggregation.

Reference: Zhang et al. "Federated Learning with Adaptive Local Aggregation" (2022)
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
import numpy as np


class FedALA(FedAvg):
    """
    FedALA Strategy - Federated Adaptive Local Aggregation
    
    Parameters:
    -----------
    eta : float
        Learning rate for adaptive weights (default: 1.0)
        Controls how fast the adaptive weights are updated
    threshold : float
        Threshold for weight adaptation (default: 0.5)
        Determines the sensitivity of weight updates
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1.0,
        threshold: float = 0.5,
        rand_percent: int = 80,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.eta = eta
        self.threshold = threshold
        self.rand_percent = rand_percent
        self.layer_weights = None  # Adaptive weights for each layer

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with FedALA parameters."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add FedALA-specific parameters
        config["eta"] = self.eta
        config["threshold"] = self.threshold
        config["rand_percent"] = self.rand_percent
        config["server_round"] = server_round
        
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using adaptive local aggregation."""
        
        if not results:
            return None, {}
        
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Initialize layer weights if first round
        if self.layer_weights is None and weights_results:
            num_layers = len(weights_results[0][0])
            self.layer_weights = np.ones(num_layers)
        
        # Adaptive aggregation with learned weights
        aggregated_ndarrays = self._adaptive_aggregate(weights_results, server_round)
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            print("No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def _adaptive_aggregate(
        self, results: List[Tuple[NDArrays, int]], server_round: int
    ) -> NDArrays:
        """Perform adaptive weighted aggregation."""
        
        # Calculate total number of examples
        num_examples_total = sum([num_examples for _, num_examples in results])
        
        # Initialize aggregated weights
        aggregated_weights = [
            np.zeros_like(layer) for layer in results[0][0]
        ]
        
        # Adaptive aggregation with layer-specific weights
        for layer_idx in range(len(aggregated_weights)):
            layer_weight = self.layer_weights[layer_idx] if self.layer_weights is not None else 1.0
            
            for client_weights, num_examples in results:
                # Weight by number of examples and adaptive layer weight
                weight = (num_examples / num_examples_total) * layer_weight
                aggregated_weights[layer_idx] += client_weights[layer_idx] * weight
        
        # Update adaptive weights based on variance (simplified approach)
        if server_round > 1:
            self._update_layer_weights(results)
        
        return aggregated_weights

    def _update_layer_weights(self, results: List[Tuple[NDArrays, int]]) -> None:
        """Update adaptive layer weights based on layer importance."""
        
        # Calculate variance for each layer across clients
        for layer_idx in range(len(self.layer_weights)):
            layer_values = [client_weights[layer_idx] for client_weights, _ in results]
            layer_variance = np.var([np.mean(lv) for lv in layer_values])
            
            # Update weight based on variance (higher variance = more important)
            # This is a simplified heuristic
            importance = 1.0 + self.eta * np.log(1.0 + layer_variance)
            self.layer_weights[layer_idx] = importance
        
        # Normalize weights
        self.layer_weights = self.layer_weights / np.sum(self.layer_weights) * len(self.layer_weights)
