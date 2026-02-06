"""
Custom StatAvg (Statistical Averaging) Strategy Implementation

StatAvg uses statistical properties of model updates to improve aggregation.
It computes statistical metrics (mean, variance) across client updates.

Reference: Statistical Federated Averaging approach
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


class StatAvg(FedAvg):
    """
    StatAvg Strategy - Statistical Federated Averaging
    
    Parameters:
    -----------
    stat_weight : float
        Weight for statistical regularization (default: 0.1)
        Controls influence of variance-based adjustments
    use_variance : bool
        Whether to use variance in aggregation (default: True)
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
        stat_weight: float = 0.1,
        use_variance: bool = True,
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
        self.stat_weight = stat_weight
        self.use_variance = use_variance
        self.historical_stats = None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with StatAvg parameters."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add StatAvg-specific parameters
        config["stat_weight"] = self.stat_weight
        config["use_variance"] = self.use_variance
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
        """Aggregate fit results using statistical averaging."""
        
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
        
        # Compute statistical aggregation
        aggregated_ndarrays = self._statistical_aggregate(weights_results, server_round)
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            print("No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def _statistical_aggregate(
        self, results: List[Tuple[NDArrays, int]], server_round: int
    ) -> NDArrays:
        """Perform statistical aggregation with variance consideration."""
        
        # Calculate total number of examples
        num_examples_total = sum([num_examples for _, num_examples in results])
        
        # Initialize aggregated weights and statistics
        num_layers = len(results[0][0])
        aggregated_weights = [np.zeros_like(layer) for layer in results[0][0]]
        layer_means = [[] for _ in range(num_layers)]
        layer_variances = [[] for _ in range(num_layers)]
        
        # Collect statistics for each layer across clients
        for layer_idx in range(num_layers):
            client_layer_values = []
            for client_weights, num_examples in results:
                client_layer_values.append(client_weights[layer_idx])
                layer_means[layer_idx].append(np.mean(client_weights[layer_idx]))
            
            # Compute variance if enabled
            if self.use_variance and len(client_layer_values) > 1:
                stacked = np.array([np.mean(v) for v in client_layer_values])
                layer_variances[layer_idx] = np.var(stacked)
        
        # Weighted aggregation with statistical adjustment
        for layer_idx in range(num_layers):
            variance_weight = 1.0
            
            if self.use_variance and layer_variances[layer_idx]:
                # Lower variance = higher confidence = higher weight
                variance_weight = 1.0 / (1.0 + self.stat_weight * layer_variances[layer_idx])
            
            for client_weights, num_examples in results:
                weight = (num_examples / num_examples_total) * variance_weight
                aggregated_weights[layer_idx] += client_weights[layer_idx] * weight
            
            # Normalize if variance weighting was used
            if self.use_variance:
                total_variance_weight = sum([
                    (num_examples / num_examples_total) * 
                    (1.0 / (1.0 + self.stat_weight * layer_variances[layer_idx]))
                    for _, num_examples in results
                ])
                if total_variance_weight > 0:
                    aggregated_weights[layer_idx] /= total_variance_weight
        
        # Store historical statistics
        self.historical_stats = {
            'means': layer_means,
            'variances': layer_variances,
            'round': server_round
        }
        
        return aggregated_weights
