"""
Custom DASHA (Distributed Asynchronous Stochastic Hamiltonian Averaging) Strategy Implementation

DASHA uses momentum-based variance reduction and asynchronous updates for improved convergence.
It combines gradient compression with variance-reduced stochastic optimization.

Reference: Tyurin et al. "DASHA: Distributed Nonconvex Optimization with Communication Compression and Optimal Oracle Complexity" (2023)
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


class DASHA(FedAvg):
    """
    DASHA Strategy - Distributed Asynchronous Stochastic Hamiltonian Averaging
    
    Parameters:
    -----------
    alpha : float
        Step size parameter (default: 0.1)
        Controls the update magnitude
    gamma : float
        Compression parameter (default: 0.5)
        Controls gradient compression level (0 to 1)
    momentum : float
        Momentum coefficient (default: 0.9)
        Used for variance reduction
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
        alpha: float = 0.1,
        gamma: float = 0.5,
        momentum: float = 0.9,
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
        self.alpha = alpha
        self.gamma = gamma
        self.momentum = momentum
        self.velocity = None  # Momentum buffer
        self.h_history = None  # Hamiltonian gradient history

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training with DASHA parameters."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add DASHA-specific parameters
        config["alpha"] = self.alpha
        config["gamma"] = self.gamma
        config["momentum"] = self.momentum
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
        """Aggregate fit results using DASHA algorithm."""
        
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
        
        # Apply DASHA aggregation
        aggregated_ndarrays = self._dasha_aggregate(weights_results, server_round)
        
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            print("No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def _dasha_aggregate(
        self, results: List[Tuple[NDArrays, int]], server_round: int
    ) -> NDArrays:
        """Perform DASHA-based aggregation with momentum and compression."""
        
        # Calculate total number of examples
        num_examples_total = sum([num_examples for _, num_examples in results])
        
        # Initialize structures
        num_layers = len(results[0][0])
        
        # Standard weighted average
        aggregated_weights = [np.zeros_like(layer) for layer in results[0][0]]
        
        for layer_idx in range(num_layers):
            for client_weights, num_examples in results:
                weight = num_examples / num_examples_total
                aggregated_weights[layer_idx] += client_weights[layer_idx] * weight
        
        # Initialize velocity if first round
        if self.velocity is None:
            self.velocity = [np.zeros_like(layer) for layer in aggregated_weights]
        
        # Apply DASHA momentum and compression
        compressed_weights = []
        for layer_idx in range(num_layers):
            # Compute gradient estimate (difference from previous)
            if self.h_history is not None:
                gradient = aggregated_weights[layer_idx] - self.h_history[layer_idx]
            else:
                gradient = aggregated_weights[layer_idx]
            
            # Apply compression (top-k or random sparsification)
            compressed_gradient = self._compress_gradient(gradient, self.gamma)
            
            # Update velocity with momentum
            self.velocity[layer_idx] = (
                self.momentum * self.velocity[layer_idx] + 
                (1 - self.momentum) * compressed_gradient
            )
            
            # Update weights with step size
            updated_layer = aggregated_weights[layer_idx] + self.alpha * self.velocity[layer_idx]
            compressed_weights.append(updated_layer)
        
        # Store history for next round
        self.h_history = [layer.copy() for layer in aggregated_weights]
        
        return compressed_weights

    def _compress_gradient(self, gradient: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gradient compression with compression ratio gamma."""
        
        if gamma >= 1.0:
            # No compression
            return gradient
        
        # Flatten gradient
        flat_grad = gradient.flatten()
        
        # Compute top-k based on absolute values
        k = max(1, int(len(flat_grad) * gamma))
        
        # Get indices of top-k absolute values
        top_k_indices = np.argpartition(np.abs(flat_grad), -k)[-k:]
        
        # Create compressed gradient (sparse)
        compressed_flat = np.zeros_like(flat_grad)
        compressed_flat[top_k_indices] = flat_grad[top_k_indices]
        
        # Reshape back to original shape
        compressed = compressed_flat.reshape(gradient.shape)
        
        return compressed
