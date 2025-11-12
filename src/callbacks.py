
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np



class PerformanceThresholdCallback(BaseCallback):
    def __init__(
        self,
        metric_weights=None,
        thresholds=None,
        patience=5000,
        fail_patience=15000,
        warmup_steps=200,
        success_streak=50,
        adaptive=True,
        verbose=1,
    ):
        """
        Custom callback for early stopping based on performance thresholds and plateau detection.

        :param metric_weights: Dictionary of metric names and their weights for combined metric calculation (default: {"error": 1.0})
        :param thresholds: Dictionary of threshold values for each metric (default: {"error": 0.5})
        :param patience: Maximum number of training steps before stopping (default: 5000)
        :param fail_patience: Number of steps without improvement before stopping (default: 15000)
        :param warmup_steps: Number of initial steps to skip before applying stopping logic (default: 200)
        :param success_streak: Number of consecutive steps below threshold required for success (default: 50)
        :param adaptive: Whether to use adaptive threshold calculation (default: True)
        :param verbose: Verbosity level (0: silent, 1: print messages) (default: 1)
        """
        super().__init__(verbose)
        self.metric_weights = metric_weights or {"error": 1.0}
        self.thresholds = thresholds or {"error": 0.5}
        self.patience = patience
        self.fail_patience = fail_patience
        self.warmup_steps = warmup_steps
        self.success_streak = success_streak
        self.adaptive = adaptive

        # tracking
        self.best_metric = np.inf
        self.last_improvement_step = 0
        self.success_counter = 0
        self.total_steps = 0

    def _on_step(self) -> bool:
        self.total_steps += 1

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        info = infos[-1]
        if self.total_steps <= self.warmup_steps:
            # just warmup, don't decide yet
            return True

        # combined metric
        metrics = []
        for key, w in self.metric_weights.items():
            if key in info:
                val = float(info[key])
                metrics.append(w * val)
        combined_metric = np.sum(metrics)

        # update best
        if combined_metric < self.best_metric:
            self.best_metric = combined_metric
            self.last_improvement_step = self.total_steps
            self.success_counter += 1
        else:
            self.success_counter = 0

        # adaptive threshold combination
        if self.adaptive:
            adaptive_threshold = np.sum(
                [self.metric_weights[k] * self.thresholds[k] for k in self.metric_weights if k in self.thresholds]
            )
        else:
            adaptive_threshold = list(self.thresholds.values())[0]

        # Stop based on consecutive success
        if combined_metric < adaptive_threshold:
            self.success_counter += 1
            if self.success_counter >= self.success_streak:
                if self.verbose:
                    print(f"[Callback] ✅ Threshold reached consistently ({self.success_streak} steps) at step {self.total_steps}: {combined_metric:.4f}")
                return False
        else:
            self.success_counter = 0  # reset counter if above threshold

        # Stop based on lack of improvement
        if self.total_steps - self.last_improvement_step > self.fail_patience:
            if self.verbose:
                print(f"[Callback] ⚠️ No improvement for {self.fail_patience} steps, stopping.")
            return False

        # Stop based on reaching total patience
        if self.total_steps >= self.patience:
            if self.verbose:
                print(f"[Callback] ⏹️ Max steps ({self.patience}) reached, stopping.")
            return False

        return True
    
    
    
class LoggerCallback(BaseCallback):
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:     
            if "energy_norm" in info:
                self.logger.record("metrics/energy_norm", info["energy_norm"])
                
            if "error" in info:
                self.logger.record("metrics/error", info["error"])
        
        return True
                