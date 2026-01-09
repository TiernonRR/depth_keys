from trial import Trial
from typing import Dict, Any, List

class Experiment:
    """
    The main object representing an entire experiment.
    Contains a collection of trials and the processing configurations.
    """
    def __init__(
        self,
        experiment_name: str,
        post_config: Dict[str, Any] = None,
        pred_config: Dict[str, Any]=None,
        trials: List[Trial] = None
    ):
        # Mandatory identifying information
        self.experiment_name: str = experiment_name

        # Configuration dictionaries
        self.post_config: Dict[str, Any] = post_config  # Post-processing config
        self.pred_config: Dict[str, Any] = pred_config  # Prediction config

        # The core data collection
        self.trials: List[Trial] = trials if trials is not None else []

    def add_trial(self, trial: Trial):
        """Adds a Trial object to the experiment's list of trials."""
        self.trials.append(trial)