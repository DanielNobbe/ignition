from monai.handlers import CheckpointSaver
import logging
import os
from shutil import rmtree
# class PeftCheckpoint:
#     """
#     Simple checkpoint that can save 
#     """
from monai.utils import IgniteInfo, is_scalar, min_version, optional_import
Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")

from ignite.engine import Engine

class PeftCheckpointSaver(CheckpointSaver):
    """
    Ignite handler to save PEFT model adapter checkpoints. Behaviour similar to checkpointsaver,
    in that it saves multiple checkpoints.

    The main change that we make is that we give a custom save function, that uses peft_model.save_pretrained() instead of saving the state dict.
    """

    def _save_checkpoint(self, model, save_name):
        # we assume that the model is a peft model, so it has the save_pretrained function
        model.save_pretrained(os.path.join(self.save_dir, save_name))


    def __init__(
        self,
        save_dir: str,
        save_dict: dict,
        name: str | None = None,
        file_prefix: str = "",
        save_final: bool = False,
        # final_filename: str | None = None,
        save_key_metric: bool = False,
        key_metric_name: str | None = None,
        key_metric_n_saved: int = 1,
        # key_metric_filename: str | None = None,
        # key_metric_save_state: bool = False,
        # key_metric_greater_or_equal: bool = False,
        key_metric_negative_sign: bool = False,
        epoch_level: bool = True,
        save_interval: int = 0,
        n_saved: int | None = None,
        **kwargs
    ) -> None:
        # we don't call the parent's init, we only reuse some of its functions
        if save_dir is None:
            raise AssertionError("must provide directory to save the checkpoints.")
        self.save_dir = save_dir
        if not (save_dict is not None and len(save_dict) > 0):
            raise AssertionError("must provide source objects to save.")
        self.save_dict = save_dict
        self.logger = logging.getLogger(name)
        self.epoch_level = epoch_level
        self.save_final = save_final
        
        self.save_interval = save_interval
        self.n_saved = n_saved
        self.iteration_checkpoints = []  # ordering should align with iterations

        self.file_prefix = file_prefix

        self.key_metric_checkpoints = {}
        self.save_key_metric = save_key_metric
        self.key_metric_name = key_metric_name
        self.key_metric_n_saved = key_metric_n_saved
        self.key_metric_negative_sign = key_metric_negative_sign

        self._name = None

    def completed(self, engine):
        if not self.save_final:
            return
        value = engine.state.iteration
        score_name = "final_iteration"
        name = "checkpoint"

        save_name = f"{self.file_prefix}_{name}_{score_name}_{value}.pth"
        self._save_checkpoint(self.save_dict["model"], save_name)

    def metrics_completed(self, engine):
        if not self.save_key_metric:
            return
        
        if self.key_metric_name is None:
            key_metric_name = engine.state.key_metric_name
        else:
            key_metric_name = self.key_metric_name
        value = engine.state.metrics[key_metric_name]
        score_name = key_metric_name
        name = "metric_checkpoint"

        save_name = f"{self.file_prefix}_{name}_{score_name}_{value:.5f}"
        self._save_checkpoint(self.save_dict["model"], save_name)

        if save_name in self.key_metric_checkpoints.values():
            # if we already have a checkpoint with this value, we save the last, so we remove it
            matching_keys = [k for k, v in self.key_metric_checkpoints.items() if v == save_name]
            # we delete all, only save the last one, in case there are multiple with the same save name
            for k in matching_keys:
                del self.key_metric_checkpoints[k]

        self.key_metric_checkpoints[value] = save_name
        if len(self.key_metric_checkpoints) > self.key_metric_n_saved:
            # remove the worst
            if self.key_metric_negative_sign:
                worst_metric = max(self.key_metric_checkpoints)
                worst_path = self.key_metric_checkpoints[worst_metric]
            else:
                worst_metric = min(self.key_metric_checkpoints)
                worst_path = self.key_metric_checkpoints[worst_metric]
            rmtree(os.path.join(self.save_dir, worst_path))

            del self.key_metric_checkpoints[worst_metric]

    def interval_completed(self, engine):
        if not self.save_interval:
            return
        if self.epoch_level:
            value = engine.state.epoch
            score_name = "epoch"
        else:
            value = engine.state.iteration
            score_name = "iteration"

        if value % self.save_interval != 0:
            return

        name = "checkpoint"

        save_name = f"{self.file_prefix}_{name}_{score_name}_{value}"
        self._save_checkpoint(self.save_dict["model"], save_name)

        if self.n_saved is not None:
            self.iteration_checkpoints.append(save_name)
            if len(self.iteration_checkpoints) > self.n_saved:
                # remove the oldest
                oldest_save_name = self.iteration_checkpoints.pop(0)
                rmtree(os.path.join(self.save_dir, oldest_save_name))

    def exception_raised(self, engine, exception):
        # we can also save a checkpoint when an exception is raised, to be able to resume from it after fixing the issue
        self.logger.info(f"Exception raised: {exception}. Saving checkpoint.")
        self.completed(engine)

    def attach(self, engine: Engine) -> None:
        """
        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.
        """
        if self._name is None:
            self.logger = engine.logger
        engine.add_event_handler(Events.COMPLETED, self.completed)
        engine.add_event_handler(Events.EXCEPTION_RAISED, self.exception_raised)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.metrics_completed)
        if self.epoch_level:
            engine.add_event_handler(Events.EPOCH_COMPLETED(every=self.save_interval), self.interval_completed)
        else:
            engine.add_event_handler(Events.ITERATION_COMPLETED(every=self.save_interval), self.interval_completed)