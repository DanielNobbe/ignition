import logging
from typing import Any, Dict, Union
from warnings import warn

import ignite.distributed as idist
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from ignite.engine import DeterministicEngine, Engine, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Metric
import ignite.distributed as idist
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from torch.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer

from ignition.datasets import PairedDataset, IgnitionDataset
from ignition.models import IgnitionModel
from ignition.utils import split_dict_at_index

# for custom vista3d trainer
from monai.utils.enums import CommonKeys as Keys
from monai.engines.utils import IterationEvents
from monai.transforms import reset_ops_id
from monai.data.utils import list_data_collate



logger = logging.getLogger(__name__)


def instantiate_post_transforms(pt_config: DictConfig, **kwargs):
    # TODO: Add typing for output (monai.Transform or torch.Transform?)
    # typically, the config should define a compose
    """
    Instantiate a post-transform based on the configuration.

    Args:
        pt_config (DictConfig): Configuration for the post-transform.
        **kwargs: Additional arguments to pass to the post-transform constructor.

    Returns:
        The instantiated post-transform.
    
    """
    if pt_config._target_ == "monai.transforms.Compose":
        # If it's a compose, instantiate each transform in the list
        transforms = []

        for transform_cfg in pt_config.transforms:
            required_args = transform_cfg.pop("_requires_", None)
            if required_args is not None:
                if not isinstance(required_args, ListConfig | list):
                    required_args = [required_args]
                required_args = {arg: kwargs[arg] for arg in required_args}
                transforms.append(instantiate(transform_cfg, **required_args))
            else:
                transforms.append(instantiate(transform_cfg))
        return instantiate(pt_config, transforms=transforms)
    else:
        # Otherwise, instantiate it directly
        required_args = pt_config.pop("_requires_", None)
        if required_args is not None:
            if not isinstance(required_args, ListConfig | list):
                required_args = [required_args]
            required_args = {arg: kwargs[arg] for arg in required_args}
            return instantiate(pt_config, **required_args)
        else:
            return instantiate(pt_config)


class Vista3DSegmentationFinetuneTrainer(SupervisedTrainer):
    """A trainer to finetune Vista3D for semantic segmentation.

    I.e., this drops all of the point-related stuff (although there may be a benefit
    to including that in the finetuning). The Vista3D model still has a 'prompt'
    with the included classes, so this wrapper needs to insert that in the forward pass.
    
    """
    def __init__(self, *args, label_map: list[tuple[int, int]] | None = None, transpose_forward: bool = False, include_background: bool = False, **kwargs):

        if label_map is None:
            raise ValueError("Must provide a label_map to Vista3DSegmentationFinetuner.")

        self.label_map = label_map
        self.label_set = torch.tensor([l for _, l in label_map])  # only take the label indices in Vista3D space

        self.transpose_forward = transpose_forward
        self.include_background = include_background

        super().__init__(*args, **kwargs)

    def _iteration(self, engine: Engine, batchdata: dict[str, torch.Tensor]):
        """Defines a single iteration update. Based on Vista3D bundle."""
        inputs, labels = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: labels}

        def _compute_pred_loss():
            label_prompt = self.label_set.to(engine.state.device).unsqueeze(-1)
            outputs = engine.inferer(
                inputs=inputs,
                network=engine.network,
                class_vector=label_prompt,  # pass the list of
                labels=labels,
                transpose=self.transpose_forward
            )
            engine.state.output[Keys.PRED] = outputs
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)

            loss, loss_n = torch.tensor(0.0, device=engine.state.device), torch.tensor(0.0, device=engine.state.device)
            for index, (data_label, vista_label) in enumerate(self.label_map):
                if not self.include_background and vista_label == 0:
                    # ahh this actually removes loss signal for bg...
                    continue
                outputs_for_id = outputs[:, [index], ...].float()  # only works if batch size is 1 (?)
                targets_for_id = (labels == data_label).float()
                loss += engine.loss_function(outputs_for_id, targets_for_id)
                loss_n += 1.0
            loss /= max(loss_n, 1.0)
            engine.state.output[Keys.LOSS] = loss
            outputs = None
            torch.cuda.empty_cache()
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        engine.network.train()
        engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)
        if engine.amp and engine.scaler is not None:
            with torch.amp.autocast("cuda", **engine.amp_kwargs):
                _compute_pred_loss()
            engine.scaler.scale(engine.state.output[Keys.LOSS]).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
        else:
            _compute_pred_loss()
            engine.state.output[Keys.LOSS].backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            engine.optimizer.step()
        engine.fire_event(IterationEvents.MODEL_COMPLETED)

        return engine.state.output

    # need to set up a transform that maps labels to the correct indices when loading the data
    # and always need to have a windowinferer, so we can batch things.
    # although perhaps the batching can be done now that we don't have a point head?
        
class Vista3DSegmentationEvaluator(SupervisedEvaluator):
    """An evaluator for validation of Vista3D for semantic segmentation.
    
    I.e., this drops all of the point-related stuff, and just does a forward pass of the 
    segmentation head.

    NOTE: The VISTA3D model does not support batching because of the class head using
    class prompts, which has inputs in the batch dimension. It should be possible 
    to allow batching with a smart einsum there..

    TODO: Instead, we can just instantiate the backbone/encoder and a segmentation head,
    all of the class-encoder stuff from vista3d will be quite redundant when finetuning.
    
    TODO: Freeze parts of the model during training, or allow for varying the learning rate?
    The encoder is trained for CT, so should be finetuned. We also need to finetune the class
    head, since there are some new classes.
    TODO: First finetune the encoder on with a frozen class head, with classes as Vista3D knows them,
    then finetune the class head with a frozen encoder.

    """
    def __init__(self, *args, label_map: list[tuple[int, int]] | None = None, transpose_forward: bool = False, **kwargs):

        if label_map is None:
            raise ValueError("Must provide a label_map to Vista3DSegmentationFinetuner.")

        self.label_map = label_map
        self.label_set = torch.tensor([l for _, l in label_map])  # only take the label indices in Vista3D space
        self.transpose_forward = transpose_forward

        super().__init__(*args, **kwargs)

    def _iteration(self, engine: Engine, batchdata: dict[str, torch.Tensor]):
        """Defines a single iteration evaluation. Based on Vista3D bundle."""
        inputs, labels = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: labels}
        label_prompt = self.label_set.to(engine.state.device).unsqueeze(-1)

        engine.network.eval()
        # put iteration outputs into engine.state
        engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: labels}
        # execute forward computation
        with engine.mode(engine.network):
            if engine.amp:
                with torch.amp.autocast("cuda", **engine.amp_kwargs):
                    engine.state.output[Keys.PRED] = engine.inferer(
                        inputs=inputs,
                        network=engine.network,
                        class_vector=label_prompt,
                        labels=labels,
                        transpose=self.transpose_forward
                    )
            else:
                engine.state.output[Keys.PRED] = engine.inferer(
                    inputs=inputs,
                    network=engine.network,
                    class_vector=label_prompt,
                    labels=labels,
                    transpose=self.transpose_forward
                )


        inputs = reset_ops_id(inputs)  # not sure why this is necessary
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return engine.state.output

        

def setup_trainer(
    config: Any,
    model: IgnitionModel,
    optimizer: Optimizer,
    loss_fn: Module,
    device: Union[str, torch.device],
    dataset: PairedDataset,
    metrics: Dict[str, Metric] = None,
) -> Union[Engine, DeterministicEngine]:

    if config.use_amp:
        scaler = GradScaler(enabled=config.use_amp)
        warn("AMP may not be fully supported in Ignition.")
    else:
        scaler = None

    match config.engine_type:
        case "ignite":
            trainer = create_supervised_trainer(
                model,
                optimizer,
                loss_fn,
                device=device,
                prepare_batch=dataset.get_prepare_batch(),
                deterministic=True,
                non_blocking=True,
                scaler=scaler,
                model_transform=model.get_model_transform(),
                output_transform=model.get_train_output_transform(),
            )
            for name, metric in (metrics or {}).items():
                metric.attach(trainer, name)
        case "monai":
            # TODO: Add option to use slidingwindow inferer during training
            if metrics is not None:
                key_metric, other_metrics = split_dict_at_index(metrics, 1)
            else:
                key_metric, other_metrics = None, None

            trainer = SupervisedTrainer(
                device=device,
                max_epochs=config.max_epochs,
                train_data_loader=idist.auto_dataloader(
                    dataset.get_train_dataset(),
                    collate_fn=list_data_collate,
                    batch_size=dataset.train_batch_size if isinstance(dataset, PairedDataset) else config.train_batch_size
                ),
                amp=config.get("use_amp", False),
                network=model,
                inferer=instantiate(config.inferer) if config.get("train_inferer") else None,
                optimizer=optimizer,
                loss_function=loss_fn,
                non_blocking=True,
                postprocessing=instantiate(config.post_transforms.get("train")),
                key_train_metric=key_metric,
                additional_metrics=other_metrics,
            )
        case "vista3d":
            if metrics is not None:
                key_metric, other_metrics = split_dict_at_index(metrics, 1)
            else:
                key_metric, other_metrics = None, None

            trainer = Vista3DSegmentationFinetuneTrainer(
                label_map=config.get("vista3d_label_map"),
                device=device,
                max_epochs=config.max_epochs,
                train_data_loader=idist.auto_dataloader(
                    dataset.get_train_dataset(),
                    collate_fn=list_data_collate,
                    batch_size=dataset.train_batch_size if isinstance(dataset, PairedDataset) else config.train_batch_size
                ),
                amp=config.get("use_amp", False),
                network=model,
                inferer=instantiate(config.train_inferer) if config.get("train_inferer") else None,
                optimizer=optimizer,
                loss_function=loss_fn,
                non_blocking=True,
                postprocessing=instantiate(config.post_transforms.get("train")),
                key_train_metric=key_metric,
                additional_metrics=other_metrics,
                transpose_forward=bool(config.get("train_inferer")),
                include_background=config.get("vista3d_include_background", False),
            )

    return trainer


def setup_evaluator(
    config: Any,
    model: IgnitionModel,
    metrics: Dict[str, Metric],
    device: Union[str, torch.device],
    dataset: PairedDataset | IgnitionDataset,
    name: str | None = None,
    output_dir: str | None = None,
) -> Engine:

    # TODO: Move metrics declaration into here?
    instantiate_kwargs = {
        "output_dir": output_dir,
    }

    match config.engine_type:
        case "ignite":
            evaluator = create_supervised_evaluator(
                model,
                metrics=metrics,
                device=device,
                non_blocking=True,
                prepare_batch=dataset.get_prepare_batch(),
                output_transform=model.get_eval_output_transform(),
            )
            for name, metric in metrics.items():
                metric.attach(evaluator, name)
        case "monai":

            key_metric, other_metrics = split_dict_at_index(metrics, 1)
            # relies on the fact that dicts are ordered in Python 3.7+

            # in a paired dataset, we get the val dataloader
            # otherwise, we get the single dataloader
            # TODO: Make this configurable?
            val_dataset = dataset.get_val_dataset() if isinstance(dataset, PairedDataset) else dataset.get_dataset()
            dataloader = idist.auto_dataloader(
                val_dataset,
                collate_fn=list_data_collate,
                batch_size=dataset.eval_batch_size if isinstance(dataset, (PairedDataset, IgnitionDataset)) else config.eval_batch_size
            )

            post_transforms = instantiate_post_transforms(
                config.post_transforms.get(name) if name is not None else config.post_transforms,
                **instantiate_kwargs,
            )

            evaluator = SupervisedEvaluator(
                device=device,
                val_data_loader=dataloader,
                amp=config.get("use_amp", False),
                network=model,
                inferer=instantiate(config.inferer) if config.get("inferer") else None,
                postprocessing=post_transforms,
                val_handlers=instantiate(config.engine_handlers) if config.get("engine_handlers", False) else None,
                non_blocking=True,
                key_val_metric=key_metric,
                additional_metrics=other_metrics,
            )
        case "vista3d":
            
            key_metric, other_metrics = split_dict_at_index(metrics, 1)
            # relies on the fact that dicts are ordered in Python 3.7+

            # in a paired dataset, we get the val dataloader
            # otherwise, we get the single dataloader
            val_dataset = dataset.get_val_dataset() if isinstance(dataset, PairedDataset) else dataset.get_dataset()
            
            dataloader = idist.auto_dataloader(
                val_dataset,
                collate_fn=list_data_collate,
                batch_size=dataset.eval_batch_size if isinstance(dataset, (PairedDataset, IgnitionDataset)) else config.eval_batch_size
            )
            post_transforms = instantiate_post_transforms(
                config.post_transforms.get(name) if name is not None else config.post_transforms,
                **instantiate_kwargs,
            )
            evaluator = Vista3DSegmentationEvaluator(
                label_map=config.get("vista3d_label_map"),
                device=device,
                val_data_loader=dataloader,
                amp=config.get("use_amp", False),
                network=model,
                inferer=instantiate(config.inferer) if config.get("inferer") else None,
                postprocessing=post_transforms,
                val_handlers=instantiate(config.engine_handlers) if config.get("engine_handlers", False) else None,
                non_blocking=True,
                key_val_metric=key_metric,
                additional_metrics=other_metrics,
                transpose_forward=bool(config.get("inferer")),
            )

    return evaluator
