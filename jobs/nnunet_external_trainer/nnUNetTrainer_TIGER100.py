from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_TIGER100(nnUNetTrainer):
    """Protocol-matched nnU-Net trainer: same 100 epoch budget as other baselines."""

    def __init__(self, plans, configuration, fold, dataset_json, device):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
