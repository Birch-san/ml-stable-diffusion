from torch import FloatTensor, nn
import torch

from diffusers.models.unet_2d_condition import UNet2DConditionModel
class UndictedDiffusersUnet(nn.Module):
    model: UNet2DConditionModel
    def __init__(self, model: UNet2DConditionModel):
        super().__init__()
        self.model = model
  
    @property
    def device(self) -> torch.device:
        return self.model.device
  
    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def forward(
        self,
        sample: FloatTensor,
        timestep: FloatTensor,
        encoder_hidden_states: FloatTensor,
        return_dict: bool = False,
    ): 
        return self.model(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=return_dict,
        )