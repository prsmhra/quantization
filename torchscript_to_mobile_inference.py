from typing import List, Dict
import torch
from mobile_cv.predictor.api import create_predictor
from d2go.utils.demo_predictor import DemoPredictor

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        coco_idx_list = [1]

        self.coco_idx = torch.tensor(coco_idx_list)

    def forward(self, inputs: List[torch.Tensor]):
        x = inputs[0].unsqueeze(0) * 255
        scale = 320.0 / min(x.shape[-2], x.shape[-1])
        x = torch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=True, recompute_scale_factor=True)
        out = self.model(x[0])
        res : Dict[str, torch.Tensor] = {}
        res["boxes"] = out[0] / scale
        res["labels"] = torch.index_select(self.coco_idx, 0, out[1])
        res["scores"] = out[2]
        return inputs, [res]


model = create_predictor('torchscript_int8@tracing')
predictor = DemoPredictor(model)
orig_model = torch.jit.load(os.path.join(predictor_path, "model.jit"))
wrapped_model = Wrapper(orig_model)

scripted_model = torch.jit.script(wrapped_model)
scripted_model.save("d2go.pt")