import torch
import torch.nn.quantized as nnq
from torch.ao.quantization import QConfigMapping
import torch.ao.quantization.quantize_fx

# original fp32 module to replace
class CustomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)

# custom observed module, provided by user
class ObservedCustomModule(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x)

    @classmethod
    def from_float(cls, float_module):
        assert hasattr(float_module, 'qconfig')
        observed = cls(float_module.linear)
        observed.qconfig = float_module.qconfig
        return observed

# custom quantized module, provided by user
class StaticQuantCustomModule(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, x):
        return self.linear(x)

    @classmethod
    def from_observed(cls, observed_module):
        assert hasattr(observed_module, 'qconfig')
        assert hasattr(observed_module, 'activation_post_process')
        observed_module.linear.activation_post_process = \
            observed_module.activation_post_process
        quantized = cls(nnq.Linear.from_float(observed_module.linear))
        return quantized

#
# example API call (Eager mode quantization)
#

#m = torch.nn.Sequential(CustomModule()).eval()
#prepare_custom_config_dict = {
#    "float_to_observed_custom_module_class": {
#        CustomModule: ObservedCustomModule
#    }
#}
#convert_custom_config_dict = {
#    "observed_to_quantized_custom_module_class": {
#        ObservedCustomModule: StaticQuantCustomModule
#    }
#}
#m.qconfig = torch.ao.quantization.default_qconfig
#mp = torch.ao.quantization.prepare(
#    m, prepare_custom_config_dict=prepare_custom_config_dict)
# calibration (not shown)
#mq = torch.ao.quantization.convert(
#    mp, convert_custom_config_dict=convert_custom_config_dict)
#
# example API call (FX graph mode quantization)
#
m = torch.nn.Sequential(CustomModule()).eval()
qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_qconfig)
PrepareCustomConfig = {
    "float_to_observed_custom_module_class": {
      "static": {
          CustomModule: ObservedCustomModule,
        }
    }
}
ConvertCustomConfig = {
    "observed_to_quantized_custom_module_class": {
        "static": {
            ObservedCustomModule: StaticQuantCustomModule,
        }
    }
}
mp = torch.ao.quantization.quantize_fx.prepare_fx(
    m, qconfig_mapping, torch.randn(3,3), prepare_custom_config=PrepareCustomConfig)
# calibration (not shown)
mq = torch.ao.quantization.quantize_fx.convert_fx(
    mp, convert_custom_config=ConvertCustomConfig)
