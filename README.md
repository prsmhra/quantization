# D2go model

## Training

wrinkles_d2go_traing.py train the quantized mask rcnn model and save the model in torchscript format. this dummy model to test if it able to save model in torchscript format.

## Testing 

d2go_pred.py test the model and gives the model outputs in console.

## Configs
contains the configs files for models in model zoo of d2go library and also added the qat_mask_rcnn_*.yaml file to quantized the mask rcnn model.

## Mobile Inference
torchscript_to_mobile_inference.py transform the trochscript model which able to run on mobile devices.
