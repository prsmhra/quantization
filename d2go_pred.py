from mobile_cv.predictor.api import create_predictor
from d2go.utils.demo_predictor import DemoPredictor

import random,cv2

model = create_predictor('torchscript_int8@tracing')
predictor = DemoPredictor(model)
im = cv2.imread('/home/prsmhra/quantization/git/Wrinkles/test/636_right.jpg')
outputs = predictor(im)
print(f'[OUTPUTS] : {outputs["instances"]}')


