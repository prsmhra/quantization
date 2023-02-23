from mobile_cv.predictor.api import create_predictor
from d2go.utils.demo_predictor import DemoPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from d2go.runner import GeneralizedRCNNRunner
import random,cv2

model = create_predictor('torchscript_int8@tracing')
#print(f'[INFO] {model}')
predictor = DemoPredictor(model)
print(predictor.NUM_CLASSES)
im = cv2.imread('/home/prsmhra/quantization/git/Wrinkles/test/636_right.jpg')
outputs = predictor(im)
print(f'[OUTPUTS] : {outputs["instances"]}')
#v = Visualizer(im[:, :, ::-1],  scale=0.8)
#v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#print('[INFO] saving the predicted image')
#cv2.write('/home/prsmhra/quantization/git/Wrinkles/test/636_right.jpg', v.get_image()[:,:,::-1])  
    #plt.figure(figsize = (14, 10))
    #plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    #plt.show()

