import cv2,os,random
from matplotlib import pyplot as plt
import gc 
gc.collect()

from d2go.model_zoo import model_zoo
from d2go.utils.demo_predictor import DemoPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from d2go.runner import GeneralizedRCNNRunner

# get model
# model = model_zoo.get('qut_mask_rcnn_fbnetv3g_fpn.yaml', trained=True)

train_dir = "/home/prsmhra/quantization/git/Wrinkles/test/" #path for train dir
test_dir = "/home/prsmhra/quantization/git/Wrinkles/test/" #path for test dir

train_json = "/home/prsmhra/quantization/git/Wrinkles/test.json" #json path for training data
test_json = "/home/prsmhra/quantization/git/Wrinkles/test.json" #json path for test data

model_path = "/home/prsmhra/quantization/git/Wrinkles_models/" #path to save the model
object_name ="wrinklesCFA_d2go_22022023/" #model name

register_coco_instances("train", {}, train_json, train_dir)
register_coco_instances("test", {}, test_json, test_dir)

train_metadata = MetadataCatalog.get("train")
train_dataset_dicts = DatasetCatalog.get("train")

test_metadata = MetadataCatalog.get("test")
test_dataset_dicts = DatasetCatalog.get("test")
print(f'[INFO] number of classes : {len(train_metadata.thing_classes)} and classes are ({train_metadata.thing_classes})')


def prepare_for_launch():
    runner = GeneralizedRCNNRunner()
    cfg = runner.get_default_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("qat_mask_rcnn_fbnetv3a_C4.yaml"))
    cfg.MODEL_EMA.ENABLED = False
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("test",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.DEVICE="cpu"
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("mask_rcnn_fbnetv3g_fpn.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10   # 600 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train_metadata.thing_classes)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg, runner

cfg, runner = prepare_for_launch()
model = runner.build_model(cfg)
model = model.to("cpu")
runner.do_train(cfg, model, resume=False)


import copy
from detectron2.data import build_detection_test_loader
from d2go.export.exporter import convert_and_export_predictor
#from d2go.export.d2_meta_arch import patch_d2_meta_arch

import logging,time
print('*'*20)
print('\n[INFO] exporting the model in torchscript and saving the model\n')
print('*'*20)
# disable all the warnings
previous_level = logging.root.manager.disable
# logging.disable(logging.INFO)

#patch_d2_meta_arch()

pytorch_model = runner.build_model(cfg, eval_only=True)
pytorch_model.cpu()

datasets = cfg.DATASETS.TRAIN[0]
data_loader = runner.build_detection_test_loader(cfg, datasets)

predictor_path = convert_and_export_predictor(
    copy.deepcopy(cfg),
    copy.deepcopy(pytorch_model),
    "torchscript_int8@tracing",
    './',
    data_loader
)
print(f'[INFO] predictor path :{predictor_path}')
print('[INFO] Toechscript model is saved')
# recover the logging level
#right before quitting
#driver.quit()
time.sleep(1)
# logging.disable(previous_level)

from mobile_cv.predictor.api import create_predictor
from d2go.utils.demo_predictor import DemoPredictor

print('[INFO] Testing the model on test dataset')
model = create_predictor(predictor_path)
predictor = DemoPredictor(model)

# dataset_dicts = DatasetCatalog.get('test')
for d in random.sample(test_dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=test_metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    print('[INFO] saving the predicted output')
    cv2.write(d['file_name'], v.get_image()[:,:,::-1])  
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()





