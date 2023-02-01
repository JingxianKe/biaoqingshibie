# 导入必要的包
import emotion_config as config, hdf5datasetgenerator, imagetoarraypreprocessor
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str,
                help="path to model checkpoint to load")
args = vars(ap.parse_args())

# 初始化测试数据生成器和图像预处理器
testAug = ImageDataGenerator(rescale=1 / 255.0)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

# 初始化测试数据集生成器
testGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.TEST_HDF5, config.BATCH_SIZE,
                                                    aug=testAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# 加载模型
print("[INFO] loading {}...".format(args["model"]))
model = load_model(args["model"])

# 评估
(loss, acc) = model.evaluate_generator(
testGen.generator(),
    steps=testGen.numImages // config.BATCH_SIZE,
    max_queue_size=config.BATCH_SIZE * 2)
print("[INFO] accuracy: {:.2f}".format(acc * 100))

testGen.close()


