import matplotlib
matplotlib.use("Agg")

# 导入必要的包
import emotion_config as config, emotionvggnet, epochcheckpoint, hdf5datasetgenerator, \
    imagetoarraypreprocessor
import trainingmonitor
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())

# 构建训练和测试图像生成器进行数据增强，然后初始化图像预处理器
trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                              horizontal_flip=True, rescale=1 / 255.0, fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1 / 255.0)
iap = imagetoarraypreprocessor.ImageToArrayPreprocessor()

# 初始化训练和验证数据集生成器
trainGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE,
                                                     aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = hdf5datasetgenerator.HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE,
                                                   aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# 如果不使用权重文件，初始化网络，编译模型
if args["model"] is None:
    print("[INFO] compiling model...")
    model = emotionvggnet.EmotionVGGNet.build(width=48, height=48, depth=1,
                                              classes=config.NUM_CLASSES)
    opt = Adam(lr=1e-3)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

# 否则，加载权重文件
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # 更新学习率
    print("[INFO] old learning rate: {}".format(
        K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-3)
    print("[INFO] new learning rate: {}".format(
        K.get_value(model.optimizer.lr)))

figPath = os.path.sep.join([config.OUTPUT_PATH,
                            "vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH,
                             "vggnet_emotion.json"])
callbacks = [
    epochcheckpoint.EpochCheckpoint(args["checkpoints"], every=5,
                                    startAt=args["start_epoch"]),
    trainingmonitor.TrainingMonitor(figPath, jsonPath=jsonPath,
                                    startAt=args["start_epoch"])]

# 训练！
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // config.BATCH_SIZE,
    epochs=60,
    max_queue_size=config.BATCH_SIZE * 2,
    callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()


