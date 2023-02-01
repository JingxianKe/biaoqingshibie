#############################################################
#         本文件的目的是把数据格式从.csv格式转换成.hdf5格式        #
#############################################################



# 导入必要的包
import emotion_config as config
import hdf5datasetwriter
import numpy as np

print("[INFO] loading input data...")
f = open(config.INPUT_PATH)
f.__next__()
(trainImages, trainLabels) = ([], [])
(valImages, valLabels) = ([], [])
(testImages, testLabels) = ([], [])

# 逐行循环输入文件
for row in f:
    # 提取标签、图像及图像用途
    (label, image, usage) = row.strip().split(",")
    label = int(label)

    # 如果忽略"反感"类别，类别标签共有6个
    if config.NUM_CLASSES == 6:
        # 将类别"生气"和"反感"合并
        if label == 1:
            label = 0

        if label > 0:
            label -= 1

    # 调整图像大小
    image = np.array(image.split(" "), dtype="uint8")
    image = image.reshape((48, 48))

    if usage == "Training":
        trainImages.append(image)
        trainLabels.append(label)

    elif usage == "PrivateTest":
        valImages.append(image)
        valLabels.append(label)

    else:
        testImages.append(image)
        testLabels.append(label)

datasets = [
    (trainImages, trainLabels, config.TRAIN_HDF5),
    (valImages, valLabels, config.VAL_HDF5),
    (testImages, testLabels, config.TEST_HDF5)]

for (images, labels, outputPath) in datasets:
    print("[INFO] building {}...".format(outputPath))
    writer = hdf5datasetwriter.HDF5DatasetWriter((len(images), 48, 48), outputPath)

    for (image, label) in zip(images, labels):
        writer.add([image], [label])

    writer.close()

f.close()






