###########################################################
#                  本文件的目的是存储配置变量                 #
###########################################################


# 导入必要的包
from os import path

# 路径设置
BASE_PATH = "fer2013"
INPUT_PATH = path.sep.join([BASE_PATH, "fer2013.csv"])


# 设置类别数
# NUM_CLASSES = 7
# 如果忽略"反感"类别，设置为6
NUM_CLASSES = 6

# 设置.HDF5文件的输出路径
TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_PATH, "hdf5/test.hdf5"])

# 设置批大小
BATCH_SIZE = 128

# 设置输出的存储路径
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])


