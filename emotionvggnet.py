#############################################################
#                                                           #
#                            模型的架构                       #
#   ————————————————————————————————————————————————————    #
#   Layer Type   Output Size   Filter Size / Stride         #
#   ————————————————————————————————————————————————————    #
#   INPUT IMAGE  48×48×1                                    #
#   CONV         48×48×32      3×3; K = 32                  #
#   CONV         48×48×32      3×3; K = 32                  #
#   POOL         24×24×32      2×2                          #
#   CONV         24×24×64      3×3; K = 64                  #
#   CONV         24×24×64      3×3; K = 64                  #
#   POOL         12×12×64      2×2                          #
#   CONV         12×12×128     3×3; K = 128                 #
#   CONV         12×12×128     3×3; K = 128                 #
#   POOL         6×6×128       2×2                          #
#   FC           64                                         #
#   FC           64                                         #
#   FC           6                                          #
#   SOFTMAX      6                                          #
#   ———————————————————————————————————————————————————     #
#                                                           #
#############################################################

# 由VGG网络启发，CONV层都是3×3且加倍
# 根据经验，激活函数ELU比ReLU更能提升分类精度，选择使用ELU
# Dropout: 0.25, 0.5


# 导入必要的包
from keras.models import Sequential
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import ELU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # 模型初始化
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # 如果使用channels first，更新输入图像大小和通道维度
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Block #1: first CONV => ELU => CONV => ELU => POOL
        # layer set
        model.add(Conv2D(32, (3, 3), padding="same",
                         kernel_initializer="he_normal", input_shape=inputShape))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV => ELU => CONV => ELU => POOL
        # layer set
        model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #3: third CONV => ELU => CONV => ELU => POOL
        # layer set
        model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",
                         padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first set of FC => ELU layers
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block #6: second set of FC => ELU layers
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Block #7: softmax分类器
        model.add(Dense(classes, kernel_initializer="he_normal"))
        model.add(Activation("softmax"))

        return model



















