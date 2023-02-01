###########################################################
#                      用于导入权重文件                      #
###########################################################



import os
from tensorflow.keras.callbacks import Callback

class EpochCheckpoint(Callback):
    def __init__(self, outputPath, every=5, startAt=0):
        # 调用父类的构造函数
        super(Callback, self).__init__()
        self.outputPath = outputPath  # 模型保存路径
        # 间隔epoch数
        self.every = every
        # 起始epoch（当前epoch）
        self.start_epoch = startAt

    def on_epoch_end(self, epoch, logs={}):
        # 检查是否要向磁盘保存模型
        if (self.start_epoch + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath,
                                  "epoch_{}.hdf5".format(self.start_epoch + 1)])
            self.model.save(p, overwrite=True)
        # 增加内部的epoch计数器
        self.start_epoch += 1