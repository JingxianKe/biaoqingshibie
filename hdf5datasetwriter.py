# 导入必要的包
import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images",
    bufSize=1000):
        # 查看输出路径是否存在
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already "
                "exists and cannot be overwritten. Manually delete "
                "the file before continuing.", outputPath)

        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims,
            dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],),
            dtype="int")

        # 存储缓存大小，然后初始化缓存以及数据集中的索引
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, images, labels):
        # 添加images和labels到缓存中
        self.buffer["data"].extend(images)
        self.buffer["labels"].extend(labels)

        # 查看是否需要将缓存刷新到磁盘
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # 将缓存写入磁盘，然后重置缓存
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        # 创建数据集以存储实际的类别标签名， 然后存储类别标签
        dt = h5py.special_dtype(vlen=unicode)
        labelSet = self.db.create_dataset("label_names",
                                          (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # 查看缓存中是否有需要刷新到磁盘的其他元素
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()


