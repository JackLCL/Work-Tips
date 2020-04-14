from milvus import *
import os
import numpy as np
milvus = Milvus()
milvus.connect('192.168.1.183',19540)


for i in range(1000):
    collection_name = 'image_search'
    file = 'fake.npy'
    vectors = np.load(file)
    status, ids = milvus.insert(collection_name=collection_name, records=vectors.tolist())
    print(file + " insert {} round".format(i))
