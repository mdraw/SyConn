# -*- coding: utf-8 -*-
# SyConn - Synaptic connectivity inference toolkit
#
# Copyright (c) 2016 - now
# Max-Planck-Institute of Neurobiology, Munich, Germany
# Authors: Philipp Schubert, Joergen Kornfeld
import sys
sys.path.append('/u/mariakaw/dev/SyConn/')
from knossos_utils import knossosdataset
from knossos_utils import chunky
from syconn.config import global_params
from syconn.extraction import object_extraction_wrapper as oew
from syconn.extraction import cs_processing_steps
from syconn.reps.super_segmentation import SuperSegmentationDataset
from syconn.reps.segmentation import SegmentationDataset
from syconn.proc.sd_proc import dataset_analysis, extract_synapse_type
from syconn.extraction import cs_extraction_steps as ces
from syconn.extraction import cs_processing_steps as cps
from syconn.handler.logger import initialize_logging
import numpy as np
from syconn.extraction import cs_extraction_stepsM
from syconn.extraction import cs_extraction_steps


if __name__ == "__main__":
    log = initialize_logging('synapse_analysis', global_params.wd + '/logs/', overwrite=False)

    kd_seg_path = global_params.kd_seg_path
    kd = knossosdataset.KnossosDataset()  # Sets initial values of object
    # # Initializes the dataset by parsing the knossos.conf in path + "mag1"
    kd.initialize_from_knossos_path(kd_seg_path)
    data = kd.from_overlaycubes_to_matrix([512, 512, 512], [200, 200, 200], datatype=np.uint64).astype(np.uint32)

    ## Python implementation
    contacts_Python = syconn.extraction.cs_extraction_steps.detect_cs(data)
    print (contacts_Python)

    ## My implementation
    contacts_Marysia = syconn.extraction.cs_extraction_stepsM.detect_cs(data)
    print (contacts_Marysia)



def printNp(myArray):
    for i in range(myArray.shape[0]):
        for j in range(myArray.shape[1]):
            for k in range(myArray.shape[2]):
                print
                myArray[i, j, k],
            print("")
        print("")



def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped

# cdef
# int[:, :, :]
# matrix1
# matrix1 = create_toy_data(8)
# matrix2 = create_toy_data(8)
#
# print("kernel_Maria")
# wrapped = wrapper(kernel_Maria, matrix1, 3)
# print(timeit.timeit(wrapped, number=1000))
#
# matrixNp1 = np.array(matrix1)
# matrixNp2 = np.array(matrix2)
#
# print("kernel")
# wrapped = wrapper(kernel, matrixNp1, 3)
# print(timeit.timeit(wrapped, number=1000))
#
# print("process_block_Maria")
# wrapped = wrapper(process_block_Maria, matrix1, matrix2, (7, 7, 3))
# print(timeit.timeit(wrapped, number=1000))
#
# print("process_block")
# wrapped = wrapper(process_block, matrixNp1, matrixNp2, (7, 7, 3))
# print(timeit.timeit(wrapped, number=1000))
#
# print("process_block_nonzero_Maria")
# wrapped = wrapper(process_block_nonzero_Maria, matrix1, matrix2, (7, 7, 3))
# print(timeit.timeit(wrapped, number=1000))
#
# print("process_block_nonzero")
# wrapped = wrapper(process_block_nonzero, matrixNp1, matrixNp2, (7, 7, 3))
# print(timeit.timeit(wrapped, number=1000))
#
# # print("Maria")
# # t1=time.clock()
# # kernel_Maria(matrix1,3)
# # t2=time.clock()
# # print(t2-t1)
# # matrixNp1 = np.array(matrix1)
# # matrixNp2 = np.array(matrix2)
# # print("Numpy")
# # printNp(process_block_nonzero(matrixNp1, matrixNp2,(7,7,3)))
# # matrixRes = np.subtract(np.array(process_block_nonzero_Maria(matrix1, matrix2,(7,7,3))), process_block_nonzero(matrixNp1, matrixNp2,(7,7,3)))
# # printNp(matrixRes)


















    # TODO: change path of CS chunkdataset
    # Initital contact site extraction
    # contact _id data extraction
    #cd_dir = global_params.wd + "/chunkdatasets/"
    #cs_cset = chunky.load_dataset(cd_dir, update_paths=True)
