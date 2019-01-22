from cython.view cimport array as cvarray
from libc.stdint cimport uint64_t
cimport cython
from libc.stdlib cimport rand
#from syconn.extraction.cs_extraction_steps import detect_cs
#from syconn.handler.compression import save_to_h5py
from libcpp.map cimport map
from cython.operator import dereference, postincrement
import sys
import numpy as np
import timeit


def printMemView(uint64_t[:, :, :] myArray):
        for i in range(myArray.shape[0]):
                for j in range(myArray.shape[1]):
                        for k in range(myArray.shape[2]):
                                print myArray[i,j,k],
                        print("")
                print("")


def kernel_Maria(int[:,:,:] chunk, uint64_t center_id):
        cdef map[int, int] unique_ids

        for i in range(chunk.shape[0]):
                for j in range(chunk.shape[1]):
                        for k in range(chunk.shape[2]):
                                unique_ids[chunk[i][j][k]] = unique_ids[chunk[i][j][k]] + 1
        unique_ids[0] = 0
        unique_ids[center_id] = 0
        cdef int theBiggest  = 0
        cdef uint64_t key = 0

        cdef map[int,int].iterator it = unique_ids.begin()
        while it != unique_ids.end():
                if dereference(it).second > theBiggest:
                        theBiggest =  dereference(it).second
                        key = dereference(it).first
                postincrement(it)

        if theBiggest > 0:
                if center_id > key:
                        return (key << 32 ) + center_id
                else:
                        return (center_id << 32) + key

        else:
                return key

def kernel(chunk, center_id):
        unique_ids, counts = np.unique(chunk, return_counts=True)

        counts[unique_ids == 0] = -1
        counts[unique_ids == center_id] = -1

        if np.max(counts) > 0:
                partner_id = unique_ids[np.argmax(counts)]

                if center_id > partner_id:
                        return (partner_id << 32) + center_id
                else:
                        return (center_id << 32) + partner_id
        else:
                return 0



def process_block_Maria(int[:, :, :] edges, int[:, :, :] arr, stencil1=(7,7,3)):
        cdef int stencil[3]
        stencil[:] = [stencil1[0], stencil1[1], stencil1[2]]
        assert (stencil[0]%2 + stencil[1]%2 + stencil[2]%2 ) == 3
        cdef uint64_t[:, :, :] out = cvarray(shape = (arr.shape[0], arr.shape[1], arr.shape[2]), itemsize = sizeof(uint64_t), format = 'Q')
        out [:, :, :] = 0
        cdef int offset[3]
        offset[:] = [stencil[0]/2, stencil[1]/2, stencil[2]/2] ### check what ype do you need
        cdef int center_id
        cdef int[:, :, :] chunk = cvarray(shape=(2*offset[0]+2, 2*offset[2]+2, 2*offset[2]+2), itemsize=sizeof(int), format='i')

        for x in range(offset[0], arr.shape[0] - offset[0]):
                for y in range(offset[1], arr.shape[1] - offset[1]):
                        for z in range(offset[2], arr.shape[2] - offset[2]):
                                if edges[x, y, z] == 0:
                                        continue

                                center_id = arr[x, y, z] #be sure that it's 32 or 64 bit intiger 
                                chunk = arr[x - offset[0]: x + offset[0] + 1, y - offset[1]: y + offset[1], z - offset[2]: z + offset[2]]
                                out[x, y, z] = kernel_Maria(chunk, center_id)

        return out

def process_block(edges, arr, stencil=(7, 7, 3)):
        stencil = np.array(stencil, dtype=np.int)
        assert np.sum(stencil % 2) == 3

        out = np.zeros_like(arr, dtype=np.uint64)
        offset = stencil // 2 #ask
        for x in range(offset[0], arr.shape[0] - offset[0]):
                for y in range(offset[1], arr.shape[1] - offset[1]):
                        for z in range(offset[2], arr.shape[2] - offset[2]):
                                if edges[x, y, z] == 0:
                                        continue

                                center_id = arr[x, y, z]
                                chunk = arr[x - offset[0]: x + offset[0] + 1, y - offset[1]: y + offset[1], z - offset[2]: z + offset[2]]
                                out[x, y, z] = kernel(chunk, center_id)
        return out



def process_block_nonzero_Maria(int[:, :, :] edges, int[:, :, :] arr, stencil1=(7,7,3)):
        cdef int stencil[3]
        stencil[:] = [stencil1[0], stencil1[1], stencil1[2]]
        assert (stencil[0]%2 + stencil[1]%2 + stencil[2]%2 ) == 3

        cdef uint64_t[:, :, :] out = cvarray(shape = (1 + arr.shape[0] - stencil[0], arr.shape[1] - stencil[1] + 1, arr.shape[2] - stencil[2] + 1), itemsize = sizeof(uint64_t), format = 'Q')
        out[:, :, :] = 0
        cdef int center_id
        cdef int offset[3]
        offset [:] = [stencil[0]/2, stencil[1]/2, stencil[2]/2]
        cdef int[:, :, :] chunk = cvarray(shape=(stencil[0]+1, stencil[1]+1, stencil[2]+1), itemsize=sizeof(int), format='i')

        for x in range(0, edges.shape[0]-2*offset[0]):
                for y in range(0, edges.shape[1]-2*offset[1]):
                        for z in range(0, edges.shape[2]-2*offset[2]):
                                if edges[x+offset[0], y+offset[1], z+offset[2]] == 0:
                                        continue
                                center_id = arr[x + offset[0], y + offset[1], z + offset[2]]
                                chunk = arr[x: x + stencil[0], y: y + stencil[1], z: z + stencil[2]]
                                out[x, y, z] = kernel_Maria(chunk, center_id)

        return out

def process_block_nonzero(edges, arr, stencil=(7, 7, 3)):
        stencil = np.array(stencil, dtype=np.int)
        assert np.sum(stencil % 2) == 3

        arr_shape = np.array(arr.shape)
        out = np.zeros(arr_shape - stencil + 1, dtype=np.uint64)
        offset = stencil // 2 # int division!
        nze = np.nonzero(edges[offset[0]: -offset[0], offset[1]: -offset[1], offset[2]: -offset[2]])
        for x, y, z in zip(nze[0], nze[1], nze[2]):
                center_id = arr[x + offset[0], y + offset[1], z + offset[2]]
                chunk = arr[x: x + stencil[0], y: y + stencil[1], z: z + stencil[2]]
                out[x, y, z] = kernel(chunk, center_id)
        return out


def create_toy_data(int size):
        cdef int[:, :, :] matrix = cvarray(shape=(size, size, size), itemsize=sizeof(int), format='i')
        for i in range(size):
                for j in range(size):
                        for k in range(size):
                                matrix[i, j, k] = rand() % 10
        return matrix

def printNp(myArray):
        for i in range(myArray.shape[0]):
                for j in range(myArray.shape[1]):
                        for k in range(myArray.shape[2]):
                                print myArray[i,j,k],
                        print("")
                print("")



def wrapper(func, *args, **kwargs):
        def wrapped():
                return func(*args, **kwargs)
        return wrapped
cdef int[:, :, :] matrix1
matrix1 = create_toy_data(8)
matrix2 = create_toy_data(8)

print ("kernel_Maria")
wrapped = wrapper(kernel_Maria, matrix1, 3)
print (timeit.timeit(wrapped, number=1000))

matrixNp1 = np.array(matrix1)
matrixNp2 = np.array(matrix2)

print ("kernel")
wrapped = wrapper(kernel,matrixNp1, 3)
print (timeit.timeit(wrapped, number=1000))

print("process_block_Maria")
wrapped = wrapper(process_block_Maria, matrix1, matrix2, (7,7,3))
print (timeit.timeit(wrapped, number=1000))

print("process_block")
wrapped = wrapper(process_block, matrixNp1, matrixNp2, (7,7,3))
print (timeit.timeit(wrapped, number=1000))

print("process_block_nonzero_Maria")
wrapped = wrapper(process_block_nonzero_Maria, matrix1, matrix2, (7,7,3))
print (timeit.timeit(wrapped, number=1000))

print("process_block_nonzero")
wrapped = wrapper(process_block_nonzero, matrixNp1, matrixNp2, (7,7,3))
print (timeit.timeit(wrapped, number=1000))


#print("Maria")
#t1=time.clock()
#kernel_Maria(matrix1,3)
#t2=time.clock()
#print(t2-t1)
#matrixNp1 = np.array(matrix1)
#matrixNp2 = np.array(matrix2)
#print("Numpy")
#printNp(process_block_nonzero(matrixNp1, matrixNp2,(7,7,3)))
#matrixRes = np.subtract(np.array(process_block_nonzero_Maria(matrix1, matrix2,(7,7,3))), process_block_nonzero(matrixNp1, matrixNp2,(7,7,3)))
#printNp(matrixRes)


