import numpy as np
import matplotlib.pyplot as plt
import os


def readSparseMatrix(filename, total_length):
    print "reading Rao's HiC "
    infile = open(filename).readlines()
    print len(infile)
    HiC = np.zeros((total_length,total_length)).astype(np.int16)
    percentage_finish = 0
    for i in range(0, len(infile)):
        if (i %  (len(infile) / 10)== 0):
            print 'finish ', percentage_finish, '%'
            percentage_finish += 10
        nums = infile[i].split('\t')
        x = int(nums[0])
        y = int(nums[1])
        val = int(float(nums[2]))

        HiC[x][y] = val
        HiC[y][x] = val
    return HiC

def readSquareMatrix(filename, total_length):
    print "reading Rao's HiC "
    infile = open(filename).readlines()
    print('size of matrix is ' + str(len(infile)))
    print('number of the bins based on the length of chromsomes is ' + str(total_length) )
    result = []
    for line in infile:
        tokens = line.split('\t')
        line_int = list(map(int, tokens))
        result.append(line_int)
    result = np.array(result)
    print(result.shape)
    return result


def divide(HiCfile):
    subImage_size = 40
    step = 25
    chrs_length = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566]
    input_resolution = 10000
    result = []
    index = []
    chrN = 21
    matrix_name = HiCfile + '_npy_form_tmp.npy'
    if os.path.exists(matrix_name):
        print 'loading ', matrix_name
        HiCsample = np.load(matrix_name)
    else:
        print matrix_name, 'not exist, creating'
        print HiCfile           
        HiCsample = readSquareMatrix(HiCfile, (chrs_length[chrN-1]/input_resolution + 1))
        #HiCsample = np.loadtxt('/home/zhangyan/private_data/IMR90.nodup.bam.chr'+str(chrN)+'.10000.matrix', dtype=np.int16)
        print HiCsample.shape
        np.save(matrix_name, HiCsample)
    print HiCsample.shape
    path = '/home/zhangyan/HiCPlus_pytorch_production/' 
    if not os.path.exists(path):
        os.makedirs(path)
    total_loci = HiCsample.shape[0]
    for i in range(0, total_loci, step):
        for j in range(0, total_loci, ):
            if (abs(i-j) > 201 or i + subImage_size >= total_loci or j + subImage_size >= total_loci):
                continue
            subImage = HiCsample[i:i+subImage_size, j:j+subImage_size]

            result.append([subImage,])
            tag = 'test'
            index.append((tag, chrN, i, j))
    result = np.array(result)
    print result.shape
    result = result.astype(np.double)
    index = np.array(index)
    return result, index

  
if __name__ == "__main__":
    main()