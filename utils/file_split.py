import random
import os
import shutil

A_path= '/content/drive/MyDrive/bootcamp/cp01/project/data'
B_path = '/content/drive/MyDrive/bootcamp/cp01/project/data/thumbnails128x128'

temp_file_path =[]
def search(dirname):
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                search(full_filename)
            else:
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.png': 
                    temp_file_path.append(full_filename)
    except PermissionError:
        pass
    return temp_file_path

temp1 = os.listdir(A_path)
temp2 = search(B_path)


def data_split(temp1,temp2,test_len):
    dataA_dir = temp1
    dataB_dir = temp2

    len_dirA = int(len(dataA_dir))
    len_dirB = int(len(dataB_dir))

    train_len = min(len_dirA,len_dirB)
    
    test_A_dir = random.sample(dataA_dir,test_len)
    test_B_dir = random.sample(dataB_dir,test_len)


    for del_value in test_A_dir:
        dataA_dir.remove(del_value) 
    train_A_dir = random.sample(dataA_dir,train_len-test_len)
    
    for del_value in test_B_dir:
        dataB_dir.remove(del_value)
    train_B_dir = random.sample(dataB_dir,train_len-test_len)



    # 디렉토리 로 파일 옮겨주면서 분류
    # A는 fullpath
    base_path = './data/'
    trainA_path = base_path+"trainA/"
    for i in train_A_dir:
        temp = i.split('/')
        shutil.move(i,trainA_path+temp[-1])
    # B는 /data 안에 담겨있다.
    trainB_path = base_path+"trainB/"
    for i in train_B_dir:
        shutil.move(str("/data/"+i),trainB_path+i)

    testA_path = base_path+"testA/"
    for i in test_A_dir:
        temp.clear()
        temp = i.spilit('/')
        shutil.move(i,testA_path+temp[-1])

    testB_path = base_path+"testB/"
    for i in test_B_dir:
        shutil.move(str("/data/"+i),testB_path+i) 


