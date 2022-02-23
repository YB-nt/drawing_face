import random
import os
import shutil

    

def search(dirname,temp_file_path):
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


def data_split(pathA,pathB,test_len):
    dataA_dir=[]
    temp_A = os.listdir(pathA)
    for i in temp_A:
        dataA_dir.append(i))
    
    temp_file_path=[]
    dataB_dir = search(pathB,temp_file_path)

    len_dirA = len(dataA_dir)
    len_dirB = len(dataB_dir)

    train_len = int(min(len_dirA,len_dirB))

    test_A_dir = random.choice(dataA_dir,test_len)
    test_B_dir = random.choice(dataB_dir,test_len)

    train_A_temp = dataA_dir.remove(test_A_dir)
    train_A_dir = random.choice(train_A_temp,train_len)
    
    train_B_temp = dataB_dir.remove(test_B_dir)
    train_B_dir = random.choice(train_B_temp,train_len)



    # 디렉토리 로 파일 옮겨주면서 분류
    save_path = "/data/"
    # A는 fullpath
    trainA_path = save_path+"train_A/"
    for i in train_A_dir:
        temp = i.split('/')
        shutil.move(i,trainA_path+temp[-1])
    # B는 /data 안에 담겨있다.
    trainB_path = save_path+"train_B/"
    for i in train_B_dir:
        shutil.move(str("/data/"+i),trainB_path+i)

    testA_path = save_path+"test_A/"
    for i in test_A_dir:
        temp.clear()
        temp = i.spilit('/')
        shutil.move(i,testA_path+temp[-1])

    testB_path = save_path+"test_B/"
    for i in test_B_dir:
       shutil.move(str("/data/"+i),testB_path+i) 


