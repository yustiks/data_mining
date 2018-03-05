import pandas as pd


def make_txt():
    t1 = open('training_latlng.txt','r')
    t2 = open('training_missingBlocks_latlng.txt','r')
    t_1 = t1.read()
    t_2 = t2.read()
    tr = open('train.txt','a')
    tr.write(t_1)
    tr.write('\n' + t_2)
    tr.close()
    t1.close()
    t2.close()


def read_meta(i):
    
    global t
    
    if i == 0:
        df = pd.read_csv('metadata_1.csv')
    elif i == 1:
        df = pd.read_csv('metadata_2.csv')
    elif i == 2:
        df = pd.read_csv('metadata_3.csv')
    elif i == 3:
        df = pd.read_csv('metadata_4.csv')
    elif i == 4:
        df = pd.read_csv('metadata_5.csv')
    elif i == 5:
        df = pd.read_csv('metadata_6.csv')
    elif i == 6:
        df = pd.read_csv('metadata_7.csv') 
    elif i == 7:
        df = pd.read_csv('metadata_8.csv')
    elif i == 8:
        df = pd.read_csv('metadata_9.csv')
    elif i == 9:
        df = pd.read_csv('metadata_missingBlocks.csv')

    data = pd.DataFrame()
    data = pd.merge(t, df)
    data.to_csv('train_{}.csv'.format(i+1), mode='a', index = False)

       
if __name__ == "__main__":   
    make_txt()
    t = pd.read_csv('train.txt', delim_whitespace = True)
    for i in range(10):
        read_meta(i)