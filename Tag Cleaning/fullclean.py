import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np
import glob
import os


def fullclean():
    for file in glob.glob('*.csv'):
        df = pd.read_csv(file, encoding='latin-1')
        df = df.drop('accuracy', 1)
        df = df.drop('views', 1)
        df = df.drop('licenseID', 1)
        df = df.drop('DateUploaded', 1)
        tags = df.photoTags.tolist()
        row=[]
        count=0

        stop_words = stopwords.words('english')
        stop_words += ['geotagged','landscape','mm','mmf', 'geostate','geocountries',
                       'flickrhq', 'admin', 'mmf','geolat','geolon','msh','exz','img',
                       'image','photo','picture','hc','day','photoidimg','pentaxk','sigma','exdf']

        for i in range (0,len(tags)):
            newrow=[]
            if tags[i] in [None,'','nan',' ']:
                df.photoTags[i]= np.nan
            else:
                row=str(tags[i]).split()
                for j in range (0,len(row)):
                    text= row[j].lower()
                    if re.match('[a-z]+:[a-z]+=[a-z0-9_]+', text):
                        text=''
                    else:
                        text = re.sub('\W+', '', text)
                        text = re.sub('[0-9]+', '', text )
                    if (text != '') and (text != 'nan') and (text not in stop_words) and (len(text)>1) and (text not in newrow):
                        newrow.append(text)
                    new=' '.join(newrow)
                    df.photoTags[i] = new
                if new in [None,'','nan',' ']:
                    df.photoTags[i]= np.nan
                print (new)
                count+=1
                if count%1000==0:
                    print (count)

        df.dropna().to_csv('fullclean.csv', encoding='utf-8', index= False)
        
folder_list = ['train_1',
             'train_2',
             'train_3',
             'train_4',
             'train_5',
             'train_6',
             'train_7',
             'train_8',
             'train_9',
             'train_10',
             'test']
         
def main():
#    for folder in folder_list:
#        os.chdir("C:\\Users\\irene\\Downloads\\Ειρήνη\\University of Southamton\\Classes\\Data Mining\\group assignment\\"+folder)
        os.chdir("C:\\Users\\irene\\Downloads\\Ειρήνη\\University of Southamton\\Classes\\Data Mining\\group assignment\\testing")

        #cleanandmachinetags()
        clean()
    
   
if __name__ == "__main__":
 main()
    