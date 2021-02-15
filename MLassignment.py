import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import tarfile
from io import BytesIO
import requests

#URL for data 
url = 'https://github.com/beoutbreakprepared/nCoV2019/blob/master/latest_data/latestdata.tar.gz?raw=true'
res = requests.get(url,stream=True)
#get data & import  as a dataframe
with tarfile.open(fileobj=BytesIO(res.raw.read()),mode='r:gz') as tar:
    for m in tar.getmembers():
        file = tar.extractfile(m)
        df = pd.read_csv(file)
        print(df)

#sdf
#sdas
#sdfs