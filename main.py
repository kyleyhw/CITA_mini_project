from question_2 import Template, Likelihood
from data_loader import Data


for i in range(10):
    filepath = 'simple-cw-master/'
    filename = 'cw_' + str(i) + '.csv'

    data = Data(filepath + filename)