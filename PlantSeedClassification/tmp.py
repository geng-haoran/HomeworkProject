import numpy as np
import torch
from utils import *
out = np.load("/data2/haoran/HW/HomeworkProject/PlantSeedClassification/test_result.npy",allow_pickle=True).item()
print(out)
# print(out[0])
# print(torch.tensor(out).shape)
# print(np.argmax(out.reshape(-1,12),axis = 1))
import csv
with open('answer.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, 
                            # delimiter=' ',
                            # quotechar=' ', 
                            # quoting=csv.QUOTE_MINIMAL
                            )
    spamwriter.writerow(["file","species"])
    for key in out.keys():
        spamwriter.writerow([key,LABEL2NAME[out[key]]])
    # spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    # spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])