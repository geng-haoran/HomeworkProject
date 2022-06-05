import numpy as np
import torch
out = np.load("/data2/haoran/HW/HomeworkProject/PlantSeedClassification/test_result.npy",allow_pickle=True)
print(out)
print(torch.tensor(out).shape)
print(np.argmax(out.reshape(-1,12),axis = 1))