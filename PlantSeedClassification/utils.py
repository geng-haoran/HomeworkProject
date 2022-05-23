import torch
LABEL_NUM = 12
NAME2LABEL = {
    'Black-grass': 0, 
    'Charlock': 1, 
    'Cleavers': 2, 
    'Common Chickweed': 3, 
    'Common wheat': 4, 
    'Fat Hen': 5, 
    'Loose Silky-bent': 6, 
    'Maize': 7, 
    'Scentless Mayweed': 8, 
    'Shepherds Purse': 9, 
    'Small-flowered Cranesbill': 10, 
    'Sugar beet': 11
}
LABELS = ['Black-grass', 'Charlock', 'Cleavers', 
'Common Chickweed', 'Common wheat', 'Fat Hen', 
'Loose Silky-bent', 'Maize', 'Scentless Mayweed', 
'Shepherds Purse', 'Small-flowered Cranesbill', 
'Sugar beet']

LABEL2NAME = {0: 'Black-grass', 1: 'Charlock', 
2: 'Cleavers', 3: 'Common Chickweed', 4: 'Common wheat', 
5: 'Fat Hen', 6: 'Loose Silky-bent', 7: 'Maize', 
8: 'Scentless Mayweed', 9: 'Shepherds Purse', 
10: 'Small-flowered Cranesbill', 11: 'Sugar beet'}

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res