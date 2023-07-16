import torch.nn as nn
class Test(nn.Module):
    def __init__(self, a, b):
        super(Test, self).__init__()
        self.a = a
        self.b = b

    def add(self):
        return self.a + self.b

if __name__ == '__main__':
    test = Test(a=1, b=2)
    print(test.add())
    print('test', test)