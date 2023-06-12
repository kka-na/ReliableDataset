

def setter():
    global ap
    ap = 1

global ap
ap = 0

class Test():
    def __init__(self):
        self.ap = 0

    def do(self):
        self.ap = ap
        print(self.ap) 

if __name__ == "__main__":
    t = Test()
    t.do()
    setter()
    t.do()