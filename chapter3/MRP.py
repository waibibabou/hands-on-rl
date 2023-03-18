import numpy as np
np.random.seed(0)
rewards=[-1,-2,-2,10,1,0]
gamma=0.5
def compute_return(chain,gamma):
    ans=0
    for i in chain[::-1]:
        ans=ans*gamma+rewards[i-1]
    return ans
chain=[1,2,3,6]
G=compute_return(chain,gamma)
print(f"{chain}得到的回报为{G}")
