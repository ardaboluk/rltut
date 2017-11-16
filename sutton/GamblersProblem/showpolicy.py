
import matplotlib.pyplot as plt
import numpy as np
import os.path
        
policy = np.genfromtxt("policyFinal.csv", delimiter=",")
plt.figure()
plt.bar(np.arange(policy.shape[0]), policy)
plt.title("policyFinal.csv")
plt.xlabel("Capital")
plt.ylabel("Stake")
                
plt.show()
