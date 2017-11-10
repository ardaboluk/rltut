
import matplotlib.pyplot as plt
import numpy as np
import os.path

for filename in os.listdir("."):
    if(filename.startswith("policy_ep")):
        policy = np.genfromtxt(filename, delimiter=",")
        plt.figure()
        plt.imshow(np.rot90((policy-5).reshape(21,21).T), cmap="hot", interpolation="nearest")
        plt.title(filename)
        plt.xlabel("Cars at location 2")
        plt.ylabel("Cars at location 1")
        
policy = np.genfromtxt("policyFinal.csv", delimiter=",")
plt.figure()
plt.imshow(np.rot90((policy-5).reshape(21,21).T), cmap="hot", interpolation="nearest")
plt.title("policyFinal.csv")
plt.xlabel("Cars at location 2")
plt.ylabel("Cars at location 1")
                
plt.show()
