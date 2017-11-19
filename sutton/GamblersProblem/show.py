
import matplotlib.pyplot as plt
import numpy as np
import os.path
        
values = np.genfromtxt("valuesFinal.csv", delimiter=",")
policy = np.genfromtxt("policyFinal.csv", delimiter=",")
plt.figure()
plt.plot(np.arange(values.shape[0]), values)
plt.title("valuesFinal.csv")
plt.xlabel("Capital")
plt.ylabel("Value estimates")

plt.savefig("valuesFinal.png")
plt.show()
        
plt.figure()
plt.bar(np.arange(policy.shape[0]), policy)
plt.title("policyFinal.csv")
plt.xlabel("Capital")
plt.ylabel("Stake")

plt.savefig("policyFinal.png")
plt.show()
