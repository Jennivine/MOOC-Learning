import matplotlib.pyplot as plt
import numpy as np
#random data
x = np.random.normal(size = 1000)
plt.figure()
plt.subplot(121)
#histogram
plt.hist(x,histtype = "step")
plt.subplot(122)
#line graph
plt.plot([0,4,6,10,16])
plt.savefig("test2.pdf")
