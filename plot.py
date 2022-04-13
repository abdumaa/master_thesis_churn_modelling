import matplotlib.pyplot as plt
import numpy as np

# 100 linearly spaced numbers
x = np.linspace(0,1,100)

# the function, which is y = sin(x) here
y = -np.log(x)

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.title.set_text('Simple Weighted Log-Loss for $\mathregular{{C}_{i}}$=1')
ax.set(xlabel='$\mathregular{\hat{C}_{i}}$', ylabel='Log-Loss')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# plot the functions
plt.plot(x,y, 'b', label='$\mathregular{w_{1}}$=1')
plt.plot(x,5*y, 'g', label='$\mathregular{w_{1}}$=5')
plt.plot(x,10*y, 'r', label='$\mathregular{w_{1}}$=10')

plt.legend(loc='upper right')

# show the plot
#plt.show()
plt.savefig('tex/images/weighted_loss.png', dpi=200, bbox_inches="tight")