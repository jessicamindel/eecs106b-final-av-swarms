import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import time

img = cv2.cvtColor(cv2.imread('/Users/jessiemindel/Desktop/School Work/College/UC Berkeley/year3/sp21/EECS 106B/final/task1.png'), cv2.COLOR_BGR2RGB)

plt.ion()

fig, ax = plt.subplots(figsize=(8,8))
fig.canvas.set_window_title('AV Swarm Simulator')
plt.show()

for i in range(100):
	print(i)
	ax.clear()
	ax.imshow(img)
	ax.add_patch(Rectangle((200 + 2*i, 300 - 2*i), 31, 70, angle=45, edgecolor='black', facecolor='white', fill=True, lw=2))

	ax.axis('off')
	plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
	plt.pause(0.0001)
