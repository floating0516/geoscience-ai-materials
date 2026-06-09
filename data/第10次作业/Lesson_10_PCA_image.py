import cv2   ### pip install opencv-python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Loading the image
img = cv2.imread('dog.jpeg') #you can use any image you want.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # #OpenCV represents the images in BGR as opposed to the RGB we expect
plt.imshow(img)  
plt.show()

# Splitting the image in R,G,B arrays.
 
red,green,blue = cv2.split(img) 
#it will split the original image into Blue, Green and Red arrays.

for K in (5,10,30,50,100,200,250):
	pca = PCA(K)
 
#Applying to red channel and then applying inverse transform to transformed array.
	red_transformed = pca.fit_transform(red) #Fit the model with X and apply the dimensionality reduction on X.
	                                         # 用X来训练PCA模型，同时返回降维后的数据
	red_inverted = pca.inverse_transform(red_transformed)  ### Transform data back to its original space
                                                 # 将降维后的数据转换成原始数据
#Applying to Green channel and then applying inverse transform to transformed array.
	green_transformed = pca.fit_transform(green)
	green_inverted = pca.inverse_transform(green_transformed)
 
#Applying to Blue channel and then applying inverse transform to transformed array.
	blue_transformed = pca.fit_transform(blue)
	blue_inverted = pca.inverse_transform(blue_transformed)


	img_compressed = (np.dstack((red_inverted, green_inverted, blue_inverted))).astype(np.uint8)

#viewing the compressed image
	plt.imshow(img_compressed)
	plt.title('K='+str(K))
	plt.show()
