import numpy as np

arr = np.arange(27).reshape(3,3,3) #3 channel image
mask = np.zeros(shape=(3,3))
mask[1,1] = 1 # binary mask
mask_3d = np.stack((mask,mask,mask),axis=0) #3 channel mask

## Answer 1
# Simply multiply the image array with the mask

masked_arr = arr*mask_3d

## Answer 2
# Use the where function in numpy

masked_arr = np.where(mask_3d==1,arr,mask_3d)

#Both answer gives
print(masked_arr)
