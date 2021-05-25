import cv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
def homography2(pts1, pts2):
    A = []
    for i in range(0, len(pts1)):
        x, y = pts1[i][0], pts1[i][1]
        u, v = pts2[i][0], pts2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H_o = L.reshape(3, 3)
    return H_o
def homography(pts1_new, pts2_new):
    A = []
    for i in range(0, len(pts1_new)):
        x, y = pts1_new[i][0], pts1_new[i][1]
        u, v = pts2_new[i][0], pts2_new[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def LS(first, second):
    A = []
    B = []

    for sp, trg in zip(first,second):
        A.append([sp[0], 0, sp[1], 0, 1, 0])
        A.append([0, sp[0], 0, sp[1], 0, 1])
        B.append(trg[0])
        B.append(trg[1])
    result, resids, rank, s = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)

    a0, a1, a2, a3, a4, a5 = result
    affinTrans = np.float32([[a0,a2,a4], [a1,a3,a5]])

    return affinTrans

    #return trans

img = cv2.imread('TajMahal.jpg',1)
plt.imshow(img[:, :, ::-1])
#data_org_img = cv2.setMouseCallback('image',click)
#cv2.waitKey()
#cv2.destroyAllWindows()
pts1 = np.float32([[100, 250], [340, 570], [480, 600]]) #source
pts2 = np.float32([[110, 270], [360, 590], [540, 610]]) #dst

aff = cv2.getAffineTransform(pts1, pts2) # get both pts for affine transform
print("AFF: ", aff)
test = cv2.warpAffine(img,aff, (img.shape[1], img.shape[0])) # warp the image with aff points

plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('Input') # display input
plt.subplot(122), plt.imshow(test[:, :, ::-1]), plt.title('Output') #display output
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
# end of affine transform partB 1

#Start of perspective transform

pts1_h = np.float32([[100, 250], [340, 570], [480, 600], [20, 10]])
pts2_h = np.float32([[125, 260], [380, 600], [500, 620], [25, 25]])


hom = cv2.getPerspectiveTransform(pts1_h, pts2_h)
print("HOM: ", hom)
test_h = cv2.warpPerspective(img,hom, (img.shape[1], img.shape[0]))

plt.subplot(121), plt.imshow(img[:, :, ::-1]), plt.title('Input') # display input
plt.subplot(122), plt.imshow(test_h[:, :, ::-1]), plt.title('Output') #display output
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
#End of perspective transform

#start of partB 2

#get points from orginal 3 points
plt.imshow(img[:, :, ::-1])
pts1_new = plt.ginput(3)
pts1_new = np.float32(pts1_new)
print(pts1_new)

#get points from test 3 points
plt.imshow(test_h[:, :, ::-1])
pts2_new = plt.ginput(3)
pts2_new = np.float32(pts2_new)
print(pts2_new)
plt.close()
cv2.destroyAllWindows()



H = homography(pts1_new, pts2_new)
#print("linsys",linsys)
print("H",H)
#plt.show()
#abs error
print(np.sum(np.abs(np.subtract(hom,H,dtype=float))))

# Homography matrix using more than 4 points

# get points from orginal image
plt.imshow(img[:, :, ::-1])
pts1_over = plt.ginput(5)
pts1_over = np.float32(pts1_over)
print(pts1_over)

#get points from test 3 points
plt.imshow(test_h[:, :, ::-1])
pts2_over = plt.ginput(5)
pts2_over = np.float32(pts2_over)
plt.close()
cv2.destroyAllWindows()
print(pts2_over)

H_o = homography2(pts1_over, pts2_over)
print("H_o: ", H_o)
#print("H Again: ",H)
#get abs error
print(np.sum(np.abs(np.subtract(hom,H_o,dtype=float))))

# HOMOGRAPHY over all things

img_h_n = cv2.warpPerspective(img,H, (img.shape[1], img.shape[0]))
img_h_o = cv2.warpPerspective(img, H_o, (img.shape[1], img.shape[0]))

#plot 4 graphs
fig = plt.figure(figsize=(7, 7))
grid = ImageGrid(fig, 111,nrows_ncols=(2, 2),axes_pad=0.4,)

for ax, im, text in zip(grid, [img[:, :, ::-1], test_h[:, :, ::-1], img_h_n[:, :, ::-1], img_h_o[:, :, ::-1]
], [['Input'], ['Original Transformed'], ['3 points'], ['OverConstrained']]):
    ax.imshow(im)
    ax.set_title(text)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#Now do affine matrix computation
# 3 points
#orginal image
plt.imshow(img[:, :, ::-1])
pts1_aff = plt.ginput(3)
#plt.close()
pts1_aff = np.float32(pts1_aff)
print(pts1_aff)

#aff image
plt.imshow(test[:, :, ::-1])
pts2_aff = plt.ginput(3)
pts2_aff = np.float32(pts2_aff)
print(pts2_aff)


res = LS(pts1_aff, pts2_aff)
print("RES: ",res)
print(res.shape)
#res = res[0]
#res = res.reshape((2,3))

print(np.sum(np.abs(np.subtract(res,aff,dtype=float))))
# affine of 4 points

plt.imshow(img[:,:,::-1])
pts1_5aff = plt.ginput(5)
pts1_5aff = np.float32(pts1_5aff)
print(pts1_5aff)

plt.imshow(test[:,:,::-1])
pts2_5aff = plt.ginput(5)
pts2_5aff = np.float32(pts2_5aff)
plt.close()
cv2.destroyAllWindows()
print(pts2_5aff)

res_over = LS(pts1_5aff, pts2_5aff)
print("RES4: ", res_over)
print(np.sum(np.abs(np.subtract(res_over,aff,dtype=float))))

img_a_n = cv2.warpAffine(img, res, (img.shape[1], img.shape[0]))
img_a_o = cv2.warpAffine(img, res_over, (img.shape[1], img.shape[0]))

fig = plt.figure(figsize=(7, 7))
grid = ImageGrid(fig, 111,
nrows_ncols=(2, 2),
axes_pad=0.4,
)
for ax, im, text in zip(grid, [img[:, :, ::-1], test[:, :, ::-1], img_a_n[:, :, ::-1], img_a_o[:, :, ::-1]],
[['Input'], ['Original Transformed'], ['3 points'], ['OverConstrained']]):
    ax.imshow(im)
    ax.set_title(text)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()