import math

import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
from pip._vendor.msgpack.fallback import xrange

#gray scale convolution
def convolution2D(img, kernel):
    # Flip the kernel
    #kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros_like(img)

    # Add zero padding to the input image
    image_padded = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    image_padded[1:-1, 1:-1] = img

    # Loop over every pixel of the image
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y: y + 3, x: x + 3]).sum()

    return output

# for converting color of each r g and b
def convolution(img, kernel):

    output_r = convolution2D(img[:, :, 0], kernel)
    output_g = convolution2D(img[:, :, 1], kernel)
    output_b = convolution2D(img[:, :, 2], kernel)
    # stack all the 2D outputs for each channels to get the color image back.
    final_output = np.dstack((np.rint(abs(output_r)),
                              np.rint(abs(output_g)),
                              np.rint(abs(output_b))))

    return final_output

def reduce(image):
  kernel = np.ones((3,3)) / 9
  convolved = convolution2D(image, kernel)

  reduced = convolved[::2, ::2]
  return reduced



# Convolve input image with kernel for Gaussian smoothing


# Subsample with numpy indexing to take every other row/column


def expand(img):


    expand = np.repeat(img,2, axis=1)
    expand = np.repeat(expand,2, axis=0)
    #print(expand.shape)
    return expand

# output = np.repeat(image, 2, axis=1)
   # output = np.repeat(output, 2, axis=0)
    #return output'

def gayssPyramid(img, n):
    output = [img]
    for i in xrange(n):
        output.append(reduce(output[i]))
    return output

def LaplacianPyramids(img,n):


    current = gayssPyramid(img,n)
    #print("CURRENT: ", current[4])
    guass = [current[4]] # start where gauss ended
    #print("guass shape: ", guass.shape)
    #print(current.shape)
    pyr = []
    for i in xrange(4,0,-1):
        #filtered = convolution2D(current, n)
        #print("filterd shape: ", filtered.shape)
        #down = reduce(filtered)
        #print("Down shape: ", down.shape)
        up = expand(current[i])
        print("up shape: ", up.shape)
        diff = current[i-1] - up # has to be whole number or will get error with shapes.
        print("diff shape: ", diff.shape)
        guass.append(diff)
    return guass


def reconstruct(Lap, n):
    reconstructed_img = Lap[0] #start from index 0
    print("recon: ",reconstructed_img.shape)
    for i in range(1, n+1):
       # size = (Lap[i].shape[1], Lap[i].shape[0])
        reconstructed_img = expand(reconstructed_img) # call expand on each lap postion
        print("Expand: ",reconstructed_img.shape)
        reconstructed_img = reconstructed_img + Lap[i] # add expand image with postion of lap on each loop
    return reconstructed_img


flag = False
def draw_boundary(img):
    startpt = []
    x_list = []
    y_list = []
    points_list = []

    def draw_bound_mouse(event, x, y, flags, param):
        global flag

        if event == cv2.EVENT_LBUTTONDOWN:
            flag = True
            startpt.append((x,y))
            x_list.append(x)
            y_list.append(y)

        elif event == cv2.EVENT_LBUTTONUP:
            startpt.append((x,y))
            x_list.append(x)
            y_list.append(y)
            flag = False

        elif (event == cv2.EVENT_MOUSEMOVE):
            if flag:
                startpt.append((x,y))
                x_list.append(x)
                y_list.append(y)

    exit_f = True
    while exit_f:
        window_name = 'image'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name,draw_bound_mouse)
        cv2.imshow(window_name,img)
        while True:
            ip = cv2.waitKey(0) & 0xFF
            points_list.append(startpt)
            startpt = []
            if ip == ord('q'):
                exit_f = False
                break
    cv2.destroyAllWindows()
    disp_img = img[:,::-1]
    plt.imshow(disp_img)
    plt.plot(x_list,y_list, '-o')
    plt.show()
    return points_list

def blend(lap1, lap2):
    water = []
    n = 0
    for img1_lap, img2_lap in zip(lap1,lap2):
        n+=1
        print(img1_lap.shape)
        cols, rows = img1_lap.shape
        lapa = np.hstack((img1_lap[:,0:int(cols/2)], img2_lap[:,int(cols/2):]))
        print("lapa: ",lapa.shape)
        water.append(lapa)
        #print("water:",water.shape)
    return water

#may be something wrong with homography function
def homography(pts1_new, pts2_new):
    A = []
    for i in range(0, len(pts1_new)):
        x, y = pts1_new[i][0], pts1_new[i][1]
        u, v = pts2_new[i][0], pts2_new[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1] #/ Vh[-1, -1]
    L /= L[-1]
    H = L.reshape((3, 3))
    return H

def affine(first, second):
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
#----------------------------------------------------------

#start of part 1 convolving image
img = cv2.imread("earth.jpg", 0)
img = cv2.resize(img,(800,800))
#print(img.shape)
#plt.imshow(img[:, :, ::-1])
#plt.show()
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#fig, ax = plt.subplots(1, figsize=(12,8))
plt.imshow(img, cmap='gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

# create kernel
print(img.shape)
kernel = np.ones((3,3)) / 9


#call convolution function
#fimg = convolution2D(img,kernel)

gray = convolution2D(img, kernel)
print("CONVOLVED MAIN: ", gray)
#color = np.uint8(color) # needed or get error
#-------------------------------------------------------
# Most for testing below
#-------------------------------------------------------
#plt.imshow(color[:,:,::-1])
#plt.show()
#cv2.imwrite('SHARPEN.jpg', color)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#cv2.imshow('CONVOLE', color)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#print("FIMG: ", fimg)

# end of testing
#-------------------------------------------------------------
#show figure of orginal and convolved
fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(img, cmap='gray'),plt.title('ORGINAL')
plt.subplot(122), plt.imshow(gray, cmap='gray'),plt.title('CONVOLVED')
plt.show()

#part 2 reduce an input image
h = reduce(img)
print("Reduce: ", h)
fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(img, cmap='gray'),plt.title('ORGINAL-REDUCE')
plt.subplot(122), plt.imshow(h, cmap='gray'),plt.title('CONVOLVED-REDUCE')
plt.show()
#-------------------------------------------------------------------
# expand part
e = expand(img)
print("e: ",e)

fig = plt.figure(figsize=(15,15))
plt.subplot(121), plt.imshow(img, cmap='gray'),plt.title('ORGINAL-EXPAND')
plt.subplot(122), plt.imshow(e, cmap='gray'),plt.title('CONVOLVED-EXPAND')
plt.show()

#-------------------------------------------------------------------
guass = gayssPyramid(img,4)
print("gauss: ",guass)

o = 1
GaussinPyramid = gayssPyramid(img,4)
fig = plt.figure(figsize=(15,15))
for i in GaussinPyramid:
    print((i.shape))
    plt.subplot(1,len(GaussinPyramid),o), plt.imshow(i, cmap='gray'), plt.title('Image' + str(o))
    o +=1 # increment
plt.show()
# plot laplacian

of = 1
LaplacianPyramid = LaplacianPyramids(img,4)
fig = plt.figure(figsize=(15,15))
for i in LaplacianPyramid:
    print("lap:",i.shape)
    plt.subplot(1,len(LaplacianPyramid),of), plt.imshow(i, cmap='gray'), plt.title('Image' + str(of))
    of +=1 # increment
plt.show()
# Plot reconstruct
rec_org = reconstruct(LaplacianPyramid,4)
fig = plt.figure(figsize=(15,15))
plt.subplot(121),plt.imshow(img, cmap= 'gray'), plt.title("Orginal")
plt.subplot(122),plt.imshow(rec_org, cmap='gray'), plt.title("Reconstructed")
plt.show()


print(np.sum(img-rec_org))

#blend part
#img0 = cv2.imread('water.jpg')
img1 = cv2.imread('mount_left.jpg',0)
img2 = cv2.imread('mount_right.jpg',0)

img1_r = cv2.resize(img1, (1200,1200))
img2_r = cv2.resize(img2,(1200,1200))
print(img1_r.shape)
fig = plt.figure(figsize=(10,10))
plt.subplot(121);plt.imshow(img1_r,cmap='gray');plt.title('Left Image')
plt.subplot(122);plt.imshow(img2_r, cmap='gray');plt.title('Right Image')
plt.show()

lap1 = LaplacianPyramids(img1_r,4)
lap2 = LaplacianPyramids(img2_r,4)

blended = blend(lap1,lap2)
reco_img = reconstruct(blended,4)

plt.imshow(reco_img,cmap='gray');plt.title('Blended')
plt.show()

#---------------------------------------------------- end of blended
#boundary_points = draw_boundary(img2_r) # not using this

#---------------------------------------------------- start of unwrapping

img1_c = cv2.imread('mount_left.jpg')
img2_c = cv2.imread('mount_right.jpg')

plt.imshow(img1_c[:,:,::-1])
pts1_h = plt.ginput(3)
pts1_h = np.float32(pts1_h)

# --------- pick points of image two
plt.imshow(img2_c[:,:,::-1])
pts2_h = plt.ginput(3)
pts2_h = np.float32(pts2_h)

#pts2_h = pts1_h

H_ = homography(pts1_h, pts2_h)
print(H_)
H_I = np.linalg.inv(H_)
print(H_I)

result = cv2.warpPerspective(img2_c, H_I, (img1_c.shape[1] + img2_c.shape[1], img1_c.shape[0]))
result[0:img1_c.shape[0], 0:img1_c.shape[1]] = img1_c

#plt doesn't display image as well as in example
plt.imshow(result[:,:,::-1])
plt.show()

#------------------------ Affine part
plt.imshow(img1_c[:,:,::-1])
pts1_a = plt.ginput(3)
pts1_a = np.float32(pts1_a)

plt.imshow(img2_c[:,:,::-1])
pts2_a = plt.ginput(3)
pts2_a = np.float32(pts2_a)

A_ = affine(pts1_a,pts2_a)

aff_mat_fin = np.vstack((A_, np.array([0,0,1])))
aff_mat_fin = np.linalg.inv(aff_mat_fin)

result_aff = cv2.warpPerspective(img2_c, aff_mat_fin, (img1_c.shape[1] + img2_c.shape[1], img1_c.shape[0] ))
result_aff[0:img1_c.shape[0], 0:img1_c.shape[1]] = img1_c

plt.imshow(result_aff[:,:,::-1])
plt.show()