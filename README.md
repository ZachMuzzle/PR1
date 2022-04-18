# Project 1 for Computer Vision

## I have provided jpeg images of results.

## PART A INSTRUCTIONS
* When part A is ran you will be greeted with the image of earth in grayscale. Press the X button to at the top of the pop to go to the next part of the program. The next part will show the original image as well as the convolved image. The next part will show the original with a reduced convolved image next to it. After this an expanded of the image will show with the original

* The following part after expand will show for graphs for our gaussian part of the earth image. Then after this we will have the laplacian pyramid of 5 graphs starting from index 4 of the gaussian part. Reconstructed image will then be display next to the original image

* Now we will have two images of mountains which are cropped from the main image. We will blend these images. We will now have the two images blended together

* Now pick 3 points for each image displayed of the left and right. Now using homography we the images will go together (My homography function may be wrong as it doesn't combine the images together)

* After this we will using affine function we created. pick another 3 points again and the same ones for the next picture the image should become full now.

### Results for Part A for Earth:
* [Original Earth](earth.jpg). This is what we input into the program.
* [Convolved Earth](convolved.jpeg). This is the Earth convolved.
* [Reduced Earth](reduced.jpeg). This is the Earth reduced.
* [Expanded Earth](expanded.jpeg). This it the Earth expanded.
* [Gaussian Earth](gaussian_earth.jpeg). This is the Earth Gaussian Pyramid.
* [Laplacian Earth](laplacian_earth.jpeg). This the Earth Laplacian Pyramid.
* [Reconstructed Earth](Reconstructed_earth.jpeg). This is Earth reconstructed from Laplacian Pyramid.

### Results for Part A for Mountain:
* [Original Mountain](mount.jpg). The original mountain photo.
* [Left Mountain](mount_left.jpg). The picture of the left side of the mountain.
* [Right Mountain](mount_right.jpg). The picture of the right side of the mountain.
* [Blended Mountain](blended_mountains.jpeg). The picture of the left and right mountains blended.
* [Affine Mountain](Affine_unwrap.jpeg). The picture of the affine warp perspective.
* [Holography Mountain](holography_unwrap.jpeg) The picture of the holography mountain warp (This is not correct output I believe)

## PART B INSTRUCTIONS
* When program is ran image for affine will be displayed once exited from that image an image for homography will be displayed these use the built in functions

* Next we will pick 3 points for two images. Try to make them the same points we will use our own homography function and display the error

* After this we will do the same thing again but this time with 5 points.

* After this is done 4 pictures will be shown showing results using our function we built.

* Now we will do the same process all over again but use our affine image and affine function we have created. Once this is done another 4 images will appear with the results.

### Results for Part B:
* [Original Picture](TajMahal.jpg). Picture that we pick our points from.

