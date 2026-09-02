# import cv2
# from pathlib import Path

# def main():
#     images_path = Path("images")

#     images = []

#     # read all images from /images
#     for path in images_path.rglob("*"):
#         image = cv2.imread(path)
#         images.append(image)

#     print(images)


# if __name__ == "__main__":
#     main()



##################### Nathen learning #####################
import cv2
import numpy

def opencv_basics(): # {
    '''
        openCV basic usage for reading, resizing, rotating, and saving an image.

        cv2.imread(), cv2.resize(), cv2.rotate(), cv2.imwrite()
    '''
    img = cv2.imread('images/camera_man_w_noise.jpg', cv2.IMREAD_COLOR)

    ### RESIZE
    # resize an image by pixels
    img = cv2.resize(img, (600, 600))
    # OR by scale
    img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5) # increase the photo by half its size

    ### ROTATE
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) # rotate 90 counter clockwise

    ### SAVE AN IMAGE
    cv2.imwrite('new_image.jpg', img)

    # show the image in a window
    cv2.imshow('Image', img)

    # close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# }

def opencv_image_fundamentals_and_manipulation():
    # you can pass -1, 0, 1, for flags IMREAD_UNCHANGED, IMREAD_GRAYSCALE, and IMREAD_COLOR, respectively
    img = cv2.imread('images/camera_man_w_noise.jpg', 1)

    '''
    Note!
    imread() returns a numpy array, which is an optimized library for python arrays.
    you can get the shape of an array using array.shape which gives you the:
    * num of rows
    * num of cols
    * num of channels
    '''
