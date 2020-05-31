import cv2

def convert_to_black_background(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j] >= 127:
                array[i][j] = 0
            else:
                array[i][j] = 255
    return array

def convert_to_white_background(image):
    im_gray = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    try:
        b, g, r = cv2.split(im_gray)
        t = [None] * 3
        u = [None] * 3
        for i, im in enumerate([b, g, r]):
            t[i], u[i] = cv2.threshold(im, 255, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        dst = cv2.merge((*u,))
        cv2.imwrite(image + "_w.png", dst)
        fileImage = image + "_w.png"
        return fileImage
    except:
        return None

def rotate_image(image , radient):
    return image.rotate(radient)
