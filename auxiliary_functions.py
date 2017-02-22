import scipy.misc

#CHECKED SAME AS GIVEN

def merge_and_save(imgs, size, image_path):

    # first we build the inverse of the images
    imgs = (imgs+1.)/2.
    
    # next we merge them
    height, width = imgs.shape[1], imgs.shape[2]

    merged_img = np.zeros((size[0]*height, size[1]*width))

    for iterator, img in enumerate(imgs):
        foo1 = iterator % size[1]
        foo2 = iterator // size[1]

        merged_img[foo2*height:foo2*height+height, foo1*width:foo1*width+width] = img
    # save merged image
    return scipy.misc.imsave(image_path, merged_img)

if __name__=="__main__":
	pass