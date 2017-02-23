import scipy.misc
import numpy as np

'''
Function: Able to display multiple Images in an neat way, as a grid of images

Usage: Provide:
    - Images in form of a 3 Dimensional Matrix: [images, height, width]
    - Size of the "grid" as a tuple, e.g. if 6 images x 6 images is desired --> (6,6)
    - Path, where the resulting image is supposed to be saved
'''

def merge_and_save(imgs, size, image_path):

    # first we build the inverse of the images
    imgs = (imgs+1.)/2.
    
    # next, we merge them
    height, width = imgs.shape[1], imgs.shape[2]

    merged_img = np.zeros((size[0]*height, size[1]*width))

    for iterator, img in enumerate(imgs):
        foo1 = iterator % size[1]
        foo2 = iterator // size[1]

        merged_img[foo2*height:foo2*height+height, foo1*width:foo1*width+width] = img
    # save merged image
    return scipy.misc.imsave(image_path, merged_img)

def arg_parser(argv):

    # default dictionary. nbatch: If not supplied, first value is false, else true.
    arg_dict = {'train':False, 'test':False, 'nbatch':[False, 0], 'load':False, \
                'vis':False, 'number':1, 'z-int':False}
    if len(argv) == 1:
        arg_dict['train'] = True
        arg_dict['vis'] = True 
    if '-train' in argv:
        arg_dict['train'] = True
        if '-load' in argv:
            arg_dict['load'] = True
        if '-vis' in argv:
            arg_dict['vis'] = True
        if '-nbatch' in argv:
            arg_dict['nbatch'] = [True, int(argv[argv.index('-nbatch')+1])]
    if '-test' in argv:
        arg_dict['test'] = True
        if len(argv) == 2:
            return arg_dict
        elif argv[2] == '-z' or argv[2] == '-Z':
            arg_dict['z-int'] = True
        else:
            arg_dict['number'] = int(argv[2])
    return arg_dict