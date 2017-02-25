import scipy.misc
import numpy as np
import sys
import os

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

# returns number of the highest model in /models
def get_highest_model(model_dir):

    high = 0

    for file_name in os.listdir(model_dir):
      
        # ignore meta
        if "model.ckpt-" in file_name and not '.meta' in file_name:
            
            # get number
            number = int(file_name.split('-')[1])
            
            # check for highest
            if number > high:
                high = number

    return high


def arg_parser(argv):

    # default dictionary. nbatch and load: If not supplied, first value is false, else true.
    arg_dict = {'train':False, 'test':False, 'nbatch':[False, 0], 'load':[False, 0], \
                'vis':False, 'number':1, 'z_int':False, 'z_change':[False,-1], 'help':False}

    # in case only "python3 dc_gan.py" is executed: train with visualisation
    if len(argv) == 1:
        arg_dict['train'] = True
        arg_dict['vis'] = True
    if '--help' in argv:
        arg_dict['help'] = True
        return arg_dict
    # in case flag was provided without either "-train" or "-test"
    if len(argv) > 1 and not ("-train" in argv or "-test" in argv): 
        raise ValueError('You cannot provide a flag without either "-train" or "-test"')
    
    # if "-train" was provided, check for corresponding tags
    if '-train' in argv:
        arg_dict['train'] = True
       
        # if "-load" was provided, set load to true
        if '-load' in argv:
            arg_dict['load'][0] = True
            
            # if also int was provided, save it ...
            try:
                arg_dict['load'][1] = int(argv[argv.index('-load')+1])
            # ... otherwise leave it
            except:
                pass
        
        # if "-vis" was provided, set it to true
        if '-vis' in argv:
            arg_dict['vis'] = True

        # if '-nbatch' was provided, set it to true...
        if '-nbatch' in argv:
            # .. if int was also provided, save it
            try:
               arg_dict['nbatch'] = [True, int(argv[argv.index('-nbatch')+1])]
            # .. in case it wasnt, raise an error
            except:
                raise ValueError('Please provide an int after "-nbatch"...')
    
    # in case '-test' was provided, set it to true. check for corresponding flags 
    if '-test' in argv:
        arg_dict['test'] = True
        
        # if only '-test' was provided --> finished
        if len(argv) == 2:
            return arg_dict
        
        # if "-z_int" was provided, set z to true
        if '-z_int' in argv: 
            arg_dict['z_int'] = True
        # z-change
        if '-z_change' in argv:
            arg_dict['z_change'][0] = True
             # if also int was provided, save it ...
            try: #-1 because array... CHANGE
                arg_dict['z_change'][1] = int(argv[argv.index('-z_change')+1])-1
                if arg_dict['z_change'][1] < 0 or arg_dict[1] > 99:
                    raise ValueError('Z_change integer has to be between 1 and 100...')
            # ... otherwise leave it
            except:
                pass
        # only number is left, check if it is in t, if yes save; otherwise ignore
        else:
            try:
                arg_dict['number'] = int(argv[argv.index('-test')+1])
            
            except:
                pass

            if arg_dict['number'] < 1 or arg_dict['number'] > 8:
                    raise ValueError('Test integer has to be between 1 and 8...')
            
    return arg_dict