import scipy.misc
import numpy as np
import sys
import os
import random



#################### Merge and Save
'''
Function: Able to display multiple Images in an neat way, as a grid of images

Provide:
    - Images in form of a 3 Dimensional Matrix: [#images, height, width]
    - Size of the "grid" as a tuple, e.g. if 6 images x 6 images is desired --> (6,6)
    - Path, where the resulting image is supposed to be saved
'''
def merge_and_save(imgs, size, image_path):

    # first we build the inverse of the images
    imgs = (imgs+1.)/2.
    
    # next, we merge them to one image
    height, width = imgs.shape[1], imgs.shape[2]

    merged_img = np.zeros((size[0]*height, size[1]*width))

    for iterator, img in enumerate(imgs):
        foo1 = iterator % size[1]
        foo2 = iterator // size[1]

        merged_img[foo2*height:foo2*height+height, foo1*width:foo1*width+width] = img
    
    # save merged image
    return scipy.misc.imsave(image_path, merged_img)



#################### Get Highest Model
'''
Function: Returns the model with the highest number from a model folder

Provide:
    - Path of the models
'''
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



#################### Argument Parser
'''
Function: Looks at arguments which were supplied with the execution
            of the programm and builds a dictionary out of it and 
            returns it to the to "dc_gan.py"

Provide:
    - argv argument
'''
def arg_parser(argv):
    
    # default dictionary. nbatch and load: If not supplied, first value is false, else true.
    arg_dict = {'train':False, 'test':False, 'nbatch':[False, 0], 'load':[False, 0], \
                'vis':False, 'number':5, 'z_int':False, 'z_change':[False,-1], 'help':False}

    # in case only "python3 dc_gan.py" is executed: train with visualisation
    if len(argv) == 1:
        arg_dict['train'] = True
        arg_dict['vis'] = True

    ########################## Help Flag
    # in case someone requests help, call "diplay_help()"
    if '--help' in argv:
        arg_dict['help'] = display_help()

    # in case flag was provided without either "-train" or "-test"
    if len(argv) > 1 and not ("-train" in argv or "-test" in argv): 
        raise ValueError('You cannot provide a flag without either "-train" or "-test"')    

    ########################## Train Flag
    # if "-train" was provided, check for corresponding tags
    if '-train' in argv:
        arg_dict['train'] = True
       
       ############# Load Flag
        # if "-load" was provided, set load to true
        if '-load' in argv:
            arg_dict['load'][0] = True
            
            # if also int was provided, save it ...
            try:
                arg_dict['load'][1] = int(argv[argv.index('-load')+1])
            # ... otherwise leave it
            except:
                pass
        
        ############# Visualisation Flag
        # if "-vis" was provided, set it to true
        if '-vis' in argv:
            arg_dict['vis'] = True

        ############# Number of Batches Flag
        # if '-nbatch' was provided, set it to true...
        if '-nbatch' in argv:
            # .. if int was also provided, save it
            try:
               arg_dict['nbatch'] = [True, int(argv[argv.index('-nbatch')+1])]
            # .. in case it wasnt, raise an error
            except:
                raise ValueError('Please provide an int after "-nbatch"...')
    
    ########################## Test Flag
    # in case '-test' was provided, set it to true. check for corresponding flags 
    if '-test' in argv:
        arg_dict['test'] = True
        
        ############# Z-Interpolation Flag
        if '-z_int' in argv: 
            arg_dict['z_int'] = True
        
        ############# Z-Change Flag
        if '-z_change' in argv:
            arg_dict['z_change'][0] = True

             # if also int was provided, save it ... (and -1 for array)
            try: 
                arg_dict['z_change'][1] = int(argv[argv.index('-z_change')+1])-1
                
            # if not, set it to a random int between 0 and 99 (because array)
            except:
                arg_dict['z_change'][1] = random.randint(0,99)

            # check if number was between 1 and 100 (after we subtracted 1)
            if arg_dict['z_change'][1] < 0 or arg_dict['z_change'][1] > 99:
                    raise ValueError('Z_change integer has to be between 1 and 100...')
        
        # if none of the other two, normal sample is requested
        if not(arg_dict['z_change'][0] or arg_dict['z_int']):

            # check if it was supplied
            try:
                arg_dict['number'] = int(argv[argv.index('-test')+1])

            # otherwise leave it at default 1
            except:
                pass

            # check if number was between 1 and 8
            if arg_dict['number'] < 1 or arg_dict['number'] > 8:
                    raise ValueError('Test integer has to be between 1 and 8...')
            
    return arg_dict



#################### Help Function
'''
Function: Displays Help Menu and shuts down the program after
'''
def display_help():
    print('--------------------- Options ---------------------\n\n')

    print('-train:\t\t Training. Without arguments equal to "-train -vis".\n')
    print('\t-nbatch x:\t\t Train "x" batches (x has to be an integer).')
    print('\t-load:\t\t\t Load most most advanced model.')
    print('\t-load x:\t\t Train model with number x (e.g. 3416).')
    print('\t-vis:\t\t\t Turn on visualizations.\n\n')

    print('-test:\t\t Testing. Return image sample.')
    print('-text x: \t Return x*x ordinary samples in one image.\n')
    print('\t-z_int:\t\t\t Enable Z-Interpolation. Img will be saved.')
    print('\t-z_change:\t\t Do Z-Change with a random paramter.')
    print('\t-z_change x:\t\t Do Z-Change with a paramter number x.')
    print('\t-vis:\t\t\t Turn on visualizations.\n\n')
    print('Note: When using a flag with "-text", no ordinary sample will be returned.')

    print('---------------------------------------------------\n\n')

    sys.exit()