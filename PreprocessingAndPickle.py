import os
import cv2
import pickle
import numpy as np

'''
Function: 
	1) Preprocessing of Celeb_A Datase
		- cropping into square
		- resizing to 64x64
		- convert to grayscale
	2) Dumps Images into a pickle 


Usage:
	- Provide path of the folder with the images to be
	preprocessed --> "pic_folder_path"

	- Provide path + name for a folder to be created, in which the pickle will dump

	- Provide Name of the file the pickled-file
'''

def preprocessing(pic_folder_path='./celebAPics', pickle_path='./pickle_Data', pickle_name="CelebA_pickle.dat"):

	

	if not os.path.isdir(pickle_path):
		# create directory for the pickled data
		os.makedirs(pickle_path)

		# list, where we save our pictures so we can dump them later
		IMG_LIST = []
		
		# iterate through every element in the directory
		for counter, file_name in enumerate(os.listdir(pic_folder_path)):

			# print progress if we did 0.1% of whole data (takes really long...)
			if (counter)%round(len(os.listdir(pic_folder_path))/1000) == 0 and counter != 0:
				print("[*] " + str(counter) + "/" + str(len(os.listdir(pic_folder_path))) +" images processed ...")

			# check whether its a picture
			if file_name.endswith(".jpg") or file_name.endswith(".png"):
				# read img
				img = cv2.imread(pic_folder_path + '/' + file_name)
				# apply cropping
				img = img[20:198,:]
				# resize img to 64x64
				img = cv2.resize(img, (64, 64)) 
				# convert to greyscale
				img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				# bring to range -1 to 1
				img = (img/255.0-0.5)*2.0
				# expand dimensions so its compatible with placeholder function
				img = np.expand_dims(img,-1)

				# append to "IMG_LIST"
				IMG_LIST.append(img)

		# Dump the img list into a file
		with open(pickle_path+'/'+pickle_name, 'wb') as f:
			pickle.dump(IMG_LIST, f)

	else:
		print('Folder with name "'+ pickle_path +'"" already exists. Please delete...')

if __name__=='__main__':
	preprocessing()