import os
import imageio

# Function: Creates Gif using Images
#
# Usage: Provide
#	- (a path and) a name for the gif (with .gif as datatype)
#	- a path to folder, containing the images
#	- number of frames per second

def createGif(gif_name='../Images-Gifs/name_me.gif', image_path='.', image_duration=1):
	
	image_list = []

	# argument with duration of each image in seconds
	kargs = { 'duration': image_duration }

	# save all images in a list
	for file_name in os.listdir(image_path):
		if file_name.endswith(".jpg") or file_name.endswith(".png"):
			image_list.append(imageio.imread(file_name))

	# create gif
	imageio.mimsave(gif_name, image_list, **kargs)

if __name__=='__main__':
	createGif()
