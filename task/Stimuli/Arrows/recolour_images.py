from scipy.misc import imread, imsave
import os

arrow_path = 'task/Stimuli/Arrows/'

for i in os.listdir(arrow_path):

    if '.png' in i:

        image = imread(os.path.join(arrow_path, i))

        image[..., :3][image[..., :3] == 0] = 255

        imsave(os.path.join('task/Stimuli/Arrows/', i), image)