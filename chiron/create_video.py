import numpy as np
from PIL import Image, ImageDraw,ImageFont
import os
import re
from tqdm import tqdm

files = os.listdir('vid')

files_ord = [(int(re.split('[_\.]', x)[1]), x) for x in files]
files_ord.sort() # (nr, fname)

font = ImageFont.truetype('liberation.ttf', 32)

fi = 0

for episode_nr, episode_file in tqdm(files_ord):
    arr = np.load('vid/' + episode_file)
    n_steps = arr.shape[0]
    for step in range(n_steps):
        i = Image.fromarray(arr[step, :,:], 'RGB')
        draw = ImageDraw.Draw(i)
        draw.text((10,10), 'PGPE\nσ=0.4\nEpisode: {}'.format(episode_nr), (255,255,255), font)
        i.save('frames/{:010d}.png'.format(fi))
        fi += 1
        del draw
        del i





# i = Image.Image.frombytes('RBG', (400,600), frame)
