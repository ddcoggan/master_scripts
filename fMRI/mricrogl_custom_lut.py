# makes custom colout look up tables for mricrogl
outdir = '/home/tonglab/MRIcroGL/Resources/lut'
RGBcols = {'red': (230, 25, 75),
           'green': (60, 180, 75),
           'yellow': (255, 225, 25),
           'blue': (0, 130, 200),
           'orange': (245, 130, 48),
           'purple': (145, 30, 180),
           'cyan': (70, 240, 240),
           'magenta': (240, 50, 230),
           'lime': (210, 245, 60),
           'pink': (250, 190, 212),
           'teal': (0, 128, 128),
           'lavender': (220, 190, 255),
           'brown': (170, 110, 40),
           'beige': (255, 250, 200),
           'maroon': (128, 0, 0),
           'mint': (170, 255, 195),
           'olive': (128, 128, 0),
           'apricot': (255, 215, 180),
           'navy': (0, 0, 128),
           'white': (255, 255, 255),
           'black': (0, 0, 0),
           'red_light': (255,127,127),
           'red_medium': (255,0,0),
           'red_dark': (127,0,0),
           'green_light': (127,255,127),
           'green_medium': (0,255,0),
           'green_dark': (0,155,0),
           'blue_light': (127,127,255),
           'blue_medium': (0,0,255),
           'blue_dark': (0,0,155),
           'purple_light': (255,127,255),
           'purple_medium': (255,0,255),
           'purple_dark': (191,0,191),
           'purple_darker': (127,0,127),
           'gold': (255,215,0),
           'red_aspect': (255,63,63),
           'green_aspect': (63,255,63),
           'blue_aspect': (63,63,255),
           'grey_light': (191,191,191),
           'grey_medium': (127,127,127),
           'grey_dark': (63,63,63)}

for colour in RGBcols:
    outpath = f'{outdir}/custom_{colour}.clut'
    text = f'[FLT]\n' \
           f'min=0\n' \
           f'max=0\n' \
           f'[INT]\n' \
           f'numnodes=3\n' \
           f'[BYT]\n' \
           f'nodeintensity0=0\n' \
           f'nodeintensity1=127\n' \
           f'nodeintensity2=255\n' \
           f'[RGBA255]\n' \
           f'nodergba0={RGBcols[colour][0]}|{RGBcols[colour][1]}|{RGBcols[colour][2]}|0\n' \
           f'nodergba1={RGBcols[colour][0]}|{RGBcols[colour][1]}|{RGBcols[colour][2]}|127\n' \
           f'nodergba2={RGBcols[colour][0]}|{RGBcols[colour][1]}|{RGBcols[colour][2]}|255'

    with open(outpath, 'w+') as f:
        f.write(text)
    f.close()

