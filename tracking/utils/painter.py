from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import random
import csv


def set_colors(n_colors, saturation=1., light=1., shuffle=True):
    hsv_tuples = [(x/n_colors, saturation, light) for x in range(n_colors)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x),hsv_tuples))
    colors = list(map(lambda x: (int(x[0]*255),int(x[1]*255),int(x[2]*255)),colors))
    if shuffle:
        random.shuffle(colors)
    return colors

def put_mask(
    image,
    mask,
    alpha = 0.6,
    color = (255, 255, 255),
    inverse = False,
    align_size = True
):

    if align_size:
        mask = mask.resize(image.size)    
    mask = np.array(mask)
    if inverse:
        mask = 255 - mask
    mask = alpha*mask

    cover = np.array(Image.new('RGBA', image.size, color))
    cover[..., 3] = mask
    cover = Image.fromarray(cover)

    image = image.convert('RGBA')
    image = Image.alpha_composite(image, cover)
    return image
    
def mark_boxes(
    image, 
    boxes: list,
    tags: list = None,
    colors: list = None,
    line_width = None,
    font_path = None,
    text_size = None,
    filling: list = None,
    filling_color: list = None,
    filling_alpha = 0.3
):
    default_color = (255, 255, 255)
    img = image.convert('RGBA')
    
    text_size = int(max(img.size)/50) if text_size is None else text_size
    line_width = int(max(img.size)/300) if line_width is None else line_width
    
    # create filling cover
    if filling:
        mask = Image.new('L', img.size, color=0)
        cover = Image.new('RGBA', img.size, default_color)
        draw_mask = ImageDraw.Draw(mask)
        draw_cover = ImageDraw.Draw(cover)
        for i, box in enumerate(boxes):
            if filling[i]:
                draw_mask.rectangle(box, fill=int(filling_alpha*255))
                if filling_color:
                    fc = tuple(filling_color[i])
                    draw_cover.rectangle(box, fill=fc)
        cover = np.array(cover)
        cover[..., 3] = mask
        cover = Image.fromarray(cover)
        img = Image.alpha_composite(img, cover)
    
    if font_path:
        font = ImageFont.truetype(font=font_path, size=text_size)
    else:
        font = ImageFont.load_default()
     
    draw = ImageDraw.Draw(img)  # draw annotations
    for i, box in enumerate(boxes):
        bc = default_color if colors is None else tuple(colors[i])  # set color
        draw.rectangle(box, fill=None, outline=bc, width=line_width)  # box

        if tags:
            tag = f'{tags[i]} '
            text_bar = [box[0], box[1]-text_size, box[0]+len(tag)*text_size//2, box[1]]
            draw.rectangle(text_bar, fill=bc, outline=bc, width=line_width)
            draw.text(text_bar[:2], tag, fill=(0,0,0), font=font)
    return img

def plot_losses(
    csv_path, 
    view=(0., 1.), 
    figsize=(8,6), 
    dpi=150, 
    skip_header=True, 
    save_fig=False
):
    """
    Read a csv log which includes column [Epoch, Training Loss, Validation Loss].
    """
    epoch, train, valid = [], [], []
    
    with open(csv_path) as cf:
        data = csv.reader(cf)
        if skip_header:
            field = next(data)
        for d in data:
            epoch.append(float(d[0]))
            train.append(float(d[1]))
            valid.append(float(d[2]))
    
    a = int(view[0]*len(epoch))
    b = int(view[1]*len(epoch))
    fontsize = sum(figsize)
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.xlabel('Epoch', fontsize=fontsize)
    plt.ylabel('Loss', fontsize=fontsize)
    plt.grid(alpha=0.6)
    plt.plot(epoch[a:b], train[a:b], 'blue', alpha=0.7, label='Train')
    plt.plot(epoch[a:b], valid[a:b], 'red', alpha=0.7, label='Valid')
    plt.legend()
    if save_fig:
        plt.savefig(csv_path.replace('.csv', '.png'))
    plt.show()