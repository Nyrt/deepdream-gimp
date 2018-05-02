#!/usr/bin/env python


from __future__ import print_function
import math
from gimpfu import *

# Code adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb

import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import sys

import tensorflow as tf
try:
    # for Python2
    from Tkinter import * ## notice capitalized T in Tkinter
except ImportError:
    # for Python3
    from tkinter import *
from ttk import *


# Get inception model
# Note: add to gitignore
# bashCommand = "wget -nc https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip -n inception5h.zip"
# process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
# output, error = process.communicate()
# print(error)

# Try doing this with different models? It might just *work*
model_fn = 'tensorflow_inception_graph.pb'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(config=config, graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# Print some data about the model
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

# print('Number of layers', len(layers))
# print('Total number of feature channels:', sum(feature_nums))


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 1)*255)
   
    PIL.Image.fromarray(a).show()
 #  display(Image(data=f.getvalue()))


def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)



def calc_grad_tiled(img, t_grad, tile_size=244):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):

            sub = img_shift[y:y+sz,x:x+sz]

            s = sub.shape
            # print(s)
            pad_x = 244 - s[0]
            pad_x *= pad_x > 0
            pad_y = 244 - s[1]
            pad_y *= pad_y > 0

            sub=np.pad(sub, ((0,pad_x),(0,pad_y),(0,0)), 'reflect')
            # print(sub.shape)

            # print(sub.shape)
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g[:s[0],:s[1]]
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


img_noise = np.random.uniform(size=(224,224,3)) + 100.0
def render_deepdream(t_obj, img0=img_noise,
                     iter_n=15, step=1.5, octave_n=10, octave_scale=1.2):
    


    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in range(octave_n):
        print(octave)
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
            if img.shape[0] < 244 or img.shape[1] < 244:
                print("Too small, skipping octave")
                continue
        for i in range(iter_n):
            pdb.gimp_progress_update(float(i + octave * iter_n )/float(iter_n * octave_n))

            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('.',end = ' ')
        clear_output()
        #showarray(img/255.0)
    return img/255.0




print("loading class names")
with open("imagenet_comp_graph_label_strings.txt") as textFile:
    class_names = [line[:-1] for line in textFile]

# Returns NP array (N,bpp) (single vector ot triplets)
def channelData(layer):
    region=layer.get_pixel_rgn(0, 0, layer.width,layer.height)
    pixChars=region[:,:] # Take whole layer
    bpp=region.bpp

    return np.frombuffer(pixChars,dtype=np.uint8).reshape(layer.height, layer.width, bpp)#.transpose((1,0,2))

def createResultLayer(image,name,result):
    rlBytes=np.uint8(result).tobytes();

    rl=gimp.Layer(image,name,image.width,image.height,image.active_layer.type,100,NORMAL_MODE)
    region=rl.get_pixel_rgn(0, 0, rl.width,rl.height,True)
    region[:,:]=rlBytes
    image.add_layer(rl,0)
    gimp.displays_flush()


def python_deepdream_legacy(timg, tdrawable, iter_n, step, layer, feature):

    width = tdrawable.width
    height = tdrawable.height

    layer ='softmax%i'%layer

    target_class = T(layer)[:,feature]

    img0 = channelData(tdrawable)
    img0 = np.float32(img0)

    result = render_deepdream(target_class, img0, iter_n, step)
    result = np.clip(result, 0, 1)

    createResultLayer(timg, "deepdream", result*255.0)

def python_deepdream(timg, tdrawable, iter_n, step, layer, features):

    width = tdrawable.width
    height = tdrawable.height

    layer ='softmax%i'%layer

    target_class = T(layer)[:,int(features[0][1:], 16) - 1]

    for feature in features[1:]:
        feature = int(feature[1:], 16) - 1
        target_class += T(layer)[:,feature]

    print(target_class)

    img0 = channelData(tdrawable)
    img0 = np.float32(img0)

    result = render_deepdream(target_class, img0, iter_n, step)
    result = np.clip(result, 0, 1)

    createResultLayer(timg, "deepdream", result*255.0)

register(
        "python_fu_deepdream_legacy",
        "Apply deepdream generative art to the specified layer",
        "Apply deepdream generative art to the specified layer",
        "Tufts LaserLemon",
        "Tufts LaserLemon",
        "2018",
        "<Image>/Filters/Deepdream_Legacy...",
        "RGB*, GRAY*",
        [
                (PF_INT, "iter_n", "Detail", 15),
                (PF_SPINNER, "step", "Strength", 1.5, (-10, 10, 0.1)),
                #(PF_INT, "octave_n", "Number of Octaves", 5),
                #(PF_SLIDER, "octave_scale", "Octave Scale", 1.2, (1, 2, 0.01)),
                (PF_OPTION, "head", "Layer depth:", 0, ["Shallow", "Medium", "Deep"]),
                (PF_OPTION, "feature", "Class:", 0, class_names)
        ],
        [],
        python_deepdream_legacy)


class gui(Tk):
    def __init__(self, timg, tdrawable, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        window = Tk()
        window.title("Deep Dream Plugin")
        window.geometry("250x480+32+32")

        detail = StringVar(window)
        detail.set("15")
        Label(window, text="Detail", font=("Arial", 10)).place(x = 20, y = 20)
        Spinbox(window, textvariable=detail, from_=1, to=100).place(x = 112, y = 20, width = 128, height = 32)

        strength = StringVar(window)
        strength.set("1.5")
        Label(window, text="Strength", font=("Arial", 10)).place(x = 20, y = 72)
        Spinbox(window, textvariable=strength, from_=0.1, to=10, increment=0.1).place(x = 112, y = 72, width = 128, height = 32)

        depth = StringVar(window)
        depths = ["", "Shallow", "Medium", "Deep"]
        depth.set(depths[1])
        Label(window, text="Depth", font=("Arial", 10)).place(x = 20, y = 104)
        OptionMenu(window, depth, *depths).place(x = 112, y = 104, width = 128, height = 32)

        Label(window, text="Class Select", font=("Arial", 10)).place(x = 20, y = 152)
        Label(window, text="Hold 'ctrl' or 'shift' to select multiple", font=("Arial", 10, "italic")).place(x = 20, y = 174)

        class_select = Treeview(window)
        i = 0

        for class_name in class_names:
            class_select.insert("", i, text=class_name)
            i += 1
        
        
        class_select.place(x = 32, y = 206)


        def run():
            iter_n = int(detail.get())
            step = float(strength.get())
            layer = depth.get()
            if layer == "Deep":
                layer = 2
            elif layer == "Medium":
                layer = 1
            else:
                layer = 0

            features = class_select.selection()
            python_deepdream(timg, tdrawable, iter_n, step, layer, features)

        Button(window, text = "Cancel", command = window.destroy).place(x = 66, y = 450)
        Button(window, text = "Run", command=run).place(x = 156, y = 450)

        ## Create class categories
        ## Add preview




def python_deepdream_gui(timg, tdrawable):
    # op = sess.graph.get_operations()
    # for m in op:
    #     print(m.values())

    top = gui(timg, tdrawable)
 
    top.mainloop()


register(
        "python_fu_deepdream",
        "Apply deepdream generative art to the specified layer",
        "Apply deepdream generative art to the specified layer",
        "Tufts LaserLemon",
        "Tufts LaserLemon",
        "2018",
        "<Image>/Filters/Deepdream...",
        "RGB*, GRAY*",
        [
                # (PF_INT, "iter_n", "Detail", 15),
                # (PF_SPINNER, "step", "Strength", 1.5, (-10, 10, 0.1)),
                # #(PF_INT, "octave_n", "Number of Octaves", 5),
                # #(PF_SLIDER, "octave_scale", "Octave Scale", 1.2, (1, 2, 0.01)),
                # (PF_OPTION, "head", "Layer depth:", 0, ["Shallow", "Medium", "Deep"]),
                # (PF_OPTION, "feature", "Class:", 0, class_names)
        ],
        [],
        python_deepdream_gui)


main()
