import random
import time
import sys
from colorama import just_fix_windows_console
just_fix_windows_console()
import math
from Variables import Weights as w  
from ASCIIart import Art
from NeuralCore import NeuralNetwork
import json


def clear_previous_frame(x):
    for _ in range(x):
        sys.stdout.write("\033[2K")
        sys.stdout.write("\033[1A")  
    sys.stdout.write("\033[2K")

with open("E:\\py\\AI training fed\\train-images.idx3-ubyte", "rb") as imgf:
    imgf.read(16)
    with open("E:\\py\\AI training fed\\train-labels.idx1-ubyte", "rb") as labf:
        labf.read(8)
        
        pixel_count = 784
        json.dump([pixel_count*16,16*16,16*10],open("E:\\py\\Config.json", "w"))

        img_bytes = imgf.read(60000 * pixel_count)
        lab_bytes = labf.read(60000)

        img_data = [list(img_bytes[i*pixel_count:(i+1)*pixel_count]) for i in range(60000)]
        lab_data = list(lab_bytes)

        _w = w().weights

        setnum = 0
        acc = 0
        lr = 0.05

        nn = NeuralNetwork(_w)
        while setnum<=60:
            correct = 0
            setnum += 1
            if setnum % 10 == 0:
                lr *= 0.9
    
            combined = list(zip(img_data, lab_data))
            random.shuffle(combined)
            img_data[:], lab_data[:] = zip(*combined)

            for i in range((setnum-1)*1000, setnum*1000):
                st_time = time.time()
                img = [px / 255.0 for px in img_data[i]]
                label = lab_data[i]

                nn.forward( img )

                pred = nn.predict()
                if pred == label:
                    correct += 1

                clear_previous_frame(29)
                frame = f"{Art.give_img(img_data[i], (28,28))}Label: {label} Prediction:{pred} Accuracy: {acc}\n"
                sys.stdout.write(frame)
                sys.stdout.flush()
                
                nn.backprop( img, label, lr)

##                t_time = time.time() - st_time
##                remaining = 6 - t_time
##                if remaining > 0:
##                    time.sleep(remaining)
            w() + [nn._w1, nn._w2, nn._w3]
            acc = correct / ( 1000)
