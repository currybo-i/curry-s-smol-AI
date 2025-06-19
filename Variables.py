import json
import random
import math

filePath = "E:\\py\\weightdata.json"
configPath = "E:\\py\\Config.json"

firstset, secondset, thirdset = json.load(open(configPath, "r"))
class Weights:
    def __init__(self, mode=None):
        self.mode = mode
        self.weights = self.load_json(filePath)
        
    def __add__(self, newWeights):
        self.save_injson(newWeights, filePath)

    def save_injson(self, data, path):
        with open(path, "w") as file:
            json.dump(data, file)

    def load_json(self, path):
        try:
            with open(path, "r") as file:
                w = json.load(file)
                if (
                    len(w) != 3 or
                    len(w[0]) != firstset or
                    len(w[1]) != secondset or
                    len(w[2]) != thirdset
                ):
                    self.initialise()
                    return self.load_json(path)
                return w
        except FileNotFoundError:
            self.initialise()
            return self.load_json(path)

    
    def he_init(self, fan_in, fan_out):
        std = math.sqrt(2 / fan_in)
        return [random.gauss(0, std) for _ in range(fan_in * fan_out)]

    
    def initialise(self):
        inp = int(firstset / 16)
        out = int(thirdset / 16)
        w1 = self.he_init(inp, 16)
        w2 = self.he_init(16, 16)
        w3 = self.he_init(16, out)
##        print(len(w1), len(w2), len(w3))  # debug
        self.save_injson([w1, w2, w3], filePath)

