

class Art():
    def pixel_char(val):
        if val > 200:
            return "@"
        elif val > 150:
            return "#"
        elif val > 100:
            return "="
        elif val > 50:
            return "+"
        else:
            return "_"
        
    def give_img( img, shape):
        st = ""
        for row in range(shape[1]):
            for col in range(shape[0]):
                pixel = img[row * 28 + col]
                st += Art.pixel_char(pixel)
            st += "\n"
        return st

