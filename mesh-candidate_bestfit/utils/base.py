import random

class element(object):
    def __init__(self, cx, cy, h, w, category, filepath):
        self.cx = cx
        self.cy = cy
        self.h = h
        self.w = w
        self.category = category
        self.filepath = filepath
        self.ratio = h / w 
        self.area  = h * w
    
    def gen_real_bbox(self):
        self.real_cx, self.real_cy = self.cx, self.cy
        self.real_w, self.real_h =  self.w*random.uniform(0.8,0.95), self.h*random.uniform(0.8,0.95)
        
    def get_real_bbox(self):
        return self.real_cx, self.real_cy, self.real_w, self.real_h
    
    def __repr__(self):
        return f'cx: {self.cx}, cy: {self.cy}, h:{self.h}, w:{self.w}, category:{self.category}'
    

class Layout(object):
    def __init__(self, cand_elements, align=None, fill=None):
        self.cand_elements = cand_elements
        self.align = align
        self.fill = fill