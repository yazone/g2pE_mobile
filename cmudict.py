
class CMUDict:
    def __init__(self):
        self.cmu_dict = {}
        with open("cmudict","r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                
                line_array = line.split(" ")
                if len(line_array) < 2:
                    continue
                
                word = line_array[0]
                pronounce = ' '.join(line_array[2:])
                
                self.cmu_dict[word] = pronounce
                
    def dict(self):
        return self.cmu_dict;    
