import json
import os

class JsonParser:
    def __init__(self, working_dir):
        json_dir = os.path.join(working_dir, "configure.json")
        if not os.path.exists(json_dir):
            print("Error - Configuration miss")
            exit(-1)

        with open(json_dir, 'r') as f:
            # content = f.read()
            config = json.load(f)
        self.train_batch = config.get("train_batch", 32)
        self.input_channel = config.get("input_channel", 3)
        self.internal_channel = config.get("internal_channel", [16, 32, 64])
        self.bottle_channel = config.get("bottle_channel", 128)
        self.class_num = config.get("class_num", 21)
        self.model_name = config.get("model_name", "model.pth")
        self.class_weight = config.get("class_weight", [])
        if len(self.class_weight) != 0 and len(self.class_weight) != self.class_num:
            print(f"Error! class number {self.class_num} is not equal to {len(self.class_weight)}")
            exit(-1)
