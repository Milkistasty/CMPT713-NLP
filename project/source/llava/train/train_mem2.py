import os,sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
print("current_dir : ",current_dir)
print("parent_dir : ",parent_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(parent_dir)
print("parent_dir : ",parent_dir)

from llava.train.train2 import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
