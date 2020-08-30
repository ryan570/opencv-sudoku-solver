import os
import random
import shutil

val_ratio = 0.2

source = 'data/train/'
dest = 'data/validation/'

count = 0
for folder in os.listdir(source):
    for img in os.listdir(os.path.join(source, folder)):
        count += 1
print(count)
count=0
for folder in os.listdir(dest):
    for img in os.listdir(os.path.join(dest, folder)):
        count+=1
print(count)