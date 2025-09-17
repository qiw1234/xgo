import json
import numpy as np
import glob
import os

files = "output_panda_fixed_gripper"
motion_full =[]
name_full = []
for i, file in enumerate(os.listdir(files)):
    name_full.append(file.split('.')[0])
    motion = np.loadtxt(os.path.join(files,file), delimiter=',')
    motion_full.append(motion)

for j in range(len(motion_full)):
    json_data={
        'frame_duration':1/50,
        'frames':motion_full[j].tolist()
    }
    with open('output_panda_fixed_gripper_json/'+name_full[j]+'.json', 'w') as f:
        json.dump(json_data, f, indent=4)

# files = 'output_panda_fixed_gripper'
# file = "panda_spacetrot.txt"
# name = file.split('.')[0]
# motion = np.loadtxt(os.path.join(files,file), delimiter=',')
# json_data={
#     'frame_duration':1/150,
#     'frames':motion.tolist()
# }
# with open(files+'_json/'+name+'.json', 'w') as f:
#     json.dump(json_data, f, indent=4)
