import glob 

path = "/home/kana/Documents/Dataset/WAYMO/data/"
l_file = "/home/kana/Documents/Dataset/WAYMO/cleaning/iter1/data.txt"
f = open(l_file, 'w')
for label in glob.glob(path+"*.png"):
  f.write(str(label)+"\n")
f.close() 

