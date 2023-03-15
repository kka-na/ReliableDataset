import glob
from tqdm import tqdm
for file in tqdm(glob.glob("/home/kana/Documents/Dataset/TS/2DOD/data/*.txt")):
    with open(file, 'r+') as f:
      lines = f.readlines()  # read all lines into a list
      f.seek(0)  # move the file pointer to the beginning of the file
      for line in lines:
          w, h = 1920, 1080  # replace with your desired width and height values
          line = line.strip()  # remove any trailing whitespace
          line_with_wh = f"{line} {w} {h}\n"  # add the "w, h" value to the line
          f.write(line_with_wh)  # write the modified line back to the file
      f.truncate()