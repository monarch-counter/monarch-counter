'''Iterate through csv files with labels 
and shifts files that have been labeled 
in a folder named labeled'''

import os
import shutil
import pandas as pd
import argparse, sys

parser = argparse.ArgumentParser()

parser.add_argument('--source', help='source images folder -- should be in_imgs')
parser.add_argument('--dest', help='destination for labeled images - should be labeled')
parser.add_argument('--labels', help='folder where labeled csv are store. should be /data/label_files.')

args = parser.parse_args()

files = os.listdir(args.labels)
files_processed = []

for fi in files:
    if ".csv" in fi:
        with open(args.labels + '/' + fi) as f:
            df_temp = pd.read_csv(f, header=None, names=['label', 'x', 'y', 'file', 'a', 'b'])
            files_processed.extend(list(df_temp.file.unique()))

del_count = 0
for f in files_processed:
    if os.path.isfile(args.source + '/' + f):
        try:
            shutil.move(args.source + '/' + f, args.dest)
        except shutil.Error:
            del_count += 1
            os.remove(args.source + '/' + f)

print('Finished. {} files overlapped'.format(del_count))
