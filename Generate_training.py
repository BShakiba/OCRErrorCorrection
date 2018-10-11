from os.path import join, exists
from os import listdir, makedirs
import json
from multiprocessing import Pool
import re
import argparse
import os
import linecache


input_info = '/home/bs643/PycharmProjects/OCRErrorCorrection/OCRErrorCorrection/out_merged/pair.x.info'
input = '/home/bs643/PycharmProjects/OCRErrorCorrection/OCRErrorCorrection/out_merged/pair.x'
output_true = '/home/bs643/PycharmProjects/OCRErrorCorrection/OCRErrorCorrection/out_merged/pair1.mt'
input_lines= []
output_lines = []
with open(input_info,'r') as infile:
    for line in infile:
        a = line.split("\t")
        if (int(a[6])> 0):
            input_lines.append(linecache.getline(input,int(a[1])+1).strip("\n"))
            if (int(a[5]) > 0):
                manualTrancriptions = linecache.getline(output_true, int(a[8]) + 1)
                first_mt = manualTrancriptions.split("\t")[0]
                output_lines.append(first_mt)
            else:
                manualTrancriptions = linecache.getline(output_true, int(a[7]) + 1)
                first_mt = manualTrancriptions.split("\t")[0]
                output_lines.append(first_mt)