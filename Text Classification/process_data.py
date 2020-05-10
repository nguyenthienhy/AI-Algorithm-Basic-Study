from __future__ import division, print_function, unicode_literals
import re
from pathlib import Path

def readData(path):
    f = open(path.__str__(), "r", encoding="latin1")
    contents = f.readlines()
    contents = list(contents)
    return contents

# remove "\n"
def remove_N(contents):
    for line in contents:
        if line == "\n":
            contents.remove(line)
    comments = []
    for line in contents:
        if line.endswith('\n'):
            newline = line.replace('\n' , '')
            comments.append(newline)
    return comments

def remove_mark_sentences(sentence):# Loại bỏ dấu câu trong một câu
    s = sentence
    s = re.sub(r'[^\w\s]', '', s)
    return s

def remove_mark_all_sentences(sentences): # loại bỏ dấu câu trong một tập văn bản
    res = []
    for cmt in sentences:
        cmt = remove_mark_sentences(cmt)
        res.append(cmt)
    return res

# chuyển sang dạng chữ thường
def convert_to_lower_case(comments): # chuyển hết 1 tập hợp các câu về chữ thường
    comments_lower = []
    for cmt in comments:
        cmt_lower = cmt.lower()
        comments_lower.append(cmt_lower)
    return comments_lower

def erase_old_data(FOLDER , file_name):
    f = open((Path(FOLDER) / file_name).__str__(), 'r+')
    f.truncate(0)

# write to out_put_folder
def write_to_out_put_folder(path , comment):
    for cmt in comment:
        with open(path.__str__() , "a" , encoding="utf-8-sig") as f:
            f.write(cmt + "\n")
