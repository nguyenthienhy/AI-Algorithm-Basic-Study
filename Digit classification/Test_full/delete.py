import os

maxTake = 2000

list_del = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
           "A", "B", "C", "D",
           "E", "F", "G", "H",
           "I", "J", "K", "L",
           "M", "N", "O", "P",
           "Q", "R", "S", "T",
           "U", "V", "W", "X",
           "Y", "Z"]


path = "Data_normalize"

def count_file(path):
	count = 0
	for entry in os.listdir(path):
		if os.path.isfile((os.path.join(path, entry))):
			count += 1
	return count

def del_file(path):
	count = count_file(path)
	for entry in os.listdir(path):
		if count == 200:
			break
		if os.path.isfile((os.path.join(path, entry))):
			count -= 1
			os.remove(path + "/" + entry)

for i in list_del:
	del_file(path + "/" + i)