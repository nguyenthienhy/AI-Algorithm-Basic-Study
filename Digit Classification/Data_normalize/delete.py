import os

maxTake = 2000

list_del = ["b_u" , "d_u" , "e_u" , "g_u" , "h_u" , "j_u" 
			, "k_u" , "l_u" , "m_u" 
			, "q_u" , "u_u" , "v_u" 
			, "w_u" , "x_u" , "z_u"]

path = "C:/Users/hydon/OneDrive/Máy tính/Data_normalize"

def count_file(path):
	count = 0
	for entry in os.listdir(path):
		if os.path.isfile((os.path.join(path, entry))):
			count += 1
	return count

def del_file(path):
	count = count_file(path)
	for entry in os.listdir(path):
		if count == 2000:
			break
		if os.path.isfile((os.path.join(path, entry))):
			count -= 1
			os.remove(path + "/" + entry)

for i in list_del:
	del_file(path + "/" + i)