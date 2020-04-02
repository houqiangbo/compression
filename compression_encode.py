# -*- coding: utf-8 -*-
import sys
import re
import os
from skimage import io
import numpy as np
import pywt,cv2
from PIL import Image

image = io.imread(sys.argv[1])
im=Image.open(sys.argv[1])
size = im.size

B = image[:, :, 0]
G = image[:, :, 1]
R = image[:, :, 2]

#YUV的转化
Y = np.around(0.299*R + 0.587*G + 0.114*B, 1)
U = np.around(-0.1687*R -0.3313*G + 0.5*B, 1)
V = np.around(0.5*R -0.4187*G -0.0813*B, 1)
#
#小波变换
coeffs_Y = pywt.wavedec2(Y, 'haar', level = 5)
coeffs_U = pywt.wavedec2(U, 'haar', level = 5)
coeffs_V = pywt.wavedec2(V, 'haar', level = 5)

#print(np.shape(coeffs_Y[1][0]))
#print(np.shape(coeffs_Y[5]))

y50 = np.shape(coeffs_Y[5])[0]
y51 = np.shape(coeffs_Y[5])[1]
y52 = np.shape(coeffs_Y[5])[2]
#print(y50,y51,y52)

v30 = np.shape(coeffs_V[3])[0]
v31 = np.shape(coeffs_V[3])[1]
v32 = np.shape(coeffs_V[3])[2]
v40 = np.shape(coeffs_V[4])[0]
v41 = np.shape(coeffs_V[4])[1]
v42 = np.shape(coeffs_V[4])[2]
v50 = np.shape(coeffs_V[5])[0]
v51 = np.shape(coeffs_V[5])[1]
v52 = np.shape(coeffs_V[5])[2]

u30 = np.shape(coeffs_U[3])[0]
u31 = np.shape(coeffs_U[3])[1]
u32 = np.shape(coeffs_U[3])[2]
u40 = np.shape(coeffs_U[4])[0]
u41 = np.shape(coeffs_U[4])[1]
u42 = np.shape(coeffs_U[4])[2]
u50 = np.shape(coeffs_U[5])[0]
u51 = np.shape(coeffs_U[5])[1]
u52 = np.shape(coeffs_U[5])[2]


def func0(name,coef):
	for i in range(1,6):
		for j in range(3):
			np.savetxt(name + str(i) + str(j) + ".txt", coef[i][j], delimiter = '\t', fmt = '%.0f')
	if np.shape(coef[0][0])[0]==np.shape(coef[1][0])[0]:
		for i in range(np.shape(coef[0])[0]):
			np.savetxt(name + str(0) + str(i) + ".txt", coef[0][i], delimiter = '\t', fmt = '%.0f')
	if np.shape(coef[0][0])[0]==np.shape(coef[1][0])[1]:
		for i in range(np.shape(coef[0])[0]):
			np.savetxt(name + str(0) + str(i) + ".txt", coef[0][i], delimiter = '\t', fmt = '%.0f')

func0("Y",coeffs_Y)
func0("U",coeffs_U)
func0("V",coeffs_V)


def combine0(name):
	list1 = []
	for i in range(np.shape(coeffs_V[1][1])[0]):
		f = open(name+ str(i) + ".txt")
		for line in f:
			list1.append(eval(line.strip('\n')))
	#print (list1)
	a = np.array(list1)
	b = a.reshape(np.shape(coeffs_V[1][1])).T
	#print (b)
	return np.savetxt(name + ".txt", b ,delimiter = '\t', fmt = '%.0f')
#
combine0("Y0")
combine0("U0")
combine0("V0")
#
def tran(name,num):
	return np.loadtxt(name+str(num)+".txt")
#
arr_list = [
			tran("Y",0), tran("Y",10), tran("Y",11), tran("Y",12), tran("Y",20), tran("Y",21), tran("Y",22), 
			tran("Y",30), tran("Y",31),tran("Y",32), tran("Y",40), tran("Y",41), tran("Y",42), 
			tran("U",0), tran("U",10), tran("U",11), tran("U",12), tran("U",20), tran("U",21), tran("U",22),
			tran("V",0), tran("V",10), tran("V",11), tran("V",12), tran("V",20), tran("V",21), tran("V",22),	
			]

index_list = ['Y0', 'Y10', 'Y11', 'Y12', 'Y20', 'Y21', 'Y22', 'Y30', 'Y31', 'Y32',
			  'Y40', 'Y41', 'Y42', 'U0', 'U10', 'U11', 'U12', 'U20', 'U21', 'U22',
			   'V0', 'V10', 'V11', 'V12','V20', 'V21', 'V22']

def find_max(arr):
	a= np.abs(arr).max()
	if a==0:
		return a+1
	else:
		return a

max_list = []
for arr in arr_list:
	max_list.append(find_max(arr))

div_max = []
for i,_ in enumerate(max_list):
	div_max.append(np.around(arr_list[i]/max_list[i],2))

new_list = []
for div in div_max:
	new_list.append((div*100).astype(np.int))

ATCG = { 
	     0:"TA",    1:"AA",    2:"AC",    3:"AG",    4:"CA",    5:"CC",    6:"CG",    7:"GA",    8:"GC",    9:"GG",
		10:"AAA",  11:"AAC",  12:"AAG",  13:"ACA",  14:"ACC",  15:"ACG",  16:"AGA",  17:"AGC",  18:"AGG",  19:"CAA",
		20:"CAC",  21:"CAG",  22:"CCA",  23:"GCCA", 24:"CCG",  25:"CGA",  26:"CGC",  27:"CGG",  28:"GAA",  29:"GAC",
		30:"GAG",  31:"GCA",  32:"GCC",  33:"GCG",  34:"GGA",  35:"GGC",  36:"GCCG", 37:"GCAA", 38:"AAAC", 39:"AAAG",
		40:"AACA", 41:"AACC", 42:"AACG", 43:"AAGA", 44:"AAGC", 45:"AAGG", 46:"ACAA", 47:"ACAC", 48:"ACAG", 49:"ACCA",
		50:"ACCC", 51:"ACCG", 52:"ACGA", 53:"ACGC", 54:"ACGG", 55:"AGAA", 56:"AGAC", 57:"AGAG", 58:"AGCA", 59:"AGCC",
		60:"AGCG", 61:"AGGA", 62:"AGGC", 63:"AGGG", 64:"CAAA", 65:"CAAC", 66:"CAAG", 67:"CACA", 68:"CACC", 69:"CACG",
		70:"CAGA", 71:"CAGC", 72:"CAGG", 73:"CCAA", 74:"CCAC", 75:"CCAG", 76:"GCGA", 77:"GCAC", 78:"GCGC", 79:"CCGA",
		80:"CCGC", 81:"CCGG", 82:"CGAA", 83:"CGAC", 84:"CGAG", 85:"CGCA", 86:"CGCC", 87:"CGCG", 88:"CGGA", 89:"CGGC",
		90:"CGGG", 91:"GAAA", 92:"GAAC", 93:"GAAG", 94:"GACA", 95:"GACC", 96:"GACG", 97:"GAGA", 98:"GAGC", 99:"GAGG",
		100:"GCAG"
		}

Row_num = {
		   0:"TA",1:"TC",2:"TG",3:"AT",4:"AC",5:"AG",6:"CG",7:"CT",8:"GT",9:"GC"	
			}

index ={
		 "Y0":"AAT","Y10":"AAC","Y11":"AAG","Y12":"ATA","Y20":"TGA","Y21":"ATC","Y22":"ATG",
		"Y30":"ACA","Y31":"ACT","Y32":"ACC","Y40":"ACG","Y41":"AGA","Y42":"AGT", "U0":"AGC",
		"U10":"AGG","U11":"TGT","U12":"TAT","U20":"TAC","U21":"TAG","U22":"TTA", "V0":"TGC",
		"V10":"TTC","V11":"TTG","V12":"TCA","V20":"TCT","V21":"TCC","V22":"TCG", "V30":"GTT",
		"U30":"GCA","U31": "GCC","U32":"GCG","V31":"GCT","V32":"GGG","U40":"GCC","U41":"GTA",
		"U42":"GAG"
	}

size_index = {"zero":"GAT", "one":"GAC", "y50":"GAA", "y51":"GAG", "v30":"CAC", "v31":"CAG",
			 "v40":"CAT", "v41":"CAA", "v50":"CTA", "v51":"CTT", "u30":"CTC", "u31":"CTG",
			 "u40":"CCC", "u41":"CCG", "u50":"CCA", "u51":"CCT", "y52":"AAT","v32":"AAC",
			 "v42":"AAG", "v52":"ATA", "u32":"TGA", "u42":"ATC", "u52":"ATG"}

def size_st(a,b):
	str1 = size_index[b]+"AAGGAAGGAA"
	for i in list(str(int(a))):
		str1+= Row_num[int(i)]
	print (str1)

def func(a,i,j):
	if a[i][j]>0:
		return "TC"+ATCG[abs(a[i][j])]
	elif a[i][j]<0:
		return "TG"+ATCG[abs(a[i][j])]
	else:
		return ATCG[a[i][j]]

def func1(str1,n,str2):
	m=0
	le=len(str1)//n
	for i in range(le+2):
		if i==0:
			continue
		else:
			print (str2+Row_num[i//10]+Row_num[i%10]+str1[m:i*n])
		m=i*n

def func3(a,b):
	rows,cols = a.shape
	for i in range(rows):
		list1 = []
		for j in range(cols):
			list1.append(func(a,i,j))
		str1 = "".join(list1)
		str1 = str1.replace("TATATATATATATATATATA","TAA")
		func1(str1,84,index[b]+Row_num[(i+1)//100]+Row_num[(i+1)%100//10]+Row_num[(i+1)%100%10])

def spl_max(a,b):
	str1 = index[b]+"GGAAGGAAGG"
	for i in list(str(int(a))):
		str1+= Row_num[int(i)]
	print (str1)

#print (np.shape(arr_list[0]))

savedStdout = sys.stdout 
f1 = open(sys.argv[1]+'.seq', 'w+')
sys.stdout = f1
for i in range(len(arr_list)):
	func3(new_list[i],index_list[i])
	spl_max(max_list[i],index_list[i])


#add size information of picture
size_st(size[0],"zero")
size_st(size[1],"one")
size_st(y50,"y50")
size_st(y51,"y51")
size_st(y52,"y52")
size_st(v30,"v30")
size_st(v31,"v31")
size_st(v32,"v32")
size_st(v40,"v40")
size_st(v41,"v41")
size_st(v42,"v42")
size_st(v50,"v50")
size_st(v51,"v51")
size_st(v52,"v52")
size_st(u30,"u30")
size_st(u31,"u31")
size_st(u32,"u32")
size_st(u40,"u40")
size_st(u41,"u41")
size_st(u42,"u42")
size_st(u50,"u50")
size_st(u51,"u51")
size_st(u52,"u52")

sys.stdout = savedStdout
sys.stdout.close()
os.system('rm -rf *.txt')

