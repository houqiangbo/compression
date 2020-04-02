import sys
import numpy as np
import re
from skimage import io
import numpy as np
import pywt,cv2
from PIL import Image


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
        90:"CGGG", 91:"GAAA", 92:"GAAC", 93:"GAAG", 94:"GACA", 95:"GACC", 96:"GACG", 97:"GAGA", 98: "GAGC", 99:"GAGG",  
        100:"GCAG"}


Index={"Y0":"AAT","Y10":"AAC","Y11":"AAG","Y12":"ATA","Y20":"TGA","Y21":"ATC","Y22":"ATG",
        "Y30":"ACA","Y31":"ACT","Y32":"ACC","Y40":"ACG","Y41":"AGA","Y42":"AGT", 
        "U0":"AGC","U10":"AGG","U11":"TGT","U12":"TAT","U20":"TAC","U21":"TAG","U22":"TTA", 
        "V0":"TGC","V10":"TTC","V11":"TTG","V12":"TCA","V20":"TCT","V21":"TCC","V22":"TCG", 
        "V30":"GTT","U30":"GCA","U31":"GCC","U32":"GCG","V31":"GCT","V32":"GGG","U40":"GCC",
        "U41":"GTA","U42":"GAG"}


Row_num = {0:"TA",1:"TC",2:"TG",3:"AT",4:"AC",5:"AG",6:"CG",7:"CT",8:"GT",9:"GC"}

size_index = {"zero":"GAT", "one":"GAC", "y50":"GAA", "y51":"GAG", "v30":"CAC", 
              "v31":"CAG","v40":"CAT", "v41":"CAA", "v50":"CTA", "v51":"CTT", 
              "u30":"CTC", "u31":"CTG","u40":"CCC", "u41":"CCG", "u50":"CCA", 
              "u51":"CCT", "y52":"AAT","v32":"AAC", "v42":"AAG", "v52":"ATA", 
                "u32":"TGA", "u42":"ATC", "u52":"ATG"}

def invert_dict(dict1):
    dict2 = dict()
    for key in dict1:
        dict2.update({dict1[key] : key})
    return dict2

invert_ATCG = invert_dict(ATCG)
invert_Index = invert_dict(Index)
invert_Row_num = invert_dict(Row_num)
invert_size_index = invert_dict(size_index)
    


max_value = {}
size_value = {}
element = {}

with open(sys.argv[1]) as f:
    for seq in f:
        if 'GGAAGGAAGG' in seq:
            max_value_split = seq.rstrip('\n').split('GGAAGGAAGG')
            num = ''
            for i in range(len(max_value_split[1])//2):
                num += str(invert_Row_num[max_value_split[1][i*2:(i+1)*2]])
            #print(num)
            max_value.update({invert_Index[max_value_split[0]]:int(num)})
        elif "AAGGAAGGAA" in seq:
            size_value_split = seq.rstrip('\n').split('AAGGAAGGAA')
            num = ''
            for i in range(len(size_value_split[1])//2):
                num += str(invert_Row_num[size_value_split[1][i*2:(i+1)*2]])
            size_value.update({invert_size_index[size_value_split[0]]:int(num)})
        else:
            element.update({invert_Index[seq[0:3]] + '|' + str(invert_Row_num[seq[3:5]]) + 
            str(invert_Row_num[seq[5:7]]) + str(invert_Row_num[seq[7:9]]) + '_' + 
            str(invert_Row_num[seq[9:11]]) + str(invert_Row_num[seq[11:13]]):seq.rstrip('\n')[13:]})
#print(size_value)


def combine(Y_0):
    num = set()
    seq = ''
    for key,value in Y_0.items():
        num.add(key.split('_')[0])
        seq += value
    #print(seq)
    #print(len(num))
    seq = seq.replace('TAA','TATATATATATATATATATA')
    #return [seq,len(num)]
    element = seq.split('T')
    number = []
    for i in element[1:]:
        if i.startswith('C'):
            number.append(invert_dict(ATCG)[i[1:]]*1)
        if i.startswith('G'):   
            number.append(invert_dict(ATCG)[i[1:]]*(-1))
        if i == 'A':
            number.append(0)
    return np.array(number).reshape(len(num),len(number)//len(num)) / 100

#print(combine(element))
#print(np.zeros((size_value['u40'], size_value['u41'], size_value['u42'], 3)))
#print(element.keys())

def reb_num(a):
    y0 = dict()
    for key in element:
        if a in key:
            y0.update({key : element[key]})
    return (combine(y0) * max_value[a])
#print(max_value)

m_Y = [[[0] * 2 for _ in range(3) ] for _ in range(6)]
#print(np.shape(m_Y))
#m_Y = np.array(list)
#print(reb_num("Y0").T.shape)

m_Y[0] = reb_num("Y0").T
m_Y[1][0] = reb_num("Y10")
m_Y[1][1] = reb_num("Y11")
m_Y[1][2] = reb_num("Y12")
m_Y[2][0] = reb_num("Y20")
m_Y[2][1] = reb_num("Y21")
m_Y[2][2] = reb_num("Y22")
m_Y[3][0] = reb_num("Y30")
m_Y[3][1] = reb_num("Y31")
m_Y[3][2] = reb_num("Y32")
m_Y[4][0] = reb_num("Y40")
m_Y[4][1] = reb_num("Y41")
m_Y[4][2] = reb_num("Y42")
m_Y[5] = np.zeros((size_value['y50'], size_value['y51'], size_value['y52']))
#print(np.shape(m_Y[0][1]))

m_U = [[[0] * 2 for _ in range(3) ] for _ in range(6)]
m_U[0] = reb_num("U0").T
m_U[1][0] = reb_num("U10")
m_U[1][1] = reb_num("U11")
m_U[1][2] = reb_num("U12")
m_U[2][0] = reb_num("U20")
m_U[2][1] = reb_num("U21")
m_U[2][2] = reb_num("U22")
m_U[3] = np.zeros((size_value['u30'], size_value['u31'], size_value['u32']))
m_U[4] = np.zeros((size_value['u40'], size_value['u41'], size_value['u42']))
m_U[5] = np.zeros((size_value['u50'], size_value['u51'], size_value['u52']))

#print(m_Y.shape)
m_V = [[[0] * 2 for _ in range(3) ] for _ in range(6)]
m_V[0] = reb_num("V0").T
m_V[1][0] = reb_num("V10")
m_V[1][1] = reb_num("V11")
m_V[1][2] = reb_num("V12")
m_V[2][0] = reb_num("V20")
m_V[2][1] = reb_num("V21")
m_V[2][2] = reb_num("V22")
m_V[3] = np.zeros((size_value['v30'], size_value['v31'], size_value['v32']))
m_V[4] = np.zeros((size_value['v40'], size_value['v41'], size_value['v42']))
m_V[5] = np.zeros((size_value['v50'], size_value['v51'], size_value['v52']))

#re_Y=pywt.waverec2(m_Y,'db1')
#print(np.shape(m_Y), np.shape(m_Y[5]))
#print(np.shape(m_U))
#print(np.shape(m_V))

re_Y=pywt.waverec2(m_Y,'db1')
re_V=pywt.waverec2(m_V,'db1')
re_U=pywt.waverec2(m_U,'db1')


R_n = np.around(re_Y + 1.402*re_V, 1)
G_n = np.around(re_Y - 0.34414*re_U -0.71414*re_V, 1)
B_n = np.around(re_Y + 1.772*re_U, 1)


image = np.zeros((size_value["one"] + 1,size_value["zero"], 3))

image[:,:,0] = R_n
image[:,:,1] = G_n
image[:,:,2] = B_n
cv2.imwrite(sys.argv[1]+".png", image)

