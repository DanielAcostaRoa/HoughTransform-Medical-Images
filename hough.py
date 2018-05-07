from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float
import sklearn

def proceso(img,a,b,c):
    im=img<c
    im=255*im
    im=np.array(im,dtype=np.uint8)
    edges2 = cv2.Canny(im,a,b)
    return edges2

def tranformada_Hough(im,rho,theta):
    cont={}
    for i in rho:
        cont[(i)]=np.zeros((im.shape[0],im.shape[1]))
    for i in range(im.shape[0]):
        print(i)
        for j in range(im.shape[1]):
            if im[i][j]>0:
                for r in rho:
                    for th in theta:
                        a = int(i - r*np.cos(th))
                        b = int(j - r*np.sin(th))
                        if a>=0 and a<im.shape[0] and b>=0 and b<im.shape[1]:
                            cont[(r)][a][b]= 1 + cont[(r)][a][b]
    return cont

def func(im,r):
    ht=proceso(im,80,120,80)
    thetas = np.deg2rad(np.arange(0, 360.0, 6))
    cont={}
    cont=tranformada_Hough(ht,r,thetas)
    return cont

def carga():
    L=[]
    for i in range(6):
        name=str(1+i)+'.png'
        img=np.array(cv2.imread(name,1)[:, :, 1])
        L.append(img)
    return L

def hist_3d(pen):
    X = np.arange(0,512, 1.0)
    Y = np.arange(0, 512, 1.0)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')   
    surf = ax.plot_surface(X, Y, pen, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()
    return surf

def resp(imagen, tr_ho, radio, name):
    mask=np.zeros((imagen.shape[0],imagen.shape[1]))
    num=0
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            if tr_ho[i][j]!=0:
                if mask[i][j]==0:
                    num=num+1
                    procesa(i,j,mask,tr_ho,imagen,num)
    centros={}
    for i in range(num):
        centros[(i+1)]=np.array((1000,-1,1000,-1))
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            if mask[i][j]!=0:
                if i<centros[mask[i][j]][0]:
                    centros[mask[i][j]][0]=i
                if i>centros[int((mask[i][j]))][1]:
                    centros[int((mask[i][j]))][1]=i
                if j<centros[int((mask[i][j]))][2]:
                    centros[int((mask[i][j]))][2]=j
                if j>centros[int((mask[i][j]))][3]:
                    centros[int((mask[i][j]))][3]=j
    l=[]
    for i in range(len(centros)):
        a=(centros[i+1][0]+centros[i+1][1])/2.0
        b=(centros[i+1][2]+centros[i+1][3])/2.0
        l.append((a,b))
    
    phi=np.arange(360)*np.pi/180.0
    for i in range(len(l)):
        for j in range(len(phi)):
            imagen[int(l[i][0]-radio*np.sin(phi[j]))][int(l[i][1]+radio*np.cos(phi[j]))]=255
    cv2.imwrite(name,imagen)
    return num
    
def procesa(i,j,mask,tr_ho,imagen,num):
    mask[i][j]=num
    if i+1<imagen.shape[0]:
        if tr_ho[i+1][j]!=0:
            if mask[i+1][j]==0:
                procesa(i+1,j,mask,tr_ho,imagen,num)
    if i-1>=0:
        if tr_ho[i-1][j]!=0:
            if mask[i-1][j]==0:
                procesa(i-1,j,mask,tr_ho,imagen,num)
    if j+1<imagen.shape[1]:
        if tr_ho[i][j+1]!=0:
            if mask[i][j+1]==0:
                procesa(i,j+1,mask,tr_ho,imagen,num)
    if j-1>=0:
        if tr_ho[i][j-1]!=0:
            if mask[i][j-1]==0:
                procesa(i,j-1,mask,tr_ho,imagen,num)
    if i+1<imagen.shape[0] and j+1<imagen.shape[1]:
        if tr_ho[i+1][j+1]!=0:
            if mask[i+1][j+1]==0:
                procesa(i+1,j+1,mask,tr_ho,imagen,num)
    if i+1<imagen.shape[0] and j-1>=0:
        if tr_ho[i+1][j-1]!=0:
            if mask[i+1][j-1]==0:
                procesa(i+1,j-1,mask,tr_ho,imagen,num)
    if j+1<imagen.shape[1] and i-1>=0:
        if tr_ho[i-1][j+1]!=0:
            if mask[i-1][j+1]==0:
                procesa(i-1,j+1,mask,tr_ho,imagen,num)
    if j-1>=0 and i-1>=0:
        if tr_ho[i-1][j-1]!=0:
            if mask[i-1][j-1]==0:
                procesa(i-1,j-1,mask,tr_ho,imagen,num)

L=carga()

r=np.arange(14,28,1)
cont=func(L[3],r)

pen=np.zeros((L[0].shape[0],L[0].shape[1]))
#pen=np.ones((L[0].shape[0],L[0].shape[1]))

for i in cont:
    pen=pen+cont[i]
#pen=np.power(pen,1.0/10.0)*2
pen=np.power(pen/10.0,3)
#cv2.imwrite("6_hougt.png",pen)
cv2.imwrite("4_promedio.png",pen)

a=cv2.GaussianBlur(pen, (5,5),15,15)
cv2.imwrite("4_gauss.png",pen)

for i in range(512):
    for j in range(512):
        if a[i][j]<100:
            a[i][j]=0
cv2.imwrite("4_blur.png",a)
r=resp(L[3],a,30,"4_deteccion.png")









