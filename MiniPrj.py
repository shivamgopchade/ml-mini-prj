import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt

def initialize_params(dims):
    params={}
    np.random.seed(3000)
    L=len(dims)
    for i in range(L-1):
        params['w'+str(i)]=np.random.randn(dims[i+1],dims[i])*0.001
        params['b'+str(i)]=np.zeros(shape=(dims[i+1],1))
    return params

def step_up(x,w,b,activation):
    z=np.dot(w,x)+b
    a=0
    error=1e-8
    if(activation=='sigmoid'):
            a=1/(1+np.exp(-z))+error
    else:
        a=np.maximum(z,float(0))
        #print("a:",a.shape)
    return a,[x,w,b,z]

def step_down(dZ,c,activation,_lambda=0.0000001):
    x,w,b,z=c
    m=dZ.shape[1]
    dW=np.dot(dZ,x.T)/m+(_lambda*w)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m

    if(activation=='sigmoid'):
        dZ*=(z*(1-z)).astype(float)
    else:
        dZ*=(z>float(0)).astype(float)

    dZ_prev=np.dot(w.T,dZ)

    return dZ_prev,dW,db

def forward_prop(X_train,params):
    layers=int(np.ceil(len(params)/2))
    x=X_train
    cach=[]
    for i in range(layers-1):
        #print(i)
        w=params['w'+str(i)]
        b=params['b'+str(i)]
        x,c=step_up(x,w,b,'relu')
        cach.append(c)
    w=params['w'+str(layers-1)]
    b=params['b'+str(layers-1)]
    x,c=step_up(x,w,b,'sigmoid')

    cach.append(c)
    #print("exited fp")
    return x,cach

def backward_prop(y_pred,Y_train,payload):
    gradients={}
    L=len(payload)
    error=1e-8
    #print("y:",y_pred)
    dZ=-np.divide(Y_train,y_pred+error)+np.divide((1-Y_train),(1-y_pred+error))
    c=payload[L-1]
    gradients['dZ'+str(L-1)],gradients['dw'+str(L)],gradients['db'+str(L)]=step_down(dZ,c,'sigmoid')

    for i in reversed(range(L-1)):
        #print(i)
        c=payload[i]
        dZ=gradients['dZ'+str(i+1)]
        gradients['dZ' + str(i)], gradients['dw' + str(i+1)], gradients['db' + str(i+1)] = step_down(dZ, c, 'relu')

    return gradients

def compute_cost(y_pred,Y_train,params,_lambda=0.0000001):
    m=y_pred.shape[1]
    error=1e-8
    j=np.sum(-1*np.multiply(Y_train,np.log(y_pred+error))+np.multiply(1-Y_train,np.log(1-y_pred+error)))
    sum=0
    L = int(np.ceil(len(params) / 2))
    for i in range(L):
        w=params['w'+str(i)]
        sum+=np.sum(np.square(w))
    j+=sum
    j/=m
    j=np.squeeze(j)
    return j

def accuracy(y_pred,y_actual,th):
    y_hat=(y_pred>th).astype('float64')
    #print(y_hat.shape,y_actual.shape)
    a=np.sum(y_hat[0]==y_actual)
    #print(y_pred.shape,y_actual.shape)
    return 100*a/y_pred.shape[1]

def optimize_params(params,gradients,a):
    L=int(np.ceil(len(params)/2))
    #print("L:",L)
    #print("params",params['w0'])
    for i in range(L):
        params['w'+str(i)]=params['w'+str(i)]-a*gradients['dw'+str(i+1)]
        params['b' + str(i)] = params['b' + str(i)] - a * gradients['db' + str(i+1)]
    #print("params size:",len(params))
    return params

def normalize(f,l):
    mean=np.mean(f,axis=1)
    std=np.std(f,axis=1)
    print(mean)
    f_norm=(f-mean)/std

    mean=np.mean(l,axis=1)
    std=np.std(l,axis=1)
    l_norm=(f-mean)/std

    return [f_norm,l_norm]

def thresholding(y,T):
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i][j]>=T:
                y[i][j]=255
            else:
                y[i][j]=0
    return y

def NN(dims,iter=100,a=0.00001):
    f=pd.read_csv('features.csv')
    l=pd.read_csv('labels.csv')
    f,l=normalize(f,l)
    X_train=np.array(f.iloc[:,1:9],dtype='uint8')
    Y_train=np.array(l.iloc[:,1:9],dtype='uint8')
    X_val=np.array(f.iloc[:,9:10],dtype='uint8')
    Y_val=np.array(l.iloc[:,9:10],dtype='uint8')
    X_test=np.array(f.iloc[:,10:],dtype='uint8')
    Y_test=np.array(l.iloc[:,10:],dtype='uint8')
    params=initialize_params(dims)
    print("Y_train:",Y_train.shape)
    #print(X_train.shape)
    #return
    cost_train=[]
    for i in range(iter):

        # print("i",i)
        y_pred,payload=forward_prop(X_train,params)
        c1=compute_cost(y_pred,Y_train,params)
        print(str(i)+" cost:",c1)
        cost_train.append(c1)
        gradients=backward_prop(y_pred,Y_train,payload)

        params=optimize_params(params,gradients,a)

    return [params,X_test,Y_test,X_val,Y_val]

def find_metric(y_pred,y_actual,th):
    y=(y_pred>th).astype(int)
    tp=np.sum(np.logical_and(y==y_actual,y==1))
    tn=np.sum(np.logical_and(y==y_actual, y==0))
    fp=np.sum(np.logical_and(y==1, y!=y_actual))
    fn=np.sum(np.logical_and(y==0 ,y!=y_actual))
    confusion_mat=np.array([[tp,fp],[fn,tn]])
    #print(confusion_mat)
    sensitivity=tp/(tp+fn)
    specificity=tn/(tn+fp)
    return [sensitivity,specificity,confusion_mat]


def BT19ECE032_linreg(arch):
    [params, X_test, Y_test, X_val, Y_val]=NN(arch,75,0.00000001)

    y1,t1=forward_prop(X_val,params)
    y2,t2=forward_prop(X_test,params)
    print("validation accuracy:",accuracy(y1,Y_val,0.5))
    print("test accuracy",accuracy(y2,Y_test,0.5))
    img=cv2.imread('cameraman.png')
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img=cv2.resize(img,(256,256))
    cv2.imshow('original img',img)
    cv2.waitKey(0)
    img_arr=np.array(img).reshape((-1,1))
    y,t=forward_prop(img_arr,params)
    y=np.reshape(y,(256,256))
    y=thresholding(y,0.5)
    print(y)
    cv2.imshow('predicted edge',y)
    cv2.waitKey(0)
    # x=[]
    # y=[]
    # for i in [0.5,0.6,0.7,0.8,0.9,1]:
    #     s1,s2,_=find_metric(y2,Y_test,i)
    #     x.append(s1)
    #     y.append(1-s2)
    # plt.plot(x,y)
    # plt.title('ROC curve')
    # plt.xlabel('FPR')
    # plt.ylabel('TPR')
    # plt.show()
    return
BT19ECE032_linreg([65536,1000,1000,65536])