import numpy as np
from pprint import pprint
import random

class neural:
    def __init__(self,input_neuro,hidden_neuro,output_neuro):
        self.weight_1st=np.random.rand(hidden_neuro,len(input_neuro))
        self.weight_2nd=np.random.rand(output_neuro,hidden_neuro)
        self.weights=np.array([self.weight_1st,self.weight_2nd])
        self.inputs=[]

    def activator(self,x):
        return 1/(1+np.exp(-x))

    def feedforward(self,input_neuro):
        hidden_input=self.activator(np.dot(self.weight_1st,input_neuro))
        output=self.activator(np.dot(self.weight_2nd,hidden_input))
        print(output)

    def disigmoid(self,x):
        return x*(1-x)

    def mult(self,a,b):
        fin=[]
        for i in range(len(a)):
            fin.append(a[i]*b[i])

        fin=np.array(fin)
        return fin

    def backPropogationModel(self,input_neuro,target):
        input_neuro=np.array(input_neuro).reshape(len(input_neuro),1)
        hidden_input=self.activator(np.dot(self.weights[0],input_neuro))
        output=self.activator(np.dot(self.weights[1],hidden_input))
        self.inputs=[input_neuro,hidden_input,output]

        error=target-output
        for i in range(len(self.weights)-1,-1,-1):#loop from 0 to (no of hidden layers + 1)
            adjusted_weight=np.dot((error*self.disigmoid(self.inputs[i+1])),self.inputs[i].T)
            self.weights[i]+=adjusted_weight
            error=np.dot(self.weights[i].T,error)

def data_generator(data,target):
    index=random.randint(0,len(target)-1)
    return data[index],target[index]

if __name__=="__main__":
    data=[[1,1],[1,0],[0,1],[0,0]]
    target=[0,1,1,0]

    obj=neural(data[0],10,1)
    epochs=2000

    for e in range(epochs):
        d,t=data_generator(data,target)
        obj.backPropogationModel(d,t)

    for d in data:
        obj.feedforward(d)
