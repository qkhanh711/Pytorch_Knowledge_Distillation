#import
import torch
from torch import nn,optim
from tqdm import tqdm
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from model import Teacher,Student,Distiller

#load data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

trainset = datasets.MNIST(root='./Security/Pytorch_Knowledge_Distillation/data',train=True,download=True,transform=transform)
trainloader = DataLoader(trainset,batch_size=64,shuffle=True)

testset = datasets.MNIST(root='./Security/Pytorch_Knowledge_Distillation/data',train=False,download=True,transform=transform)
testloader = DataLoader(testset,batch_size=64,shuffle=False)

# Train teacher
teacher = Teacher()
teacher_optimizer = optim.Adam(teacher.parameters(),lr=0.001)
teacher.train()

for epoch in range(5):
    running_loss = 0.0
    for i,data in enumerate(tqdm(trainloader)):
        inputs,labels = data
        teacher_optimizer.zero_grad()
        outputs = teacher(inputs)
        loss = nn.functional.cross_entropy(outputs,labels)
        loss.backward()
        teacher_optimizer.step()
        running_loss += loss.item()
    print("Loss: %.3f" % (running_loss/len(trainloader)))
print('Finished Training')

# Distill teacher to student
student = Student()
distiller = Distiller(student,teacher)
optimizer = optim.Adam(student.parameters(),lr=0.001)
distiller.train(trainloader,optimizer,epochs=2,T=20,alpha=0.5)

#test
distiller.test(testloader)

#save
distiller.save('./Security/Pytorch_Knowledge_Distillation/student.pth')

#load
student = Student()
distiller = Distiller(student,teacher)
distiller.load('./Security/Pytorch_Knowledge_Distillation/student.pth')

#test
distiller.test(testloader)

# Path: test.py.ipynb
