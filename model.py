#import
import torch
from torch import nn,optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# Construct Distiller class

class Distiller(nn.Module):
    def __init__(self,student,teacher):
        super(Distiller,self).__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.KL = nn.KLDivLoss(reduction='batchmean')

    def forward(self,x):
        student_logits = self.student(x)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        return student_logits,teacher_logits
    
    def loss(self,student_logits,teacher_logits,T=20,alpha=0.5):
        student_softmax = nn.functional.softmax(student_logits/T,dim=1)
        
        teacher_softmax = nn.functional.softmax(teacher_logits/T,dim=1)
        loss = alpha*T*T*self.KL(student_softmax,teacher_softmax) + nn.functional.cross_entropy(student_logits,torch.argmax(teacher_softmax,dim=1))*(1. - alpha)
        return loss

    def train(self,trainloader,optimizer,epochs=10,T=20,alpha=0.5):
        self.student.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for i,data in enumerate(tqdm(trainloader)):
                inputs,labels = data
                optimizer.zero_grad()
                student_logits,teacher_logits = self(inputs)
                loss = self.loss(student_logits,teacher_logits,T,alpha)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("Student loss: %.3f" % (running_loss/len(trainloader)))
            print('Teacher loss: %.3f' % (nn.functional.cross_entropy(teacher_logits,labels)))
            print('Distillation loss: %.3f' % (self.KL(nn.functional.softmax(student_logits/T,dim=1),nn.functional.softmax(teacher_logits/T,dim=1))))
            print('Epoch: %d, Loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
        print('Finished Training')

    def test(self,testloader):
        self.student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data
                outputs = self.student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    def save(self,path):
        torch.save(self.student.state_dict(),path)

    def load(self,path):
        self.student.load_state_dict(torch.load(path))


# Creat Student and Teacher
class Student(nn.Module):
    def __init__(self, infeatures=784, outfeatures=10):
        super().__init__()
        self.fc1 = nn.Linear(infeatures, 256)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, outfeatures)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class Teacher(nn.Module):
    def __init__(self, infeatures=784, outfeatures=10):
        super().__init__()
        self.fc1 = nn.Linear(infeatures, 256)
        self.LeakyReLU = nn.LeakyReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, outfeatures)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.LeakyReLU(x)
        x = self.fc2(x)
        x = self.LeakyReLU(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x