#!/usr/bin/env python
# coding: utf-8

# In[54]:


#get_ipython().system('pip install split-folders')
#get_ipython().system('pip3 install torch torchaudio torchvision sklearn')


# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, Dataset, DataLoader
import time
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models, utils
from torchvision.transforms import Compose, ToTensor, Resize
from collections import OrderedDict
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.optim import lr_scheduler
from copy import deepcopy
import json
import os
from os.path import exists
from random import shuffle
from numpy.lib.function_base import copy
import glob


# # **We kept our Data in our HPC and then using splitfolder library we are splitting our data in ratio of 8:2

# In[ ]:


#data_dir='/scratch/jb7854/pytorch-example/deep/Dataset'
# import splitfolder
# splitfolders.ratio(data_dir, output="Output", seed=1337, ratio=(.8, 0.2)) 


# In[2]:


output= '/scratch/jb7854/pytorch-example/deep/Output'
cate_path = '/scratch/jb7854/pytorch-example/deep/'
train_dir = output + '/train'
valid_dir = output + '/val'

benign_dir = '/scratch/jb7854/pytorch-example/deep/Output/val/benign/'
malignant_dir = '/scratch/jb7854/pytorch-example/deep/Output/val/malignant/'


# In[3]:


# images = []
# print(output)
# i = 0
# for filename in glob.glob('/scratch/jb7854/pytorch-example/deep/Output/val/malignant' + '/*.png'):
#     im = Image.open(filename)
#     im = np.asarray(im)
    
#     if im.shape[0] != 460:
# #         os.remove(filename)
#         print("HI")


# In[4]:


cate_file = cate_path + "cate.json"
with open(cate_file, 'r') as f:
    cat_to_name = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"


# In[5]:


normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

data_transforms = {
                    'train': transforms.Compose([transforms.RandomRotation(30), transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize]),
                    'val': transforms.Compose([transforms.ToTensor(), normalize,])
                  }


# In[6]:


train_dataset=ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset=ImageFolder(valid_dir, transform=data_transforms['val'])

dataset_sizes = {'train':len(train_dataset), 'val':len(val_dataset)}
dataloaders = {'train':DataLoader(train_dataset, 32, shuffle=True, num_workers=0), 'val':DataLoader(val_dataset, 32, shuffle=True, num_workers=0)}

print(dataset_sizes)
# torch.size([1,3,256,256])


# In[7]:


model = models.resnet152(pretrained=True)
print(model)
classifiers = nn.Sequential(OrderedDict([
                          ('0', nn.Linear(25088, 4096)),
                          ('1', nn.ReLU()),
                          ('2', nn.Dropout(p=0.5)),
                          ('3', nn.Linear(4096, 2048)),
                          ('4', nn.ReLU()),
                          ('5', nn.Dropout(p=0.5)),
                          ('6', nn.Linear(2048,2)),
                          ('7', nn.LogSoftmax(dim=1)),
                          ]))

model.classifier = classifiers
# classifier = nn.Sequential(OrderedDict([
#                           ('fc1', nn.Linear(512, 256)),
#                           ('relu', nn.ReLU()),
#                           ('dropout1', nn.Dropout(p=0.5)),
#                           ('fc2', nn.Linear(256, 2)),
#                           ('output', nn.LogSoftmax(dim=1))
#                           ]))


# model.fc = classifier
print(model)


# In[28]:


#Function to train the model
def train_val_model(model, criterion, optimizer, scheduler, num_epochs=5):
    
    train_loss = []
    val_loss = []
    loss_dict = {'train':train_loss, 'val':val_loss}
    
    train_acc = []
    val_acc = []
    acc_dict = {'train':train_acc, 'val':val_acc}

    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    # Training phase
  
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        scheduler.step()
        running_loss = 0.0
        running_corrects = 0

        model.train()
        for inputs, labels in dataloaders['train']:
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_epoch_loss = running_loss / dataset_sizes['train']
        train_epoch_acc = running_corrects.double() / dataset_sizes['train']  
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc.item())
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format('train', train_epoch_loss, train_epoch_acc))
        phase = "val"
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloaders['val']:
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)    

        val_epoch_loss = running_loss / dataset_sizes['val']
        val_epoch_acc = running_corrects.double() / dataset_sizes['val'] 
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc.item())

        print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', val_epoch_loss, val_epoch_acc))

        # deep copy the model
        if phase == 'val' and val_epoch_acc > best_acc:
             best_model_wts = deepcopy(model.state_dict())


    best_acc = max(val_acc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid accuracy: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_dict, loss_dict


# In[29]:


# # Train a model with a pre-trained network
num_epochs = 10
model = model.to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
#optimizer = optim.Adagrad(model.parameters(), lr = 0.1)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
model_ft, acc_dict, loss_dict = train_val_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

## save model weights 
torch.save(model_ft.state_dict(), './model_weightsR1520.001.pth')


# In[30]:


def test(model, dataloaders):
    model.eval()
    accuracy = 0
    model.to(device)

    for images, labels in dataloaders['val']:

        images, labels = images.to(device), labels.to(device)
        output = model(images)
        ps = torch.exp(output)
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    testaccuracy = (accuracy/len(dataloaders['val']))*100
    print("Testing Accuracy: {:.3f}".format(testaccuracy))
    return accuracy.item()


# In[31]:


test_acc = test(model, dataloaders)


# In[32]:


plt.plot(range(num_epochs),acc_dict['train'],'b',label="training acc")
plt.plot(range(num_epochs),acc_dict['val'],'g',label="validation acc")
plt.savefig('Accuracy.png')
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy of Resnet152")


# In[33]:


plt.plot(range(num_epochs),loss_dict['train'],'b',label="Training loss")
plt.plot(range(num_epochs),loss_dict['val'],label="Validation Loss")
plt.savefig('Loss.png')
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.title("Loss of Resnet152")


# In[14]:


model.class_to_idx = dataloaders['train'].dataset.class_to_idx
model.epochs = num_epochs
checkpoint = {'input_size': [3, 700, 460],
                 'batch_size': dataloaders['train'].batch_size,
                  'output_size': 2,
                  'state_dict': model.state_dict(),
                  'data_transforms': data_transforms,
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs}
torch.save(checkpoint, './8960_checkpoint.pth')


# In[15]:


# Write a function that loads a checkpoint and rebuilds the model

def load_model(filepath):
    model = models.resnet152(pretrained=True)
    input_size = 2048
    output_size = 2
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 512)),
                          ('relu', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(512, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.fc = classifier
       
    model.load_state_dict(torch.load(filepath))
    return model, checkpoint['class_to_idx']

# Get index to class mapping
loaded_model, class_to_idx = load_model('./model_weightsR152.pth')
idx_to_class = {v: k for k, v in class_to_idx.items()}


# In[16]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
    
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    
    return npImage


# In[17]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[18]:


def predict(image_path, model, topk=2):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class


# In[19]:


benign_image_file = benign_dir + 'SOB_B_F-14-25197-400-037.png'
print(predict(benign_image_file, loaded_model))


# In[20]:


cat_to_name = {'benign':'BENIGN', 'malignant':'MALIGNANT'}


# In[21]:


# Display an image along with the top 2 classes
def view_classify(img, probabilities, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img_filename = img.split('/')[-2]
    img = Image.open(img)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    cancer_type = mapper[img_filename]
    
    ax1.set_title(cancer_type)
    ax1.imshow(img)
    ax1.axis('off')
    
    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in classes])
    ax2.invert_yaxis()

    fig.savefig('save.png')


# In[22]:


benign_img_name = 'SOB_B_TA-14-3411F-40-008.png'
benign_image_file = benign_dir + benign_img_name
p, c = predict(benign_image_file, loaded_model)
view_classify(benign_image_file, p, c, cat_to_name)


# In[23]:


malignant_img_name = 'SOB_M_LC-14-13412-200-026.png'
malignant_img_file = malignant_dir + malignant_img_name

p, c = predict(malignant_img_file, loaded_model)
view_classify(malignant_img_file, p, c, cat_to_name)


# In[24]:


malignant_img_name = 'SOB_M_DC-14-2985-40-005.png'
malignant_img_file = malignant_dir + malignant_img_name

p, c = predict(malignant_img_file, loaded_model)
view_classify(malignant_img_file, p, c, cat_to_name)


# In[25]:


benign_img_name = 'SOB_B_TA-14-3411F-100-007.png'
benign_image_file = benign_dir + benign_img_name
p, c = predict(benign_image_file, loaded_model)
view_classify(benign_image_file, p, c, cat_to_name)


# In[26]:


benign_img_name = 'SOB_B_TA-14-21978AB-400-006.png'
benign_image_file = benign_dir + benign_img_name
p, c = predict(benign_image_file, loaded_model)
view_classify(benign_image_file, p, c, cat_to_name)


# In[27]:


benign_img_name = 'SOB_B_TA-14-16184CD-200-004.png'
benign_image_file = benign_dir + benign_img_name
p, c = predict(benign_image_file, loaded_model)
view_classify(benign_image_file, p, c, cat_to_name)


# In[96]:


malignant_img_name = 'SOB_M_MC-14-13418DE-400-007.png'
malignant_img_file = malignant_dir + malignant_img_name

p, c = predict(malignant_img_file, loaded_model)
view_classify(malignant_img_file, p, c, cat_to_name)


# In[ ]:




