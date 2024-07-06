#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTModel
import cv2
from tqdm import tqdm
import torch
import seaborn
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTModel
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix


# In[32]:


build_train = False #Should the train data be preprocessed or just read 
build_test = False #same for test
feature_extractor = 'hog' #Choose between 'hog' and 'vit'
train_path = './train' # Path of training samples
test_path = './test' #Path of test samples

img_size = None 


# All images are resized prior to training
if feature_extractor == 'hog':
    img_size = 128 #hog image size
else:
    img_size = 224 #vit image size
    
#Vit preprocessor

preprocessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k',do_resize = True,do_rescale = True,do_normalize = True,image_mean = [0.5,0.5,0.5],image_std = [0.5,0.5,0.5])
#Does all the preprocessing done at the pretraining step 
    


# In[33]:


import gc 

if build_train: # If images should be processed
    train_samples = []
    train_labels = []

    for cls in tqdm(os.listdir(train_path)):

        fullPath = os.path.join(train_path,cls)

        #Scan through all training folders
        for item in os.listdir(fullPath):

            img = cv2.imread(os.path.join(fullPath,item))
            #read image
            img = cv2.resize(img,(img_size,img_size))
            #resize image
            train_samples.append(img)
            train_labels.append(cls)

    train_features = []
    cnt = 0
    if feature_extractor == 'hog':

        if not os.path.exists('./train_extracted_hog'):
            os.mkdir('./train_extracted_hog') #preprocessed train images
            
        extractor = cv2.HOGDescriptor() #feature extractor


        

        
        
        for i in range(len(train_labels)):

            x = train_samples[i]
            y = train_labels[i]
            current_features = extractor.compute(x) #extracted features with hog

            #train_features.append(current_features)

            if not os.path.exists(os.path.join("./train_extracted_hog",y)):
                os.mkdir(os.path.join("./train_extracted_hog",y)) #save features to npy file
            fileName = os.path.join(os.path.join("./train_extracted_hog",y),f'sample_{cnt}.npy')
            cnt+=1
            #save in npys in the save folder
            np.save(fileName,current_features)
            
    else:
        if not os.path.exists('./train_extracted_vit'):
            os.mkdir('./train_extracted_vit')
        
        extractor = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').cuda() #move to cuda for speed
        
        for i in range(len(train_labels)):

            x = train_samples[i]
            y = train_labels[i]
            
            x = torch.tensor(x).unsqueeze(0) #transform to tensor and add a dimension


            x = preprocessor(x) #preprocess the image

            x['pixel_values'] = torch.tensor(x['pixel_values'][0]).unsqueeze(0).cuda() #move the data to cuda

            


            x = extractor(**x) #feature extraction
            
            x = x.pooler_output.cpu().detach().numpy() # move data back to cpu
            
            torch.cuda.empty_cache() #clear gpu memory
            gc.collect() #and python garbage collector
            
            if not os.path.exists(os.path.join("./train_extracted_vit",y)):
                os.mkdir(os.path.join("./train_extracted_vit",y))
            fileName = os.path.join(os.path.join("./train_extracted_vit",y),f'sample_{cnt}.npy')
            cnt+=1 
            #save the result in a .npy file
            np.save(fileName,x)
            


# In[34]:


if build_test:
    test_samples = []
    test_labels = []

    for cls in tqdm(os.listdir(test_path)):

        fullPath = os.path.join(test_path,cls)


        for item in os.listdir(fullPath):

            img = cv2.imread(os.path.join(fullPath,item)) #same as for train
            img = cv2.resize(img,(img_size,img_size))
            test_samples.append(img)
            test_labels.append(cls)

    test_features = []
    
    cnt = 0
    
    if feature_extractor == 'hog':

        if not os.path.exists('./test_extracted_hog'):
            os.mkdir('./test_extracted_hog')
            
        extractor = cv2.HOGDescriptor()

        #same as for training

        
        
        for i in range(len(test_labels)):

            x = test_samples[i]
            y = test_labels[i]
            current_features = extractor.compute(x)

            #train_features.append(current_features)

            if not os.path.exists(os.path.join("./test_extracted_hog",y)):
                os.mkdir(os.path.join("./test_extracted_hog",y))
                #same as for training
                
            fileName = os.path.join(os.path.join("./test_extracted_hog",y),f'sample_{cnt}.npy')
            cnt+=1
            np.save(fileName,current_features)
    else:
        if not os.path.exists('./test_extracted_vit'):
            os.mkdir('./test_extracted_vit')
        
        extractor = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
        #same as for training
        
        for i in range(len(test_labels)):

            x = test_samples[i]
            y = test_labels[i]
            
            x = torch.tensor(x).unsqueeze(0)


            x = preprocessor(x)
            
            #same as for training
            
            x['pixel_values'] = torch.tensor(x['pixel_values'][0]).unsqueeze(0).cuda()

            


            x = extractor(**x)
          
            x = x.pooler_output.cpu().detach().numpy()
           
            torch.cuda.empty_cache()
            gc.collect()
            
            if not os.path.exists(os.path.join("./test_extracted_vit",y)):
                os.mkdir(os.path.join("./test_extracted_vit",y))
                
            #same as for training
            
            fileName = os.path.join(os.path.join("./test_extracted_vit",y),f'sample_{cnt}.npy')
            cnt+=1
            np.save(fileName,x)


# In[35]:


train_samples = []
train_labels = []
train_features_path = f'./train_extracted_{feature_extractor}'

#Load the data from the presaved .npy files

for cnt,cls in enumerate(os.listdir(train_features_path)):
    
    fullPath = os.path.join(train_features_path,cls)
    
    for x in os.listdir(fullPath):
        vect = np.load(os.path.join(fullPath,x))
        train_samples.append(vect)
        train_labels.append(cnt)


# In[36]:


train_samples = np.array(train_samples)
#print(train_samples.shape)
if feature_extractor=='vit':
    train_samples = np.array(train_samples).reshape((train_samples.shape[0],train_samples.shape[2]))
    #little issue with the save, the arrays have an extra dimensions which i get rid of


# In[37]:


train_samples.shape


# In[38]:


from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
data_scaler = StandardScaler() #use standardscaler for the vectors

train_samples = data_scaler.fit_transform(train_samples) #transform the train data

supervised_model = RandomForestClassifier() #used for the supervised baselines

supervised_model.fit(train_samples,train_labels) #fit the training model


# In[39]:


test_samples = []
test_labels = []
test_features_path = f'./test_extracted_{feature_extractor}'

# Load the test data from the pre saved files

for cnt,cls in enumerate(os.listdir(test_features_path)):
    
    fullPath = os.path.join(test_features_path,cls)
    
    for x in os.listdir(fullPath):
        vect = np.load(os.path.join(fullPath,x))
        test_samples.append(vect)
        test_labels.append(cnt)



# In[40]:


test_samples = np.array(test_samples)
if feature_extractor=='vit':
    test_samples = test_samples.reshape((test_samples.shape[0],test_samples.shape[2]))
    #same dimensional issue


# In[41]:


test_samples = data_scaler.transform(test_samples) #transform the test data
predictions = supervised_model.predict(test_samples) #and predict on them


# In[42]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
print(accuracy_score(predictions, test_labels)) #supervised baselines
print(classification_report(predictions,test_labels)) #supervised baselines


# In[43]:


from sklearn.dummy import DummyClassifier #used for the random baseline, simply assigns random classes
random_baseline = DummyClassifier().fit(train_samples,train_labels) #"fitting" the model
predictions = random_baseline.predict(test_samples)#predicting on test


# In[44]:


print(accuracy_score(predictions, test_labels)) #Random baseline 
print(classification_report(predictions,test_labels))#random baseline


# In[45]:


from sklearn_extra.cluster import KMedoids
from sklearn.metrics import consensus_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score #model and scores imports


# # K-Medoids

# In[16]:


no_clusters = []
silh_scores = []
dist_scores = []
for k in tqdm(range(2,50)): #GridSearch for the best number of clusters for the KMedoids model
    unsupervised_model1 = KMedoids(n_clusters = k).fit(train_samples) #fit the model
    train_preds = unsupervised_model1.predict(train_samples) #predict
    no_clusters.append(k) #save the nr of clusters
    silh_scores.append(silhouette_score(train_samples,train_preds))
    dist_scores.append(unsupervised_model1.inertia_) #and silhouette and distortion
    


# In[17]:


import matplotlib.pyplot as plt #Graph sillhouette and keep max
plt.plot(no_clusters,silh_scores)
plt.title('Silhouette score for K Medoids')
plt.show()
print(f"The value which maximizes the silhouette score is k = {no_clusters[np.argmax(np.array(silh_scores))]}")


# In[18]:


import matplotlib.pyplot as plt
plt.plot(no_clusters,dist_scores) #Graph distortion and keep min
plt.title('Distortion score for K Medoids')
plt.show()
print(f"The value which minimizes the distortion score is k = {no_clusters[np.argmin(np.array(dist_scores))]}")


# # Agglomerative Clustering

# In[19]:


no_clusters = []
silh_scores = []
for k in tqdm(range(2,50)):
    unsupervised_model1 = AgglomerativeClustering(n_clusters = k).fit(train_samples) #Grid search for the nr of clusters
    train_preds = unsupervised_model1.labels_
    no_clusters.append(k)
    silh_scores.append(silhouette_score(train_samples,train_preds))
 
    


# In[20]:


import matplotlib.pyplot as plt
plt.plot(no_clusters,silh_scores)
plt.title('Silhouette score for Agglomerative Clustering') #maximize shillouette
plt.show()
print(f"The value which maximizes the silhouette score is k = {no_clusters[np.argmax(np.array(silh_scores))]}")


# In[21]:


no_clusters = []
silh_scores = []
for eps in tqdm([1,2,3,4,5,10,25,50]):#grid search over distance for merge
    unsupervised_model1 = AgglomerativeClustering(distance_threshold = eps,n_clusters=None).fit(train_samples) 
    train_preds = unsupervised_model1.labels_ #maximize shillouette
    no_clusters.append(eps)
    silh_scores.append(silhouette_score(train_samples,train_preds))
 
    


# In[22]:


import matplotlib.pyplot as plt
plt.plot(no_clusters,silh_scores)
plt.title('Silhouette score for Agglomerative Clustering') #Plot shillouette
plt.show()
print(f"The value which maximizes the silhouette score is k = {no_clusters[np.argmax(np.array(silh_scores))]}")


# In[18]:


from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import adjusted_rand_score #Metrics
from sklearn.metrics import davies_bouldin_score


# In[23]:


unsupervised_model1 = KMedoids(n_clusters=13).fit(test_samples) #the best parameter for shillouette
print(rand_score(unsupervised_model1.labels_,test_labels)) #rand score
print(adjusted_rand_score(unsupervised_model1.labels_,test_labels)) #adjusted rand score
print(silhouette_score(test_samples,unsupervised_model1.labels_)) #shillouette score
print(unsupervised_model1.inertia_) #distortion
print(davies_bouldin_score(test_samples,unsupervised_model1.labels_)) #davies bouldin


# In[24]:


unsupervised_model1 = KMedoids(n_clusters=47).fit(test_samples) #the best parameter for distortion
print(rand_score(unsupervised_model1.labels_,test_labels)) #rand score
print(adjusted_rand_score(unsupervised_model1.labels_,test_labels)) #adjusted rand score
print(silhouette_score(test_samples,unsupervised_model1.labels_)) #shillouette score
print(unsupervised_model1.inertia_)
print(davies_bouldin_score(test_samples,unsupervised_model1.labels_))#davies bouldin


# In[25]:


unsupervised_model1 = AgglomerativeClustering(n_clusters=22).fit(test_samples) #Same thing, but best parameters for HOG
print(rand_score(unsupervised_model1.labels_,test_labels))
print(adjusted_rand_score(unsupervised_model1.labels_,test_labels))
print(silhouette_score(test_samples,unsupervised_model1.labels_))
print(davies_bouldin_score(test_samples,unsupervised_model1.labels_))


# In[26]:


unsupervised_model1 = AgglomerativeClustering(distance_threshold = 25,n_clusters=None).fit(test_samples) #Same thing, but best parameters for HOG
print(rand_score(unsupervised_model1.labels_,test_labels))
print(adjusted_rand_score(unsupervised_model1.labels_,test_labels))
print(silhouette_score(test_samples,unsupervised_model1.labels_))
print(davies_bouldin_score(test_samples,unsupervised_model1.labels_))


# # Graphs

# In[ ]:





# In[19]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA #PCA for 2d projection


# In[22]:


colors = ['blue','red','green','yellow','orange','brown','purple','pink','cyan','olive','gray','orange','brown','blue','red']
print(len(colors))
unsupervised_model1 =  KMedoids(n_clusters=13).fit(test_samples) #best model for VIT
points_colors = []
preds = unsupervised_model1.labels_
for x in preds:
    points_colors.append(colors[x%len(colors)]) #Colors based on predictions


#print(points_colors)
pca = PCA(n_components = 2)
points = pca.fit_transform(test_samples)
plt.figure(figsize=(10,10))
plt.grid()
plt.scatter(points[:,0],points[:,1],c=points_colors,alpha = 0.25) #plot points


medoids = unsupervised_model1.cluster_centers_ #plot medoids

plt.scatter(medoids[:,0],medoids[:,1],c='red',marker='X')

plt.show()


# In[23]:


colors = ['blue','red','green','yellow','orange','brown','purple','pink','cyan','olive','gray','orange','brown','blue','red']
print(len(colors))
unsupervised_model1 =  AgglomerativeClustering(n_clusters=49).fit(test_samples) #best AgglomerativeCLustering model for VIT
points_colors = []
preds = unsupervised_model1.labels_
for x in preds:
    points_colors.append(colors[x%len(colors)]) #colors based on predictions


#print(points_colors)
pca = PCA(n_components = 2) #projection
points = pca.fit_transform(test_samples)
plt.figure(figsize=(10,10))
plt.grid()
plt.scatter(points[:,0],points[:,1],c=points_colors,alpha = 0.25)


#medoids = unsupervised_model1.cluster_centers_ #No medoids/centroids here

#plt.scatter(medoids[:,0],medoids[:,1],c='red',marker='X')

plt.show()


# In[29]:


test_samples2 = []
test_labels2 = []

for cls in tqdm(os.listdir(test_path)): #a visualisation of a few cluster for conclussions based on the eye test

    fullPath = os.path.join(test_path,cls)


    for item in os.listdir(fullPath):

        img = cv2.imread(os.path.join(fullPath,item))
        img = cv2.resize(img,(img_size,img_size))
        test_samples2.append(img)
        test_labels2.append(cls)


for i in range(len(test_samples)):
    if preds[i]==1:
        plt.imshow(test_samples2[i])
        plt.show()
        


# In[46]:


unsupervised_model1 = KMedoids(n_clusters=13).fit(test_samples) #best parameters for VIT
preds = unsupervised_model1.labels_
mappings = dict() #mapping between clusters and labels
test_labels = np.array(test_labels)
for x in range(unsupervised_model1.n_clusters):
    pos = preds==x #Elements within that clusters
    
    y = test_labels[pos] #their corespondent
    
    tr = np.bincount(y).argmax() #Most frequent element
    
    mappings[x] = tr #we map the entire cluster to a lbel
    
transformed_labels = [mappings[x] for x in preds] #tranform
print(accuracy_score(transformed_labels,test_labels)) #get scores
print(classification_report(transformed_labels,test_labels))

confusionMatrix = confusion_matrix(transformed_labels,test_labels)
seaborn.heatmap(confusionMatrix,annot=True)


# In[29]:


unsupervised_model1 = AgglomerativeClustering(n_clusters=49).fit(test_samples) #Best params for ViT
preds = unsupervised_model1.labels_
mappings = dict() #mapping between clusters and labels
test_labels = np.array(test_labels)
for x in range(unsupervised_model1.n_clusters):
    pos = preds==x #Elements within that clusters
    
    y = test_labels[pos] #their corespondent
    
    tr = np.bincount(y).argmax() #Most frequent element
    
    mappings[x] = tr #we map the entire cluster to a lbel
    
transformed_labels = [mappings[x] for x in preds] #tranform
print(accuracy_score(transformed_labels,test_labels)) #get scores
print(classification_report(transformed_labels,test_labels))


# In[30]:


confusionMatrix = confusion_matrix(transformed_labels,test_labels)
seaborn.heatmap(confusionMatrix,annot=True)


# # K Means

# In[17]:


no_clusters = []
silh_scores = []
dist_scores = []
for k in tqdm(range(2,50,10)): #Grid search on the number of clusters
    unsupervised_model1 = KMeans(n_clusters = k,n_init=10).fit(train_samples)
    train_preds = unsupervised_model1.labels_
    no_clusters.append(k)
    silh_scores.append(silhouette_score(train_samples,train_preds)) #Keep silhouette
    dist_scores.append(unsupervised_model1.inertia_) #And distortion
    


# In[18]:


import matplotlib.pyplot as plt
plt.plot(no_clusters,silh_scores)
plt.title('Silhouette score for K Means') #plot and maximize shilouette
plt.show()
print(f"The value which maximizes the silhouette score is k = {no_clusters[np.argmax(np.array(silh_scores))]}")


# In[19]:


import matplotlib.pyplot as plt
plt.plot(no_clusters,dist_scores)
plt.title('Distortion score for K Means') #plot and minimize distortion
plt.show()
print(f"The value which minimizes the distortion score is k = {no_clusters[np.argmin(np.array(dist_scores))]}")


# In[27]:


unsupervised_model1 = KMeans(n_clusters=12).fit(test_samples) #best params for HOG
print(rand_score(unsupervised_model1.labels_,test_labels))
print(adjusted_rand_score(unsupervised_model1.labels_,test_labels))
print(silhouette_score(test_samples,unsupervised_model1.labels_))
print(unsupervised_model1.inertia_)
print(davies_bouldin_score(test_samples,unsupervised_model1.labels_))
#compute scores


# In[25]:


unsupervised_model1 = KMeans(n_clusters=42).fit(test_samples) #best params for ViT
print(rand_score(unsupervised_model1.labels_,test_labels))
print(adjusted_rand_score(unsupervised_model1.labels_,test_labels))
print(silhouette_score(test_samples,unsupervised_model1.labels_))
print(unsupervised_model1.inertia_)
print(davies_bouldin_score(test_samples,unsupervised_model1.labels_))
#compute scores


# In[30]:


colors = ['blue','red','green','yellow','orange','brown','purple','pink','cyan','olive','gray','orange','brown','blue','red']
print(len(colors))
unsupervised_model1 =  KMeans(n_clusters=42).fit(test_samples) #Best HOG model 
points_colors = []
preds = unsupervised_model1.labels_
for x in preds:
    points_colors.append(colors[x%len(colors)]) #based on predictions


#print(points_colors)
pca = PCA(n_components = 2) #project in 2d
points = pca.fit_transform(test_samples)
plt.figure(figsize=(10,10))
plt.grid()
plt.scatter(points[:,0],points[:,1],c=points_colors,alpha = 0.25) #plot points


medoids = unsupervised_model1.cluster_centers_ #plot centroids

plt.scatter(medoids[:,0],medoids[:,1],c='red',marker='X')

plt.show()


# In[47]:


unsupervised_model1 = KMeans(n_clusters=42).fit(test_samples)
preds = unsupervised_model1.labels_
mappings = dict() #mapping between clusters and labels
test_labels = np.array(test_labels)
for x in range(unsupervised_model1.n_clusters):
    pos = preds==x #Elements within that clusters
    
    y = test_labels[pos] #their corespondent
    
    tr = np.bincount(y).argmax() #Most frequent element
    
    mappings[x] = tr #we map the entire cluster to a lbel
    
transformed_labels = [mappings[x] for x in preds] #tranform
print(accuracy_score(transformed_labels,test_labels)) #get scores
print(classification_report(transformed_labels,test_labels))
confusionMatrix = confusion_matrix(transformed_labels,test_labels)
seaborn.heatmap(confusionMatrix,annot=True)


# In[ ]:





# In[ ]:





# In[ ]:




