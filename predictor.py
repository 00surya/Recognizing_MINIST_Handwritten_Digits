import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('train.csv')
# print(df.shape)


data = df.values


X = data[:,1:]
Y = data[:,0] 

# dividng data into traing and test
split = int(0.8*X.shape[0])

X_train = X[:split,:]
Y_train = Y[:split]

X_test = X[split:,:]
Y_test = Y[split:]



# Visualise Some Samples

def drawImg(sample):
    img = sample.reshape(28,-1)
    plt.imshow(img,cmap='gray')
    plt.show()
    
# drawImg(X_train[3])    




def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5): 
    vals = []
    m = X.shape[0]
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append([d,Y[i]])      
    vals = sorted(vals)

    vals = vals[:k]
    vals = np.array(vals)
#     print(vals)
    new_vals = np.unique(vals[:,1],return_counts=True)
#     print("new_vals",new_vals)
    index = new_vals[1].argmax()
#     print("index",index)
    pred = new_vals[0][index]
    return pred


pred = knn(X_train,Y_train,X_test[3])
print(pred)
print(Y_test[3])



## Finding accuracy

#correctCase = 0


#for i in range(X_test.shape[0]):
 #   pred = knn(X_train,Y_train,X_test[i])
  #  print(X_test[i])
   # if pred == Y_test[i]:
    #    correctCase += 1
           

#accuracy = (correctCase) / (len(X_test))
#print(accuracy)
