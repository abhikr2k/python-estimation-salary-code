 #import libraries
 import pandas as pd #useful for loading the dataset
 import numpy as np #to oreform array
 
 #chose dataset from local directory
 #if using google colab perform this
 from google.colab import files
 uploaded = files.upload()
 #salary.csv file uploaded
 #load dataset
 dataset=pd.read_csv('salary.csv')
 
 #summarize dataset
 print(dataset.shape)
 print(dataset.head(5))
 
 #mapping salry to binary values
 income_set = set(datset['income'])
 dataset['income'] = dataset['income'].map({'<=50k':0,'>50k':1}).astype(int)
 print(dataset.head)
 
 #segregate dataset into X(input/independent variable)&Y(output/dependent variable)
 X= dataset.iloc[:, :-1].values
 X
 Y=dataset.iloc[:, -1].values
 Y
 #splitting dataset into train and test
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.25, random_state = 0)
 
 #feature scaling
 #fit_transform- fit method is calculating the mean and variance of each of the feature
  from sklearn.preprocessing import StandardScalar
  sc = StandardScalar()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)
  
  #Finding the best k value
  error = []
  from sklearn.neighbors import KNeighborClassifier
  import matplotlib.pyplot as plt
  #calculating error for k values between 1 and 40
  for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,Y_train)
    pred_i = model.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
  plt.figure(figsize=(12,6))  
  plt.plot(range(1,40), error, color='red', linestyle='dashed', marker='o',markerfacecolor='blue', markersize=10)
  plt.title('Error Rate k Value')
  plt.xlabel('K Value')
  plt.ylabel('Mean Error')
  
  #Training
  from sklearn.neighbours import KNeighborsClassifier
  model = KNeighborsClassifier(n_neighbors = 16,metric = 'minkowski',p = 2)
  model.fit(X_train,y_train)
  
  #Predicting Whether new customer with age and salary 
  
  age=int(input("Enter New Employee's Age:"))
  edu= int(input("Enter New Employee's Education:"))
  cg = int(input("Enter new Employee's capital Gain:"))
  wh = int(input("Enter new employee's Hour per week:"))
  newEmp = [[age,edu,cg,wh]]
  result = model.predict(sc.transform(newEmp))
  print(result)
  
  if result ==1:
    print("Employee might got salary above 50k")
  else:
    print("customer might not getsalry above 50k")
  
 #predict for all test data
 y_pred = model.predict(X_test)
 print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
 
 #Evaluating model confusion matrix
 from sklearn.metrics import confusion_matrix, accuracy_score
 cm = confusion_matrix(y_test,y_pred)
 print("confusion matrix:")
 print(cm)
 
 print("Accuracy of the model:{0}%".format(accuracy_score(y-test,y_pred)*100))
   
 
