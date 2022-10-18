####################################################### TRAINING #######################
print "training..."
cc=1.
Gamma=0.1

clf = svm.SVC(C=cc, gamma=Gamma, kernel='linear')

print "Training..."
clf.fit(X_train, y_train)
fin1 = (time.time() - debut)/60
print("[TIME] Training: {:.2f}".format(fin1)), " min"

######################################################### TEST ########################
#print(clf.predict(X_test))
accuracy=0
#rempli la matrice de confusion
prediction=np.empty(len(y_test), dtype=np.int16)
for i in range(0, len(y_test)) :
    # classify the digit
    prediction[i] = clf.predict(X_test[np.newaxis, i])
    if prediction[i] == y_test[i] :
        accuracy=accuracy+1
fin = (time.time() - debut) / 60-fin1
print("[TIME] Testing: {:.2f}".format(fin)), " min"
########################################################################################