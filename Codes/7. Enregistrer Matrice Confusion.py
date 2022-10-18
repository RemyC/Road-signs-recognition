#rempli la matrice de confusion
print ("[INFO] Confusion matrix : Wait a moment...")
prediction=np.empty (len (testLabels), dtype=np.int16)
for i in range (0, len (testLabels)) :
    # classify the digit
    proba = model.predict(testData[np.newaxis, i])
    prediction[i] = proba.argmax(axis=1)
cm = metrics.confusion_matrix(testlabel,prediction)


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, rotation=90)
    plt.yticks(tick_marks)
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print("Normalized confusion matrix")
    #print(cm)
    
    thresh = cm.max() / 2.
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
#np.set_printoptions(precision=2)
class_names = [i for i in range(0,42)]
#print class_names
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=class_names, normalize=True,title='Normalized confusionmatrix')
plt.savefig('/home/remy/SYS843/L_'+str(lr)+'_'+str(batch)+'_'+str(epoch)+'_Matrice.png', bbox_inches='tight')