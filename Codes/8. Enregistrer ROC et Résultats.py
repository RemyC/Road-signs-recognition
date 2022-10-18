##################################### ROC #################################
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

testlabel = np_utils.to_categorical(testlabel, 43)
prediction = np_utils.to_categorical(prediction, 43)

for i in range(0,43):
    fpr[i], tpr[i], _ = roc_curve(testlabel[:, i], prediction[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(testlabel.ravel(), prediction.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

lw=2

plt.figure()
plt.plot(fpr["micro"], tpr["micro"],label='microaverage ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),color='darkorange', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('/home/remy/SYS843/L_'+str(lr)+'_'+str(batch)+'_'+str(epoch)+'_ROC.png', bbox_inches='tight')

for i in range(0,43):
    plt.plot(fpr[i], tpr[i], lw=lw)#,label='ROC {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('/home/remy/SYS843/L_'+str(lr)+'_'+str(batch)+'_'+str(epoch)+'_ROCfull.png', bbox_inches='tight')


##################################### Resultats #################################

print("[RESULTAT] accuracy: {:.2f}%".format(accuracy * 100))
print"[RESULTAT] lr=", lr
print"[RESULTAT] batch=", batch
print "[RESULTAT] epochs=", epoch
print mylist1[0]
fin = (time.time() - debut)/60
print("[RESULTAT] temps: {:.2f}".format(fin)), " min"