mylist1 = []
today = datetime.date.today ()
mylist1.append (today)
debut = time.time ()

###############################################################################
mypath = '/home/remy/SYS843/data'
#mypath = '/home/remy/SYS843/data_full'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print len(onlyfiles)

dim = len(onlyfiles)
#dim = 1000

data = np.empty([dim, 784], dtype=np.int16)
label = np.empty(dim, dtype=np.int16)

# Rentre les datas dans la bonne forme
for a in range(0, dim):
    image = cv2.imread(join(mypath, onlyfiles[a]), 0)
    name = onlyfiles[a]
    labels = re.search('000(.+?)_', name)
    if labels:
        found = labels.group(1)
    label[a] = found
    
    image = cv2.resize(image, (28, 28))
    # height, width = image.shape[:2]
    # plt.subplot(121), plt.imshow(image, cmap='gray')
    # plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # A = np.asarray(image)
    # A=A[np.newaxis,np.newaxis,:,:]
    # trainData[0,:,:,:]=A[0,0,:,:]
    
    for b in range(0, 28):
        for c in range(0, 28):
            data[a, b + c * 28] = image[b, c]
            c += 1
        b += 1
    adv = a / dim * 100
    if (a) % 1000 == 0:
        print ("[INFO] Reading data: ") + str(a) + " /" + str(dim)
    a += 1
# print label

dataset = (data, label)
data = dataset[0].reshape((dataset[0].shape[0], 28, 28))
data = data[:, np.newaxis, :, :]

(trainData, testData, trainLabels, testLabels) = train_test_split(data / 255.0, dataset[1].astype("int"),test_size=0.33)

testlabel=testLabels
trainLabels = np_utils.to_categorical(trainLabels, 43)
testLabels = np_utils.to_categorical(testLabels, 43)

fin = (time.time() - debut) / 60
print("[TIME] Reading data: {:.2f}".format(fin)), " min"
###############################################################################