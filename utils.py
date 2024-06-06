import os
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pyefd import elliptic_fourier_descriptors
import joblib
import numpy as np

class SmoothIrregular:
    def __init__(self, degree=30):
        self.degree = degree
        self.model = None
        self.scaler = StandardScaler()
        self.descriptors_list = [[] for _ in range(degree)]
        self.species = []
        self.num = []

    def calcEFDs(self, contour):
        efd = elliptic_fourier_descriptors(contour, order=self.degree, normalize=True)
        return efd.flatten()

    def calcDescriptors(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        segArray = 1 * (th == 0)
        segImage = sitk.GetImageFromArray(segArray)
        segmentedImage = segImage
        hole_filling_filter = sitk.VotingBinaryHoleFillingImageFilter()
        hole_filling_filter.SetRadius(3)
        hole_filling_filter.SetMajorityThreshold(1)
        hole_filling_filter.SetBackgroundValue(0)
        hole_filling_filter.SetForegroundValue(1)
        hole_filled_image = hole_filling_filter.Execute(segmentedImage)
        convertToLabelMap = sitk.BinaryImageToLabelMapFilter()
        labelMap = convertToLabelMap.Execute(hole_filled_image)
        labelImageFilter = sitk.LabelMapToLabelImageFilter()
        labelImageFilter.SetNumberOfThreads(4)
        labelImage = labelImageFilter.Execute(labelMap)
        castFilter = sitk.CastImageFilter()
        castFilter.SetOutputPixelType(sitk.sitkUInt8)
        labelImage = castFilter.Execute(labelImage)
        labelMap = labelImage
        labelArray = sitk.GetArrayFromImage(labelMap)
        Obj = []
        for label in range(1, np.amax(labelArray) + 1):
            singleLabel = (labelMap == label)
            Obj.append(singleLabel)
        area = 0
        for i in range(len(Obj)):
            contours, hierarchy = cv2.findContours(sitk.GetArrayFromImage(Obj[i]), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            temp_area = cv2.contourArea(cnt)
            if area < temp_area:
                area = temp_area
                image_temp = Obj[i]
        img = sitk.GetArrayFromImage(image_temp)
        contour = []
        contour, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, contour)
        contour_array = contour[0][:, 0, :]
        efd_descriptors = self.calcEFDs(contour_array)
        return efd_descriptors[:self.degree]

    def load_data(self, lisse_dir, anneau_dir):
        lisse_entries = os.listdir(lisse_dir)
        for entry in lisse_entries:
            image = cv2.imread(os.path.join(lisse_dir, entry))
            descriptors = self.calcDescriptors(image)
            for i, d in enumerate(self.descriptors_list):
                d.append(descriptors[i])
            self.species.append(0)
            self.num.append(entry)

        anneau_entries = os.listdir(anneau_dir)
        for entry in anneau_entries:
            image = cv2.imread(os.path.join(anneau_dir, entry))
            descriptors = self.calcDescriptors(image)
            for i, d in enumerate(self.descriptors_list):
                d.append(descriptors[i])
            self.species.append(1)
            self.num.append(entry)

        data_dict = {f'd{i+1}': self.descriptors_list[i] for i in range(self.degree)}
        data_dict['species'] = self.species
        data_dict['num'] = self.num
        df = pd.DataFrame(data_dict)

        X = df.drop(columns=['species', 'num']).values
        y = np.array(self.species)

        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.degree,)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_test, y_test, epochs=25, batch_size=32):
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        return history

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

        y_pred = (self.model.predict(X_test) > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Lisse', 'Anneau'])
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    def save_model(self, model_path, scaler_path):
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)  # Sauvegarder le scaler

    def load_model(self, model_path, scaler_path):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)  # Charger le scaler

    def predict(self, image):
        descriptors = self.calcDescriptors(image)
        descriptors = np.array(descriptors).reshape(1, -1)
        descriptors = self.scaler.transform(descriptors)
        prediction = (self.model.predict(descriptors) > 0.5).astype(int)
        return 'Lisse' if prediction[0][0] == 0 else 'Irregular'


