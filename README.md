# Brain-Tumor

Brain Tumor is one of the deadliest disease any human can suffer from.
Machine Learning bought a whole new dimension to the field of Disease Detection.

### Identification:
Many a times there has been various errors from doctors while giving reports
be it like sometimes.

* The doctor makes a mistake and falsely diagnose a patient.
* The doctor fails to diagnose a patient.

Our objective in this project is to create an image classification model that
can predict Brain MRI scans that belong to one of the four classes with a
reasonably high accuracy. Our dataset has more than 3000 Brain MRI scans
which are categorized in four classes - Glioma Tumor, Meningioma Tumor,
Pituitary Tumor and No Tumor.

### Result

● Developed a Machine Learning model to help diagnose a person has Brain Tumor using his Brain MRI scan 
with a reasonable accuracy of over 97%.
● Trained state-of-art CNN model EfficientNet B1 on the MRI scanned images to predict the brain tumor with 
F1-score 82%. The implementation was done in Keras.
● Later build a production built of the model using Streamlit and hosted it on GitHub.

https://share.streamlit.io/apoorv-17/brain-tumor/app.py

Just Upload your Brain MRI Scan file and It will show you your the Type of Brain Tumor.
