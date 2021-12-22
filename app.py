import streamlit as st 
import streamlit.components.v1 as components 
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras
    

components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

    <style>
        .jumbotron{
            background: lightcoral;
        }
        .display-4{
            display: flex;
            justify-content: center;
            color: black;
            font-weight: bold;
        }
        p{
            display: flex;
            justify-content: center;
        }
    </style>

    <div class="jumbotron">
        <h2 class="display-4">Brain-Tumour Detection And classification</h2>
        <p class="lead">Upload The Image</p>
        <hr class="my-4">
        <p>It uses utility classes for typography and spacing to space content out within the larger container.</p>
        <p class="lead">
            <a class="btn btn-primary btn-lg" href="#" role="button">Learn more</a>
        </p>
    </div>


    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    """,
    height=200,
)


@st.cache(allow_output_mutation = True)
# Loading the saved Model

def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = img.convert('RGB')
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability

menu = ["Image"]
choice = st.sidebar.selectbox("Menu",menu)

if choice == "Image":
    st.subheader("Image")
    image_file = st.file_uploader("Upload Image", type=["jpg","PNG"])

    if image_file is not None:

        # To See details
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
        st.write(file_details)        
        image = Image.open(image_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    
    button = st.button("CLASSIFY")  
    with st.spinner("Finding Answer..."):
        if button:
            label = teachable_machine_classification(image, 'my_model')
            st.write(label)
            if label == 0:
                st.write("It is Glioma_Tumor")
            if label == 1:
                st.write("It is Meningioma_Tumour")
            if label == 2:
                st.write("Good Newz! No Tumour Found")
            if label == 3:
                st.write("It is Pituitary_Tumour")
            
            st.success("Success")