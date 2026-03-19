# Credit Score Streamlit App

## Overview
This project is a Streamlit application designed to classify credit scores into three categories: bad (0), medium (1), and good (2). The application utilizes an artificial neural network (ANN) model to make predictions based on user input.

## Project Structure
The project consists of the following files:

- **app.py**: The main Streamlit application that sets up the user interface with a title, subtitle, input sliders for credit score classification, and a button labeled "predecir" to trigger the prediction.
  
- **models/ann_model.h5**: The trained artificial neural network model used for predicting credit scores based on the input data.
  
- **models/label_encoders.joblib**: Contains the label encoders necessary for transforming categorical input data into a format suitable for the model.
  
- **assets/credit_card_image_url.txt**: A text file containing the URL of a royalty-free image of an open credit card, which will be displayed in the Streamlit app.
  
- **requirements.txt**: Lists the dependencies required to run the Streamlit application, including Streamlit and any other necessary libraries.

## Setup Instructions
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using the following command:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit application with the command:
   ```
   streamlit run app.py
   ```

## Usage
Once the application is running, you will see a user interface where you can input various parameters using sliders. After entering the data, click the "predecir" button to classify the credit score.

## Author
This project was developed by Juan Escobar.