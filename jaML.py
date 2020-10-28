import altair as alt
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pre_eda.visualize import Visualize

class jaML(object):
    def __init__(self):
        self.hide_menu_style = """
                            <style>
                            #MainMenu {visibility: hidden;}
                            footer {visibility: hidden;}
                            </style>
                            
                        """

    def get_csv_download_link(self, df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href = "data:file/csv;base64,{b64}" download="Transformed_Data.csv">Download Transformed Data</a>'
        return href


    def main(self):
        st.markdown(self.hide_menu_style, unsafe_allow_html=True)
        
        st.sidebar.title('Sections')
        page = st.sidebar.radio("Go To Page",['Homepage', 'Data View', 'EDA', 'Model Training', 'Test CSV Prediction'])
        data = st.file_uploader('Upload Your Dataset', type = 'csv', key = 'Train')

        if data == None:
            pass
        else:
            df = st.cache(pd.read_csv)(data)

            if page == 'Homepage':
                st.sidebar.title('Uploaded Wrong File?')
                st.sidebar.info("You Will Have To Refresh The Page To Re-Upload")

                st.title("Welcome To The jaML App!")
                st.write("The Purpose Of This App Is To Simplify The EDA Process With Just The Click Of The Mouse.")
                st.write("The jaML App Has A Lot Of Features That Can Be Used For EDA.")

                st.subheader("View Your Original Data") 
                st.write("")
                st.write("What Can You Do With Your Data View?")
                st.write("1. View Entire Data")
                st.write("2. View First N Rows Of Data")
                st.write("3. Show Dimensions Of Data")
                st.write("4. View Single Column")
                st.write("5. View Data Summary")
                st.write("")
                st.write("")               
        
                st.subheader("You Can Vizualize Your Data With Just A Click.")
                st.write("")
                st.write("The Different Types Of Plots You Can Generate Are - ")
                st.write("1. Barplot")
                st.write("2. Scatterplot")
                st.write("3. Lineplot")
                st.write("4. Areaplot")
                st.write("5. Boxplot")
                st.write("6. Countplot")
                st.write("7. Correlation Heatmap")
                st.write("8. Selection Plot")
                st.write("")
                st.write("")

                st.subheader("You Can Train Your Own ML Model")
                st.write("")
                st.write("The Different Types Of Models You Can Generate Are - ")
                st.write("1. Linear Regression")
                st.write("2. Logistic Regression")
                st.write("3. Decision Tree Classifier/Regressor")
                st.write("4. Random Forest Classifier/Regressor")
                st.write("")
                st.write("You Can, Also, Choose To Hyper Tune The Model")
                st.write("")
                st.write("")

                if st.button("About App"):
                    st.subheader("jaML App")
                    st.text("Built with Streamlit")
                    st.text("Thanks to Abhilash, Manali, and Janavi")
                    st.balloons()

            elif page == 'Data View':
                st.title("Data View")
                st.subheader("Here You Can Take Different Views Of Your Data")
                display = st.selectbox("Choose What You What Data You Want To View", 
                                        ['View Entire Data', 'View First N Rows Of Data', 'Show Dimensions of Data', 'View Single Column', 'View Data Summary'],
                                        index = 0)

                if display == 'View Entire Data':
                    st.write(df)
                
                elif display == 'View First N Rows Of Data':
                    rows = st.slider('How Many Rows Do You Want To See?',5,50)
                    st.write("Your Data is Displayed Below")
                    st.write(df.head(rows))

                elif display == 'Show Dimensions of Data':
                        data_dim = st.radio("What Dimension Do You Want to Show", ("Rows", "Columns"))
                        if data_dim == "Rows":
                            st.text("Showing Number of Rows")
                            st.write(len(df))
                        if data_dim == "Columns":
                            st.text("Showing Number of Columns")
                            st.write(df.shape[1])

                elif display == 'View Single Column':
                    col = st.selectbox("Select Column You Want To View", df.columns, index = 0)
                    st.write(df[col])

                elif display == 'View Data Summary':
                    st.write(df.describe())

            elif page == 'EDA':
                viz = Visualize()

                st.sidebar.title('What To Do?')
                st.sidebar.info("You Can Choose Your Inputs In The Dropdown For Which You Want The Plot For")
 
                st.title('Exploratory Data Analysis')
                plot_types = ['Barplot','Scatterplot','Lineplot','Areaplot','Boxplot','Countplot','Correlation Heatmap','Selection Plot', 'Distribution Plot']
                type_of_plot = st.selectbox("Choose Type Of Plot", plot_types,index = 0)

                if type_of_plot == 'Barplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    viz.visualize_bar(df, x_axis, y_axis, legend)

                elif type_of_plot == 'Scatterplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    viz.visualize_circle(df, x_axis, y_axis, legend)

                elif type_of_plot == 'Lineplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    viz.visualize_line(df, x_axis, y_axis, legend)

                elif type_of_plot == 'Areaplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    viz.visualize_area(df, x_axis, y_axis, legend)

                elif type_of_plot == 'Boxplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    viz.visualize_box(df, x_axis, y_axis, legend)

                elif type_of_plot == 'Countplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    viz.visualize_count(df, x_axis)

                elif type_of_plot == 'Correlation Heatmap':
                    st.text("Below Is The Heatmap That Indicates The Correlation Amongst The Data Columns")
                    viz.visualize_heatmap(df)

                elif type_of_plot == 'Selection Plot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    viz.visualize_selection(df, x_axis, y_axis, legend)
                
                elif type_of_plot == 'Distribution Plot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    viz.visualize_distribution(df, x_axis)

            elif page == 'Model Training':
                st.sidebar.title('What To Do?')
                st.sidebar.info("You Can Select Your Target And Train A Model On Your Dataset")

                st.title("Model Training")
                st.subheader("Here You Can Train A Model On Uploaded Train Data-Set")
                display = st.selectbox("Choose What Type Of Training You Want To Do", 
                                        ['Classification', 'Regression'],
                                        index = 0)

                if display == 'Classification':
                    model = st.selectbox("Choose The Algorithm", 
                                        ['Linear Regression', 'Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier'],
                                        index = 0)

                    target = st.selectbox("Select Target Column", 
                                          df.columns,
                                          index = len(df.columns)-1)

                    if model != 'Linear Regression': 
                        hypertune = st.selectbox("Do You Want To Hyper Tune The Model?", 
                                                ['Yes', 'No'],
                                                index = 1)

                        if hypertune == 'Yes':
                            kind = st.selectbox("What Kind Of Hyper Tuning?", 
                                                ['RandomizedSearchCV', 'GridSearchCV'],
                                                index = 1)

            elif page == 'Test CSV Prediction':
                data = st.file_uploader('Upload Your Test Dataset', type = 'csv', key = 'Test')

                if data == None:
                    pass
                else:
                    df = st.cache(pd.read_csv)(data)
                                            
                    

jaml = jaML()
jaml.main()
