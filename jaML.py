import altair as alt
import base64
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import sys

class jaML(object):
    def __init__(self):
        self.hide_menu_style = """
                            <style>
                            #MainMenu {visibility: hidden;}
                            footer {visibility: hidden;}
                            </style>
                            
                        """
        self.width = 750
        self.height = 500

    '''
    @st.cache
    def load_data(self):
        df = data.cars()
        return df
    '''

    def get_csv_download_link(self, df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href = "data:file/csv;base64,{b64}" download="Transformed_Data.csv">Download Transformed Data</a>'
        return href

    def visualize_bar(self, df, x_axis, y_axis, legend, tooltips):
        graph = alt.Chart(df).mark_bar().encode(
            x = x_axis,
            y = y_axis,
            color = legend
        ).interactive().properties(width = self.width, height = self.height)
        st.text("")
        st.text("")
        st.write(graph)


    def visualize_circle(self, df, x_axis, y_axis, legend):
        graph = alt.Chart(df).mark_circle(size = 60).encode(
            x = x_axis,
            y = y_axis,
            color = legend
        ).interactive().properties(width = self.width, height = self.height)
        st.text("")
        st.text("")
        st.write(graph)

    def visualize_line(self, df, x_axis, y_axis, legend):
        graph = alt.Chart(df).mark_line().encode(
            x = x_axis,
            y = y_axis,
            color = legend
        ).interactive().properties(width = self.width, height = self.height)
        st.text("")
        st.text("")
        st.write(graph)

    def visualize_area(self, df, x_axis, y_axis, legend):
        graph = alt.Chart(df).mark_area().encode(
            x = x_axis,
            y = y_axis,
            color = legend
        ).interactive().properties(width = self.width, height = self.height)
        st.text("")
        st.text("")
        st.write(graph)

    def visualize_box(self, df, x_axis, y_axis, legend, tooltips):
        graph = alt.Chart(df).mark_boxplot().encode(
            x = x_axis,
            y = y_axis,
            color = legend
        ).interactive().properties(width = self.width, height = self.height)
        st.text("")
        st.text("")
        st.write(graph)

    def visualize_count(self, df, x_axis):
        graph = alt.Chart(df).mark_bar().encode(
            x = x_axis,
            y = 'count('+x_axis+'):Q',
            color = x_axis+':N',
        ).interactive().properties(width = self.width, height = self.height)
        st.text("")
        st.text("")
        st.write(graph)
    
    def visualize_heatmap(self, df):
        corrMatrix = df.corr().reset_index().melt('index')
        corrMatrix.columns = ['X','Y','Correlation']

        base = alt.Chart(corrMatrix).transform_filter(
            alt.datum.X < alt.datum.Y).encode(
                x = 'X',
                y = 'Y',
            ).properties(
                width = self.width,
                height = self.height
            )

        rects = base.mark_rect().encode(
            color = 'Correlation'
        )

        text = base.mark_text(
            size = 30
        ).encode(
            text = alt.Text('Correlation', format = '.2f'),
            color = alt.condition(
                "datum.Correlation > 0.5",
                alt.value('white'),
                alt.value('black')
            )
        )
        st.text("")
        st.text("")
        st.write(rects + text)

    def visualize_selection(self, df, x_axis, y_axis, legend):
        brush = alt.selection_interval()
        graph = alt.Chart(df).mark_point().encode(
            x = x_axis+':Q',
            y = y_axis+':Q',
            color = alt.condition(brush, legend+':N', alt.value('lightgray'))
            #tooltip = tooltips
        ).properties(width = self.width, height = self.height).add_selection(
            brush
        )
        st.text("")
        st.text("")
        st.write(graph)

    def visualize_distribution(self, df, x_axis):
        df = df.dropna()
        try:
            sns.distplot(df[x_axis])
            plt.xlabel(x_axis)
            st.pyplot()
        except:
            st.subheader("Cannot Build Distribution Plot")


    def main(self):
        st.markdown(self.hide_menu_style, unsafe_allow_html=True)
        
        st.sidebar.title('Sections')
        page = st.sidebar.radio("Go To Page",['Homepage', 'Data View', 'EDA'])
        data = st.file_uploader('Upload Your Dataset', type = 'csv')
        


        if data == None:
            pass
        else:
            df = st.cache(pd.read_csv)(data)
            if page == 'Homepage':
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
                st.sidebar.title('What To Do?')
                st.sidebar.info("You Can Choose Your Inputs In The Dropdown For Which You Want The Plot For")
                st.sidebar.title('Note:')
                st.sidebar.info("Some Plots Are Interactive. You Can Choose Your Tooltips and Hover Over The Plot To View The Selected Tooltips.")

                st.title('Exploratory Data Analysis')
                plot_types = ['Barplot','Scatterplot','Lineplot','Areaplot','Boxplot','Countplot','Correlation Heatmap','Selection Plot', 'Distribution Plot']
                type_of_plot = st.selectbox("Choose Type Of Plot", plot_types,index = 0)

                if type_of_plot == 'Barplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    tooltips = st.multiselect("Choose Variable(s) For Tooltips", list(df.columns), default = [list(df.columns)[0]])
                    self.visualize_bar(df, x_axis, y_axis, legend, tooltips)

                elif type_of_plot == 'Scatterplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    self.visualize_circle(df, x_axis, y_axis, legend)

                elif type_of_plot == 'Lineplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    tooltips = st.multiselect("Choose Variable(s) For Tooltips", list(df.columns), default = [list(df.columns)[0]])
                    self.visualize_line(df, x_axis, y_axis, legend, tooltips)

                elif type_of_plot == 'Areaplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    tooltips = st.multiselect("Choose Variable(s) For Tooltips", list(df.columns), default = [list(df.columns)[0]])
                    self.visualize_area(df, x_axis, y_axis, legend, tooltips)

                elif type_of_plot == 'Boxplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    tooltips = st.multiselect("Choose Variable(s) For Tooltips", list(df.columns), default = [list(df.columns)[0]])
                    self.visualize_box(df, x_axis, y_axis, legend, tooltips)

                elif type_of_plot == 'Countplot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    self.visualize_count(df, x_axis)

                elif type_of_plot == 'Correlation Heatmap':
                    st.text("Below Is The Heatmap That Indicates The Correlation Amongst The Data Columns")
                    self.visualize_heatmap(df)

                elif type_of_plot == 'Selection Plot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    y_axis = st.selectbox("Choose A Variable For Y-Axis", df.columns, index = len(df.columns)-2)
                    legend = st.selectbox("Choose A Variable For The Legend", df.columns, index = len(df.columns)-3)
                    self.visualize_selection(df, x_axis, y_axis, legend)
                
                elif type_of_plot == 'Distribution Plot':
                    x_axis = st.selectbox("Choose A Variable For X-Axis", df.columns, index = len(df.columns)-1)
                    self.visualize_distribution(df, x_axis)

jaml = jaML()
jaml.main()
