import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class Visualize(object):

    def __init__(self):
        self.width = 750
        self.height = 500

    def visualize_bar(self, df, x_axis, y_axis, legend):
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

    def visualize_box(self, df, x_axis, y_axis, legend):
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