# jaML

## Abstract

“Ughh! I want to work on data science and machine learning, but I don’t know how to code!!” Have you ever been in such a situation? I know we have! 

Introducing our idea - **jaML**. We have seen many autoML applications where the developer/newbie can just import some modules and write one line of code to create their very own ML model. But, have you realized how tiresome it is? Install dependencies, set up the environment, import modules, and, moreover, you cannot just pass in any dataset without having any preprocessing or EDA done. Well, that’s where we come in!

We are planning to develop a full-fledged web application where the user(Basically ANYONE!) can upload the dataset and using drop-down UI/UX can just select what he wants to do with the dataset ranging from viewing the data to EDA to creating your own machine learning model, without any coding experience!! 


What’s the catch though? **NOTHING!** Moreover, we have made it **open-source** to test the extents that this application can actually go to!!

Sounds far fetched? We have a **Proof-Of-Concept** ready for the basic tasks of our web application.

Here is the system architecture of our application -

![System Architecture jaML](https://github.com/Abhilash2000/jaML/blob/main/Sys_Arc.png?raw=true)

## Tech Stack

- For the front-end of the web application, we will be using **Streamlit**
- For the back-end - Data Science, Machine Learning; we will be using **Python**
- For version control and maintenance, we will be using **Git (Gitkraken)**. 
- Code will be pushed from **Gitkraken to Github** on frequent commits

## Instructions To Run Code

1. Clone Repo and Change Into Cloned Directory

2. Install Required Dependencies

```
pip install -r requirements.txt
```

3. Run The Web Application

```
streamlit run jaML.py
```

In case you do not want to run through the above procedures to test on localhost, you can check out the Prototype Web Application Deployed at https://share.streamlit.io/abhilash2000/jaml/main/jaML.py
