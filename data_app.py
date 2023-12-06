import streamlit as st

import os
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS

dataset_loc='data/Tweets.csv'
image_loc='data/airline.jpeg'

def load_sidebar():
    
    st.sidebar.subheader("Twitter Us Airline Sentiment")
    st.sidebar.subheader('Analyze  how travels in Febuary  2015 expected')
    st.sidebar.info("This data originally came from Crowdflower's Data for Everyone library.")
    st.sidebar.warning("Made with :heart: for JIET :sunglasses:")

    
# loading the data

# @st.cache
@st.cache_data
def load_data(dataset_loc):
    df=pd.read_csv(dataset_loc)
    df = df.loc[:, ['airline_sentiment', 'airline', 'text']]
    return df

def load_description(df):
    st.header('Data Preview')
    preview = st.radio('Choose Head/Tail',("TOP","BOTTOM"))
    
    if (preview=='TOP'):
        st.write(df.head())
    if (preview=="BOTTOM"):
        st.write(df.tail())
        
    if (st.checkbox("Show complete Dataset")):
        st.write(df)

    if (st.checkbox('Display the shape')):
        st.write(df.shape)
        dim=st.radio('Rows/Columns',("Rows",'Columns'))
        if (dim=='Rows'):
            st.write('no of Rows',df.shape[0])
        if (dim=="Columns"):
            st.write('No of Columns',df.shape[1])  
            
    if (st.checkbox("show the columns")):
        st.write(df.columns) 
    
def load_wordcloud(df,kind):
    temp_df=df.loc[df['airline_sentiment']==kind,:]
    words=" ".join(temp_df['text'])
    clean_word=" ".join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word !='"RT'])
    wc=WordCloud(stopwords=STOPWORDS,background_color="black",width=1600,height=800).generate(clean_word)

def load_viz(df):

    st.header("Data Visualisation")

    # Show tweet sentiment count
    st.subheader("Seaborn - Tweet Sentiment Count")

    # Create a countplot using Seaborn
    sns.countplot(x='airline_sentiment', data=df)

    # Display the plot in Streamlit
    st.pyplot()

    # Disable matplotlib's global use warning
    st.set_option('deprecation.showPyplotGlobalUse', False)
          
        # ***************
    # st.subheader("Plotly - Tweet Sentiment Count")
    # temp = pd.DataFrame(df['airline_sentiment'].value_counts())
    # fig = px.bar(temp, x=temp.index, y='airline_sentiment')
    # st.plotly_chart(fig, use_container_width=True)
        # ***************

        # show airline count
    st.subheader("Airline Count")
    st.write(sns.countplot(x='airline', data=df))
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
        # Show sentiment based on airline
    st.subheader("Airline Count")
    airline = st.radio("Choose an Airline?", ("US Airways", "United", "American", "Southwest", "Delta", "Virgin America"))
    temp_df = df.loc[df['airline']==airline, :]
    st.write(sns.countplot(x='airline_sentiment', order=['neutral', 'positive', 'negative'], data=temp_df))
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
        # Show WordCloud
    st.subheader("Word Cloud")
    type = st.radio("Choose the sentiment?", ("positive", "negative"))
    load_wordcloud(df, type)
    st.image("data/wc.png", use_column_width = True)

   
def main():
    load_sidebar()
    
    st.title("Airline Sentiment Analysis")
    st.image(image_loc,use_column_width=True)
    st.text('Analyze how tracelersin Feb 2015')
    
    df=load_data(dataset_loc)
    
    load_description(df)
    
    load_viz(df)
    
if __name__ =='__main__':
    main()
    