import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import os

#Connect to the database
#conn = sqlite3.connect('Prject2.db')


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load data from csv file
    Input:
    messages_filepath: path of messages data
    categories_filepath: path of categories data
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on ='id')
    
    return df

def clean_data(df):
    '''
    Function to clean data into cleaned data
    Input: data need to be cleaned
    Output: cleaned data
    '''
    
    #Split categories columns into 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    
    #select the first row of the categories dataframe
    row = categories.iloc[0]
    
    #Use this row to extract a list of new column names for categories dataframe
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1:])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

        #The values in your cleaned data_frame should contain only binary values(i.e either 0 or 1)
        #You will find that certain "related" values are "related-2" which doesn't make any sense. Therefore, replace 2 by 1
        categories[column] = categories[column].map(lambda x: 1 if x==2 else x)
    
    #Drop the original categories column from df
    df = df.drop('categories',axis = 1)
    
    #Concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df,categories], axis =1)
    
    #Remove duplicates
    df.drop_duplicates(inplace=True)


    
    return df


def save_data(df, database_filename):
    '''
    Function to save cleaned to database
    INPUT:
       df: cleaned data
       database_filename: database filename of sqlite database with .db file type
       
     OUTPUT:
       None: save cleaned data into sqlite database
    '''
    
    engine = create_engine('sqlite:///' + os.path.join(os.getcwd(), database_filename))
    df.to_sql('DisasterResponse_table', engine, index = False, if_exists = 'replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()