import streamlit as st 
import pandas as pd 
import json
import datetime
import pickle
from datetime import datetime as dt

st.set_page_config(page_title = "Ingredient Procurement Prediction for Pizza Shops",
                   page_icon = "pizza",
                   layout = "wide",
                   initial_sidebar_state = "expanded",
                   menu_items = None)

@st.cache_resource()

def get_start_of_week(year, week_number):
    # Find the first day of the week
    start_of_week = dt.strptime(f'{year}-W{week_number}-1', "%Y-W%U-%w")
    return start_of_week

def to_encoding(df, week, week_date):
    
    with open(r'src/Category_Columns_Encoded_Data.json', 'r') as file:
        encoded_data = json.load(file)

    df.drop('quantity', axis =1, inplace =True)
    
    test_data = df[df.week_number==51].reset_index(drop = True)

    test_data['week_number'] = [week for i in range(test_data.shape[0])]
    test_data['year'] = df['year'].apply(lambda x: week_date.year)
    test_data['month'] = df['month'].apply(lambda x: week_date.month)
    test_data['day'] = df['day'].apply(lambda x: week_date.day)

    test_data1 = test_data.copy()
    #Converting Categorical data into Numerical data
    for col_name in [col for col in df.columns if df[col].dtype == 'object']:
        test_data[col_name] = test_data[col_name].map(encoded_data[col_name])

    return test_data, test_data1

def to_get_pizza_name_id(Pizza_Size, Pizza_Name):
    sales_df = pd.read_csv(r'src/Pizza_sales.csv')
    PizzaNameID = [sales_df[(sales_df['pizza_name']==name) & (sales_df['pizza_size']==size)]['pizza_name_id'].mode()[0] for name, size in zip(Pizza_Name, Pizza_Size)]
    return PizzaNameID

def to_get_ingredients(test_data1, y_pred):

    df2 = pd.read_csv(r'src/Pizza_ingredients.csv')
    
    test_data1['predicted_quantity'] = y_pred
    test_data1['min_predicted_quantity'] = y_pred -2.0
    test_data1['max_predicted_quantity'] = y_pred +2.0
    
    Pizza_Size =test_data1['pizza_size']
    Pizza_Name = test_data1['pizza_name']
    test_data1['pizza_name_id'] = to_get_pizza_name_id(Pizza_Size, Pizza_Name)

    df2 =df2.merge(test_data1[['pizza_name_id', 'min_predicted_quantity','predicted_quantity', 'max_predicted_quantity']], how = 'inner', on = 'pizza_name_id')

    df2['Predicted_Quantity_In_KilloGrams'] =round(( df2['Items_Qty_In_Grams'] * df2['predicted_quantity'])/1000, 2)
    df2['Min_Excepted_Quantity_In_KilloGrams'] =round(( df2['Items_Qty_In_Grams'] * df2['min_predicted_quantity'])/1000, 2)
    df2['Max_Excepted_Quantity_In_KilloGrams'] =round(( df2['Items_Qty_In_Grams'] * df2['max_predicted_quantity'])/1000, 2)

    ingredients =df2.groupby('pizza_ingredients')[['Min_Excepted_Quantity_In_KilloGrams', 'Predicted_Quantity_In_KilloGrams','Max_Excepted_Quantity_In_KilloGrams']].sum().sort_values(by = 'Max_Excepted_Quantity_In_KilloGrams', ascending = False).reset_index().head(60)
    
    return ingredients
    

df = pd.read_csv(r'src/Weekly_Pizza_Sales.csv')


with open('Pizza_Selling_Quantity_Prediction_Model.pkl', 'rb') as file:
    model = pickle.load(file)

      
st.title(":red[Ingredient Procurement] :blue[Prediction for] :orange[Pizza Shops] :pizza:")
date = st.date_input('Select the **Date**:',   datetime.date(2015,7,1), min_value= datetime.date(2015,1,1))

week = date.isocalendar().week
year = date.year
week_date = get_start_of_week(year, week)

test_data, test_data1 =to_encoding(df, week, week_date)
st.markdown('Click Below Button To Predict the Ingredients')
if st.button('Predict'):
    y_pred = model.predict(test_data)


    st.table(to_get_ingredients(test_data1, y_pred))

