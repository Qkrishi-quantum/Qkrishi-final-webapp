import numpy as np     
import pandas as pd   
import requests
import json 
from datetime import datetime,timedelta



class calculate_portfolio_value:
    
    def __init__(self, portfolio):
        self.porfolio = portfolio
        
        
    def get_portfolio_vals(self):
        portfolio_create_date = self.porfolio.date_created
        print("portfolio create date ", portfolio_create_date)
        
        portfolio_assets = list(self.porfolio.optimal_num_stocks.keys())
        portfolio_returns = self.porfolio.risk_ret['returns']
        print(portfolio_assets)
        # print("portfolio ret ", portfolio_returns )
    
        s3_bucket_url = 'https://portfolio-data-bucet.s3.amazonaws.com/all_dates_assets_data.json'

        # Call the function to read the JSON file
        json_data = self.read_json_from_s3_url(s3_bucket_url)
        
        asssets_data_df = pd.DataFrame(json_data)
        # print(asssets_data_df["Date"].values)

        while portfolio_create_date not in asssets_data_df["Date"].values:
            # Subtract one day from the date
            portfolio_create_date = (pd.to_datetime(portfolio_create_date) - pd.DateOffset(days=1)).strftime("%Y-%m-%d")  ## this is correct data format, y-d-m 

        print("Found date:", portfolio_create_date)
        
                
        portfolio_start_date = portfolio_create_date
        portfolio_values_dict = {}  # Dictionary to store portfolio values on each day

        for entry in json_data:
            date = entry["Date"]
            # date = portfolio_start_date
            portfolio_value = 0 #self.porfolio.initial_investment
            
            for asset, num_stocks in self.porfolio.optimal_num_stocks.items():
                if asset in entry:
                    stock_price = entry[asset]
                    portfolio_value += stock_price * num_stocks

            portfolio_values_dict[date] = portfolio_value
                    
        # # To get the portfolio values as a dictionary
        # portfolio_values_dict = portfolio_values
        print("len ", len(portfolio_values_dict))
        # print(portfolio_values_dict)
        
        fd_initial_price = list(portfolio_values_dict.values())[0]
        fd_initial_date = list(portfolio_values_dict.keys())[0]
        fd_last_date = list(portfolio_values_dict.keys())[-1]
        # fd_initial_date = portfolio_values.keys[0]
        # print("fd_initial_price ", fd_initial_price)
        

        def calculate_years_difference(start_date, end_date):
            # Convert date strings to datetime objects
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

            # Calculate the difference between dates
            date_difference = end_date_obj - start_date_obj

            # Extract the number of years
            years_difference = date_difference.days / 365

            # Return the result as an integer
            return int(years_difference)
        
        year_diff = calculate_years_difference(fd_initial_date, fd_last_date)

        def calculate_future_dates(start_date, num_years):
            # Convert the start date string to a datetime object
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")

            # Calculate future dates for each year
            future_dates = [start_date_obj + timedelta(days=365 * year) for year in range(1, num_years + 1)]

            # Convert datetime objects back to date strings in the original format
            future_date_strings = [date_obj.strftime("%Y-%m-%d") for date_obj in future_dates]

            return future_date_strings
        
        dates_for_fd = calculate_future_dates(fd_initial_date, year_diff)
        
                
        def find_nearest_date(target_date, available_dates):
            target_date = datetime.strptime(target_date, "%Y-%m-%d")
            available_dates = [datetime.strptime(date, "%Y-%m-%d") for date in available_dates]

            closest_date = min(available_dates, key=lambda date: abs(date - target_date))

            return closest_date.strftime("%Y-%m-%d")
        
        dates_list =  list(portfolio_values_dict.keys())
        print(len(dates_list))
        # print("dates for fd ", dates_for_fd)
        
        updated_fd_dates = [find_nearest_date(date, dates_list) for date in dates_for_fd]

        
        fd_returns = {}
        def calculate_fd_returns(principal, annual_interest_rate, dates):

            for date_str in dates:
                # Convert the date string to a datetime object
                current_date = datetime.strptime(date_str, "%Y-%m-%d")

                # Calculate interest for the current date
                interest = principal * (annual_interest_rate / 100)

                # Add interest to the principal
                principal += interest

                # Add the result to the dictionary
                fd_returns[current_date.strftime("%Y-%m-%d")] = round(principal, 2)

            return fd_returns
        
        # Example usage
        initial_investment = fd_initial_price  # Initial principal amount
        fd_annual_interest_rate = 6.5  # Annual interest rate (in percentage)
        # investment_period_years = year_diff  # Investment period in years
        # print("dates dict 0", updated_fd_dates)

        fd_returns_dict = calculate_fd_returns(fd_initial_price, fd_annual_interest_rate, updated_fd_dates)
        print("tyype of ", type(fd_returns))
        
        fd_returns_dict= {fd_initial_date: np.round(fd_initial_price, 2), **fd_returns_dict}
        
        
        #  Initialize a dictionary to store gain or loss for each stock
        stock_gain_loss = {}
        
        key_index = list(json_data.keys()).index(portfolio_create_date) if portfolio_create_date in json_data else -1

        # Calculate gain or loss for each stock
        for stock in portfolio_assets:
            initial_price = asssets_data_df[stock].iloc[key_index]
            final_price = asssets_data_df[stock].iloc[-1]
            gain_loss = (final_price - initial_price) / initial_price
            stock_gain_loss[stock] = gain_loss
            
        print("stocks gain loss ", stock_gain_loss)
        
        def calculate_investment_returns(initial_investment, annual_returns_percentage, years):
            returns = {}
            principal = initial_investment

            for year in range(1, years + 1):
                # Calculate returns for the current year
                returns_percentage = principal * (annual_returns_percentage / 100)
                # Update principal for the next year
                principal += returns_percentage

                # Store the returns for the current year in the dictionary
                returns[year] = round(principal, 2)

            return returns
        
        projection_fd_start_price  = list(portfolio_values_dict.values())[-1]
        
        projection_years_investment = 5
    
        fd_projection = calculate_investment_returns(projection_fd_start_price, fd_annual_interest_rate, projection_years_investment)
        portfolio_projection = calculate_investment_returns(projection_fd_start_price, portfolio_returns, projection_years_investment)
        
        fd_projection = {'0':np.round(projection_fd_start_price, 2) , **fd_projection}
        portfolio_projection = {'0':projection_fd_start_price, **portfolio_projection}
        
        return portfolio_values_dict, stock_gain_loss, portfolio_start_date, fd_returns_dict, fd_projection, portfolio_projection
        
        
            
    def read_json_from_s3_url(self, s3_bucket_url):
        try:
            # Send an HTTP GET request to the S3 bucket URL
            response = requests.get(s3_bucket_url)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                # Parse the JSON data
                json_data = json.loads(response.text)
                return json_data
            else:
                print(f"HTTP request failed with status code {response.status_code}")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None        
    