# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product
import json

import numpy as np
from pprint import pprint

# from pandas_datareader.data import DataReader
from dimod import Integer
from dimod import quicksum
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler
import pandas as pd

import requests
import json
import time


class SinglePeriod:
    """Define and solve a  single-period portfolio optimization problem."""

    def __init__(self, stocks_data_json, sampler_args=None, verbose=True):
        # self.stocks = [f"stock_{i}" for i in range(len(mu))]
        self.model_type = "CQM"
        self.verbose = verbose
        self.stocks_data = stocks_data_json

        self.stocks_names = list(self.stocks_data.stock_names)
        print("stock names ", self.stocks_names)
        
        self.budget = self.stocks_data.investment_amount

        print("budget is ", self.budget)
        self.alpha = self.stocks_data.risk_factor_value
        
        # Specify the S3 bucket URL (publicly accessible)
        s3_bucket_url = 'https://portfolio-data-bucet.s3.amazonaws.com/all_dates_assets_data.json'

        # Call the function to read the JSON file
        json_data = self.read_json_from_s3_url(s3_bucket_url)
        # print("ddaaaaaaaaaaaaaaattttttttttttttaaaaaaaaaaa ", json_data[:-1])
        
        historical_stocks_prices = pd.DataFrame(json_data)
        historical_stocks_prices = historical_stocks_prices[self.stocks_names]
        # print("hist ", historical_stocks_prices )
        self.stock_prices = np.array(historical_stocks_prices.tail(1))[0]
        # print("stock_prices ", self.stock_prices)

        self.max_num_shares = np.array((self.budget / self.stock_prices).astype(int))
        # print("max num shares ", self.max_num_shares)

        self.mu = np.array(historical_stocks_prices.pct_change().mean())

        self.covariance_matrix = np.array(historical_stocks_prices.pct_change().cov())


        # Initial holdings, or initial portfolio state.
        self.init_holdings = {s: 0 for s in self.stocks_names}

        if sampler_args:
            self.sampler_args = json.loads(sampler_args)
        else:
            self.sampler_args = {}

        self.sampler = LeapHybridCQMSampler(
            **self.sampler_args, token="DEV-3092b857364ac14474b6cca827ed602eda60252d"
        )

        self.solution = {}
        self.precision = 2

        
            
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
        

    def build_cqm(self):
        """Build and store a CQM.
        This method allows the user a choice of 3 problem formulations:
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk
            3) min risk s.t. return >= min_return

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
        """
        # Instantiating the CQM object
        cqm = ConstrainedQuadraticModel()

        # Defining and adding variables to the CQM model

        x = [
            Integer("%s" % s, lower_bound=0, upper_bound=self.max_num_shares[i])
            for i, s in enumerate(self.stocks_names)
        ]

        # Defining risk expression
        risk = 0
        stock_indices = list(range(len(self.stocks_names)))
        print(stock_indices)
        for s1, s2 in product(stock_indices, stock_indices):
            coeff = (
                self.covariance_matrix[s1][s2]
                * self.stock_prices[s1]
                * self.stock_prices[s2]
            )
            risk = risk + coeff * x[s1] * x[s2]

        # Defining the returns expression
        returns = 0
        for i, s in enumerate(self.stocks_names):
            returns = returns + self.stock_prices[i] * self.mu[i] * x[i]

        cqm.add_constraint(
            quicksum(
                [x[i] * self.stock_prices[i] for i, s in enumerate(self.stocks_names)]
            )
            <= self.budget,
            label="upper_budget",
        )
        cqm.add_constraint(
            quicksum(
                [x[i] * self.stock_prices[i] for i, s in enumerate(self.stocks_names)]
            )
            >= 0.997 * self.budget,
            label="lower_budget",
        )

        # Objective: minimize mean-variance expression
        cqm.set_objective(self.alpha * risk - returns)
        cqm.substitute_self_loops()

        self.model = cqm

    def solve_cqm(self):
        """Solve CQM.
        This method allows the user to solve one of 3 cqm problem formulations:
            1) max return - alpha*risk (default formulation)
            2) max return s.t. risk <= max_risk
            3) min risk s.t. return >= min_return

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.

        Returns:
            solution (dict): This is a dictionary that saves solutions in desired format
                e.g., solution = {'stocks': {'IBM': 3, 'WMT': 12}, 'risk': 10, 'return': 20}
        """
        self.build_cqm()

        self.sample_set = self.sampler.sample_cqm(
            self.model, label="CQM - Portfolio Optimization"
        )

        n_samples = len(self.sample_set.record)

        feasible_samples = self.sample_set.filter(lambda d: d.is_feasible)

        if not feasible_samples:
            raise Exception(
                "No feasible solution could be found for this problem instance."
            )
        else:
            best_feasible = feasible_samples.first
            solution = {}
            solution["stocks"] = {
                k: int(best_feasible.sample[k]) for k in self.stocks_names
            }

            spending = sum(
                [
                    self.stock_prices[i]
                    * max(0, solution["stocks"][s] - self.init_holdings[s])
                    for i, s in enumerate(self.stocks_names)
                ]
            )
            sales = sum(
                [
                    self.stock_prices[i]
                    * max(0, self.init_holdings[s] - solution["stocks"][s])
                    for i, s in enumerate(self.stocks_names)
                ]
            )

            self.investment_amount = spending + sales

            if self.verbose:
                print("")
            #     print(
            #         f"Number of feasible solutions: {len(feasible_samples)} out of {n_samples} sampled."
            #     )
            #     print(f"\nBest energy: {self.sample_set.first.energy: .2f}")
            #     print(f"Best energy (feasible): {best_feasible.energy: .2f}")

            # print(f"\nBest feasible solution:")
            # print(
            #     "\n".join(
            #         "{}\t{:>3}".format(k, v) for k, v in solution["stocks"].items()
            #     )
            # )

            # print(f"\nEstimated Returns: {solution['return']}")

            # print(f"Sales Revenue: {sales:.2f}")

            # print(f"Purchase Cost: {spending:.2f}")

            # print(f"investment_amount Cost: {self.investment_amount:.2f}")

            # print(f"Variance: {solution['risk']}\n")

            return solution

    def _get_risk_ret(self):
        # returns of a portfolio after optimum weight allocation
        ret = np.sum(self.mu * self.asset_weights) * 252

        # risk of a portfolio after optimum weight allocation
        vol = np.sqrt(
            np.dot(
                np.array(self.asset_weights).T,
                np.dot(self.covariance_matrix * 252, self.asset_weights),
            )
        )

        # sharpe ratio of a portfolio after optimum weight allocation_qu
        sharpe_ratio = ret / vol

        risk_ret_dict = {
            "returns": np.round(ret * 100, 2),
            "risk": np.round(vol * 100, 2),
            "sharpe_ratio": np.round(sharpe_ratio, 2),
        }

        return risk_ret_dict

    def run(self):
        """Execute sequence of load_data --> build_model --> solve.

        Args:
            max_risk (int): Maximum risk for the risk bounding formulation.
            min_return (int): Minimum return for the return bounding formulation.
            num (int): Number of stocks to be randomnly generated.
            init_holdings (float): Initial holdings, or initial portfolio state.
        """

        print(f"\nCQM run...")
        self.solution = self.solve_cqm()

        # Calculate the total value of the portfolio
        self.portfolio_value = sum(
            shares * price
            for shares, price in zip(
                self.solution["stocks"].values(), self.stock_prices
            )
        )

        # Calculate individual asset weigh  ts
        self.asset_weights = [
            shares * price / self.portfolio_value
            for shares, price in zip(
                self.solution["stocks"].values(), self.stock_prices
            )
        ]
        # print(self.asset_weights)
        # print("type ", type(self.asset_weights))

        # Initialize an empty dictionary to store the results
        optimal_num_stocks = self.solution["stocks"].copy()

        optimal_weights = dict(
            zip(self.solution["stocks"].keys(), np.round(self.asset_weights, 2))
        )
        
        
        optimal_num_stocks = {k: v for k, v in optimal_num_stocks.items() if v != 0}  # excludes stocks who got 0 alllocation.
        optimal_weights = {k: v for k, v in optimal_weights.items() if v != 0}  # excludes stocks who got 0 alllocation.

        risk_returns = self._get_risk_ret()

        pprint(optimal_weights)
        pprint(optimal_num_stocks)
        pprint(risk_returns)

        # # Loop through the stocks and their weights
        # for i, (stock_name, shares) in enumerate(self.solution["stocks"].items()):
        #     # Create a key-value pair in the dictionary
        #     results_dict[stock_name] = np.round(self.asset_weights[i], 2)

        # # Now, results_dict contains the results as a dictionary
        # # print(results_dict)

        # # Print the results
        # for i, (stock_name, shares) in enumerate(self.solution["stocks"].items()):
        #     print(f"{stock_name} Weight: {self.asset_weights[i]:.2%}")

        # opt_weights = {stock_name, self.asset_weights[i]}
        ## we cannot hide the stocks which are allocated 0 weight at this level. we need to pass it in order to get risk and returns.
        ## we have to do this before it stores to the db. and after calculating the risk_returns thing, then only we can do this. so find a gap to do this.
        ## also need to multiply with 100, may be in the dict.

        investment_amount = np.round(self.investment_amount, 3)

        return {
            "holdings": optimal_weights,
            "optimal_num_stocks": optimal_num_stocks,
            "risk_ret": risk_returns,
            "initial_investment": investment_amount,
        }
