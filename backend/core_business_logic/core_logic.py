import numpy as np 
import pandas as pd   
import requests 
import json 

import numpy as np
import copy

from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

# Problem modelling imports
from docplex.mp.model import Model
from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSampler

# Qiskit imports
from qiskit.algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit.utils.algorithm_globals import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit_optimization import QuadraticProgram

from qiskit_optimization.problems.variable import VarType
from qiskit_optimization.converters.quadratic_program_to_qubo import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp

from scipy.optimize import minimize


class dwave_classical_portfolio:
    
    def __init__(self, stocks_data_json):
        # self.stocks = [f"stock_{i}" for i in range(len(mu))]
        
        self.stocks_data = stocks_data_json

        self.stocks_names = list(self.stocks_data.stock_names)
        print("stock names ", self.stocks_names)
        
        self.budget = self.stocks_data.investment_amount

        print("budget is ", self.budget)
        self.alpha = self.stocks_data.risk_factor_value
        
        # Specify the S3 bucket URL (publicly accessible)
        s3_bucket_url = 'https://portfolio-data-bucet.s3.amazonaws.com/all_dates_assets_data.json'

        # Call the function to read the JSON files
        json_data = self.read_json_from_s3_url(s3_bucket_url)
        # print("ddaaaaaaaaaaaaaaattttttttttttttaaaaaaaaaaa ", json_data[:-1])
        
        historical_stocks_prices = pd.DataFrame(json_data)
        historical_stocks_prices = historical_stocks_prices[self.stocks_names]
        
        last_date_prices= historical_stocks_prices.iloc[-1]
        
        self.last_date_prices = last_date_prices.to_dict()
        # print("hist ", historical_stocks_prices )
        
        self.stock_prices = np.array(historical_stocks_prices.tail(1))[0]
        # self.last_date_prices  = self.stock_prices 
        self.stock_prices  = self.last_date_prices
        print("last stock_prices ", self.stock_prices)

        # self.max_num_shares = np.array((self.budget / self.stock_prices).astype(int))
        # print("max num shares ", self.max_num_shares)

        self.mu = np.array(historical_stocks_prices.pct_change().mean())

        self.covariance_matrix = np.array(historical_stocks_prices.pct_change().cov())
        
        self.historical_stocks_prices = historical_stocks_prices


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
        
        
    def create_problem(self, mu: np.array, sigma: np.array, qiskit_budget: int, alpha: float) -> QuadraticProgram:
        """Solve the quadratic program using docplex."""

        mdl = Model()
        x = [mdl.binary_var("x%s" % i) for i in range(len(sigma))]

        objective = mdl.sum([mu[i] * x[i] for i in range(len(mu))])
        objective -= alpha * mdl.sum(
            [sigma[i, j] * x[i] * x[j] for i in range(len(mu)) for j in range(len(mu))]
        )
        mdl.maximize(objective)
        cost = mdl.sum(x)
        mdl.add_constraint(cost == qiskit_budget)

        qp = from_docplex_mp(mdl)
        return qp
        
    def assets_selection(self):
        qiskit_budget = len(self.mu)//2
        
        qubo = self.create_problem(self.mu, self.covariance_matrix, qiskit_budget, self.alpha)
        
                
        qp2qubo = QuadraticProgramToQubo()
        dwave_qubo = qp2qubo.convert(qubo)
        
        bqm_qubo = dimod.as_bqm(dwave_qubo.objective.linear.to_array(), dwave_qubo.objective.quadratic.to_array(), dimod.BINARY)
        
        device = "SA"
        if device == "SA":
            sampler_sa = SimulatedAnnealingSampler()
            result_using_dwave = sampler_sa.sample(bqm_qubo, label="example_qp", num_reads=5000)
            print("executed using simulated annealing.. ")
            
        else:
            sampler_dw = DWaveSampler(solver='Advantage_system4.1', token = 'DEV-3092b857364ac14474b6cca827ed602eda60252d')
            embedded_sampler = EmbeddingComposite(sampler_dw)
            result_using_dwave = embedded_sampler.sample(bqm_qubo, label="example_qp", num_reads=5000)
            print("executed on real qpu")

        opt_output_dwave = result_using_dwave.first.sample.values()
        dwave_result = list(opt_output_dwave)

        return dwave_result
    
    
    def assets_allocation(self):
        selected_assets = self.assets_selection()
        selected_bitstring = [i for i, e in enumerate(selected_assets) if e == 1]
        my_assets = [self.stocks_names[i] for i in selected_bitstring]
        print(" selected assets ", my_assets)
        
        returns = self.historical_stocks_prices[my_assets]
        returns = returns.pct_change()
        
        self.returns = returns
        
        weights = np.array(np.random.random(len(my_assets)))

        print('normalised weights :')
        weights = weights/np.sum(weights)
        
        
        cons = ({'type':'eq','fun':self.check_sum})
        bounds = tuple((0, 1) for self.stocks_names in range(len(my_assets)))
        init_guess = list(np.random.dirichlet(np.ones(len(my_assets)),size=1))
        init_guess = np.array(init_guess).flatten()
        print("x0", init_guess)
        
        opt_weights = minimize(self.neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        opt_weights = np.round(opt_weights.x, 3)

        my_weights = np.array(opt_weights)
        # my_weights = my_weights.tolist()
        
        return my_assets, my_weights, returns 
        

    def portfolio_dwave(self):
        my_assets, my_weights, returns  = self.assets_allocation()
        weights_alloc = dict(zip(my_assets, my_weights))
        
        ret = np.sum(returns.mean() * my_weights) * 252 # returns of a portfolio after optimum weight allocation
        vol = np.sqrt(np.dot(my_weights.T,np.dot(returns.cov()*252,my_weights))) # risk of a portfolio after optimum weight allocation
        sharpe_ratio = ret/vol # sharpe ratio of a portfolio after optimum weight allocation
        print("sharpe ratio of your porrtfolio after optimization is ", sharpe_ratio)
        
        risk_returns = {
            'returns' : ret*100,
            'risk' : vol*100,
            'sharpe_ratio' : sharpe_ratio
        }
        
        opt_num_stocks, investment_amount = self.calculate_number_of_stocks(weights_alloc)
        
        optimal_weights = {k: v for k, v in weights_alloc.items() if v != 0}  # excludes stocks who got 0 alllocation.
        print("optimal weights ", optimal_weights)
                
        optimal_num_stocks = {k: v for k, v in opt_num_stocks.items() if v != 0}  # excludes stocks who got 0 alllocation.
        print("opt number of stocks ", optimal_num_stocks)
        
        return {
            "holdings": optimal_weights,
            "optimal_num_stocks": optimal_num_stocks,
            "risk_ret": risk_returns,
            "initial_investment": investment_amount,
        }

        

    def get_ret_vol_sr(self, weights):
        weights = np.array(weights)
        ret = np.sum(self.returns.mean() * weights) * 252
        vol = np.sqrt(np.dot(weights.T,np.dot(self.returns.cov()*252,weights)))
        sr = ret/vol
        return np.array(sr)


    # minimize negative Sharpe Ratio
    def neg_sharpe(self, weights):
        return self.get_ret_vol_sr(weights)*-1

    # check allocation sums to 1
    def check_sum(self, weights):
        return np.sum(weights) - 1
    
    
    def calculate_number_of_stocks(self, weights):
        # Create a dictionary to store the number of stocks
        number_of_stocks = {}
        
        # Iterate through each stock symbol
        for stock_symbol in self.stock_prices:

            if stock_symbol in weights:
                # Calculate the number of stocks based on the weight and last date price
                last_price = self.stock_prices[stock_symbol]
                weight = weights[stock_symbol]
                number_of_stocks[stock_symbol] = int(weight*self.budget / last_price)
        
        investment_dict = {key: number_of_stocks[key] * self.stock_prices[key] for key in number_of_stocks.keys()}
        investment_amount = sum(investment_dict.values())
        # print(number_of_stocks)     
        # print(investment_amount) 
        return number_of_stocks, investment_amount
