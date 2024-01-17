# ft_linear_regression
First Ecole 42 project in the AI and Machine Learning branch. A program that predicts the price of a car by using a linear function train with a gradient descent algorithm. 

## Description  

There are several programs created:  

- First script  `predict.py` that prompts users for mileage of a car in kilometers and output its estimated price.
- Script `learn.py` that launches the linear regression algorithm defined in `linear_regression.py`and computes optimal model parameters.
  It takes an optional argument `-d` or `--plot-debug`. When they are used it plots learning curve graph to help find optimal learning rate and number of iterations.  
- Script `evaluate.py` that evaluates precision of the model's predictions using R2 Score metric.
  It takes two possible optional arguments: `-d` or `--plot-data` that plots input data into scatter graph and `-r` or `--plot-result`that additionally plots the model output over the input data.

The program requires a database .csv file with two parameters: input feature and output. By default the expected db file is `data.csv` with fields "km,price". Other databases can be accepted; for that the `config.py` file has to be adjusted.
All scripts fetch initial model parameters from `regression_parameters.pkl` with pickle module. If the model hasn't been learned the parameters will be taken as 0.

## Algorithm

The algorithm used is gradient descent with linear regression taking one input parameter. Cost function is mean square error.

## Usage
The following input:
```
git clone https://github.com/bklvsky/ft_linear_regression.git
cd ft_linear_regression
python predict.py
```

Will give the output:
```
Enter the mileage of the car: [your value, e.g. 65000]
Estimated price for the mileage of 65000.0 km = 0.0
```
As the model hasn't learned yet, the output will always be 0.0.
To get meaningful results launch:
```
python learn.py
```
or launch the script with debug parameters.
```
python learn.py --plot-debug
```
This will output plot like this:  
![plot_learning_curve](/.plots/learning_curve.png)

To evaluate the result you can run:
```
python evaluate.py
```
This can give you an output like this:  
```
Estimated precision of the model: 73.058%.
```
with plots if you add the flags:

- -d/--plot-data:  
![plot_data](/.plots/data.png)

- -r/--plot-result:  
![plot_result](/.plots/result.png)
