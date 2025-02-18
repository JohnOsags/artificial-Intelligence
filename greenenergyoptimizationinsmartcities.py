#importing required modules
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# path of the green energy data file for a smart city 
# you can change the path according to the data file which you want to test

file_path = "/Users/johnosags/Downloads/Junior and Senior Year of College /Spring Semester 2024/Artificial Intelligence/Project/Artifical Intelligence Project/Code/EnergyData.csv"

# defining Function to load energy data for a smart city
def load_energy_data():
    try:
        energy_df = pd.read_csv(file_path)
        return energy_df
    except FileNotFoundError:
        print(f"Error: Energy not found.")
        return None
    
#part A. AI-Driven Energy Consumption Analysis

# Function to perform detailed energy consumption analysis for a specific smart city
    

def perform_energy_consumption_analysis_detailed():
    # Load data for the city
    energy_df = load_energy_data()
    
    if energy_df is not None:
        # Features include are  weather conditions, population density, urban activities, etc.
        features = energy_df[['WeatherCondition', 'PopulationDensity', 'UrbanActivities']]
        target = energy_df['EnergyConsumption']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train a machine learning model (Random Forest Regressor in this example)
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        # Print detailed analysis
        print(f"Detailed Analysis for green energy consumption in smart cities:")
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")

        # Visualize actual vs predicted values
        plt.scatter(y_test, predictions)
        plt.xlabel('Actual Green Energy Consumption')
        plt.ylabel('Predicted Green Energy Consumption')
        plt.title('Actual vs. Predicted Green  Energy Consumption')
        plt.show()
        #an analysis on the data plotted
        if mse < 100:
            print("The high MSE indicates potential discrepancies in the model's predictions.")
        elif 100 <= mse <= 500:
            print("The moderate MSE suggests reasonable accuracy in predicting energy consumption.")
        else:
            print("The low MSE indicates that the model provides accurate predictions for energy consumption")

        if mae < 20:
            print("The low MAE suggests good accuracy in predicting energy consumption values.")
        elif 20 <= mae <= 50:
            print("The moderate MAE suggests reasonable accuracy in predicting energy consumption values.")
        else:
            print("The high MAE indicates potential discrepancies in the model's predictions.")
perform_energy_consumption_analysis_detailed()


#part B. Dynamic Energy Distribution


# Function to load energy distribution data

def load_energy_distribution_data():
    try:
        distribution_df = pd.read_csv(file_path)
        return distribution_df
    except FileNotFoundError:
        print(f"Error: Distribution data file not found.")
        return None

# Function to perform dynamic energy distribution analysis for a smart city
    
def perform_dynamic_energy_distribution():
    # Load energy data for the smart city
    energy_df = load_energy_data()
    
    if energy_df is not None:
        # Features included are weather conditions, population density, urban activities, etc.
        features = energy_df[['WeatherCondition', 'PopulationDensity', 'UrbanActivities']]
        target_energy = energy_df['EnergyConsumption']

        # Load energy distribution data for the specified city
        distribution_df = load_energy_distribution_data()

        if distribution_df is not None:
            # Feature used is energy distribution
            feature_distribution = distribution_df['EnergyDistribution']

            # Concatenate features and distribution data
            X = pd.concat([features, feature_distribution], axis=1)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, target_energy, test_size=0.2, random_state=42)

            # Train a machine learning model (Random Forest Regressor in this example)
            model_energy = RandomForestRegressor()
            model_energy.fit(X_train, y_train)

            # Make predictions on the test set
            predictions_energy = model_energy.predict(X_test)

            # Evaluate the model for energy consumption
            mse_energy = mean_squared_error(y_test, predictions_energy)
            mae_energy = mean_absolute_error(y_test, predictions_energy)

            # Print detailed analysis for energy consumption
            print(f"Detailed Analysis for dynamic energy distribution for a smart city :")
            print(f"Mean Squared Error: {mse_energy}")
            print(f"Mean Absolute Error: {mae_energy}")

            # Interpretation for Dynamic Energy Distribution

            if mse_energy < 100:
                print("The high MSE raises concerns about accuracy during dynamic distribution.")
            elif 100 <= mse_energy <= 500:
                print("The moderate MSE suggests reasonable accuracy in predicting energy consumption during dynamic distribution.")
            else:
                print("The low MSE for energy consumption suggests good accuracy in dynamic distribution predictions.")

            if mae_energy < 20:
                print("The low MAE for energy consumption suggests good accuracy in dynamic distribution predictions.")
            elif 20 <= mae_energy <= 50:
                print("The moderate MAE indicates reasonable accuracy in predicting energy consumption during dynamic distribution.")
            else:
                print("The high MAE implies potential discrepancies in dynamic distribution predictions.")    

            # Train a machine learning model for energy distribution
            # In this example, let's use a simplified model (linear regression) for illustration
            model_distribution = RandomForestRegressor()
            model_distribution.fit(features, feature_distribution)

            # Make predictions on the energy features
            predictions_distribution = model_distribution.predict(features)

            # Visualize actual vs. predicted energy distribution values
            plt.scatter(feature_distribution, predictions_distribution)
            plt.xlabel('Actual Green Energy Distribution')
            plt.ylabel('Predicted Green Energy Distribution')
            plt.title('Actual vs. Predicted Green Energy Distribution')
            plt.show()
            
perform_dynamic_energy_distribution()


#Part C. Optimised energy storage

# Function to load energy storage data for a smart city

def load_energy_storage_data():
    try:
        storage_df = pd.read_csv(file_path)
        return storage_df
    except FileNotFoundError:
        print(f"Error: Storage data file  not found.")
        return None

# Function to perform optimized energy storage analysis for a smart city
def perform_optimized_energy_storage():

    # Load energy distribution data for the city

    distribution_df = load_energy_distribution_data()

    if distribution_df is not None:
        # Feature used is  energy distribution
        feature_distribution = distribution_df['EnergyDistribution']

        # Load energy storage data for the specified city
        storage_df = load_energy_storage_data()

        if storage_df is not None:
            # Features involved are weather conditions, population density, urban activities, etc.
            features = storage_df[['WeatherCondition', 'PopulationDensity', 'UrbanActivities']]
            target_storage_health = storage_df['EnergyStorageHealth']

            # Concatenate features and distribution data
            X = pd.concat([features, feature_distribution], axis=1)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, target_storage_health, test_size=0.2, random_state=42)

            # Train a machine learning model (Random Forest Regressor in this example)
            model_storage = RandomForestRegressor()
            model_storage.fit(X_train, y_train)

            # Make predictions on the test set
            predictions_storage = model_storage.predict(X_test)

            # Evaluate the model for energy storage health
            mse_storage = mean_squared_error(y_test, predictions_storage)
            mae_storage = mean_absolute_error(y_test, predictions_storage)

            # Print detailed analysis for energy storage health
            print(f"Detailed Analysis for Energy Storage Health for a smart city :")
            print(f"Mean Squared Error: {mse_storage}")
            print(f"Mean Absolute Error: {mae_storage}")

            # Interpretation and analysis for Optimized Energy Storage

            if mse_storage < 100:
                print("The low MSE for energy storage health indicates accurate predictions in optimized storage scenarios.")
            elif 100 <= mse_storage <= 500:
                print("The moderate MSE suggests reasonable accuracy in predicting energy storage health in optimized scenarios.")
            else:
                print("The high MSE raises concerns about accuracy in predicting energy storage health in optimized scenarios.")

            if mae_storage < 20:
                print("The low MAE for energy storage health suggests good accuracy in predictions for optimized storage scenarios.")
            elif 20 <= mae_storage <= 50:
                print("The moderate MAE indicates reasonable accuracy in predicting energy storage health in optimized scenarios.")
            else:
                print("The high MAE implies potential discrepancies in predictions for energy storage health in optimized scenarios.")


            # Visualization of actual vs. predicted energy storage health values
                
            plt.scatter(y_test, predictions_storage)
            plt.xlabel('Actual Green Energy Storage Health')
            plt.ylabel('Predicted Green Energy Storage Health')
            plt.title('Actual vs. Predicted Green Energy Storage Health')
            plt.show()
perform_optimized_energy_storage()


#Part D.Demand side management

# Function to load demand response data for a smart city

def load_demand_response_data():
    try:
        demand_response_df = pd.read_csv(file_path)
        return demand_response_df
    except FileNotFoundError:
        print(f"Error: Demand response data file not found.")
        return None
    
# Function to perform demand-side management analysis for a smart city
    
def perform_demand_side_management():

    # Load energy distribution data for the city

    distribution_df = load_energy_distribution_data()

    if distribution_df is not None:
        feature_distribution = distribution_df['EnergyDistribution']

        # Load demand response data for the specified smart city

        demand_response_df = load_demand_response_data()

        if demand_response_df is not None:
            # Feature used is demand response
            feature_demand_response = demand_response_df['DemandResponse']

            # Concatenate features and distribution data
            X = pd.concat([feature_distribution, feature_demand_response], axis=1)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, demand_response_df ['EnergyConsumption'], test_size=0.2, random_state=42)

            # Train a machine learning model (Random Forest Regressor in this example)
            model_demand = RandomForestRegressor()
            model_demand.fit(X_train, y_train)

            # Make predictions on the test set
            predictions_demand = model_demand.predict(X_test)

            # Evaluate the model for energy consumption
            mse_demand = mean_squared_error(y_test, predictions_demand)
            mae_demand = mean_absolute_error(y_test, predictions_demand)

            # Print detailed analysis for demand-side management
            print(f"Detailed Analysis for Demand-Side Management for a smart city:")
            print(f"Mean Squared Error: {mse_demand}")
            print(f"Mean Absolute Error: {mae_demand}")

            # Visualize actual vs. predicted energy consumption values
            plt.scatter(y_test, predictions_demand)
            plt.xlabel('Actual Green Energy Consumption demand')
            plt.ylabel('Predicted Green Energy Consumption demand')
            plt.title('Actual vs. Predicted Green Energy Consumption demand')
            plt.show()
            if mse_demand < 100:
                print("The high MSE implies potential discrepancies in predictions for energy consumption in demand-side management scenarios.")
            elif 100 <= mse_demand <= 500:
                print("The moderate MSE suggests reasonable accuracy in predicting energy consumption in demand-side management scenarios.")
            else:
                print("The low MSE for energy consumption indicates accurate predictions in demand-side management scenarios.")

            if mae_demand < 20:
                print("The high MAE implies potential discrepancies in predictions for energy consumption in demand-side management scenarios.")
            elif 20 <= mae_demand <= 50:
                print("The moderate MAE indicates reasonable accuracy in predicting energy consumption in demand-side management scenarios.")
            else:
                print("The low MAE for energy consumption indicates accurate predictions in demand-side management scenarios.")
perform_demand_side_management()
print("\n----------------------------------------")
print("Overall Summary:")
print("The analysis for various aspects of green energy optimization in smart cities has been completed.")
print("Detailed results and visualizations for each analysis are provided above.")
print("Thank you for using our program!")


