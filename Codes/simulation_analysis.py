import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

TARGET_OPTIONS = {
    1: 'before_springback_Secondary axis[mm]',
    2: 'before_springback_Main axis[mm]',
    3: 'before_springback_Out-of-roundness[-]',
    4: 'before_springback_Collapse[mm]',
    5: 'after_springback_Secondary axis[mm]',
    6: 'after_springback_Main axis[mm]',
    7: 'after_springback_Out-of-roundness[-]',
    8: 'after_springback_Collapse[mm]'
}

def get_simulation_data(df, sim_nums, target_column, num_points_per_simulation=2000):
    sim_data_list = []
    for num in sim_nums:
        start = (num - 1) * num_points_per_simulation
        end = num * num_points_per_simulation
        sim_data = df.iloc[start:end].drop([col for col in TARGET_OPTIONS.values() if col != target_column], axis=1)
        sim_data_list.append(sim_data)
    return pd.concat(sim_data_list)

def train_and_test_simulations(df):
    # Get user input for training and testing simulations
    train_sims = input("Enter the training simulations (comma-separated): ")
    train_sim_nums = list(map(int, train_sims.split(',')))

    test_sims = input("Enter the testing simulations (comma-separated): ")
    test_sim_nums = list(map(int, test_sims.split(',')))

    # Get user input for target option
    target_option = int(input(f"Enter the target option (1-{len(TARGET_OPTIONS)}): "))
    target_column = TARGET_OPTIONS[target_option]

    # Get training and testing data
    train_data = get_simulation_data(df, train_sim_nums, target_column)
    test_data = get_simulation_data(df, test_sim_nums, target_column)

    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    # Create and train the model
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    # Test the model
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Print results
    print(f"Mean Squared Error for test simulations {test_sim_nums}: {mse}")

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.legend()
    plt.show()
