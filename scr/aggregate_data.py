import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import multiprocessing

# Ensure all functions are defined at the module level

def aggregate_runner_data(runner_group):
    runner_group = runner_group.sort_values('Date')
    aggregated_data = []
    
    for i, (_, current_race) in enumerate(runner_group.iterrows()):
        current_date = current_race['Date']
        prior_events = runner_group.iloc[:i]
        
        if prior_events.empty:
            aggregates = pd.Series({
                'Avg_Performance_Ratio': np.nan,
                'Avg_Finish_Percentage': np.nan,
                'Total_Races': 0,
                'Total_Distance': 0,
                'Preferred_Terrain': 'Unknown',
                'Avg_Speed': np.nan,
                'Days_Since_Last_Race': np.nan,
                'Is_First_Race': True
            })
        else:
            aggregates = pd.Series({
                'Avg_Performance_Ratio': prior_events['Performance Ratio'].mean(),
                'Avg_Finish_Percentage': prior_events['Finish Percentage'].mean(),
                'Total_Races': len(prior_events),
                'Total_Distance': prior_events['Distance KM'].sum(),
                'Preferred_Terrain': prior_events['Terrain'].mode().iloc[0] if not prior_events['Terrain'].mode().empty else 'Unknown',
                'Avg_Speed': prior_events['Avg.Speed km/h'].mean(),
                'Days_Since_Last_Race': (current_date - prior_events['Date'].iloc[-1]).days,
                'Is_First_Race': False
            })
        
        aggregated_data.append(pd.concat([current_race, aggregates]))
    
    return pd.DataFrame(aggregated_data)

def process_runner_group(group):
    return aggregate_runner_data(group)

def process_dataset_parallel(data, n_jobs=None):
    if n_jobs is None:
        n_jobs = cpu_count()
    
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    
    grouped = [group for _, group in data.groupby('Runner ID')]
    
    with Pool(n_jobs) as pool:
        results = pool.map(process_runner_group, grouped)
    
    return pd.concat(results, ignore_index=True)

if __name__ == '__main__':
    # Set the start method to 'spawn'
    multiprocessing.set_start_method('spawn')
    
    # Load your data
    data = pd.read_csv("./data/data_cleaned.csv", low_memory=False)
    data = data[data['Event Type']=='Distance']
    
    # Process the data
    processed_data = process_dataset_parallel(data)
    processed_data.to_csv('./output/aggdata.csv', index=False)

    # Do something with the processed data
    print(processed_data.head())