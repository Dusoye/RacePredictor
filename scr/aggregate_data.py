import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import multiprocessing
from tqdm import tqdm

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

def concat_in_chunks(dataframes, chunk_size=100):
    concatenated = pd.DataFrame()
    for i in range(0, len(dataframes), chunk_size):
        chunk = pd.concat(dataframes[i:i+chunk_size], ignore_index=True)
        concatenated = pd.concat([concatenated, chunk], ignore_index=True)
        print(f"Concatenated {i+chunk_size} results...")  # Debug print
    return concatenated

def process_dataset_parallel(data, n_jobs=None):
    if n_jobs is None:
        n_jobs = cpu_count()

    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    grouped = [group for _, group in data.groupby('Runner ID')]

    results = []
    with tqdm(total=len(grouped), desc="Processing Groups") as pbar:
        if n_jobs == 1:
            # Single-process mode for debugging
            for group in grouped:
                result = process_runner_group(group)
                results.append(result)
                pbar.update()
        else:
            # Multiprocessing mode
            with Pool(n_jobs) as pool:
                for result in pool.imap_unordered(process_runner_group, grouped):
                    results.append(result)
                    pbar.update()
                pool.close()
                pool.join()

    # Filter out None and empty DataFrames
    valid_results = [res for res in results if res is not None and not res.empty]
    print(f"Number of valid results: {len(valid_results)}")

    # Concatenate in chunks to avoid memory issues
    processed_data = concat_in_chunks(valid_results, chunk_size=10000)
    
    return processed_data

if __name__ == '__main__':
    # Set the start method to 'spawn'
    multiprocessing.set_start_method('spawn')
    
    # Load your data
    data = pd.read_csv("./data/data_cleaned.csv", low_memory=False)
    data = data[data['Event Type']=='Distance']
    
    # Process the data
    processed_data = process_dataset_parallel(data)
    processed_data.to_csv('./output/aggdata.csv', index=False)

    print("Data processing completed and saved successfully!")