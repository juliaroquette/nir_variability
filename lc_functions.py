import sys
sys.path.append('/Users/marie/Documents/Research/sge/Julia/VariabilityIndexes')
from variability.lightcurve import LightCurve
from tqdm import tqdm
import pandas as pd
import os
import re
import numpy as np

def stats_lightcurves_paralalel(folder, filter='J', batch_size=1000, **columns):
    """
    Process light curve files in batches, extracting statistical properties.
    
    Parameters:
    -----------
    folder : str
        Path to the directory containing light curve CSV files
    filter : str, optional (default='J')
        Filter to process (e.g., 'J', 'H', 'K'). Files are filtered by this value
    batch_size : int, optional (default=1000)
        Number of files to process in each batch for memory management
    **columns : dict
        Column mapping dictionary with keys:
        - 'time': column name for time values
        - 'mag': column name for magnitude values  
        - 'err': column name for error values
    
    Returns:
    --------
    None
        Saves results to CSV files in the parent directory:
        - props_{filter}_temp.csv: Temporary file updated after each batch
        - props_{filter}_final.csv: Final complete results
    
    Output Columns:
    ---------------
    - file: Original filename
    - file_number: Numeric identifier extracted from filename
    - filter: Filter used for processing
    - N: Number of data points
    - SNR: Signal-to-noise ratio
    - max: Maximum magnitude value
    - mean: Mean magnitude value
    - mean_err: Mean error value
    - median: Median magnitude value
    - min: Minimum magnitude value
    - ptp: Peak-to-peak magnitude variation
    - range: Magnitude range
    - std: Standard deviation of magnitudes
    - time_max: Time of maximum magnitude
    - time_min: Time of minimum magnitude
    - time_span: Total time span of observations
    - weighted_average: Weighted average magnitude
    - error: Error message if processing failed, None otherwise
    
    Example:
    --------
    >>> process_light_curves_optimized(
    ...     folder='/data/light_curves',
    ...     filter='H',
    ...     batch_size=500,
    ...     time='mjd',
    ...     mag='mag',
    ...     err='err'
    ... )
    """
    #.Extract column names from parameters
    time_col, mag_col, err_col = columns['time'], columns['mag'], columns['err']
    parent_folder = os.path.dirname(folder)
    
    #.List and sort files numerically by embedded number in filename
    all_files = sorted(
        [f for f in os.listdir(folder) if f.startswith('UKIRT2007_lc_') and f.endswith('.csv')],
        key=lambda x: int(re.search(r'UKIRT2007_lc_(\d+)\.csv', x).group(1)) if re.search(r'UKIRT2007_lc_(\d+)\.csv', x) else 0
    )
    
    print(f"Processing {len(all_files)} files in batches of {batch_size}...")
    
    results = []  #.To store all processing results
    
    #.Processing files in batches
    for i in tqdm(range(0, len(all_files), batch_size)):
        batch_results = []
        
        #.Processing files in current batch
        for file in all_files[i:i+batch_size]:
            #.Extracting numeric identifier from filename
            file_number = int(re.search(r'UKIRT2007_lc_(\d+)\.csv', file).group(1))
            file_data = {'file': file, 'file_number': file_number, 'filter': filter}
            
            try:
                #.Loading columns used
                data = pd.read_csv(os.path.join(folder, file), 
                                  usecols=['Filter', time_col, mag_col, err_col])
                #.Filter data for specified filter band
                data_filtered = data[data['Filter'] == filter]
                
                if not data_filtered.empty:
                    #.Create LightCurve object and extract properties
                    lc = LightCurve(data_filtered[time_col], 
                                   data_filtered[mag_col], 
                                   data_filtered[err_col])
                    
                    #.LightCurve properties to extract
                    props = ['N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min', 
                            'ptp', 'range', 'std', 'time_max', 'time_min', 
                            'time_span', 'weighted_average']
                    
                    #.Extract each property (returns NaN if missing)
                    for prop in props:
                        file_data[prop] = getattr(lc, prop, np.nan)
                    file_data['error'] = None  #.no error occurred
                    file_data['RA'] = np.nanmean(data_filtered['RA'])
                    file_data['DEC'] = np.nanmean(data_filtered['DEC'])                    
                else:
                    #.If no data is available for specified filter
                    file_data.update({prop: np.nan for prop in [
                        'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                        'ptp', 'range', 'std', 'time_max', 'time_min',
                        'time_span', 'weighted_average'
                    ]})
                    file_data['N'] = 0  #.set count to zero
                    file_data['error'] = 'No data for filter'
                    file_data['RA'] = np.nan
                    file_data['DEC'] = np.nan
                    
            except Exception as e:
                #.If error, fill with NaN
                file_data.update({prop: np.nan for prop in [
                    'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                    'ptp', 'range', 'std', 'time_max', 'time_min',
                    'time_span', 'weighted_average', 'RA', 'DEC'
                ]})
                file_data['error'] = str(e)  #.storing error message
                print(f"✗ Error in {file}: {e}")
            
            batch_results.append(file_data)
        
        #.Add to main results list
        results.extend(batch_results)
        
        #.Save temporary results (for safe)
        temp_df = pd.DataFrame(results).sort_values('file_number')
        temp_df.to_csv(os.path.join(parent_folder, f'props_{filter}_temp.csv'), index=False)
    
    #.Save final results sorted by file number
    final_df = pd.DataFrame(results).sort_values('file_number')
    output_path = os.path.join(parent_folder, f'props_{filter}_final.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f" {filter}: {len(results)} files processed successfully")
    print(f" Output saved to: {output_path}")
    

def stats_lightcurves(folder, filter='J', **columns):
    """
    Process light curves files extracting statistical properties.
    
    Parameters:
    -----------
    folder : str
        Path to the directory containing light curve CSV files
    filter : str, optional (default='J')
        Filter to process (e.g., 'J', 'H', 'K'). Files are filtered by this value
    **columns : dict
        Column mapping dictionary with keys:
        - 'time': column name for time values
        - 'mag': column name for magnitude values  
        - 'err': column name for error values
    
    Returns:
    --------
    None
        Saves results to CSV files in the parent directory:
        - props_{filter}_final.csv: Final complete results
    
    Output Columns:
    ---------------
    - file: Original filename
    - file_number: Numeric identifier extracted from filename
    - filter: Filter used for processing
    - N: Number of data points
    - SNR: Signal-to-noise ratio
    - max: Maximum magnitude value
    - mean: Mean magnitude value
    - mean_err: Mean error value
    - median: Median magnitude value
    - min: Minimum magnitude value
    - ptp: Peak-to-peak magnitude variation
    - range: Magnitude range
    - std: Standard deviation of magnitudes
    - time_max: Time of maximum magnitude
    - time_min: Time of minimum magnitude
    - time_span: Total time span of observations
    - weighted_average: Weighted average magnitude
    - error: Error message if processing failed, None otherwise
    - RA: Average Right Ascension
    - DEC: Average Declination

    Example:
    --------
    stats_lightcurves(
    ...     folder='/data/light_curves',
    ...     filter='H',
    ...     time='mjd',
    ...     mag='mag',
    ...     err='err'
    ... )
    
    """
    #.Extract column names from parameters
    time_col, mag_col, err_col = columns['time'], columns['mag'], columns['err']
    parent_folder = os.path.dirname(folder)
    
    #.List and sort files numerically by embedded number in filename
    all_files = sorted(
        [f for f in os.listdir(folder) if f.startswith('UKIRT2007_lc_') and f.endswith('.csv')],
        key=lambda x: int(re.search(r'UKIRT2007_lc_(\d+)\.csv', x).group(1)) if re.search(r'UKIRT2007_lc_(\d+)\.csv', x) else 0
    )
    
    print(f"Processing {len(all_files)} files...")
    
    results = []  #.To store all processing results
    
    #.Processing all files at once (no batches)
    for file in tqdm(all_files, desc="Processing files", miniters=5000):
        # Extracting numeric identifier from filename
        file_number = int(re.search(r'UKIRT2007_lc_(\d+)\.csv', file).group(1))
        file_data = {'file': file, 'file_number': file_number, 'filter': filter}
        
        try:
            #.Loading columns used
            data = pd.read_csv(os.path.join(folder, file), 
                              usecols=['Filter', time_col, mag_col, err_col])
            #.Filter data for specified filter band
            data_filtered = data[data['Filter'] == filter]
            
            if not data_filtered.empty:
                #.Create LightCurve object and extract properties
                lc = LightCurve(data_filtered[time_col], 
                               data_filtered[mag_col], 
                               data_filtered[err_col])
                
                #.LightCurve properties to extract
                props = ['N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min', 
                        'ptp', 'range', 'std', 'time_max', 'time_min', 
                        'time_span', 'weighted_average']
                
                #.Extract each property (returns NaN if missing)
                for prop in props:
                    file_data[prop] = getattr(lc, prop, np.nan)
                file_data['error'] = None  # no error occurred
                # add average coordinate
                file_data['RA'] = np.nanmean(data_filtered['RA'])
                file_data['DEC'] = np.nanmean(data_filtered['DEC'])

            else:
                #.If no data is available for specified filter
                file_data.update({prop: np.nan for prop in [
                    'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                    'ptp', 'range', 'std', 'time_max', 'time_min',
                    'time_span', 'weighted_average'
                ]})
                file_data['N'] = 0  #.set count to zero
                file_data['error'] = 'No data for filter'
                file_data['RA'] = np.nan
                file_data['DEC'] = np.nan
        except Exception as e:
            #.If error, fill with NaN
            file_data.update({prop: np.nan for prop in [
                'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                'ptp', 'range', 'std', 'time_max', 'time_min',
                'time_span', 'weighted_average', 'RA', 'DEC'
            ]})
            file_data['error'] = str(e)  #.storing error message
            print(f"✗ Error in {file}: {e}")
        
        results.append(file_data)
    
    #.Save final results sorted by file number
    final_df = pd.DataFrame(results).sort_values('file_number')
    output_path = os.path.join(parent_folder, f'props_{filter}_final.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f" {filter}: {len(results)} files processed successfully")
    print(f" Output saved to: {output_path}")