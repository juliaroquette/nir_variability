import sys
sys.path.append('/Users/marie/Documents/Research/sge/Julia/VariabilityIndexes')
from variability.lightcurve import LightCurve
from variability.indexes import VariabilityIndex
from tqdm import tqdm
import pandas as pd
import os
import re
import glob
import numpy as np
# to prevent it from printing the Q-index warning
VariabilityIndex.suppress_warnings_globally()

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
        - 'ra': column name for RA values
        - 'dec': column name for DEC values
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
    - ra_mean: Mean RA value
    - dec_mean: Mean DEC value
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
    ra_col, dec_col, time_col, mag_col, err_col = columns['ra'], columns['dec'], columns['time'], columns['mag'], columns['err']
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
                #.Loading columns used (including RA and DEC)
                data = pd.read_csv(os.path.join(folder, file), 
                                  usecols=['Filter', ra_col, dec_col, time_col, mag_col, err_col])
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
                    
                    #.Calculate mean RA and DEC
                    file_data['ra_mean'] = data_filtered[ra_col].mean()
                    file_data['dec_mean'] = data_filtered[dec_col].mean()
                    file_data['error'] = None  #.no error occurred
                    
                else:
                    #.If no data is available for specified filter
                    file_data.update({prop: np.nan for prop in [
                        'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                        'ptp', 'range', 'std', 'time_max', 'time_min',
                        'time_span', 'weighted_average', 'ra_mean', 'dec_mean'
                    ]})
                    file_data['N'] = 0  #.set count to zero
                    file_data['error'] = 'No data for filter'
                    
            except Exception as e:
                #.If error, fill with NaN
                file_data.update({prop: np.nan for prop in [
                    'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                    'ptp', 'range', 'std', 'time_max', 'time_min',
                    'time_span', 'weighted_average', 'ra_mean', 'dec_mean'
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

def get_varindexes(folder, filter='J', **columns):
    """
    Process light curves files, extracting statistical properties and variability indexes.
    
    Parameters:
    -----------
    folder : str
        Path to the directory containing light curve CSV files
    filter : str, optional (default='J')
        Filter to process (e.g., 'J', 'H', 'K'). Files are filtered by this value
    **columns : dict
        Column mapping dictionary with keys:
        - 'ra': column name for RA values
        - 'dec': column name for DEC values
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
    - ra_mean: Mean RA value
    - dec_mean: Mean DEC value
    - error: Error message if processing failed, None otherwise
    - Abbe: Abbe Index
    - IQR: Interquantile Range
    - Lag1AutoCorr: Autocorrelation lag-1
    - RoMS: Robust median statistic
    - ShapiroWilk: Shapiro-Wilk statistic
    - andersonDarling: Anderson-Darling statistics
    - chisquare: Chi-square
    - kurtosis: Kurtosis
    - mad: median absolute deviation 
    - norm_ptp: Normalized peak-to-peak variability
    - normalisedExcessVariance: Normalized excessive variance
    - reducedChiSquare: reduced Chi Square
    - skewness: Skewness
    - Q_index: Q index
    
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
    #.Validação das colunas obrigatórias
    required_columns = ['ra', 'dec', 'time', 'mag', 'err']
    for col in required_columns:
        if col not in columns:
            raise ValueError(f"Coluna obrigatória '{col}' não fornecida")
    
    #.Extract column names from parameters
    ra_col, dec_col, time_col, mag_col, err_col = columns['ra'], columns['dec'], columns['time'], columns['mag'], columns['err']
    parent_folder = os.path.dirname(folder)
    
    #.List and sort files numerically by embedded number in filename
    #.Melhoria no filtro de arquivos - abordagem mais robusta
    all_files = []
    for f in os.listdir(folder):
        if f.startswith('UKIRT2007_lc_') and f.endswith('.csv'):
            match = re.search(r'(\d+)', f)
            if match:  #.Só incluir se tiver número
                all_files.append(f)
    
    #.Ordenar arquivos numericamente
    all_files = sorted(
        all_files,
        key=lambda x: int(re.search(r'UKIRT2007_lc_(\d+)\.csv', x).group(1)) if re.search(r'UKIRT2007_lc_(\d+)\.csv', x) else 0
    )
    
    print(f"Processing {len(all_files)} files...")
    
    results = []  #.To store all processing results
    
    #.Processing all files at once (no batches)
    for file in tqdm(all_files, desc="Processing files"):
        # Extracting numeric identifier from filename
        file_number = int(re.search(r'UKIRT2007_lc_(\d+)\.csv', file).group(1))
        file_data = {'file': file, 'file_number': file_number, 'filter': filter}
        
        try:
            #.Loading columns used (including RA and DEC)
            data = pd.read_csv(os.path.join(folder, file), 
                              usecols=['Filter', ra_col, dec_col, time_col, mag_col, err_col])
            #.Filter data for specified filter band
            data_filtered = data[data['Filter'] == filter]
            
            if not data_filtered.empty:
                #.Create LightCurve and Variability indexes objects and extract properties
                lc = LightCurve(data_filtered[time_col], 
                               data_filtered[mag_col], 
                               data_filtered[err_col])
                var = VariabilityIndex(lc)
                var.suppress_warnings_globally(cls)
                #.LightCurve properties to extract
                props_lc = ['N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min', 
                        'ptp', 'range', 'std', 'time_max', 'time_min', 
                        'time_span', 'weighted_average']
                props_var = ['Abbe', 'IQR', 'Lag1AutoCorr', 'RoMS', 'ShapiroWilk',
                        'andersonDarling', 'chisquare', 'kurtosis', 'mad',
                        'norm_ptp', 'normalisedExcessVariance', 'reducedChiSquare',
                        'skewness', 'Q_index']
                
                #.Extract each property (returns NaN if missing)
                for prop in props_lc:
                    file_data[prop] = getattr(lc, prop, np.nan)
                for prop in props_var:
                    file_data[prop] = getattr(var, prop, np.nan)
                
                #.Calculate mean RA and DEC
                file_data['ra_mean'] = data_filtered[ra_col].mean()
                file_data['dec_mean'] = data_filtered[dec_col].mean()
                file_data['error'] = None  #.no error occurred
                
            else:
                #.If no data is available for specified filter
                file_data.update({prop: np.nan for prop in [
                    'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                    'ptp', 'range', 'std', 'time_max', 'time_min',
                    'time_span', 'weighted_average', 'ra_mean', 'dec_mean'
                ]})
                file_data['N'] = 0  #.set count to zero
                file_data['error'] = 'No data for filter'
                
        except Exception as e:
            #.If error, fill with NaN
            file_data.update({prop: np.nan for prop in [
                'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                'ptp', 'range', 'std', 'time_max', 'time_min',
                'time_span', 'weighted_average', 'ra_mean', 'dec_mean'
            ]})
            file_data['error'] = str(e)  #.storing error message
            print(f"✗ Error in {file}: {e}")
        
        results.append(file_data)
    
    #.Save final results sorted by file number
    final_df = pd.DataFrame(results).sort_values('file_number')
    output_path = os.path.join(parent_folder, f'variability_{filter}_final.csv')
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
        - 'ra': column name for RA values
        - 'dec': column name for DEC values
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
    - ra_mean: Mean RA value
    - dec_mean: Mean DEC value
    - error: Error message if processing failed, None otherwise
    
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
    ra_col, dec_col, time_col, mag_col, err_col = columns['ra'], columns['dec'], columns['time'], columns['mag'], columns['err']
    parent_folder = os.path.dirname(folder)
    
    #.List and sort files numerically by embedded number in filename
    all_files = sorted(
        [f for f in os.listdir(folder) if f.startswith('UKIRT2007_lc_') and f.endswith('.csv')],
        key=lambda x: int(re.search(r'UKIRT2007_lc_(\d+)\.csv', x).group(1)) if re.search(r'UKIRT2007_lc_(\d+)\.csv', x) else 0
    )
    
    print(f"Processing {len(all_files)} files...")
    
    results = []  #.To store all processing results
    
    #.Processing all files at once (no batches)
    for file in tqdm(all_files, desc="Processing files"):
        # Extracting numeric identifier from filename
        file_number = int(re.search(r'UKIRT2007_lc_(\d+)\.csv', file).group(1))
        file_data = {'file': file, 'file_number': file_number, 'filter': filter}
        
        try:
            #.Loading columns used (including RA and DEC)
            data = pd.read_csv(os.path.join(folder, file), 
                              usecols=['Filter', ra_col, dec_col, time_col, mag_col, err_col])
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
                
                #.Calculate mean RA and DEC
                file_data['ra_mean'] = data_filtered[ra_col].mean()
                file_data['dec_mean'] = data_filtered[dec_col].mean()
                file_data['error'] = None  #.no error occurred
                
            else:
                #.If no data is available for specified filter
                file_data.update({prop: np.nan for prop in [
                    'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                    'ptp', 'range', 'std', 'time_max', 'time_min',
                    'time_span', 'weighted_average', 'ra_mean', 'dec_mean'
                ]})
                file_data['N'] = 0  #.set count to zero
                file_data['error'] = 'No data for filter'
                
        except Exception as e:
            #.If error, fill with NaN
            file_data.update({prop: np.nan for prop in [
                'N', 'SNR', 'max', 'mean', 'mean_err', 'median', 'min',
                'ptp', 'range', 'std', 'time_max', 'time_min',
                'time_span', 'weighted_average', 'ra_mean', 'dec_mean'
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

def clean_light_curves(input_folder, output_folder, bad_hjds):
    """
    Removes specifics HJD from lightcurves.
    
    Parameters:
    input_folder: folder with lcs
    output_folder: folder to save clean lcs
    bad_hjds: HJDs list to be removed
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    stats = {
        'processed': 0,
        'cleaned': 0,
        'rows_removed': 0
    }
    
    files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    for file_path in tqdm(files, desc="Processando arquivos"):
        try:
            df = pd.read_csv(file_path)
            
            if 'HJD' not in df.columns:
                continue
            
            original_len = len(df)
            df_clean = df[~df['HJD'].isin(bad_hjds)]
            removed = original_len - len(df_clean)
            
            if removed > 0:
                stats['cleaned'] += 1
                stats['rows_removed'] += removed
            
            output_path = os.path.join(output_folder, os.path.basename(file_path))
            df_clean.to_csv(output_path, index=False)
            stats['processed'] += 1
            
        except Exception as e:
            print(f"Error: {os.path.basename(file_path)}: {e}")
    
    return stats


def quantiles_comp(df, coluna_x, coluna_y, q=0.01, min_points=200, bins=None, max_bin_value=20.1):
    """
    Calculates the percentile of column_y as a function of coluna_x using adaptive or user-defined bins.
    All values above max_bin_value are grouped into a single bin.
    
    Parameters:
        df: DataFrame
        coluna_x: str - reference column (e.g., 'mean')
        coluna_y: str - column to calculate percentile (e.g., 'mean_err')
        q: float - desired quantile
        min_points: int - used only if bins is None
        bins: int or array-like, optional - number of bins or array of bin edges
        max_bin_value: float - all values above this go into a single bin
    
    Returns:
        DataFrame with columns:
            coluna_x: bin center
            coluna_y: percentile value
    """
    df = df.copy()

    #.bins
    if bins is None:
        N = len(df[df[coluna_x] <= max_bin_value])
        n_bins = max(10, N // min_points)
        bins = np.linspace(df[coluna_x].min(), max_bin_value, n_bins)
    elif isinstance(bins, int):
        bins = np.linspace(df[coluna_x].min(), max_bin_value, bins)
    else:
        bins = np.array(bins)
        bins = bins[bins <= max_bin_value]

    #.last bin added with all values above max_bin_value
    bins = np.append(bins, np.inf)

    #.bin the data
    df['bin'] = pd.cut(df[coluna_x], bins, include_lowest=True, right=True)

    #.quantile per bin
    result = df.groupby('bin')[coluna_y].quantile(q).reset_index()

    #.center of the bin
    def bin_center(x):
        if isinstance(x, pd.Interval):
            if np.isfinite(x.right):
                return x.mid
            else:
                return max_bin_value + 0.5  #.last bin representative value
        else:
            return x

    result[coluna_x] = result['bin'].apply(bin_center)
    return result[[coluna_x, coluna_y]]

def iqr_comp(df, coluna_x, coluna_y, min_points=200, bins=None, max_bin_value=20.1):
    """
    Calculates the IQR (Q3 - Q1) of column_y as a function of coluna_x using adaptive or user-defined bins.
    All values above max_bin_value are grouped into a single bin.
    """
    df = df.copy()

    #.bins
    if bins is None:
        N = len(df[df[coluna_x] <= max_bin_value])
        n_bins = max(10, N // min_points)
        bins = np.linspace(df[coluna_x].min(), max_bin_value, n_bins)
    elif isinstance(bins, int):
        bins = np.linspace(df[coluna_x].min(), max_bin_value, bins)
    else:
        bins = np.array(bins)
        bins = bins[bins <= max_bin_value]

    bins = np.append(bins, np.inf)

    #bin the data
    df['bin'] = pd.cut(df[coluna_x], bins, include_lowest=True, right=True)

    #.q1 and q3
    grouped = df.groupby('bin')[coluna_y]
    Q1 = grouped.quantile(0.25)
    Q3 = grouped.quantile(0.75)

    result = pd.DataFrame({
        'bin': Q1.index,
        'IQR': Q3.values - Q1.values
    })

    #.bin center
    def bin_center(x):
        if isinstance(x, pd.Interval):
            if np.isfinite(x.right):
                return x.mid
            else:
                return max_bin_value + 0.1
        else:
            return x

    result[coluna_x] = result['bin'].apply(bin_center)
    return result[[coluna_x, 'IQR']]

#computing peak-to-peake using the median of the top and bottom percentage of points
def comp_ptp(data, magnitudes, percentage):
    
    mag = data[magnitudes]
    
    #.numerical order
    mag_s = np.sort(mag)
    #.number of points in 5%
    n = len(mag_s)
    k = max(1, int(np.ceil(percentage * n)))
    
    val_inf = mag_s[:k]     
    val_sup = mag_s[-k:]    
    
    median_inf = np.median(val_inf)
    median_sup = np.median(val_sup)
    
    amp = median_sup - median_inf
    
    return amp




