"""
A class for reading light curves from the UKIRT 2007 campaign
"""
import os
import pandas as pd

print(os.getcwd())
class read_ukirt_2007:
    def __init__(self, 
                 #ids are 6 digits long integers between 000000 and 594494
                 id=-999, 
                 **kwargs):
        assert (id >= 0) and (id <= 594494),\
            "ID must be a 6-digit integer between 000000 and 594494"
        if 'file_path' not in kwargs:
            file_path = '/Users/juliaroquette/Work/Data/CygnusOB2/UKIRT_2007/clean_lc/'
        else:
            file_path = kwargs['file_path']
        os.getcwd()
        self.file_path = file_path
        self.filename = f'UKIRT2007_lc_{id:06d}.csv'
        # initialize light curves
        self.read_base_light_curves()
    
    def read_base_light_curves(self):
        try:
            # Read the light-curves file into a pandas dataframe
            df = pd.read_csv(os.path.join(
                self.file_path, 
                self.filename))
            # get the label of the sector containing the source
            self.sector = df['grid_label'].mode().values[0]
            self.n_detections = df['n_detections'].mode().values[0]
            # get magnitudes, uncertainties and julian dates
            self.mag_J = df.loc[df.Filter == 'J', 'MAG_AUTO']
            self.mag_H = df.loc[df.Filter == 'H', 'MAG_AUTO']
            self.mag_K = df.loc[df.Filter == 'K', 'MAG_AUTO']
            self.err_J = df.loc[df.Filter == 'J', 'MAGERR_AUTO']
            self.err_H = df.loc[df.Filter == 'H', 'MAGERR_AUTO']
            self.err_K = df.loc[df.Filter == 'K', 'MAGERR_AUTO']
            self.hjd_J = df.loc[df.Filter == 'J', 'HJD']
            self.hjd_H = df.loc[df.Filter == 'H', 'HJD']
            self.hjd_K = df.loc[df.Filter == 'K', 'HJD']
        except FileNotFoundError:
            print("File for light curve not found.")
            print(os.path.join(
                self.file_path, 
                self.filename))

    def get_colours(self):
        # get colours. Note that Julian dates are not exactly the same, 
        # need to match them
        
        pass