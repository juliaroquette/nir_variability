"""
A class for reading light curves from the UKIRT 2007 campaign
"""
import os
import pandas as pd

print(os.getcwd())
class Read_UKIRT_2007:
    def __init__(self, 
                 #ids are 6 digits long integers between 000000 and 594494
                 id=-999, 
                 **kwargs):
        assert (id >= 0) and (id <= 594494),\
            "ID must be a 6-digit integer between 000000 and 594494"
        if 'file_path' not in kwargs:
            file_path = '/Users/juliaroquette/Work/Data/CygnusOB2/UKIRT_2007/lc/'
        else:
            file_path = kwargs['file_path']
        os.getcwd()
        self.file_path = file_path
        self.filename = f'UKIRT2007_lc_{id:06d}.csv'
    
    def read_light_curves(self):
        try:
            # Read the light-curves file into a pandas dataframe
            df = pd.read_csv(os.path.join(
                self.file_path, 
                self.filename))
            
            # Process the dataframe if needed
            
            return df
        except FileNotFoundError:
            print("File not found.")
            print(os.path.join(
                self.file_path, 
                self.filename))
            return None