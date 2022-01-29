import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA, TruncatedSVD
import time
from datetime import datetime

class scTraceModel(object):
    def __init__(self, params):
        super(scTraceModel, self).__init__()
        self.params = params
        self.timepoint_scdata = {}
        self.__timepoint_scdata_num = 1
        
    def add_timepoint_scdata(self,timepoint_scdata):
        data=pd.DataFrame(timepoint_scdata)
        self.timepoint_scdata[self.__timepoint_scdata_num]=ad.AnnData(data)
        self.__timepoint_scdata_num += 1

    
    def __create_folder(self):
        Output_Dir = self.params['Output_Dir']
        Output_Name = self.params['Output_Name']
        if not os.path.exists(Output_Dir):
            raise ValueError(f'{Output_Dir} : No such directory')
        elif not os.path.exists(Output_Dir+Output_Name):
            os.makedirs(Output_Dir+Output_Name)
        else:
            print(f"The result folder {Output_Dir+Output_Name} exists!")

        if not os.path.exists(Output_Dir+Output_Name+"/"+"ClusteringResults"):
            os.makedirs(Output_Dir+Output_Name+"/"+"ClusteringResults")
        if not os.path.exists(Output_Dir+Output_Name+"/"+"TrajactoryResults"):
            os.makedirs(Output_Dir+Output_Name+"/"+"TrajactoryResults")
    
    def cluster_timepoint_data(self):
        self.__create_folder()
        for (timepoint, adata) in self.timepoint_scdata.items():
            if self.params['clustering_method'] == 'Louvain':
                adata_copy=adata.copy()
                adata_copy.obs_names_make_unique()
                adata_copy.var_names_make_unique()
                # normalize log-transform
                sc.pp.normalize_per_cell(adata_copy, counts_per_cell_after=1e4)
                sc.pp.log1p(adata_copy)
                
    def get_trajactory(self):
        pass
    
    def run_sctrace(self):
        
        self.cluster_timepoint_data()
        
        self.get_trajactory()
        
                
        
