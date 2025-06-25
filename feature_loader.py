#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:24:50 2025

@author: dativa
"""
all_features= ['sex', 'age', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q']
no_lab_features = ['sex', 'age', 'b', 'c', 'h', 'i', 'j', 'k','l', 'm', 'n', 'o', 'p', 'q']
lasso_regression = ['sex', 'age', 'f', 'b', 'g', 'm', 'c', 'j', 'p', 'o', 'q', 'e']
crammers_v_correlation = ['sex', 'age', 'c', 'd', 'i', 'j', 'k', 'm', 'n']
mutual_information= ['sex', 'age', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'j', 'k', 'l', 'p', 'q']
backward_elimination = ['sex', 'age',  'b', 'c', 'd', 'g', 'h', 'j', 'k', 'm', 'n', 'p']
intersection = ['sex', 'age', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'j', 'k', 'm', 'n', 'p', 'q']

feature_groups = {
    "all_features": all_features,
    "no_lab_features": no_lab_features,
    "lasso_regression": lasso_regression,
    "crammers_v_correlation": crammers_v_correlation,
    "mutual_information": mutual_information,
    "backward_elimination": backward_elimination,
    "intersection": intersection
}


