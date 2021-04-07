#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:33:26 2020

@author: afo
"""

from sagemaker.sklearn.estimator import SKLearn
from inspect import getsourcefile
from os.path import abspath
import os

import config

# Get the working path of script
p = abspath(getsourcefile(lambda:0))
p = p.rsplit('/', 1)[0]
os.chdir(p)
print('Working Directory is: %s' % os.getcwd())

model_name = 'rf'


FRAMEWORK_VERSION = '0.23-1'  # framework version
role = config.aws_role  # get execution role
aws_sklearn = SKLearn(entry_point=p+'/model_scripts_aws/'+ model_name +'.py',  # change script name for different model
                      train_instance_type='ml.m4.2xlarge',
                      framework_version=FRAMEWORK_VERSION,
                      base_job_name= config.job_name + model_name,  # change for any name
                      role=role
                      # source_dir='./',
                      # requirements_file='requirements.txt'
                      )

# Send model to train
aws_sklearn.fit({'train': config.train_path, 'test': config.test_path}, wait=False)


