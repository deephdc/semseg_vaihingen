# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 - 2019 Karlsruhe Institute of Technology - Steinbuch Centre for Computing
# This code is distributed under the MIT License
# Please, see the LICENSE file
#
"""
Created on Sat Aug 10 08:47:51 2019
@author: vykozlov
"""

import unittest
import semseg_vaihingen.models.deepaas_api as deepaas_api

class TestModelMethods(unittest.TestCase):
    
    def setUp(self):
        self.meta = deepaas_api.get_metadata()
        
    def test_model_metadata_type(self):
        """
        Test that get_metadata() returns dict
        """
        self.assertTrue(type(self.meta) is dict)
        
    def test_model_metadata_values(self):
        """
        Test that get_metadata() returns right values (subset)
        """
        self.assertEqual(self.meta['Name'].replace('-','_'),
                        'semseg_vaihingen'.replace('-','_'))
        self.assertEqual(self.meta['Author'], 'G.Cavallaro (FZJ), M.Goetz (KIT), V.Kozlov (KIT), A.Grupp (KIT)')
        self.assertEqual(self.meta['Author-email'], 'valentin.kozlov@kit.edu')


if __name__ == '__main__':
    unittest.main()
