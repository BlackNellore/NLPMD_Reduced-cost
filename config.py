OUTPUT_FILE = 'output.xlsx'
SOLVER = 'cplex'
# RNS_FEED_PARAMETERS = {}
RNS_FEED_PARAMETERS = {'source': 'RNS/image.RData',
                       'report_diff': False,
                       'on_error': 0}  # 0: quit; 1: report and continue with NRC; -1: silent continue

INPUT_FILE = {'filename': {'name': 'input.xlsx'},
              'sheet_feed_lib': {'name': 'Feed Library',
                                 'headers': [
                                     'ID',
                                     'Feed',
                                     'Forage, %DM',
                                     'DM, %AF',
                                     'CP, %DM',
                                     'SP, %CP',
                                     'ADICP, %CP',
                                     'Sugars, %DM',
                                     'OA, %DM',
                                     'Fat, %DM',
                                     'Ash, %DM',
                                     'Starch, %DM',
                                     'NDF, %DM',
                                     'Lignin, %DM',
                                     'TDN, %DM',
                                     'NEma, Mcal/kg',
                                     'NEga, Mcal/kg',
                                     'RUP, %CP',
                                     'pef, %NDF',
                                     'NPN, %DM']},
              'sheet_feeds': {'name': 'Feeds',
                              'headers': ['Feed Scenario',
                                          'ID',
                                          'Min %DM',
                                          'Max %DM',
                                          'Cost [US$/kg AF]',
                                          'Name']},
              'sheet_scenario': {'name': 'Scenario',
                                 'headers': ['ID',
                                             'Feed Scenario',
                                             'Batch',
                                             'Breed',
                                             'SBW',
                                             'Feeding Time',
                                             'Target Weight',
                                             'BCS',
                                             'BE',
                                             'L',
                                             'SEX',
                                             'a2',
                                             'PH',
                                             'Selling Price [US$]',
                                             'Algorithm',
                                             'Identifier',
                                             'LB',
                                             'UB',
                                             'Tol',
                                             'DMI Equation',
                                             'Obj',
                                             'Find Reduced Cost',
                                             'Ingredient Level',
                                             'LCA ID']},
              'sheet_batch': {'name': 'Batch',
                              'headers': ['Batch ID',
                                          'Filename',
                                          'Period col',
                                          'Initial Period',
                                          'Final Period',
                                          'Only Costs Batch']},
              'sheet_lca': {'name': 'LCA',
                            'headers': ['ID',
                                        'weight_LCA_Phosphorous consumption (kg P)',
                                        'weight_LCA_CED 1.8 non renewable fossil+nuclear (MJ)',
                                        'weight_LCA_Climate change ILCD (kg CO2 eq)',
                                        'weight_LCA_Acidification ILCD (molc H+ eq)',
                                        'weight_LCA_Eutrophication CML baseline (kg PO4- eq)',
                                        'weight_LCA_Land competition CML non baseline (m2a)',
                                        'Methane_Equation',
                                        'N2O_Equation',
                                        'Normalize']},
              'sheet_lca_lib': {'name': 'LCA Library',
                                'headers': ['ID',
                                            'Name',
                                            'LCA_Phosphorous consumption (kg P)',
                                            'LCA_CED 1.8 non renewable fossil+nuclear (MJ)',
                                            'LCA_Climate change ILCD (kg CO2 eq)',
                                            'LCA_Acidification ILCD (molc H+ eq)',
                                            'LCA_Eutrophication CML baseline (kg PO4- eq)',
                                            'LCA_Land competition CML non baseline (m2a)']}

              }

