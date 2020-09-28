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
                                     'pef, %NDF']},
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
                                             'Ingredient Level']},
              'sheet_batch': {'name': 'Batch',
                              'headers': ['Batch ID',
                                          'Filename',
                                          'Period col',
                                          'Initial Period',
                                          'Final Period',
                                          'Only Costs Batch']}

              }
OUTPUT_FILE = 'output.xlsx'
SOLVER = 'HiGHS'
