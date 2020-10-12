""" Mathematical model """
#from optimizer import optimizer
import pandas
from model import data_handler
from model.nrc_equations import NRC_eq as nrc
import logging
import math
import pyomo.environ as pyo


cnem_lb, cnem_ub = 0.8, 3
default_special_cost = 10.0
bigM = 100000


def model_factory(ds, parameters, special_product = -1):
    if special_product > 0:
        return Model_ReducedCost(ds, parameters, special_product)
    else:
        return Model(ds, parameters)


class Model:
    # TODO 1.2: Acho que os Dataframes não precisam mais ser globais se vamos substiruir por dicts, mas os dicts precisam
    ds: data_handler.Data = None
    headers_feed_lib: data_handler.Data.IngredientProperties = None  # Feed Library
    data_feed_lib: pandas.DataFrame = None  # Feed Library
    data_feed_scenario: pandas.DataFrame = None  # Feeds
    headers_feed_scenario: data_handler.Data.ScenarioFeedProperties = None  # Feeds
    data_scenario: pandas.DataFrame = None  # Scenario
    headers_scenario: data_handler.Data.ScenarioParameters = None  # Scenario

    p_id, p_feed_scenario, p_batch, p_breed, p_sbw, p_feed_time, p_target_weight, \
    p_bcs, p_be, p_l, p_sex, p_a2, p_ph, p_selling_price, \
    p_algorithm, p_identifier, p_lb, p_ub, p_tol, p_dmi_eq, p_obj, p_find_reduced_cost, p_ing_level = [None for i in range(23)]

    _batch_map: dict = None
    # batch_map = {batch_ID:
    #                  {"data_feed_scenario": {Feed_Scenario: {Feed_id: {col_name: [list_from_batch_file]}}},
    #                   "data_scenario": {ID: {col_name: [list_from_batch_file]}}
    #                   }
    #              }
    _diet = None
    _p_mpmr = None
    _p_mpgr = None
    _p_dmi = None
    _p_nem = None
    _p_neg = None
    _p_pe_ndf = None
    _p_cnem = None
    _p_cneg = None
    _var_names_x = None
    _p_swg = None
    _model_feeding_time = None
    _model_final_weight = None
    _p_fat_orient = None

    _print_model_lp = False
    _print_model_lp_infeasible = False
    _print_solution_xml = False

    opt_sol = None
    prefix_id = ""

    def __init__(self, out_ds, parameters):
        self._cast_data(out_ds, parameters)

    @staticmethod
    def _remove_inf(vector):
        for i in range(len(vector)):
            if vector[i] == float("-inf"):
                vector[i] = -bigM
            elif vector[i] == float("inf"):
                vector[i] = bigM

    def run(self, p_id, p_cnem):
        """Either build or update model, solve it and return solution = {dict xor None}"""
        logging.info("Populating and running model")
        try:
            self.opt_sol = None
            self._p_cnem = p_cnem
            if self.p_batch > 0:
                self._setup_batch()
            if not self._compute_parameters(p_id):
                self._infeasible_output(p_id)
                return None
            if self._diet is None:
                self._build_model()
            self._update_model()
            return self._solve(p_id)
        except Exception as e:
            logging.error("An error occurred in lp_model.py L86:\n{}".format(str(e)))
            return None

    def _get_params(self, p_swg):
        if p_swg is None:
            return dict(zip(["CNEm", "CNEg", "NEm", "NEg", "DMI", "MPm",  "peNDF"],
                            [self._p_cnem, self._p_cneg, self._p_nem, self._p_neg,
                             self._p_dmi, self._p_mpmr * 0.001, self._p_pe_ndf]))
        else:
            return dict(zip(["CNEm", "CNEg", "NEm", "NEg", "SWG", "DMI", "MPm",  "peNDF"],
                            [self._p_cnem, self._p_cneg, self._p_nem, self._p_neg, p_swg,
                             self._p_dmi, self._p_mpmr * 0.001, self._p_pe_ndf]))

    def _solve(self, problem_id):
        # TODO 2.2: Modificar o solve implementando as funções do Pyomo
        """Return None if solution is infeasible or Solution dict otherwise"""
#        diet = self._diet
        solver = pyo.SolverFactory('glpk')
        # self._diet.pprint()
        solver.solve(self._diet) 
        
        # diet.write_lp(name="CNEm_{}.lp".format(str(self._p_cnem)))
#        diet.solve()
#         status = self._diet.get_solution_status()
#         logging.info("Solution status: {}".format(status))
#         if status.__contains__("infeasible"):
#             self._infeasible_output(problem_id)
#             return None

        sol_id = {"Problem_ID": problem_id,
                  "Feeding Time": self._model_feeding_time,
                  "Initial weight": self.p_sbw,
                  "Final weight": self._model_final_weight}
        
#        print("Funcao objetivo: ",pyo.value(model.obj))
#for v in model.component_objects(pyo.Var, active=True):
#    print("Variavel", v)  
#    for index in v:
#        print ("    ", index, ": valor = ", pyo.value(v[index]), 
#               ", custo reduzido = ", model.rc[v[index]])
#
#for c in model.component_objects(pyo.Constraint, active=True):
#    print("Restricao:", c,  ", dual = ", model.dual[c], 
#          ", folga inferior = ", c.lslack(), ", folga superior =", c.uslack())
        sol = dict(zip([i for i in self._diet.v_x], [pyo.value(self._diet.v_x[i]) for i in self._diet.v_x]))
        sol["obj_func"] = pyo.value(self._diet.f_obj)
        sol["obj_cost"] = - pyo.value(self._diet.f_obj) + self.cst_obj
        if self.p_obj == "MaxProfitSWG" or self.p_obj == "MinCostSWG":
            sol["obj_cost"] *= self._p_swg
        sol["obj_revenue"] = self.revenue
#        for i in self.ingredient_ids:
#            sol["obj_cost"] += pyo.value(self._diet.v_x[i]) * self.dc_cost[i]

        params = self._get_params(self._p_swg)

#        sol_rhs = dict(zip(["{}_rhs".format(constraint) for constraint in self.constraints_names],
#                           diet.get_constraints_rhs(self.constraints_names)))
        sol_rhs = {}
        sol_rhs["fat orient"] = self._p_fat_orient
#        sol_red_cost = dict(zip(["{}_red_cost".format(var) for var in diet.get_variable_names()],
#                                diet.get_dual_reduced_costs())) #get dual values
#         sol_red             [self._diet.r_cost = dict(zip(["{}_red_cost".format(var) for var in self._diet.v_x],
        # #                    c[var] for var in self._diet.v_x]))
#        sol_dual = dict(zip(["{}_dual".format(const) for const in diet.get_constraints_names()],
#                            diet.get_dual_values())) # get dual reduced costs
        #TODO JOAO: Arrumar lslack e uslack 
#        sol_slack = dict(zip(["{}_slack".format(const) for const in diet.get_constraints_names()],
#                             diet.get_dual_linear_slacks()))
        sol = {**sol_id, **params, **sol, **sol_rhs}
#        sol = {**sol_id, **params, **sol, **sol_rhs, **sol_activity,
#               **sol, **sol_dual, **sol_red_cost, **sol_slack}
#        self.opt_sol = diet.get_solution_obj()

        return sol

    def _infeasible_output(self, problem_id):
        sol_id = {"Problem_ID": self.prefix_id + str(problem_id)}
        params = self._get_params(p_swg=None)
        sol = {**sol_id, **params}
        self.opt_sol = None
        # diet.write_lp(f"lp_infeasible_{str(problem_id)}.lp")
        logging.warning("Infeasible parameters:{}".format(sol))

    # Parameters filled by inner method ._cast_data()
    n_ingredients = None
    cost_vector = None
    cost_obj_vector = None
    constraints_names = None
    # revenue_obj_vector = None
    revenue = None
    expenditure_obj_vector = None
    dm_af_conversion = None
    cst_obj = None
    batch_execution_id = None
    scenario_parameters = None

    def __set_parameters(self, parameters):
        if isinstance(parameters, dict):
            [self.p_id, self.p_feed_scenario, self.p_batch, self.p_breed, self.p_sbw, self.p_feed_time,
             self.p_target_weight,self.p_bcs, self.p_be, self.p_l,
             self.p_sex, self.p_a2, self.p_ph, self.p_selling_price,
             self.p_algorithm, self.p_identifier, self.p_lb, self.p_ub, self.p_tol, self.p_dmi_eq, self.p_obj,
             self.p_find_reduced_cost, self.p_ing_level] = parameters.values()
        elif isinstance(parameters, list):
            [self.p_id, self.p_feed_scenario, self.p_batch, self.p_breed, self.p_sbw, self.p_feed_time,
             self.p_target_weight,self.p_bcs, self.p_be, self.p_l,
             self.p_sex, self.p_a2, self.p_ph, self.p_selling_price,
             self.p_algorithm, self.p_identifier, self.p_lb, self.p_ub, self.p_tol, self.p_dmi_eq, self.p_obj,
             self.p_find_reduced_cost] = parameters

    def _cast_data(self, out_ds, parameters):
        # TODO 1.0: Substituir os DataFrames por dicts
        """Retrieve parameters data from table. See data_handler.py for more"""
        self.ds = out_ds

        self.data_feed_scenario = self.ds.data_feed_scenario
        self.headers_feed_scenario = self.ds.headers_feed_scenario

        self.scenario_parameters = parameters
        self.__set_parameters(parameters)

        headers_feed_scenario = self.ds.headers_feed_scenario
        self.data_feed_scenario = self.ds.filter_column(self.ds.data_feed_scenario,
                                                        self.ds.headers_feed_scenario.s_feed_scenario,
                                                        self.p_feed_scenario)
        self.data_feed_scenario = self.ds.sort_df(self.data_feed_scenario, self.headers_feed_scenario.s_ID)

        self.ingredient_ids = list(
            self.ds.get_column_data(self.data_feed_scenario, self.headers_feed_scenario.s_ID, int))

        self.headers_feed_lib = self.ds.headers_feed_lib
        self.data_feed_lib = self.ds.filter_column(self.ds.data_feed_lib, self.headers_feed_lib.s_ID,
                                                   self.ingredient_ids)

        self.n_ingredients = self.data_feed_scenario.shape[0]
        self.dc_cost = self.ds.sorted_column(self.data_feed_scenario,
                                             headers_feed_scenario.s_feed_cost,
                                             self.ingredient_ids,
                                             self.headers_feed_scenario.s_ID,
                                             return_dict=True)
        
        self.dc_ub = self.ds.sorted_column(self.data_feed_scenario,
                                           self.headers_feed_scenario.s_max,
                                           self.ingredient_ids,
                                           self.headers_feed_scenario.s_ID,
                                           return_dict=True)
        
        self.dc_lb = self.ds.sorted_column(self.data_feed_scenario,
                                           self.headers_feed_scenario.s_min,
                                           self.ingredient_ids,
                                           self.headers_feed_scenario.s_ID,
                                           return_dict=True)
        
        self.dc_dm_af_conversion = self.ds.sorted_column(self.data_feed_lib,
                                                        self.headers_feed_lib.s_DM,
                                                        self.ingredient_ids,
                                                        self.headers_feed_lib.s_ID,
                                                        return_dict=True)
        
        self.dc_nem = self.ds.sorted_column(self.data_feed_lib,
                                            self.headers_feed_lib.s_NEma, 
                                            self.ingredient_ids,
                                            self.headers_feed_lib.s_ID,
                                            return_dict=True)
        
        self.dc_mpm = self.ds.sorted_column(self.data_feed_lib,
                                            [self.headers_feed_lib.s_DM,
                                             self.headers_feed_lib.s_TDN,
                                             self.headers_feed_lib.s_CP,
                                             self.headers_feed_lib.s_RUP,
                                             self.headers_feed_lib.s_Forage,
                                             self.headers_feed_lib.s_Fat,
                                             self._p_fat_orient],
                                            self.ingredient_ids,
                                            self.headers_feed_lib.s_ID,
                                            return_dict=True)
        
        self.dc_fat = self.ds.sorted_column(self.data_feed_lib,
                                            self.headers_feed_lib.s_Fat,
                                            self.ingredient_ids,
                                            self.headers_feed_lib.s_ID,
                                            return_dict=True)
        
        mp_properties = self.ds.sorted_column(self.data_feed_lib,
                                              [self.headers_feed_lib.s_DM,
                                               self.headers_feed_lib.s_TDN,
                                               self.headers_feed_lib.s_CP,
                                               self.headers_feed_lib.s_RUP,
                                               self.headers_feed_lib.s_Forage,
                                               self.headers_feed_lib.s_Fat,
                                               self._p_fat_orient],
                                              self.ingredient_ids,
                                              self.headers_feed_lib.s_ID,
                                              return_dict=True)
        
        rup = self.ds.sorted_column(self.data_feed_lib,
                                    self.headers_feed_lib.s_RUP,
                                    self.ingredient_ids,
                                    self.headers_feed_lib.s_ID,
                                    return_dict=True)
        
        cp = self.ds.sorted_column(self.data_feed_lib,
                                   self.headers_feed_lib.s_CP,
                                   self.ingredient_ids,
                                   self.headers_feed_lib.s_ID,
                                   return_dict=True)
        
        ndf = self.ds.sorted_column(self.data_feed_lib,
                                    self.headers_feed_lib.s_NDF,
                                    self.ingredient_ids,
                                    self.headers_feed_lib.s_ID,
                                    return_dict=True)
        
        pef = self.ds.sorted_column(self.data_feed_lib,
                                    self.headers_feed_lib.s_pef,
                                    self.ingredient_ids,
                                    self.headers_feed_lib.s_ID,
                                    return_dict=True)
        self.dc_mpm = {}
        self.dc_rdp = {}
        self.dc_pendf = {}
        for ids in self.ingredient_ids:
            self.dc_mpm[ids] = nrc.mp(*mp_properties[ids])
            self.dc_rdp[ids] = (1 - rup[ids]) * cp[ids]
            self.dc_pendf[ids] = ndf[ids] * pef[ids]

        if self.p_batch > 0:
            # TODO 1.1: Aconselho a substituir esses DFs por dicts tb e manter a mesma lógica de busca
            try:
                batch_feed_scenario = self.ds.batch_map[self.p_id]["data_feed_scenario"][self.p_feed_scenario]
                # {Feed_id: {col_name: [list_from_batch_file]}}
            except KeyError:
                logging.warning(f"No Feed_scenario batch for scenario {self.p_id},"
                                f" batch {self.p_batch}, feed_scenario{self.p_feed_scenario}")
                batch_feed_scenario = {}
            try:
                batch_scenario = self.ds.batch_map[self.p_id]["data_scenario"][self.p_id]
                # {col_name: [list_from_batch_file]}}
            except KeyError:
                logging.warning(f"No Scenario batch for scenario {self.p_id},"
                                f" batch {self.p_batch}, scenario{self.p_feed_scenario}")
                batch_scenario = {}

            self._batch_map = {"data_feed_scenario": batch_feed_scenario,
                               "data_scenario": batch_scenario}

    def _compute_parameters(self, problem_id):
        # TODO 1.3: Talvez precise atualizar algo aqui depois de mudar os DFs para dicts, checar.

        """Compute parameters variable with CNEm"""
        self._p_mpmr, self._p_dmi, self._p_nem, self._p_pe_ndf = \
            nrc.get_all_parameters(self._p_cnem, self.p_sbw, self.p_bcs, self.p_be, self.p_l, self.p_sex, self.p_a2,
                                   self.p_ph, self.p_target_weight, self.p_dmi_eq)

        self._p_cneg = nrc.cneg(self._p_cnem)
        self._p_neg = nrc.neg(self._p_cneg, self._p_dmi, self._p_cnem, self._p_nem)
        if self._p_neg is None:
            return False
        if math.isnan(self.p_feed_time) or self.p_feed_time == 0:
            self._model_final_weight = self.p_target_weight
            self._p_swg = nrc.swg(self._p_neg, self.p_sbw, self._model_final_weight)
            self._model_feeding_time = (self.p_target_weight - self.p_sbw)/self._p_swg
        elif math.isnan(self.p_target_weight) or self.p_target_weight == 0:
            self._model_feeding_time = self.p_feed_time
            self._p_swg = nrc.swg_time(self._p_neg, self.p_sbw, self._model_feeding_time)
            self._model_final_weight = self._model_feeding_time * self._p_swg + self.p_sbw
        else:
            raise Exception("target weight and feeding time cannot be defined at the same time")

        self._p_mpgr = nrc.mpg(self._p_swg, self._p_neg, self.p_sbw, self._model_final_weight, self._model_feeding_time)

#        self.cost_obj_vector = self.cost_vector.copy()
        self.revenue = self.p_selling_price * (self.p_sbw + self._p_swg * self._model_feeding_time)
        # self.revenue_obj_vector = self.cost_vector.copy()
        self.dc_expenditure = self.dc_cost.copy()
        
        if self.p_obj == "MaxProfit" or self.p_obj == "MinCost":
            for i in self.ingredient_ids:
                self.dc_expenditure[i] = - self.dc_cost[i] * self._p_dmi * self._model_feeding_time / self.dc_dm_af_conversion[i]
        elif self.p_obj == "MaxProfitSWG" or self.p_obj == "MinCostSWG":
            for i in self.ingredient_ids:
                self.dc_expenditure[i] = - self.dc_cost[i] * self._p_dmi * self._model_feeding_time / (self.dc_dm_af_conversion[i] * self._p_swg)
        
        if self.p_obj == "MaxProfit":
            self.cst_obj = self.revenue
        elif self.p_obj == "MaxProfitSWG":
            self.cst_obj = self.revenue/self._p_swg
        elif self.p_obj == "MinCost" or self.p_obj == "MinCostSWG":
            self.cst_obj = 0
            
#        for i in range(len(self.cost_vector)):
#            # self.revenue_obj_vector[i] = self.p_selling_price * (self.p_sbw + self._p_swg * self._model_feeding_time)
#            self.expenditure_obj_vector[i] = self.cost_obj_vector[i] * self._p_dmi * self._model_feeding_time
        # r = [self.revenue_obj_vector[i] - self.expenditure_obj_vector[i] for i in range(len(self.revenue_obj_vector))]
#        if self.p_obj == "MaxProfit":
##            for i in range(len(self.cost_vector)):
##                self.cost_obj_vector[i] = - self.expenditure_obj_vector[i]
#            self.cst_obj = self.revenue
#        elif self.p_obj == "MinCost":
##            for i in range(len(self.cost_vector)):
##                self.cost_obj_vector[i] = - self.expenditure_obj_vector[i]
#            self.cst_obj = 0
#        elif self.p_obj == "MaxProfitSWG":
#            for i in self.ingredient_ids:
#                self.dc_cost[i] /= self._p_swg
#            self.cst_obj = self.revenue/self._p_swg
#        elif self.p_obj == "MinCostSWG":
#            for i in range(len(self.cost_vector)):
#                self.cost_obj_vector[i] = -(self.expenditure_obj_vector[i])/self._p_swg
#            self.cst_obj = 0

#         self.cost_obj_vector_mono = self.cost_obj_vector.copy()
        return True

    def _build_model(self):
        # TODO 2.0: Implementar construção do modelo pelo Pyomo
        """Build model (initially based on CPLEX 12.8.1)"""
#        self._remove_inf(self.dc_cost) PRECISA ARRUMAR
        
        self._diet = pyo.ConcreteModel()
        self._diet.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT) # resgatar duais
        self._diet.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT) # resgatar custos reduzidos
        ### Conjuntos ###
        self._diet.s_var_set = pyo.Set(initialize=self.ingredient_ids)
        ### Parametros ###
        self._diet.p_model_offset = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_model_cost = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_model_lb = pyo.Param(self._diet.s_var_set, initialize=self.dc_lb)
        self._diet.p_model_ub = pyo.Param(self._diet.s_var_set, initialize=self.dc_ub)
        self._diet.p_model_nem = pyo.Param(self._diet.s_var_set, initialize=self.dc_nem)
        self._diet.p_model_mpm = pyo.Param(self._diet.s_var_set, initialize=self.dc_mpm)
        self._diet.p_model_rdp = pyo.Param(self._diet.s_var_set, initialize=self.dc_rdp) 
        self._diet.p_model_fat = pyo.Param(self._diet.s_var_set, initialize=self.dc_fat)
        self._diet.p_model_pendf = pyo.Param(self._diet.s_var_set, initialize=self.dc_pendf)
        self._diet.p_rhs_cnem_ge = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_cnem_le = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_sum_1 = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_mpm = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_rdp = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_fat = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_alt_fat_ge = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_alt_fat_le = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_pendf = pyo.Param(within=pyo.Any, mutable=True)
        ### Functions ###
        def bound_function(model, i):
            return (model.p_model_lb[i], model.p_model_ub[i])
        ### Variaveis ###
        self._diet.v_x = pyo.Var(self._diet.s_var_set, bounds=bound_function)
        ### Objetivo ###
        self._diet.f_obj = pyo.Objective(expr=(self._diet.p_model_offset + pyo.summation(self._diet.p_model_cost, self._diet.v_x)), sense=pyo.maximize)
        ### Restricoes ###
        self._diet.c_cnem_ge = pyo.Constraint(expr=pyo.summation(self._diet.p_model_nem, self._diet.v_x) >= self._diet.p_rhs_cnem_ge)
        self._diet.c_cnem_le = pyo.Constraint(expr=pyo.summation(self._diet.p_model_nem, self._diet.v_x) >= self._diet.p_rhs_cnem_le)
        self._diet.c_sum_1 = pyo.Constraint(expr=pyo.summation(self._diet.v_x) == self._diet.p_rhs_sum_1)
        self._diet.c_mpm = pyo.Constraint(expr=pyo.summation(self._diet.p_model_mpm, self._diet.v_x) >= self._diet.p_rhs_mpm)
        self._diet.c_rdp = pyo.Constraint(expr=pyo.summation(self._diet.p_model_rdp, self._diet.v_x) >= self._diet.p_rhs_rdp)
        self._diet.c_fat = pyo.Constraint(expr=pyo.summation(self._diet.p_model_fat, self._diet.v_x) <= self._diet.p_rhs_fat)
        self._diet.c_alt_fat_ge = pyo.Constraint(expr=pyo.summation(self._diet.p_model_fat, self._diet.v_x) >= self._diet.p_rhs_alt_fat_ge)
        self._diet.c_alt_fat_le = pyo.Constraint(expr=pyo.summation(self._diet.p_model_fat, self._diet.v_x) <= self._diet.p_rhs_alt_fat_le)
        self._diet.c_pendf = pyo.Constraint(expr=pyo.summation(self._diet.p_model_pendf, self._diet.v_x) <= self._diet.p_rhs_pendf)
        
#        self._var_names_x = ["x" + str(f_id)
#                             for f_id in self.ingredient_ids]

#        diet = self._diet
#        diet.set_sense(sense="max")

        

#        x_vars = list(diet.add_variables(obj=self.cost_obj_vector,
#                                         lb=self.ds.sorted_column(self.data_feed_scenario,
#                                                                  self.headers_feed_scenario.s_min,
#                                                                  self.ingredient_ids,
#                                                                  self.headers_feed_scenario.s_ID),
#                                         ub=self.ds.sorted_column(self.data_feed_scenario,
#                                                                  self.headers_feed_scenario.s_max,
#                                                                  self.ingredient_ids,
#                                                                  self.headers_feed_scenario.s_ID),
#                                         names=self._var_names_x))
        
        
#        diet.set_obj_offset(self.cst_obj)

#        "Constraint: sum(x a) == CNEm"
#        diet.add_constraint(names=["CNEm GE"],
#                            lin_expr=[[x_vars, self.ds.sorted_column(self.data_feed_lib,
#                                                                     self.headers_feed_lib.s_NEma,
#                                                                     self.ingredient_ids,
#                                                                     self.headers_feed_lib.s_ID)]],
#                            rhs=[self._p_cnem * 0.999],
#                            senses=["G"]
#                            )
#        diet.add_constraint(names=["CNEm LE"],
#                            lin_expr=[[x_vars, self.ds.sorted_column(self.data_feed_lib,
#                                                                     self.headers_feed_lib.s_NEma,
#                                                                     self.ingredient_ids,
#                                                                     self.headers_feed_lib.s_ID)]],
#                            rhs=[self._p_cnem * 1.001],
#                            senses=["L"]
#                            )
#        "Constraint: sum(x) == 1"
#        diet.add_constraint(names=["SUM 1"],
#                            lin_expr=[[x_vars, [1] * len(x_vars)]],
#                            rhs=[1],
#                            senses=["E"]
#                            )
#        "Constraint: sum(x a)>= MPm"
#        mp_properties = self.ds.sorted_column(self.data_feed_lib,
#                                              [self.headers_feed_lib.s_DM,
#                                               self.headers_feed_lib.s_TDN,
#                                               self.headers_feed_lib.s_CP,
#                                               self.headers_feed_lib.s_RUP,
#                                               self.headers_feed_lib.s_Forage,
#                                               self.headers_feed_lib.s_Fat,
#                                               self._p_fat_orient],
#                                              self.ingredient_ids,
#                                              self.headers_feed_lib.s_ID)
#        mpm_list = [nrc.mp(*row) for row in mp_properties]
#
#        diet.add_constraint(names=["MPm"],
#                            lin_expr=[[x_vars, mpm_list]],
#                            rhs=[(self._p_mpmr + self._p_mpgr) * 0.001 / self._p_dmi],
#                            senses=["G"]
#                            )
#
#        rdp_data = [(1 - self.ds.sorted_column(self.data_feed_lib,
#                                               self.headers_feed_lib.s_RUP,
#                                               self.ingredient_ids,
#                                               self.headers_feed_lib.s_ID)[x_index])
#                    * self.ds.sorted_column(self.data_feed_lib,
#                                            self.headers_feed_lib.s_CP,
#                                            self.ingredient_ids,
#                                            self.headers_feed_lib.s_ID)[x_index]
#                    for x_index in range(len(x_vars))]
#
#        "Constraint: RUP: sum(x a) >= 0.125 CNEm"
#        diet.add_constraint(names=["RDP"],
#                            lin_expr=[[x_vars, rdp_data]],
#                            rhs=[0.125 * self._p_cnem],
#                            senses=["G"]
#                            )
#
#        "Constraint: Fat: sum(x a) <= 0.06 DMI"
#        diet.add_constraint(names=["Fat"],
#                            lin_expr=[[x_vars, self.ds.sorted_column(self.data_feed_lib,
#                                                                     self.headers_feed_lib.s_Fat,
#                                                                     self.ingredient_ids,
#                                                                     self.headers_feed_lib.s_ID)]],
#                            rhs=[0.06],
#                            senses=["L"]
#                            )
#
#        "Constraint: Alternative fat: sum(x a) <= 0.039 DMI or sum(x a) >= 0.039 DMI"
#        diet.add_constraint(names=["alternative_fat"],
#                            lin_expr=[[x_vars, self.ds.sorted_column(self.data_feed_lib,
#                                                                     self.headers_feed_lib.s_Fat,
#                                                                     self.ingredient_ids,
#                                                                     self.headers_feed_lib.s_ID)]],
#                            rhs=[0.039],
#                            senses=[self._p_fat_orient]
#                            )
#
#        "Constraint: peNDF: sum(x a) <= peNDF DMI"
#        pendf_data = [self.ds.sorted_column(self.data_feed_lib,
#                                            self.headers_feed_lib.s_NDF,
#                                            self.ingredient_ids,
#                                            self.headers_feed_lib.s_ID)[x_index]
#                      * self.ds.sorted_column(self.data_feed_lib,
#                                              self.headers_feed_lib.s_pef,
#                                              self.ingredient_ids,
#                                              self.headers_feed_lib.s_ID)[x_index]
#                      for x_index in range(len(x_vars))]
#        diet.add_constraint(names=["peNDF"],
#                            lin_expr=[[x_vars, pendf_data]],
#                            rhs=[self._p_pe_ndf],
#                            senses=["G"]
#                            )

#        self.constraints_names = diet.get_constraints_names()
        # diet.write_lp(name="file.lp")
        pass

    def _update_model(self):
        # TODO 2.1: Atualizar utilizando as funções do Pyomo.
        # TODO 2.2 Aqui vai existir uma gambiarra. Vc vai utilizar os nomes dos objetos que foram criados em _build()
        """Update RHS values on the model based on the new CNEm and updated parameters"""
        self._diet.p_model_offset = self.cst_obj
        for i in self._diet.s_var_set:
            self._diet.p_model_cost[i] = self.dc_expenditure[i]
        
        self._diet.p_rhs_cnem_ge = self._p_cnem * 0.999
        self._diet.p_rhs_cnem_le = self._p_cnem * 1.001
        self._diet.p_rhs_sum_1 = 1
        self._diet.p_rhs_mpm = (self._p_mpmr + self._p_mpgr) * 0.001 / self._p_dmi
        self._diet.p_rhs_rdp = 0.125 * self._p_cnem
        self._diet.p_rhs_fat = 0.06
        self._diet.p_rhs_alt_fat_ge = 0.039
        self._diet.p_rhs_alt_fat_le = 0.039
        self._diet.p_rhs_pendf = self._p_pe_ndf
        
        if self._p_fat_orient == "G":
            self._diet.c_alt_fat_ge.activate()
            self._diet.c_alt_fat_le.deactivate()
        else:
            self._diet.c_alt_fat_ge.deactivate()
            self._diet.c_alt_fat_le.activate()
        
#        new_rhs = {
#            "CNEm GE": self._p_cnem * 0.999,
#            "CNEm LE": self._p_cnem * 1.001,
#            "SUM 1": 1,
#            "MPm": self._p_mpmr * 0.001 / self._p_dmi,
#            "RDP": 0.125 * self._p_cnem,
#            "Fat": 0.06,
#            "peNDF": self._p_pe_ndf}

#        seq_of_pairs = tuple(zip(new_rhs.keys(), new_rhs.values()))
#        self._diet.set_constraint_rhs(seq_of_pairs)
#        self._diet.set_constraint_sense("alternative_fat", self._p_fat_orient)
#        self._diet.set_objective_function(list(zip(self._var_names_x, self.cost_obj_vector)), self.cst_obj)


    def set_fat_orient(self, direction):
        self._p_fat_orient = direction

    def set_batch_params(self, i):
        self.batch_execution_id = i

    def _setup_batch(self):
        # TODO 1.4: Mudar de dataframe para o dict
        # batch_map = {"data_feed_scenario": {Feed_id: {col_name: [list_from_batch_file]}}},
        #              "data_scenario": {col_name: [list_from_batch_file]}}
        #              }

        for col_name, vector in self._batch_map["data_scenario"].items():
            self.scenario_parameters[col_name] = vector[self.batch_execution_id]
        self.__set_parameters(self.scenario_parameters)

        for ing_id, data in self._batch_map["data_feed_scenario"].items():
            for col_name, vector in data.items():
                if col_name == self.headers_feed_scenario.s_feed_cost:
                    self.cost_vector[self.ingredient_ids.index(ing_id)] = vector[self.batch_execution_id]
                elif col_name == self.headers_feed_scenario.s_min:
                    self.data_feed_scenario.loc[
                        self.data_feed_scenario[self.headers_feed_scenario.s_ID] == ing_id,
                        self.headers_feed_scenario.s_min
                    ] = vector[self.batch_execution_id]
                elif col_name == self.headers_feed_scenario.s_max:
                    self.data_feed_scenario.loc[
                        self.data_feed_scenario[self.headers_feed_scenario.s_ID] == ing_id,
                        self.headers_feed_scenario.s_max
                    ] = vector[self.batch_execution_id]

class Model_ReducedCost(Model):
    # TODO 1.X: Mesmas coisas que foram feitas na classe pai
    # TODO 2.X: Mesmas coisas que foram feitas na classe pai

    _special_ingredient = None
    _special_id = None
    _special_cost = None

    def __init__(self, out_ds, parameters, special_id, special_cost = default_special_cost):
        Model.__init__(self, out_ds, parameters)
        self._special_id = special_id
        for i in range(len(self.ingredient_ids)):
            if self.ingredient_ids[i] == self._special_id:
                self._special_ingredient = i
        self._special_cost = special_cost

    def _solve(self, problem_id):
        sol = Model._solve(self, problem_id)
        sol["x" + str(self._special_id) + "_price_" + str(int(100 * self.p_ing_level))] = self._special_cost
        return sol

    def _compute_parameters(self, problem_id):
        if not Model._compute_parameters(self, problem_id):
            return False
        else:
            self.cost_obj_vector[self._special_ingredient] = self._special_cost/self.dm_af_conversion[self._special_ingredient]
            self.expenditure_obj_vector[self._special_ingredient] = self.cost_obj_vector[self._special_ingredient] * self._p_dmi * self._model_feeding_time        

            if self.p_obj == "MaxProfit" or self.p_obj == "MinCost":
                self.cost_obj_vector[self._special_ingredient] = - self.expenditure_obj_vector[self._special_ingredient]   
            elif self.p_obj == "MaxProfitSWG" or self.p_obj == "MinCostSWG":
                self.cost_obj_vector[self._special_ingredient] = -(self.expenditure_obj_vector[self._special_ingredient])/self._p_swg
            return True
        
    def set_special_cost(self, cost=default_special_cost):
        self._special_cost = cost

    def get_special_cost(self):
        return self._special_cost

    def get_special_id(self):
        return self._special_id

    def get_special_ingredient(self):
        return self._special_ingredient