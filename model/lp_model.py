""" Mathematical model """
# from optimizer import optimizer
import pandas

from config import SOLVER
from model import data_handler
from model.nrc_equations import NRC_eq as nrc
import logging
import math
import pyomo.environ as pyo
from pyomo.opt.results import SolverResults

default_special_cost = 10.0


def model_factory(ds, parameters, special_product=-1):
    if special_product > 0:
        return ModelReducedCost(ds, parameters, special_product)
    else:
        return Model(ds, parameters)


class Model:
    _diet: pyo.ConcreteModel = None

    _print_model_lp = False
    _print_model_lp_infeasible = False
    _print_solution_xml = False

    opt_sol = None
    prefix_id = ""

    data = None
    parameters = None
    computed = None

    def __init__(self, out_ds, parameters):
        self.parameters = self.Parameters(parameters)
        self.data = self.Data(out_ds, self.parameters)
        self.computed = self.ComputedArrays()

    def run(self, p_id, p_cnem):
        """Either build or update model, solve it and return solution = {dict xor None}"""
        logging.info("Populating and running model")
        try:
            self.opt_sol = None
            self.parameters.cnem = p_cnem
            if self.parameters.p_batch > 0:
                self._setup_batch()
            if not self._compute_parameters():
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
            return dict(zip(["CNEm", "CNEg", "NEm", "NEg", "DMI", "MPm", "peNDF"],
                            [self.parameters.cnem, self.parameters.cneg, self.parameters.nem, self.parameters.neg,
                             self.parameters.dmi, self.parameters.mpmr * 0.001, self.parameters.pe_ndf]))
        else:
            return dict(zip(["CNEm", "CNEg", "NEm", "NEg", "SWG", "DMI", "MPm", "peNDF"],
                            [self.parameters.cnem, self.parameters.cneg, self.parameters.nem, self.parameters.neg,
                             p_swg, self.parameters.dmi, self.parameters.mpmr * 0.001, self.parameters.pe_ndf]))

    def _solve(self, problem_id):
        """Return None if solution is infeasible or Solution dict otherwise"""
        solver = pyo.SolverFactory(SOLVER)
        results = SolverResults()
        r = solver.solve(self._diet)
        results.load(r)
        if not (results.solver.status == pyo.SolverStatus.ok or
                pyo.TerminationCondition.optimal == results.solver.termination_condition):
            logging.info("Solution status: {}".format(results.solver.termination_condition))
            self._infeasible_output(problem_id)
            return None

        sol_id = {"Problem_ID": problem_id,
                  "Feeding Time": self.parameters.c_model_feeding_time,
                  "Initial weight": self.parameters.p_sbw,
                  "Final weight": self.parameters.c_model_final_weight}

        params = self._get_params(self.parameters.c_swg)

        sol = dict(zip(["x{}".format(i) for i in self._diet.v_x], [self._diet.v_x[i].value for i in self._diet.v_x]))
        sol["obj_func"] = self._diet.f_obj.expr()
        sol["obj_cost"] = - self._diet.f_obj.expr() + self.computed.cst_obj
        if self.parameters.p_obj == "MaxProfitSWG" or self.parameters.p_obj == "MinCostSWG":
            sol["obj_cost"] *= self.parameters.c_swg
        sol["obj_revenue"] = self.computed.revenue

        is_active_constraints = []
        l_slack = {}
        u_slack = {}
        lower = {}
        upper = {}
        duals = {}

        for c in self._diet.component_objects(pyo.Constraint):
            is_active_constraints.append(c.active)
            if c.active:
                duals["{}_dual".format(c)] = self._diet.dual[c]
                l_slack["{}_lslack".format(c)] = c.lslack()
                u_slack["{}_uslack".format(c)] = c.uslack()
                if c.has_lb():
                    lower["{}_lower".format(c)] = c.lower()
                    upper["{}_upper".format(c)] = "None"
                else:
                    lower["{}_lower".format(c)] = "None"
                    upper["{}_upper".format(c)] = c.upper()
            else:
                duals["{}_dual".format(c)] = "None"
                l_slack["{}_lslack".format(c)] = "None"
                u_slack["{}_uslack".format(c)] = "None"
                lower["{}_lower".format(c)] = "None"
                upper["{}_upper".format(c)] = "None"

        sol_red_cost = dict(zip(["x{}_red_cost".format(i) for i in self._diet.v_x],
                                [self._diet.rc[self._diet.v_x[i]] for i in self._diet.v_x]))

        sol_fat_orient = {"fat orient": self.parameters.p_fat_orient}

        sol = {**sol_id, **params, **sol, **sol_red_cost, **duals,
               **sol_fat_orient, **l_slack, **u_slack, **lower, **upper}

        return sol

    def _infeasible_output(self, problem_id):
        sol_id = {"Problem_ID": self.prefix_id + str(problem_id),
                  "Feeding Time": self.parameters.c_model_feeding_time,
                  "Initial weight": self.parameters.p_sbw,
                  "Final weight": self.parameters.c_model_final_weight}
        params = self._get_params(p_swg=None)
        sol = {**sol_id, **params}
        self.opt_sol = None
        logging.warning("Infeasible parameters:{}".format(sol))

    class ComputedArrays:
        revenue = None
        cst_obj = None
        dc_expenditure = None
        dc_mpm = None

        def __init__(self):
            pass

    class Parameters:
        # Initialized in Model
        mpmr = None
        mpgr = None
        dmi = None
        nem = None
        neg = None
        pe_ndf = None
        cnem = None
        cneg = None

        # External assignment
        p_batch_execution_id = None
        p_fat_orient = None

        # Computed in Model
        c_swg = None
        c_model_feeding_time = None
        c_model_final_weight = None
        c_batch_map: dict = None

        # From outer scope
        p_id, p_feed_scenario, p_batch, p_breed, p_sbw, p_feed_time, p_target_weight, p_bcs, p_be, p_l, p_sex, p_a2, \
        p_ph, p_selling_price, p_algorithm, p_identifier, p_lb, p_ub, p_tol, p_dmi_eq, p_obj, p_find_reduced_cost, \
        p_ing_level \
            = [None for i in range(23)]

        init_parameters = None

        def __init__(self, parameters):
            self.init_parameters = parameters
            self.set_parameters(parameters)

        def set_parameters(self, parameters):
            if isinstance(parameters, dict):
                [self.p_id, self.p_feed_scenario, self.p_batch, self.p_breed, self.p_sbw, self.p_feed_time,
                 self.p_target_weight, self.p_bcs, self.p_be, self.p_l,
                 self.p_sex, self.p_a2, self.p_ph, self.p_selling_price,
                 self.p_algorithm, self.p_identifier, self.p_lb, self.p_ub, self.p_tol, self.p_dmi_eq, self.p_obj,
                 self.p_find_reduced_cost, self.p_ing_level] = parameters.values()
            elif isinstance(parameters, list):
                [self.p_id, self.p_feed_scenario, self.p_batch, self.p_breed, self.p_sbw, self.p_feed_time,
                 self.p_target_weight, self.p_bcs, self.p_be, self.p_l,
                 self.p_sex, self.p_a2, self.p_ph, self.p_selling_price,
                 self.p_algorithm, self.p_identifier, self.p_lb, self.p_ub, self.p_tol, self.p_dmi_eq, self.p_obj,
                 self.p_find_reduced_cost] = parameters

        def compute_nrc_parameters(self):
            self.mpmr, self.dmi, self.nem, self.pe_ndf = \
                nrc.get_all_parameters(self.cnem, self.p_sbw, self.p_bcs, self.p_be, self.p_l, self.p_sex, self.p_a2,
                                       self.p_ph, self.p_target_weight, self.p_dmi_eq)

            self.cneg = nrc.cneg(self.cnem)
            self.neg = nrc.neg(self.cneg, self.dmi, self.cnem, self.nem)

    class Data:
        ds: data_handler.Data = None
        headers_scenario: data_handler.Data.ScenarioParameters = None  # Scenario

        ingredient_ids = None
        dc_mp_properties = None
        dc_cost = None
        dc_ub = None
        dc_lb = None
        dc_dm_af_conversion = None
        dc_nem = None
        dc_fat = None

        def __init__(self, out_ds, parameters):
            self._cast_data(out_ds, parameters)

        def _cast_data(self, out_ds, parameters):
            """Retrieve parameters data from table. See data_handler.py for more"""
            self.ds = out_ds

            self.headers_feed_scenario = self.ds.headers_feed_scenario

            headers_feed_scenario = self.ds.headers_feed_scenario
            data_feed_scenario = self.ds.filter_column(self.ds.data_feed_scenario,
                                                       self.ds.headers_feed_scenario.s_feed_scenario,
                                                       parameters.p_feed_scenario)
            data_feed_scenario = self.ds.sort_df(data_feed_scenario, self.headers_feed_scenario.s_ID)

            self.ingredient_ids = list(
                self.ds.get_column_data(data_feed_scenario, self.headers_feed_scenario.s_ID, int))

            headers_feed_lib = self.ds.headers_feed_lib
            data_feed_lib = self.ds.filter_column(self.ds.data_feed_lib, headers_feed_lib.s_ID,
                                                  self.ingredient_ids)

            [self.dc_cost,
             self.dc_ub,
             self.dc_lb] = self.ds.multi_sorted_column(data_feed_scenario,
                                                       [headers_feed_scenario.s_feed_cost,
                                                        self.headers_feed_scenario.s_max,
                                                        self.headers_feed_scenario.s_min],
                                                       self.ingredient_ids,
                                                       self.headers_feed_scenario.s_ID,
                                                       return_dict=True
                                                       )
            [self.dc_dm_af_conversion,
             self.dc_nem,
             self.dc_fat,
             self.dc_mp_properties,
             rup,
             cp,
             ndf,
             pef] = self.ds.multi_sorted_column(data_feed_lib,
                                                [headers_feed_lib.s_DM,
                                                 headers_feed_lib.s_NEma,
                                                 headers_feed_lib.s_Fat,
                                                 [headers_feed_lib.s_DM,
                                                  headers_feed_lib.s_TDN,
                                                  headers_feed_lib.s_CP,
                                                  headers_feed_lib.s_RUP,
                                                  headers_feed_lib.s_Forage,
                                                  headers_feed_lib.s_Fat],
                                                 headers_feed_lib.s_RUP,
                                                 headers_feed_lib.s_CP,
                                                 headers_feed_lib.s_NDF,
                                                 headers_feed_lib.s_pef
                                                 ],
                                                self.ingredient_ids,
                                                self.headers_feed_scenario.s_ID,
                                                return_dict=True
                                                )
            self.dc_rdp = {}
            self.dc_pendf = {}
            for ids in self.ingredient_ids:
                self.dc_rdp[ids] = (1 - rup[ids]) * cp[ids]
                self.dc_pendf[ids] = ndf[ids] * pef[ids]

            if parameters.p_batch > 0:
                try:
                    batch_feed_scenario = self.ds.batch_map[parameters.p_id]["data_feed_scenario"][
                        parameters.p_feed_scenario]
                except KeyError:
                    logging.warning(f"No Feed_scenario batch for scenario {parameters.p_id},"
                                    f" batch {parameters.p_batch}, feed_scenario{parameters.p_feed_scenario}")
                    batch_feed_scenario = {}
                try:
                    batch_scenario = self.ds.batch_map[parameters.p_id]["data_scenario"][parameters.p_id]
                except KeyError:
                    logging.warning(f"No Scenario batch for scenario {parameters.p_id},"
                                    f" batch {parameters.p_batch}, scenario{parameters.p_feed_scenario}")
                    batch_scenario = {}

                parameters.c_batch_map = {"data_feed_scenario": batch_feed_scenario,
                                          "data_scenario": batch_scenario}

        def setup_batch(self, parameters):
            for ing_id, data in parameters.c_batch_map["data_feed_scenario"].items():
                for col_name, vector in data.items():
                    if col_name == self.headers_feed_scenario.s_feed_cost:
                        self.dc_cost[ing_id] = \
                            vector[parameters.p_batch_execution_id]
                    elif col_name == self.headers_feed_scenario.s_min:
                        self.dc_lb[ing_id] = vector[parameters.p_batch_execution_id]
                    elif col_name == self.headers_feed_scenario.s_max:
                        self.dc_ub[ing_id] = vector[parameters.p_batch_execution_id]

    def _compute_parameters(self):
        """Compute parameters variable with CNEm"""
        self.parameters.compute_nrc_parameters()

        if self.parameters.neg is None:
            return False
        if math.isnan(self.parameters.p_feed_time) or self.parameters.p_feed_time == 0:
            self.parameters.c_model_final_weight = self.parameters.p_target_weight
            self.parameters.c_swg = nrc.swg(self.parameters.neg, self.parameters.p_sbw,
                                            self.parameters.c_model_final_weight)
            self.parameters.c_model_feeding_time = \
                (self.parameters.p_target_weight - self.parameters.p_sbw) / self.parameters.c_swg
        elif math.isnan(self.parameters.p_target_weight) or self.parameters.p_target_weight == 0:
            self.parameters.c_model_feeding_time = self.parameters.p_feed_time
            self.parameters.c_swg = nrc.swg_time(self.parameters.neg, self.parameters.p_sbw,
                                                 self.parameters.c_model_feeding_time)
            self.parameters.c_model_final_weight = \
                self.parameters.c_model_feeding_time * self.parameters.c_swg + self.parameters.p_sbw
        else:
            raise Exception("target weight and feeding time cannot be defined at the same time")

        self.parameters.mpgr = nrc.mpg(self.parameters.c_swg, self.parameters.neg, self.parameters.p_sbw,
                                       self.parameters.c_model_final_weight, self.parameters.c_model_feeding_time)
        self.computed.dc_mpm = {}
        for ids in self.data.ingredient_ids:
            self.computed.dc_mpm[ids] = nrc.mp(*self.data.dc_mp_properties[ids], self.parameters.p_fat_orient)

        self.computed.revenue = self.parameters.p_selling_price * (
                self.parameters.p_sbw + self.parameters.c_swg * self.parameters.c_model_feeding_time)
        self.computed.dc_expenditure = self.data.dc_cost.copy()

        if self.parameters.p_obj == "MaxProfit" or self.parameters.p_obj == "MinCost":
            for i in self.data.ingredient_ids:
                self.computed.dc_expenditure[i] = - self.data.dc_cost[
                    i] * self.parameters.dmi * self.parameters.c_model_feeding_time / \
                                                  self.data.dc_dm_af_conversion[i]
        elif self.parameters.p_obj == "MaxProfitSWG" or self.parameters.p_obj == "MinCostSWG":
            for i in self.data.ingredient_ids:
                self.computed.dc_expenditure[i] = \
                    - self.data.dc_cost[i] * self.parameters.dmi * self.parameters.c_model_feeding_time / \
                    (self.data.dc_dm_af_conversion[i] * self.parameters.c_swg)

        if self.parameters.p_obj == "MaxProfit":
            self.computed.cst_obj = self.computed.revenue
        elif self.parameters.p_obj == "MaxProfitSWG":
            self.computed.cst_obj = self.computed.revenue / self.parameters.c_swg
        elif self.parameters.p_obj == "MinCost" or self.parameters.p_obj == "MinCostSWG":
            self.computed.cst_obj = 0

        return True

    def _build_model(self):
        """Build model"""
        self._diet = pyo.ConcreteModel()
        self._diet.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)
        self._diet.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        # Set
        self._diet.s_var_set = pyo.Set(initialize=self.data.ingredient_ids)

        # Parameters
        self._diet.p_model_offset = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_model_cost = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_model_lb = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_model_ub = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_model_nem = pyo.Param(self._diet.s_var_set, initialize=self.data.dc_nem)
        self._diet.p_model_mpm = pyo.Param(self._diet.s_var_set, within=pyo.Any, mutable=True)
        self._diet.p_model_rdp = pyo.Param(self._diet.s_var_set, initialize=self.data.dc_rdp)
        self._diet.p_model_fat = pyo.Param(self._diet.s_var_set, initialize=self.data.dc_fat)
        self._diet.p_model_pendf = pyo.Param(self._diet.s_var_set, initialize=self.data.dc_pendf)
        self._diet.p_rhs_cnem_ge = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_cnem_le = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_sum_1 = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_mpm = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_rdp = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_fat = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_alt_fat_ge = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_alt_fat_le = pyo.Param(within=pyo.Any, mutable=True)
        self._diet.p_rhs_pendf = pyo.Param(within=pyo.Any, mutable=True)

        # Functions
        def bound_function(model, i):
            return model.p_model_lb[i], model.p_model_ub[i]

        # Variables
        self._diet.v_x = pyo.Var(self._diet.s_var_set, bounds=bound_function)

        # Objective
        self._diet.f_obj = pyo.Objective(
            expr=(self._diet.p_model_offset + pyo.summation(self._diet.p_model_cost, self._diet.v_x)),
            sense=pyo.maximize)

        # Constraints
        self._diet.c_cnem_ge = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_nem, self._diet.v_x) >= self._diet.p_rhs_cnem_ge)
        self._diet.c_cnem_le = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_nem, self._diet.v_x) <= self._diet.p_rhs_cnem_le)
        self._diet.c_sum_1 = pyo.Constraint(expr=pyo.summation(self._diet.v_x) == self._diet.p_rhs_sum_1)
        self._diet.c_mpm = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_mpm, self._diet.v_x) >= self._diet.p_rhs_mpm)
        self._diet.c_rdp = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_rdp, self._diet.v_x) >= self._diet.p_rhs_rdp)
        self._diet.c_fat = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_fat, self._diet.v_x) <= self._diet.p_rhs_fat)
        self._diet.c_alt_fat_ge = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_fat, self._diet.v_x) >= self._diet.p_rhs_alt_fat_ge)
        self._diet.c_alt_fat_le = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_fat, self._diet.v_x) <= self._diet.p_rhs_alt_fat_le)
        self._diet.c_pendf = pyo.Constraint(
            expr=pyo.summation(self._diet.p_model_pendf, self._diet.v_x) >= self._diet.p_rhs_pendf)

    def _update_model(self):
        """Update RHS values on the model based on the new CNEm and updated parameters"""
        self._diet.p_model_offset = self.computed.cst_obj
        for i in self._diet.s_var_set:
            self._diet.p_model_lb[i] = self.data.dc_lb[i]
            self._diet.p_model_ub[i] = self.data.dc_ub[i]
            self._diet.p_model_cost[i] = self.computed.dc_expenditure[i]
            self._diet.p_model_mpm[i] = self.computed.dc_mpm[i]

        self._diet.p_rhs_cnem_ge = self.parameters.cnem * 0.999
        self._diet.p_rhs_cnem_le = self.parameters.cnem * 1.001
        self._diet.p_rhs_sum_1 = 1
        self._diet.p_rhs_mpm = (self.parameters.mpmr + self.parameters.mpgr) * 0.001 / self.parameters.dmi
        self._diet.p_rhs_rdp = 0.125 * self.parameters.cnem
        self._diet.p_rhs_fat = 0.06
        self._diet.p_rhs_alt_fat_ge = 0.039
        self._diet.p_rhs_alt_fat_le = 0.039
        self._diet.p_rhs_pendf = self.parameters.pe_ndf

        if self.parameters.p_fat_orient == "G":
            self._diet.c_alt_fat_ge.activate()
            self._diet.c_alt_fat_le.deactivate()
        else:
            self._diet.c_alt_fat_ge.deactivate()
            self._diet.c_alt_fat_le.activate()

    def set_fat_orient(self, direction):
        self.parameters.p_fat_orient = direction

    def set_batch_params(self, i):
        self.parameters.p_batch_execution_id = i

    def _setup_batch(self):
        parameters = self.parameters.init_parameters.copy()
        for col_name, vector in self.parameters.c_batch_map["data_scenario"].items():
            parameters[col_name] = vector[self.parameters.p_batch_execution_id]
        self.parameters.set_parameters(parameters)
        self.data.setup_batch(self.parameters)


class ModelReducedCost(Model):
    _special_id = None
    _special_cost = None

    def __init__(self, out_ds, parameters, special_id, special_cost=default_special_cost):
        Model.__init__(self, out_ds, parameters)
        self._special_id = special_id
        self._special_cost = special_cost

    def _solve(self, problem_id):
        sol = Model._solve(self, problem_id)
        sol["x{0}_price_{1}".format(self._special_id, str(int(100 * self.parameters.p_ing_level)))] = self._special_cost
        return sol

    def _compute_parameters(self):
        if not Model._compute_parameters(self):
            return False
        else:
            self.data.dc_cost[self._special_id] = self._special_cost
            self.computed.dc_expenditure[self._special_id] = - self.data.dc_cost[self._special_id] * \
                self.parameters.dmi * self.parameters.c_model_feeding_time / self.data.dc_dm_af_conversion[self._special_id]
            if self.parameters.p_obj == "MaxProfitSWG" or self.parameters.p_obj == "MinCostSWG":
                self.computed.dc_expenditure[self._special_id] /= self.parameters.c_swg
            return True

    def set_special_cost(self, cost=default_special_cost):
        self._special_cost = cost

    def get_special_cost(self):
        return self._special_cost

    def get_special_id(self):
        return self._special_id
