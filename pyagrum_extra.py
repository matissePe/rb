# Module required

# -*- coding: utf-8 -*-
import pyAgrum as gum
import numpy as np
import itertools
import pandas as pd
import sys
import pkg_resources
import tqdm
import json

installed_pkg = {pkg.key for pkg in pkg_resources.working_set}
if 'ipdb' in installed_pkg:
    import ipdb  # noqa: 401


# Constructor
# ===========

@classmethod
def create_from_df(cls,
                   data_df,
                   model_name="",
                   exclude_variables=[]):

    bn = cls(model_name)
    for var in data_df:
        if not(var in exclude_variables):
            bn_var = gum.LabelizedVariable(var, var, 0)

            if data_df[var].dtype.name == 'category':
                var_dom = data_df[var].cat.categories.to_list()
            else:
                var_dom = data_df[var].unique()

            [bn_var.addLabel(d) for d in var_dom]

            bn.add(bn_var)

    return bn


gum.BayesNet.from_df = create_from_df

# Aggregators
# ===========


def addSUM(bn_self,
           var_name,
           var_ope_names,
           var_desc="",
           ope_factors=None,
           lower_bound=None,
           upper_bound=None):
    """
    This method aims to create a CPT representing the deterministic sum of parent's values.

    TODO COMMENTS
    """
    var_ope = [bn_self.variable(vn) for vn in var_ope_names]
    nb_var_ope = len(var_ope_names)

    if ope_factors is None:
        ope_factors = [1]*nb_var_ope
    elif len(ope_factors) != nb_var_ope:
        error_msg = "Operand factors should None or a list of same size than operand variables list"
        raise ValueError(error_msg)

    var_dom_int = [[int(lab) for lab in var.labels()] for var in var_ope]
    parent_conf_list = list(itertools.product(*reversed(var_dom_int)))
    nb_parent_conf = len(parent_conf_list)
    var_conf_sum = [np.dot(pc, list(reversed(ope_factors)))
                    for pc in parent_conf_list]

    lower_bound = min(var_conf_sum) if lower_bound is None else lower_bound
    upper_bound = max(var_conf_sum) if upper_bound is None else upper_bound
    var_conf_sum_adj = [max(min(v, upper_bound), lower_bound)
                        for v in var_conf_sum]

    var_sum_dom_int = list(range(lower_bound, upper_bound + 1))
    var_sum_dom_lab = [str(v) for v in var_sum_dom_int]
    var_sum_dom_size = len(var_sum_dom_lab)

    var_sum = gum.LabelizedVariable(var_name, var_desc, 0)
    [var_sum.addLabel(lab) for lab in var_sum_dom_lab]

    if var_name in bn_self.names():
        bn_self.erase(var_name)

    bn_self.add(var_sum)
    [bn_self.addArc(v_op.name(), var_name) for v_op in var_ope]

    var_conf_idx = [var_sum_dom_int.index(val) for val in var_conf_sum_adj]
    prob_1_idx = np.ravel_multi_index([range(nb_parent_conf), var_conf_idx],
                                      [nb_parent_conf, var_sum_dom_size])

    cpt_arr_flat = bn_self.cpt(var_name).toarray().flatten()
    cpt_arr_flat[prob_1_idx] = 1

    bn_self.cpt(var_name).fillWith(cpt_arr_flat)


gum.BayesNet.addSUM = addSUM


# Special structures
# ==================

def add_pgdm(bn,
             state_name="X",
             state_domain=["0", "1"],
             sojourn_time_dist=None,
             sojourn_time_limit=0):

    # Create random variables
    state_var_name = state_name + "_"
    sojourn_var_name = state_name + "_" + "S" + "_"
    sojourn_domain_size = sojourn_time_limit + 1
    for slice in ["0", "t"]:

        state_var = gum.LabelizedVariable(state_var_name + slice, "", 0)
        [state_var.addLabel(d) for d in state_domain]
        bn.add(state_var)

        sojourn_time_var = gum.RangeVariable(sojourn_var_name + slice,
                                             "", 0, sojourn_time_limit)
        bn.add(sojourn_time_var)

    # Dependencies
    bn.addArc(state_var_name + "0",
              sojourn_var_name + "0")
    bn.addArc(state_var_name + "0",
              state_var_name + "t")
    bn.addArc(state_var_name + "t",
              sojourn_var_name + "t")

    bn.addArc(sojourn_var_name + "0",
              sojourn_var_name + "t")
    bn.addArc(sojourn_var_name + "0",
              state_var_name + "t")

    # Conditional probabilities
    nb_states = len(state_domain)

    # X_0
    bn.cpt(state_var_name + "0")[0] = 1

    # S_0
    if sojourn_time_dist is None:
        sojourn_time_dist = \
            pd.np.array([[1/(sojourn_domain_size)] *
                         sojourn_domain_size]*(nb_states - 1))

    bn.cpt(sojourn_var_name + "0")[:-1, :] = sojourn_time_dist
    bn.cpt(sojourn_var_name + "0")[-1, 0] = 1

    # X_{t} (state transition model)
    # S_{t-1} = 1 : sojourn time over => jumps to the next state
    bn.cpt(state_var_name + "t")[0, :-1, 1:] = pd.np.eye(nb_states - 1)
    bn.cpt(state_var_name + "t")[0, -1, -1] = 1
    # S_{t-1} > 1 : stays in the current state
    bn.cpt(state_var_name + "t")[1:, :, :] = pd.np.eye(nb_states)

    # S_{t}
    # S_{t-1} > 1 => Deterministic counting down the sojourn time
    bn.cpt(sojourn_var_name + "t")[1:, :, :-1] = \
        pd.np.eye(sojourn_time_limit)\
             .reshape(sojourn_time_limit,
                      1,
                      sojourn_time_limit)\
             .repeat(nb_states, axis=1)

    # S_{t-1} == 1 => Sojourn time for each states
    bn.cpt(sojourn_var_name + "t")[0, :-1, :] = sojourn_time_dist
    bn.cpt(sojourn_var_name + "t")[0, -1, 0] = 1

    return bn


gum.BayesNet.add_pgdm = add_pgdm


def add_rul_var(pgdm,
                state_name="X"):

    # Create random variables
    state_var_name = state_name + "_"
    state_var = pgdm.variable(state_var_name + "0")
    sojourn_var_name = state_name + "_" + "S" + "_"
    sojourn_var = pgdm.variable(sojourn_var_name + "0")
    sojourn_var_limit = sojourn_var.domainSize() - 1
    rul_var_name = state_var_name + "rul_"
    rul_domain_size = (state_var.domainSize() - 1)*sojourn_var_limit

    for slice in ["0", "t"]:

        rul_var = gum.RangeVariable(rul_var_name + slice,
                                    "",
                                    0, rul_domain_size)
        pgdm.add(rul_var)

        # Dependencies
        pgdm.addArc(state_var_name + slice,
                    rul_var_name + slice)
        pgdm.addArc(sojourn_var_name + slice,
                    rul_var_name + slice)

    XS_dom_int = [list(range(state_var.domainSize())),
                  list(range(sojourn_var.domainSize()))]
    XS_conf_list = list(itertools.product(*XS_dom_int))
    print(XS_conf_list)
    for x_val, s_val in XS_conf_list:

        r_val = np.max(
            [0, rul_domain_size - ((x_val + 1)*sojourn_var_limit - s_val)])
        print(x_val, s_val, r_val)

        pgdm.cpt(rul_var_name + "0")[{state_var_name + "0": x_val,
                                      sojourn_var_name + "0": s_val,
                                      rul_var_name + "0": r_val}] = 1

    #pgdm.cpt(rul_var_name + "0")[:,-1,0] = 1

    pgdm.cpt(rul_var_name + "t")[:] = pgdm.cpt(rul_var_name + "0")[:]

    return pgdm


gum.BayesNet.add_rul_var = add_rul_var


# Fitting/Predict methods
# =======================

def fit_cpt_bis(bn, df, var_name,
                apriori_coef="smart",
                apriori_dist="uniform",
                apriori_data_threshold=30,
                verbose_mode=False):
    """
    This function aims to compute the maximum likelihood estimation of CPT parameters from a Pandas
    dataframe.

    Parameters
    - =df=: a Pandas DataFrame consisting only of categorical variables.
    - =var_name=: the variable name associated to the CPT to be fitted.
    - =apriori_coef=: this parameter represents the apriori weight in the fitting process. if
    apriori_coef is a non negative real number :
    - the higher it is, the closer to the apriori distribution the resulting configuration
    distribution will be.  
    - the lower it is, the closer to the distribution fitted by data the resulting configuration
    distribution will be.  
    User can only pass a string associated to an apriori coefficient strategy. Possible values are:
    - "smart": in this case, the apriori coefficient is set equal to 1/nb_data_conf if nb_data_conf >
    0 else 1 where nb_data_conf is the number of data observed for a given configuration.
    - =apriori_dist=: shape of the apriori distribution. Possible values are: "uniform". Passing None to
    this parameter disables apriori consideration in the fitting process.
    - =apriori_data_threshold=: apply apriori for a conditional distribution if the number of oberserved
    corresponding configurations is lower or equal than this parameter.

    Note: this method is an adaptation of codes found at http://www-desir.lip6.fr/~phw/aGrUM/officiel/notebooks/
    """
    if verbose_mode:
        sys.stdout.write("- Learn CPT {0}\n".format(var_name))

    parents = list(reversed(bn.cpt(var_name).names))
    parents.pop()

    # Check if df consists of catagorical variables only
    for data_s in df[[var_name] + parents].columns:
        if str(df[data_s].dtype) != "category":
            err_msg = "Variable {0} is not categorical : type {1}\n".format(
                data_s, str(df[data_s].dtype))
            raise TypeError(err_msg)

    # Warning : df variables must be categorical here if not unobserved but possible configurations will be dropped
    # and then lead to unconsistent CPT
    if len(parents) == 0:
        joint_counts = np.array(df[var_name].value_counts().loc[df[var_name].cat.categories], dtype=float)
    else:   
        joint_counts = np.array(pd.crosstab(
            df[var_name], [df[parent] for parent in parents], dropna=False), dtype=float)

    cond_counts = joint_counts.sum(axis=0)
    # if cond_counts is monodimensionnal then cond_counts will be a float and not a Series
    # if type(cond_counts) == pd.core.series.Series:
    #     cond_counts = cond_counts.apply(np.float32)
    # else:
    #     cond_counts = float(cond_counts)

    # A priori management
    if not(apriori_dist is None):
        # Select apriori distribution
        apriori_joint_arr = np.zeros(joint_counts.shape)
        if apriori_dist == "uniform":
            apriori_joint_arr[:] = 1/apriori_joint_arr.shape[0]
        else:
            err_msg = "apriori distribution {0} is not supported. Possible values are : 'uniform'\n".format(
                apriori_dist)
            raise ValueError(err_msg)

        # Build the apriori coefficient array
        apriori_coef_arr = np.ones(cond_counts.shape)
        if isinstance(apriori_coef, str):
            if apriori_coef == "smart":
                if len(cond_counts.shape) == 0:
                    apriori_coef_arr = 1/cond_counts if cond_counts > 0 else 1.0
                else:
                    idx_cond_count_sup_0 = np.where(cond_counts > 0)
                    apriori_coef_arr = np.ones(cond_counts.shape)
                    apriori_coef_arr[idx_cond_count_sup_0] = \
                        1/cond_counts[idx_cond_count_sup_0]
            else:
                err_msg = "apriori coef {0} is not supported. Possible values are : 'smart' or non negative value\n".format(
                    apriori_coef)
                raise ValueError(err_msg)
        else:
            if len(cond_counts.shape) == 0:
                apriori_coef_arr = abs(apriori_coef)
            else:
                apriori_coef_arr[:] = abs(apriori_coef)

        # Check coordinate that need apriori
        if len(cond_counts.shape) == 0:
            if cond_counts > apriori_data_threshold:
                apriori_coef_arr = 0.
        else:
            apriori_counts_idx = cond_counts <= apriori_data_threshold
            apriori_coef_arr[~apriori_counts_idx] = 0.

        # Update joint and cond counts
        joint_counts += apriori_joint_arr*apriori_coef_arr
        cond_counts = joint_counts.sum(axis=0)

    # Normalization of counts to get a consistent CPT
    # Note: np.nan_to_num is used only in the case where no apriori is requested to force nan value to 0
    #       => this is of course highly unsafe to work in this situation as CPTs may not sum to 1 for all configurations
    cpt_shape = bn.cpt(var_name)[:].shape
    bn.cpt(var_name)[:] = np.nan_to_num(
        (joint_counts/cond_counts).transpose().reshape(cpt_shape))


def fit_cpt(bn, df, var_name, verbose_mode=False):
    """
    This function aims to compute the maximum likelihood estimation of CPT parameters from a Pandas
    dataframe.

    Parameters
    - =df=: a Pandas DataFrame consisting only of categorical variables.
    - =var_name=: the variable name associated to the CPT to be fitted.
    """
    if verbose_mode:
        sys.stdout.write("- Learn CPT {0}\n".format(var_name))

    parents = list(reversed(bn.cpt(var_name).names))
    parents.pop()

    # Check if df consists of catagorical variables only
    for data_s in df[[var_name] + parents].columns:
        if str(df[data_s].dtype) != "category":
            err_msg = "Variable {0} is not categorical : type {1}\n".format(
                data_s, str(df[data_s].dtype))
            raise TypeError(err_msg)

    # Warning : df variables must be categorical here if not unobserved but possible configurations will be dropped
    # and then lead to unconsistent CPT

    if len(parents) == 0:
        cpt_df = df[var_name].value_counts(normalize=True)\
                                   .loc[df[var_name].cat.categories].astype(float)
        bn.cpt(var_name)[:] = cpt_df.values[:]
    else:
        joint_counts = pd.crosstab(
            df[var_name], [df[parent] for parent in parents], dropna=False).astype(float)

        cond_counts = joint_counts.sum(axis=0)

        cpt_df = (joint_counts/cond_counts).fillna(1/joint_counts.shape[0])

        #ipdb.set_trace()
        bn_cpt_shape = bn.cpt(var_name)[:].shape
        bn.cpt(var_name)[:] = \
            cpt_df.values[:].transpose()\
                            .reshape(bn_cpt_shape)


gum.BayesNet.fit_cpt = fit_cpt
gum.BayesNet.fit_cpt_bis = fit_cpt_bis

def fit_bis(bn, df,
        apriori_coef="smart",
        apriori_dist="uniform",
        apriori_data_threshold=30,
        exclude_variables=[],
        verbose_mode=False,
        progress_mode=False):
    """
    Fit the CPTs of every variable in the BN bn from the database df.
    """
    for name in tqdm.tqdm(bn.names(),
                          disable=not(progress_mode),
                          desc="CPT fitting"):
        if not(name in exclude_variables):
            bn.fit_cpt_bis(df, name,
                       apriori_coef=apriori_coef,
                       apriori_dist=apriori_dist,
                       apriori_data_threshold=apriori_data_threshold,
                       verbose_mode=verbose_mode)

def fit(bn, df,
        exclude_variables=[],
        verbose_mode=False,
        progress_mode=False):
    """
    Fit the CPTs of every variable in the BN bn from the database df.
    """
    for name in tqdm.tqdm(bn.names(),
                          disable=not(progress_mode),
                          desc="CPT fitting"):
        if not(name in exclude_variables):
            bn.fit_cpt(df, name,
                       verbose_mode=verbose_mode)


gum.BayesNet.fit = fit
gum.BayesNet.fit_bis = fit_bis


def predict(bn, data, var_target, returns="map", show_progress=False, debug=False):
    """
    This function is used to predict the value of a target variable from observations 
    using a bayesian network model. 

    Inputs:
    - =bn=: the predictive model given as a =pyAgrum.BayesNet= object
    - =data=: the data containing the observations used to predict the target variable 
    as a =pandas.DataFrame= object
    - =var_target=: the name of the target variable as a =str= object
    - =returns=: string indicating which elements are returned by the method. Possible values are:
    "map": array of maximum a posteriori predictions
    "probs": matrix of label probabilities for each data 

    Returns:
    - a =numpy.array= containing the predictions of the target variables maximising the 
    maximum a posteriori criterion 
    - a =numpy.array= containing the posterior probability distribution of the target 
    variable given each observation in =data=.
    """

    # Initialize the inference engine
    inf_bn = gum.LazyPropagation(bn)
    inf_bn.setTargets({var_target})
    nb_data = len(data)
    target_size = bn.variable(var_target).domainSize()
    target_dom = np.array([bn.variable(var_target).label(i)
                           for i in range(target_size)])
    data_records = data.to_dict("records")
    post_prob = np.zeros((nb_data, target_size))
    for i in range(nb_data):
        if debug:
            print(i)
        if debug:
            print(data_records[i])
        # Set the evidence
        inf_bn.setEvidence(dict_np2native(data_records[i]))
        # Run inference
        inf_bn.makeInference()
        # Compute posterior probability of target variable
        post_prob[i, :] = inf_bn.posterior(var_target).toarray()
        if debug:
            print(post_prob[i, :])
        # Erase evidence
        inf_bn.eraseAllEvidence()
        if show_progress:
            sys.stdout.write("predict progress: {0:3.0%}\r".format(i/nb_data))

    if returns == "map":
        # Find out the predictions
        target_pred = target_dom[np.argmax(post_prob, axis=1)]
        return target_pred
    elif returns == "probs":
        return post_prob
    else:
        err_msg = "Returns type {0} is unsupported, possible values are 'map', 'probs'".format(
            returns)
        raise ValueError(err_msg)


gum.BayesNet.predict = predict


def predict_proba(bn, data, var_target, show_progress=False, debug=False):
    """ Util function for compatibility with the general sklearn interface. """
    return predict(bn, data, var_target=var_target, returns="probs", show_progress=show_progress, debug=debug)


gum.BayesNet.predict_proba = predict_proba


# Variable utils
# ==============
def series_to_lv(series):
    """
    This function creates a =pyAgrum.LabelizedVariable= object from a discrete and finite
    =Pandas.series= object.

    This function assumes the series contains string or integer values.
    """

    if isinstance(series.dtype, pd.core.dtypes.dtypes.CategoricalDtype):
        labels = list(series.cat.categories)
    else:
        labels = series.unique()
        labels.sort()
    lv = gum.LabelizedVariable(series.name, "", len(labels))
    #if series.dtype == "O": [lv.changeLabel(i,labels[i]) for i in range(len(labels))]
    # To be tested on pure numeric labels
    [lv.changeLabel(i, str(labels[i])) for i in range(len(labels))]
    return lv


# Utils
# =====

def dict_np2native(d):
    """ This function converts (in-place) a dictionnary containing =numpy= types into 
    a dictionnary containing only =Python= native types.

    Note : for now it supports only numpy.integer.
    """
    for k, v in d.items():
        if issubclass(type(v), np.integer):
            d[k] = int(v)
        else:
            d[k] = str(v)
    return d


def compute_jointdist(bn):
    """ This method compute the natural joint probability distribution represented by the BN. """

    joint = gum.Potential()
    for v in bn.names():
        joint *= bn.cpt(v)

    return joint


gum.BayesNet.compute_jointdist = compute_jointdist

# Input/output
# ============


def to_dict(bn):
    """ This method converts BN data into a dictionnary. """
    bn_dict = dict()

    property_list = ["name"]
    bn_dict["property"] = {
        prop: bn.property(prop)
        for prop in property_list}

    bn_dict["variables"] = []
    for var_name in bn.names():
        var = bn.variable(var_name)
        cpt = bn.cpt(var_name)
        bn_dict["variables"].append(
            {"specs": {
                "id": var.name(),
                "description": var.description(),
                "domain": list(var.labels()),
                "parents_var": cpt.var_names[0:-1],
                "cpt": cpt[:].flatten().tolist(),
            }}
        )

    return bn_dict


gum.BayesNet.to_dict = to_dict


# WARNING: DOES NOT WORK !!!!!
@classmethod
def create_bn_from_dict(cls, bn_specs_dict):
    """ This method build a BN from a dictionnary containing model specs. """
    bn_specs_dict.setdefault("property", {})
    bn = cls(bn_specs_dict["property"].get("name", "bn"))

    # Create variables
    for variable in bn_specs_dict["variables"]:
        var_specs = variable["specs"]

        var = gum.LabelizedVariable(var_specs["id"],
                                    var_specs["description"], 0)

        [var.addLabel(d) for d in var_specs["domain"]]

        bn.add(var)

    # Set arcs
    for variable in bn_specs_dict["variables"]:
        var_specs = variable["specs"]
        [bn.addArc(pv, var_specs["id"]) for pv in var_specs["parents_var"]]
        bn.cpt(var_specs["id"]).fillWith(var_specs["cpt"])

    return bn


gum.BayesNet.from_dict = create_bn_from_dict


def bn_to_json(bn, filename):

    with open(filename, 'w') as model_file:
        bn_dict = bn.to_dict()
        json.dump(bn_dict, model_file)


gum.BayesNet.to_json = bn_to_json


@classmethod
def create_bn_from_json(cls, filename):

    with open(filename, 'r') as model_file:
        bn_dict = json.load(model_file)
        bn = cls.from_dict(bn_dict)

    return bn


gum.BayesNet.from_json = create_bn_from_json
