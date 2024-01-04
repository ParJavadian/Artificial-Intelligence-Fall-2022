import copy
import random
import pandas as pd


class BN(object):
    """
    Bayesian Network implementation with sampling methods as a class
    
    Attributes
    ----------
    n: int
        number of variables
        
    G: dict
        Network representation as a dictionary. 
        {variable:[[children],[parents]]} # You can represent the network in other ways. This is only a suggestion.

    topological order: list
        topological order of the nodes of the graph

    CPT: list
        CPT Table
    """

    def __init__(self, G, CPT) -> None:
        ############################################################
        # Initialize Bayesian Network                              #
        # (1 Points)                                               #
        ############################################################
        self.n = len(G)
        self.G = G
        self.CPT = CPT
        self.topological_order = []
        self.set_topological_order()

    def cpt(self, node) -> pd.DataFrame:
        """
        This is a function that returns cpt of the given node
        
        Parameters
        ----------
        node:
            a variable in the bayes' net
            
        Returns
        -------
        result: dict
            {value1:{{parent1:p1_value1, parent2:p2_value1, ...}: prob1, ...}, value2: ...}
        """
        ############################################################
        # (3 Points)                                               #
        ############################################################
        for each_cpt in self.CPT:
            if each_cpt.columns[0] == node:
                return each_cpt
        return None

    def pmf(self, query, evidence) -> float:
        """
        This function gets a variable and its value as query and a list of evidences and returns probability mass
        function P(Q=q|E=e)
        
        Parameters
        ----------
        query:
            a variable and its value
            e.g. ('a', 1)
        evidence:
            list of variables and their values
            e.g. [('b', 0), ('c', 1)]
        
        Returns
        -------
        PMF: float
            P(query|evidence)
        """
        ############################################################
        # (3 Points)                                               #
        ############################################################
        initial_factors = self.remove_nonmatching_evidences(evidence, self.CPT)
        hidden_vars = copy.copy(self.topological_order)
        for q in query.keys():
            hidden_vars.remove(q)
        for e in evidence.keys():
            hidden_vars.remove(e)
        for h in hidden_vars:
            h_factors = self.get_var_factors(h, initial_factors)
            new_factor = self.join_and_eliminate(h, h_factors, evidence)
            new_factors = []
            for factor in initial_factors:
                is_hidden = False
                for h_factor in h_factors:
                    if h_factor.equals(factor):
                        is_hidden = True
                        break
                if not is_hidden:
                    new_factors.append(factor)
            new_factors.append(new_factor)
            initial_factors = new_factors
        for q in query.keys():
            joined_all = self.get_joined_factor(initial_factors, q, evidence)[0]
        self.normalize(joined_all)
        return self.get_row_consistent_with_query(query, joined_all).iloc[0, -1]

    def get_row_consistent_with_query(self, query: dict, factor: pd.DataFrame) -> pd.DataFrame:
        new_factor = factor.copy(deep=True)
        for q in query:
            new_factor = new_factor[new_factor[q] == query[q]]
        return new_factor

    def sampling(self, query: dict, evidence: dict, sampling_method, num_iter, num_burnin=1e2) -> float:

        """
        Parameters
        ----------
        query: list
            list of variables and their values
            e.g. [('a', 0), ('e', 1)]
        evidence: list
            list of observed variables and their values
            e.g. [('b', 0), ('c', 1)]
        sampling_method:
            "Prior", "Rejection", "Likelihood Weighting", "Gibbs"
        num_iter:
            number of the generated samples 
        num_burnin:
            (used only in gibbs sampling) number of samples that we ignore at the start for gibbs method to converge
            
        Returns
        -------
        probability: float
            approximate P(query|evidence) calculated by sampling
        """
        ############################################################
        # (27 Points)                                              #
        #     Prior sampling (6 points)                            #
        #     Rejection sampling (6 points)                        #
        #     Likelihood weighting (7 points)                      #
        #     Gibbs sampling (8 points)                      #
        ############################################################

        if sampling_method == "Prior":
            samples = self.prior_sample(query, evidence, num_iter)
            count = 0
            for sample in samples:
                if self.sample_consistent_with_query(sample, query):
                    count += 1
            return count / num_iter
        elif sampling_method == "Rejection":
            samples = self.rejection_sample(query, evidence, num_iter)
            count = 0
            for sample in samples:
                if self.sample_consistent_with_query(sample, query):
                    count += 1
            return count / num_iter
        elif sampling_method == "Likelihood Weighting":
            samples = self.likelihood_sample(query, evidence, num_iter)
            count = 0
            total_w = 0
            for sample in samples:
                total_w += sample[1]
                if self.sample_consistent_with_query(sample[0], query):
                    count += sample[1]
            return count / total_w
        else:
            samples = self.gibbs_sample(query, evidence, num_iter, num_burnin)
            count = 0
            for sample in samples:
                if self.sample_consistent_with_query(sample, query):
                    count += 1
            return count / num_iter

    def prior_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of generated samples

            Returns
            -------
            prior samples
        """
        samples = []
        for i in range(num_iter):
            sample = self.get_prior_sample()
            samples.append(sample)
        return samples

    def sample_consistent_with_evidence(self, sample, evidence):
        """
            To check if a sample is consistent with evidence or not?

            Parameters
            ----------
            sample:
                a sample
            evidence:
                evidence set
            
            Returns
            -------
            True if the sample is consistent with evidence, False otherwise.
        """
        for e in evidence.keys():
            if evidence[e] != sample[e]:
                return False
        return True

    def sample_consistent_with_query(self, sample, query):
        """
            To check a sample is consistent with query or not?

            Parameters
            ----------
            sample:
                a sample
            query:
                query set
            
            Returns
            -------
            True if the sample is consistent with query, False otherwise.
        """
        for q in query.keys():
            if query[q] != sample[q]:
                return False
        return True

    def get_prior_sample(self):
        """
            Returns
            -------
            Returns a set which is the prior sample. 
        """
        sample = {}
        for node in self.topological_order:
            cpt = self.cpt(node).copy(deep=True)
            for parent in self.G[node][1]:
                cpt = cpt[cpt[parent] == sample[parent]]
            rand = random.random()
            if rand < cpt.iloc[0, -1]:
                sample[node] = cpt[node].iloc[0]
            else:
                sample[node] = cpt[node].iloc[1]
        return sample

    def rejection_sample(self, query, evidence: dict, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of generated samples

            Returns
            -------
            rejection samples
        """
        samples = []
        while num_iter > 0:
            sample, sample_is_consistent = self.get_rejection_sample(evidence)
            if not sample_is_consistent:
                continue
            samples.append(sample)
            num_iter -= 1
        return samples

    def get_rejection_sample(self, evidence: dict):
        sample_is_consistent = True
        sample = {}
        for node in self.topological_order:
            cpt = self.cpt(node).copy(deep=True)
            for parent in self.G[node][1]:
                cpt = cpt[cpt[parent] == sample[parent]]
            rand = random.random()
            if rand < cpt.iloc[0, -1]:
                sample[node] = cpt[node].iloc[0]
            else:
                sample[node] = cpt[node].iloc[1]
            if node in evidence.keys() and sample[node] != evidence[node]:
                sample_is_consistent = False
                break
        return sample, sample_is_consistent

    def likelihood_sample(self, query, evidence, num_iter):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of generated samples

            Returns
            -------
            likelihood samples
        """
        samples = []
        for i in range(num_iter):
            samples.append(self.get_likelihood_sample(evidence))
        return samples

    def get_likelihood_sample(self, evidence: dict):
        sample = {}
        w = 1.0
        for node in self.topological_order:
            cpt = self.cpt(node).copy(deep=True)
            for parent in self.G[node][1]:
                cpt = cpt[cpt[parent] == sample[parent]]
            if node in evidence.keys():
                sample[node] = evidence[node]
                cpt = cpt[cpt[node] == evidence[node]]
                w *= cpt.iloc[0, -1]
            else:
                rand = random.random()
                if rand < cpt.iloc[0, -1]:
                    sample[node] = cpt[node].iloc[0]
                else:
                    sample[node] = cpt[node].iloc[1]
        return sample, w

    def gibbs_sample(self, query, evidence: dict, num_iter, num_burnin):
        """
            Parameters
            ----------
            query:
                query set
            evidence:
                evidence set
            num_iter:
                number of generated samples

            Returns
            -------
            gibbs samples
        """
        init_sample = {}
        samples = []
        for node in self.topological_order:
            if node in evidence.keys():
                init_sample[node] = evidence[node]
                continue
            rand = random.random()
            if rand < 0.5:
                init_sample[node] = 0
            else:
                init_sample[node] = 1
        i = 0
        while i < num_burnin + num_iter:
            for node in self.topological_order:
                new_sample = copy.deepcopy(init_sample)
                if node in evidence.keys():
                    continue
                other_vars = {}
                for other_node in new_sample.keys():
                    if node != other_node:
                        other_vars[other_node] = new_sample[other_node]
                var_factors = self.get_var_factors(node, self.CPT)
                joined_factor = self.get_joined_factor(var_factors, node, other_vars)[0]
                joined_factor = self.remove_nonmatching_evidences(other_vars, [joined_factor])[0]
                self.normalize(joined_factor)
                rand = random.random()
                if rand < joined_factor.iloc[0, -1]:
                    new_sample[node] = joined_factor[node].iloc[0]
                else:
                    new_sample[node] = joined_factor[node].iloc[1]
                if i >= num_burnin:
                    samples.append(new_sample)
                i += 1
                if i >= num_burnin + num_iter:
                    return samples
                init_sample = new_sample
        return samples

    def topological_sort(self, node, visited):
        """
        This function wants to make a topological sort of the graph and set the topological_order parameter of the
        class.

            Parameters
            ----------
            node:
                the list of nodes
            visited:
                the list of visited(1)/not visited(0) nodes

        """
        visited.append(node)
        for child in self.G[node][0]:
            if child not in visited:
                self.topological_sort(child, visited)
        self.topological_order.append(node)

    def set_topological_order(self):
        """
            This function calls topological sort function and set the topological sort.
        """
        visited = list()
        for node in self.G.keys():
            if node not in visited:
                self.topological_sort(node, visited)
        self.topological_order.reverse()

    def all_parents_visited(self, node, visited) -> bool:
        """
            This function checks if all parents are visited or not?

            Parameters
            ----------
            node:
                the list of nodes
            visited:
                the list of visited(1)/not visited(0) nodes

            Return
            ----------
            return True if all parents of node are visited, False otherwise.
        """
        for parent in self.G[node][1]:
            if parent not in visited:
                return False
        return True

    def remove_nonmatching_evidences(self, evidence, factors):
        new_factors = list()
        for factor in factors:
            for node in evidence.keys():
                value = evidence[node]
                if node in list(factor):
                    factor = factor[factor[node] == value]
            new_factors.append(factor)
        return new_factors

    def join_and_eliminate(self, var, factors, evidence) -> pd.DataFrame:
        joined_factor, other_vars = self.get_joined_factor(factors, var, evidence)
        eliminated_factor = self.get_eliminated_factor(joined_factor, var, other_vars)
        return eliminated_factor

    def get_eliminated_factor(self, joined_factor, var, other_vars) -> pd.DataFrame:
        joinedd_factor = joined_factor.copy(deep=True)
        p_name = joinedd_factor.columns[-1]
        eliminated_factor = pd.DataFrame(columns=(other_vars + [p_name]))
        num = int(joined_factor.shape[0] / 2)
        for i in range(num):
            first = joinedd_factor.head(1)
            second = joinedd_factor.copy(deep=True)
            new_row = {}
            for other_var in other_vars:
                second = second[joinedd_factor[other_var] == first[other_var].iloc[0]]
                second = second[second[var] != first[var].iloc[0]]
                new_row[other_var] = first[other_var].iloc[0]
            joint_p = first[p_name].iloc[0] + second[p_name].iloc[0]
            new_row[p_name] = joint_p
            eliminated_factor = eliminated_factor.append(new_row, ignore_index=True)
            joinedd_factor = pd.concat([joinedd_factor, first, first]).drop_duplicates(keep=False)
            joinedd_factor = pd.concat([joinedd_factor, second, second]).drop_duplicates(keep=False)
        return eliminated_factor

    def get_joined_factor(self, var_factors, var, evidence):
        if not var_factors:
            return None
        joined_factor = var_factors[0]
        for i in range(1, len(var_factors)):
            joined_factor = joined_factor.merge(var_factors[i], how='right')
        non_var_cols = []
        var_cols = []
        for col in joined_factor.columns:
            if col not in self.topological_order:
                non_var_cols.append(col)
            else:
                var_cols.append(col)
        if len(var_factors) > 1:
            name = "P("
            for var_col in var_cols:
                name += var_col + ","
            name += ")"
            joined_factor[name] = joined_factor[non_var_cols[0]]
            joined_factor.drop(non_var_cols[0], axis=1, inplace=True)
            for i in range(1, len(non_var_cols)):
                joined_factor[name] *= joined_factor[non_var_cols[i]]
                joined_factor.drop(non_var_cols[i], axis=1, inplace=True)
        other_vars = copy.copy(var_cols)
        other_vars.remove(var)
        return joined_factor, other_vars

    def get_rows_factor(self, factor, var, evidence, values, variables_in_joined_factor):
        pass

    def get_var_factors(self, var, factors):
        var_factors = list()
        for factor in factors:
            if var in factor.columns:
                var_factors.append(factor)
        return var_factors

    def get_variables_in_joined_factor(self, var_factors, var, evidence) -> list:
        to_return = list()
        for factor in var_factors:
            for variable in list(factor):
                if variable not in to_return and variable != var and variable not in evidence.keys():
                    to_return.append(variable)
        return to_return

    def get_join_all_factors(self, factors, query, evidence) -> pd.DataFrame:
        pass

    def get_row_factor(self, factor, query_vars, evidence, values):
        pass

    def normalize(self, joint_factor: pd.DataFrame):
        sum_of_p = joint_factor.iloc[:, -1:].sum()
        joint_factor.iloc[:, -1:] /= sum_of_p
