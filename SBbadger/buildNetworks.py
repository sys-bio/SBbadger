
import random
import warnings
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
from scipy.stats import norm, lognorm, uniform, loguniform
from collections import defaultdict
from scipy.optimize import linprog


# todo: reversible edges
# todo: adaptable probabilities
# todo: update edge lists to include modifiers


@dataclass
class Settings:
    """ Settings to control some properties of the network generation"""
    rateConstantScale = 1.0
    """ How much the rate constants are scaled by. By default rate constants ge values between 0 and 1.0"""
    allowMassViolatingReactions = False
    """ If set to true, reactions such as A + B -> A are allowed"""
    addDegradationSteps = False
    """Set true if you want every floating node (not boundary nodes) to have a degradation step"""
    # todo: add degradation step for species with no outlet (downstream boundaries)
    # todo: add function to as inputs (upstream boundaries)
    removeBoundarySpecies = True
    """Set true if you want and sink and source species to be classed as boundary species"""

    @dataclass
    class ReactionProbabilities:
        """ Defines the probabilities of generating different reaction mechanisms.
         Current probabilities are:
         
         UniUni = 0.35
         BiUni = 0.3
         UniBi = 0.3
         BiBi  = 0.05
         """
        UniUni = 0.35
        BiUni = 0.3
        UniBi = 0.3
        BiBi = 0.05


def _get_mmr_rate_law(k, s1, s2):
    return 'Vm' + str(k) + '/Km' + str(k) + '0*(' + s1 + '-' + s2 + '/Keq' + str(k) + \
           ')/(' + '1 + ' + s1 + '/' + 'Km' + str(k) + '0' + ' + ' \
           + s2 + '/' + 'Km' + str(k) + '1' + ')'


def _get_mar_rate_law(k, s1, s2):
    return 'k' + str(k) + '0*' + s1 + ' - k' + str(k) + '1' + '*' + s2


@dataclass
class TReactionType:
    UNIUNI = 0
    BIUNI = 1
    UNIBI = 2
    BIBI = 3


def _pick_reaction_type(prob=None):
    rt = random.random()
    if prob:
        if rt < prob[0]:
            return TReactionType.UNIUNI
        if rt < prob[0] + prob[1]:
            return TReactionType.BIUNI
        if rt < prob[0] + prob[1] + prob[2]:
            return TReactionType.UNIBI
        return TReactionType.BIBI
    else:
        if rt < Settings.ReactionProbabilities.UniUni:
            return TReactionType.UNIUNI
        if rt < Settings.ReactionProbabilities.UniUni + Settings.ReactionProbabilities.BiUni:
            return TReactionType.BIUNI
        if rt < Settings.ReactionProbabilities.UniUni + Settings.ReactionProbabilities.BiUni + \
                Settings.ReactionProbabilities.UniBi:
            return TReactionType.UNIBI
        return TReactionType.BIBI


def generate_distributions(n_species, in_dist, out_dist, joint_dist, min_node_deg, in_range, out_range, joint_range):

    def single_unbounded_pmf(sdist):
        """Assumes starting degree of 1 and extends until cutoff found"""

        deg = 1
        while True:
            dist = []
            for j in range(deg):
                dist.append(sdist(j + 1))
            distsum = sum(dist)
            dist_n = [x * n_species / distsum for x in dist]
            if any(elem < min_node_deg for elem in dist_n[:-1]) and dist_n[-1] >= min_node_deg:
                raise Exception("\nThe provided distribution is invalid; consider revising.")
            elif dist_n[-1] < min_node_deg:
                pmf0 = dist[:-1]
                sum_dist_f = sum(pmf0)
                pmf0 = [x / sum_dist_f for x in pmf0]
                break
            else:
                deg += 1

        pmf_range = [i + 1 for i, each in enumerate(pmf0)]

        return pmf0, pmf_range

    def single_bounded_pmf(sdist, drange):
        """Start with given degree range and trim until cutoffs found"""

        dist_ind = [j for j in range(drange[0], drange[1] + 1)]
        pmf0 = [sdist(j) for j in range(drange[0], drange[1] + 1)]
        dist_sum = sum(pmf0)
        pmf0 = [x / dist_sum for x in pmf0]
        dist = [x * n_species for x in pmf0]

        while any(freq < 1 for freq in dist):

            min_ind = pmf0.index(min(pmf0))
            del dist_ind[min_ind]
            del pmf0[min_ind]
            dist_sum = sum(pmf0)
            pmf0 = [x / dist_sum for x in pmf0]
            dist = [x * n_species for x in pmf0]

        return pmf0, dist_ind

    def find_edges_expected_value(x_dist, x_range, num_species=n_species):

        edge_ev = 0
        for j, item in enumerate(x_dist):
            if isinstance(x_range, list):
                edge_ev += item * x_range[j] * num_species
            elif isinstance(x_range, int):
                edge_ev += item * (j + x_range) * num_species
            else:
                edge_ev += item * (j+1) * num_species

        return edge_ev

    def trim_pmf_general(edge_count_target, dist, dist_range=None):

        if not dist_range:
            dist_range = [i + 1 for i in range(len(dist))]

        edge_ev = find_edges_expected_value(dist, dist_range)
        reduced_species = deepcopy(n_species)

        dist_0 = None
        dist_range_0 = None
        edge_ev_0 = None

        while edge_ev > edge_count_target:

            dist_0 = deepcopy(dist)
            dist_range_0 = deepcopy(dist_range)
            edge_ev_0 = deepcopy(edge_ev)
            reduced_species -= 1
            freqs = [reduced_species * dist[i] for i in range(len(dist))]

            while any(freq < min_node_deg for freq in freqs):
                rm_ind = freqs.index(min(freqs))
                del dist[rm_ind]
                del dist_range[rm_ind]
                dist_sum = sum(dist)
                dist = [dist[i]/dist_sum for i in range(len(dist))]
                freqs = [reduced_species * dist[i] for i in range(len(dist))]

            edge_ev = find_edges_expected_value(dist, dist_range, reduced_species)

        if abs(edge_ev - edge_count_target) < abs(edge_ev_0 - edge_count_target):

            return dist, dist_range

        if abs(edge_ev - edge_count_target) >= abs(edge_ev_0 - edge_count_target):

            return dist_0, dist_range_0

    def joint_unbounded_pmf(joint_dist1):

        dist = [(1, 1)]
        dscores = [joint_dist1(1, 1)]
        dsum = dscores[-1]
        edge = []
        edge_scores = []

        while True:

            for item in dist:
                item1 = (item[0] + 1, item[1])
                item2 = (item[0], item[1] + 1)
                item3 = (item[0] + 1, item[1] + 1)
                if item1 not in dist and item1 not in edge:
                    edge.append(item1)
                    edge_scores.append(joint_dist1(item1[0], item1[1]))
                if item2 not in dist and item2 not in edge:
                    edge.append(item2)
                    edge_scores.append(joint_dist1(item2[0], item2[1]))
                if item3 not in dist and item3 not in edge:
                    edge.append(item3)
                    edge_scores.append(joint_dist1(item3[0], item3[1]))

            tiles = []
            low_score = 0
            for j, item in enumerate(edge_scores):
                if item == low_score:
                    tiles.append(j)
                elif item > low_score:
                    tiles = [j]
                    low_score = item

            new_dist = deepcopy(dist)
            new_dscores = deepcopy(dscores)

            for j in tiles:
                new_dist.append(edge[j])
                new_dscores.append(joint_dist1(edge[j][0], edge[j][1]))
                dsum += joint_dist1(edge[j][0], edge[j][1])

            scaled_dscores = []
            for item in new_dscores:
                scaled_dscores.append(n_species * item / dsum)

            if any(x < min_node_deg for x in scaled_dscores[:len(dist)]):
                raise Exception("\nThe provided distribution appears to be malformed; consider revising.")
            if any(x < min_node_deg for x in scaled_dscores[len(dist):]):
                break

            dist = new_dist
            dscores = new_dscores

            new_edge = []
            new_edge_scores = []

            for j, item in enumerate(edge):
                if j not in tiles:
                    new_edge.append(item)
                    new_edge_scores.append(edge_scores[j])

            edge = new_edge
            edge_scores = new_edge_scores

        joint_pmf = []
        dsum = sum(dscores)
        for j, item in enumerate(dist):
            joint_pmf.append([item[0], item[1], dscores[j] / dsum])

        return joint_pmf

    def joint_bounded_pmf(joint_dist1, joint_range1):

        joint_pmf = []

        for j in range(joint_range1[0], joint_range1[1]+1):
            for k in range(joint_range1[0], joint_range1[1]+1):
                joint_pmf.append([joint_dist1(j, k), 0., (j, k)])

        pmf_sum = sum(joint_pmf[j][0] for j in range(len(joint_pmf)))
        joint_pmf = [[joint_pmf[j][0]/pmf_sum, joint_pmf[j][0]*n_species/pmf_sum, joint_pmf[j][2]]
                     for j in range(len(joint_pmf))]
        joint_pmf.sort(key=lambda x: x[1])

        while joint_pmf[0][1] < min_node_deg:
            value = joint_pmf[0][1]
            joint_pmf = [x for x in joint_pmf if x[1] != value]
            pmf_sum = sum(joint_pmf[j][0] for j in range(len(joint_pmf)))
            joint_pmf = [[joint_pmf[j][0]/pmf_sum, joint_pmf[j][0]*n_species/pmf_sum, joint_pmf[j][2]]
                         for j in range(len(joint_pmf))]

        joint_pmf_temp = []
        for item in joint_pmf:
            joint_pmf_temp.append([item[2][0], item[2][1], item[0]])
        joint_pmf = joint_pmf_temp

        return joint_pmf

    input_case = None
    pmf_out = None
    pmf_in = None
    pmf_joint = None
    range_out = None
    range_in = None
    edge_ev_out = None
    edge_ev_in = None

    if out_dist == 'random' and in_dist == 'random':
        input_case = 0

    if callable(out_dist) and in_dist == 'random' and out_range is None:
        input_case = 1

    if callable(out_dist) and in_dist == 'random' and isinstance(out_range, list):
        input_case = 2

    if isinstance(out_dist, list) and in_dist == 'random' and all(isinstance(x[1], float) for x in out_dist):
        input_case = 3

    if isinstance(out_dist, list) and in_dist == 'random' and all(isinstance(x[1], int) for x in out_dist):
        input_case = 4

    if out_dist == 'random' and callable(in_dist) and in_range is None:
        input_case = 5

    if out_dist == 'random' and callable(in_dist) and isinstance(in_range, list):
        input_case = 6

    if out_dist == 'random' and isinstance(in_dist, list) and all(isinstance(x[1], float) for x in in_dist):
        input_case = 7

    if out_dist == 'random' and isinstance(in_dist, list) and all(isinstance(x[1], int) for x in in_dist):
        input_case = 8

    if callable(out_dist) and callable(in_dist):

        if in_dist == out_dist and in_range is None and out_range is None:
            input_case = 9
        if in_dist == out_dist and in_range and in_range == out_range:
            input_case = 10
        if in_dist == out_dist and in_range != out_range:
            input_case = 11
        if in_dist != out_dist and in_range is None and out_range is None:
            input_case = 12
        if in_dist != out_dist and in_range and out_range and in_range == out_range:
            input_case = 13
        if in_dist != out_dist and in_range != out_range:
            input_case = 14

    if isinstance(out_dist, list) and isinstance(in_dist, list):
        if all(isinstance(x[1], int) for x in out_dist) and all(isinstance(x[1], int) for x in in_dist):
            input_case = 15
        if all(isinstance(x[1], float) for x in out_dist) and all(isinstance(x[1], float) for x in in_dist):
            input_case = 16

    if callable(joint_dist):
        if not joint_range:
            input_case = 17
        if joint_range:
            input_case = 18

    if isinstance(joint_dist, list):
        if all(isinstance(x[2], float) for x in joint_dist):
            input_case = 19
        if all(isinstance(x[2], int) for x in joint_dist):
            input_case = 20

    # ---------------------------------------------------------------------------

    if input_case == 1:

        pmf_out, range_out = single_unbounded_pmf(out_dist)

    if input_case == 2:

        pmf_out, range_out = single_bounded_pmf(out_dist, out_range)

    if input_case == 3:

        pmf_out = [x[1] for x in out_dist]
        range_out = [x[0] for x in out_dist]

    if input_case == 4:
        pass

    if input_case == 5:

        pmf_in, range_in = single_unbounded_pmf(in_dist)

    if input_case == 6:

        pmf_in, range_in = single_bounded_pmf(in_dist, in_range)

    if input_case == 7:

        pmf_in = [x[1] for x in in_dist]
        range_in = [x[0] for x in in_dist]

    if input_case == 8:
        pass

    if input_case == 9:

        pmf_out, range_out = single_unbounded_pmf(out_dist)
        pmf_in, range_in = single_unbounded_pmf(in_dist)

    if input_case == 10:

        pmf_out, range_out = single_bounded_pmf(out_dist, out_range)
        pmf_in, range_in = single_bounded_pmf(in_dist, in_range)

    if input_case == 11:

        pmf_out, range_out = single_bounded_pmf(out_dist, out_range)
        pmf_in, range_in = single_bounded_pmf(in_dist, in_range)

        edge_ev_out = find_edges_expected_value(pmf_out, range_out)
        edge_ev_in = find_edges_expected_value(pmf_in, range_in)

        if edge_ev_in < edge_ev_out:
            pmf_out, range_out = trim_pmf_general(edge_ev_in, pmf_out)

        if edge_ev_in > edge_ev_out:
            pmf_in, range_in = trim_pmf_general(edge_ev_out, pmf_in)

    if input_case == 12:

        pmf_out, range_out = single_unbounded_pmf(out_dist)
        pmf_in, range_in = single_unbounded_pmf(in_dist)

        edge_ev_out = find_edges_expected_value(pmf_out, range_out)
        edge_ev_in = find_edges_expected_value(pmf_in, range_in)

        if edge_ev_in < edge_ev_out:
            pmf_out, range_out = trim_pmf_general(edge_ev_in, pmf_out)

        if edge_ev_in > edge_ev_out:
            pmf_in, range_in = trim_pmf_general(edge_ev_out, pmf_in)

    if input_case == 13:

        pmf_out, range_out = single_bounded_pmf(out_dist, out_range)
        pmf_in, range_in = single_bounded_pmf(in_dist, in_range)

        edge_ev_out = find_edges_expected_value(pmf_out, range_out)
        edge_ev_in = find_edges_expected_value(pmf_in, range_in)

        if edge_ev_in < edge_ev_out:
            pmf_out, range_out = trim_pmf_general(edge_ev_in, pmf_out, range_out)

        if edge_ev_in > edge_ev_out:
            pmf_in, in_range = trim_pmf_general(edge_ev_out, pmf_in, range_in)

    if input_case == 14:

        pmf_out, range_out = single_bounded_pmf(out_dist, out_range)
        pmf_in, range_in = single_bounded_pmf(in_dist, in_range)

        edge_ev_out = find_edges_expected_value(pmf_out, range_out)
        edge_ev_in = find_edges_expected_value(pmf_in, range_in)

        if edge_ev_in < edge_ev_out:
            pmf_out, range_out = trim_pmf_general(edge_ev_in, pmf_out)

        if edge_ev_in > edge_ev_out:
            pmf_in, range_in = trim_pmf_general(edge_ev_out, pmf_in)

    if input_case == 15:

        pass

    if input_case == 16:

        pmf_out = [x[1] for x in out_dist]
        pmf_in = [x[1] for x in in_dist]

        range_out = [x[0] for x in out_dist]
        range_in = [x[0] for x in in_dist]

        edge_ev_out = find_edges_expected_value(pmf_out, range_out)
        edge_ev_in = find_edges_expected_value(pmf_in, range_in)

        if edge_ev_in < edge_ev_out:
            pmf_out, range_out = trim_pmf_general(edge_ev_in, pmf_out, range_out)

        if edge_ev_in > edge_ev_out:
            pmf_in, range_in = trim_pmf_general(edge_ev_out, pmf_in, range_in)

    if input_case == 17:

        pmf_joint = joint_unbounded_pmf(joint_dist)

    if input_case == 18:

        pmf_joint = joint_bounded_pmf(joint_dist, joint_range)

    if input_case == 19:

        pass

    if input_case == 20:

        pass

    return input_case, pmf_out, pmf_in, pmf_joint, range_out, range_in, edge_ev_out, edge_ev_in


def generate_samples(n_species, in_dist, out_dist, joint_dist, input_case, pmf_out, pmf_in, pmf_joint,
                     range_out, range_in, edge_ev_out, edge_ev_in, independent_sampling, distribution_attempts):

    in_samples = []
    out_samples = []
    joint_samples = []

    def sample_single_pmf(pmf0, drange):

        samplest = [0 for _ in pmf0]
        outind = [j for j in range(len(pmf0))]

        j = 0
        while j < n_species:
            ind = random.choices(outind, pmf0)[0]
            samplest[ind] += 1
            j += 1

        samples = []
        for j in range(len(pmf0)):
            if samplest[j] > 0:
                samples.append((drange[j], samplest[j]))

        return samples

    def sample_both_pmfs(pmf01, drange1, pmf02, drange2):

        ind1 = [j for j in range(len(pmf01))]
        ind2 = [j for j in range(len(pmf02))]

        num_tries = 0
        while True:

            samples1t = [0 for _ in pmf01]

            j = 0
            while j < n_species:
                ind = random.choices(ind1, pmf01)[0]
                samples1t[ind] += 1
                j += 1

            samples1 = []
            for j in range(len(pmf01)):
                if samples1t[j] > 0:
                    samples1.append((drange1[j], samples1t[j]))

            edges1 = 0
            for item in samples1:
                edges1 += item[0] * item[1]

            num_tries += 1
            edges2 = 0
            nodes = 0
            samples2t = [0 for _ in pmf02]

            while edges2 < edges1 and nodes < n_species:
                ind = random.choices(ind2, pmf02)[0]
                samples2t[ind] += 1
                edges2 += drange2[ind]
                nodes += 1

            if edges2 == edges1:
                samples2 = []
                for j in range(len(pmf02)):
                    if samples2t[j] > 0:
                        samples2.append((drange2[j], samples2t[j]))
                break

            if num_tries == distribution_attempts:
                raise Exception("\nReconciliation of the input and output distributions was attempted" + str(n_species)
                                + "times.\n" "Consider revising these distributions.")

        return samples1, samples2

    def indep_sample_both_pmfs(pmf_o, range_o, pmf_i, range_i):

        num_tries = 0
        while True:
            num_tries += 1
            out_samp = sample_single_pmf(pmf_o, range_o)
            in_samp = sample_single_pmf(pmf_i, range_i)
            out_edge_count = 0
            in_edge_count = 0
            for each in out_samp:
                out_edge_count += each[0] * each[1]
            for each in in_samp:
                in_edge_count += each[0] * each[1]
            if out_edge_count == in_edge_count:
                break

            if num_tries == distribution_attempts:
                raise Exception("\nReconciliation of the input and output distributions was attempted" + str(n_species)
                                + "times.\n" "Consider revising these distributions.")

        return out_samp, in_samp

    def find_edge_count(dist):

        edge_count = 0
        for item in dist:
            edge_count += item[0] * item[1]

        return edge_count

    def sample_joint(joint_pmf):

        cells, joint_pmf = [[x[0], x[1]] for x in joint_pmf], [x[2] for x in joint_pmf]

        ind = [j for j, item in enumerate(joint_pmf)]

        count = 0
        while True:
            count += 1
            samplest = [0 for _ in joint_pmf]
            j = 0
            while j < n_species:
                sample = random.choices(ind, joint_pmf)[0]
                samplest[sample] += 1
                j += 1

            out_edges = 0
            in_edges = 0
            samples = []
            for j, item in enumerate(samplest):

                out_edges += item*cells[j][0]
                in_edges += item*cells[j][1]
                samples.append((cells[j][0], cells[j][1], item))

            if out_edges == in_edges:

                return samples

            if count == distribution_attempts:
                raise Exception("\nYour joint distribution was sampled" + str(n_species) + "times.\n"
                                "Reconciliation of the outgoing and incoming edges was not achieved.\n"
                                "Consider revising this distribution.")

    # ---------------------------------------------------------------------------

    if input_case == 1:

        out_samples = sample_single_pmf(pmf_out, range_out)

    if input_case == 2:

        out_samples = sample_single_pmf(pmf_out, range_out)

    if input_case == 3:

        out_samples = sample_single_pmf(pmf_out, range_out)

    if input_case == 4:
        out_samples = out_dist

    if input_case == 5:

        in_samples = sample_single_pmf(pmf_in, range_in)

    if input_case == 6:

        in_samples = sample_single_pmf(pmf_in, range_in)

    if input_case == 7:

        in_samples = sample_single_pmf(pmf_in, range_in)

    if input_case == 8:
        in_samples = in_dist

    if input_case == 9:

        if not independent_sampling:
            in_or_out = random.randint(0, 1)  # choose which distribution is guaranteed n_species
            if in_or_out:
                in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)
            else:
                out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)
        else:
            out_samples, in_samples = indep_sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

    if input_case == 10:

        if not independent_sampling:
            in_or_out = random.randint(0, 1)  # choose which distribution is guaranteed n_species
            if in_or_out:
                in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)
            else:
                out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)
        else:
            out_samples, in_samples = indep_sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

    if input_case == 11:

        if edge_ev_in < edge_ev_out:
            in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)

        if edge_ev_in > edge_ev_out:
            out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

        if edge_ev_in == edge_ev_out:
            if not independent_sampling:
                in_or_out = random.randint(0, 1)  # choose which distribution is guaranteed n_species
                if in_or_out:
                    in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)
                else:
                    out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)
            else:
                out_samples, in_samples = indep_sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

    if input_case == 12:

        if edge_ev_in < edge_ev_out:
            in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)

        if edge_ev_in > edge_ev_out:
            out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

        if edge_ev_in == edge_ev_out:
            if not independent_sampling:
                in_or_out = random.randint(0, 1)  # choose which distribution is guaranteed n_species
                if in_or_out:
                    in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)
                else:
                    out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)
            else:
                out_samples, in_samples = indep_sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

    if input_case == 13:

        if edge_ev_in < edge_ev_out:
            in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)

        if edge_ev_in > edge_ev_out:
            out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

        if edge_ev_in == edge_ev_out:
            if not independent_sampling:
                in_or_out = random.randint(0, 1)  # choose which distribution is guaranteed n_species
                if in_or_out:
                    in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)
                else:
                    out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)
            else:
                out_samples, in_samples = indep_sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

    if input_case == 14:

        if edge_ev_in < edge_ev_out:
            in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)

        if edge_ev_in > edge_ev_out:
            out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

        if edge_ev_in == edge_ev_out:
            if not independent_sampling:
                in_or_out = random.randint(0, 1)  # choose which distribution is guaranteed n_species
                if in_or_out:
                    in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)
                else:
                    out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)
            else:
                out_samples, in_samples = indep_sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

    if input_case == 15:

        if find_edge_count(out_dist) != find_edge_count(in_dist):
            raise Exception("The edges counts for the input and output distributions must match.")

        out_samples = out_dist
        in_samples = in_dist

    if input_case == 16:

        if edge_ev_in < edge_ev_out:
            in_samples, out_samples = sample_both_pmfs(pmf_in, range_in, pmf_out, range_out)

        if edge_ev_in > edge_ev_out:
            out_samples, in_samples = sample_both_pmfs(pmf_out, range_out, pmf_in, range_in)

    if input_case == 17:

        joint_samples = sample_joint(pmf_joint)

    if input_case == 18:

        joint_samples = sample_joint(pmf_joint)

    if input_case == 19:

        joint_samples = sample_joint(joint_dist)

    if input_case == 20:

        joint_samples = joint_dist

    return in_samples, out_samples, joint_samples


def generate_reactions(in_samples, out_samples, joint_samples, n_species, n_reactions, rxn_prob, mod_reg, gma_reg, 
                       sc_reg, mass_violating_reactions, unaffected_nodes, edge_type, mass_balanced, connected):

    in_nodes_count = []
    if bool(in_samples):
        for each in in_samples:
            for i in range(each[1]):
                in_nodes_count.append(each[0])

    out_nodes_count = []
    if bool(out_samples):
        for each in out_samples:
            for i in range(each[1]):
                out_nodes_count.append(each[0])

    if bool(joint_samples):
        for each in joint_samples:
            for i in range(each[2]):
                out_nodes_count.append(each[0])
                in_nodes_count.append(each[1])

    max_count = max(len(in_nodes_count), len(out_nodes_count))

    in_nodes_list = []
    out_nodes_list = []
    for i in range(max_count):
        in_nodes_list.append(i)
        out_nodes_list.append(i)

    while len(out_nodes_count) < len(in_nodes_count):
        out_nodes_count.append(0)

    while len(in_nodes_count) < len(out_nodes_count):
        in_nodes_count.append(0)

    if not bool(joint_samples):
        random.shuffle(in_nodes_count)
        random.shuffle(out_nodes_count)

    reaction_list = []
    reaction_list2 = []
    edge_list = []

    s_matrix = np.empty((0, n_species), int)
    c_groups = []

    def consistency_check(reactants, products):

        s_matrix_copy = deepcopy(s_matrix)
        s_matrix_row = np.array([0 for _ in range(n_species)])
        for react in reactants:
            s_matrix_row[react] -= 1
        for prod in products:
            s_matrix_row[prod] += 1
        s_matrix_copy = np.append(s_matrix_copy, [s_matrix_row], axis=0)
        b = np.zeros(s_matrix_copy.shape[0])
        c = np.ones(s_matrix_copy.shape[1])
        ulb = (1, None)

        warnings.filterwarnings("ignore")
        res = linprog(c, A_eq=s_matrix_copy, b_eq=b, bounds=ulb)
        warnings.filterwarnings("default")

        return res.success, s_matrix_copy

    def iterative_connected_check(reactants, products, modifiers, out_list, in_list):

        c_groups_copy = deepcopy(c_groups)
        elems = []

        for item in reactants:
            if item not in elems:
                elems.append(item)
        for item in products:
            if item not in elems:
                elems.append(item)
        for item in modifiers:
            if item not in elems:
                elems.append(item)

        attached_groups = []
        unattached_groups = []

        for j, item in enumerate(c_groups_copy):
            if set(item).intersection(set(elems)):
                attached_groups.append(j)
            else:
                unattached_groups.append(item)
        for j in attached_groups:
            for item in c_groups_copy[j]:
                if item not in elems:
                    elems.append(item)

        c_groups_copy = unattached_groups
        c_groups_copy.append(elems)

        edge_availability = []
        for item in c_groups_copy:
            edge_availability.append(0)
            for every in item:
                if out_list[every] > 0 or in_list[every] > 0:
                    edge_availability[-1] = 1
                    break

        closed_gr = True
        if not all(out_list[j] == 0 for j in range(len(out_list))) and \
                all(edge_availability[j] == 1 for j in range(len(edge_availability))):
            closed_gr = False
        if all(out_list[j] == 0 for j in range(len(out_list))) and \
                len(edge_availability) == 1:
            closed_gr = False

        return closed_gr, c_groups_copy

    def connected_check(reactants, products, modifiers):

        c_groups_copy = deepcopy(c_groups)
        elems = []

        for item in reactants:
            if item not in elems:
                elems.append(item)
        for item in products:
            if item not in elems:
                elems.append(item)
        for item in modifiers:
            if item not in elems:
                elems.append(item)

        attached_groups = []
        unattached_groups = []

        for j, item in enumerate(c_groups_copy):
            if set(item).intersection(set(elems)):
                attached_groups.append(j)
            else:
                unattached_groups.append(item)
        for j in attached_groups:
            for item in c_groups_copy[j]:
                if item not in elems:
                    elems.append(item)

        c_groups_copy = unattached_groups
        c_groups_copy.append(elems)

        return c_groups_copy

    # ---------------------------------------------------------------------------------------------------

    nodes_list = [i for i in range(n_species)]

    if not bool(out_samples) and not bool(in_samples):

        node_set = set()
        pick_continued = 0
        while True:

            # todo: This is an issue for larger networks: link cutoff with number of species
            if pick_continued == n_species:
                return [None], [out_samples, in_samples, joint_samples]

            if rxn_prob:
                rt = _pick_reaction_type(rxn_prob)
            else:
                rt = _pick_reaction_type()

            mod_num = 0
            if mod_reg:
                mod_num = random.choices([0, 1, 2, 3], mod_reg[0])[0]

            gma_num = 0
            if gma_reg:
                gma_num = random.choices([0, 1, 2, 3], gma_reg[0])[0]

            sc_num = 0
            if sc_reg:
                sc_num = random.choices([0, 1, 2, 3], sc_reg[0])[0]

            # -----------------------------------------------------------------------------

            if rt == TReactionType.UNIUNI:

                product = random.choice(nodes_list)
                reactant = random.choice(nodes_list)

                if [[reactant], [product]] in reaction_list2 or reactant == product:
                    pick_continued += 1
                    continue

                if mass_balanced:

                    res_result, s_mat_c = consistency_check([reactant], [product])
                    if not res_result:
                        pick_continued += 1
                        continue
                    else:
                        s_matrix = s_mat_c

                mod_species = []
                reg_signs = []
                reg_type = []

                if mod_reg:
                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                if gma_reg:
                    mod_species = random.sample(nodes_list, gma_num)
                    reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                    reg_type = ['gma' for _ in mod_species]

                if sc_reg:
                    mod_species = random.sample(nodes_list, sc_num)
                    reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                    reg_type = ['sc' for _ in mod_species]

                reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                reaction_list2.append([[reactant], [product]])

                if edge_type == 'generic':
                    edge_list.append((reactant, product))
                    node_set.add(reactant)
                    node_set.add(product)
                    node_set.update(mod_species)
                if edge_type == 'metabolic':
                    edge_list.append((reactant, product))
                    node_set.add(reactant)
                    node_set.add(product)

                if connected:
                    c_groups = connected_check([reactant], [product], mod_species)

            if rt == TReactionType.BIUNI:

                product = random.choice(nodes_list)
                reactant1 = random.choice(nodes_list)
                reactant2 = random.choice(nodes_list)

                if [[reactant1, reactant2], [product]] in reaction_list2:
                    pick_continued += 1
                    continue

                if not mass_violating_reactions and product in {reactant1, reactant2}:
                    pick_continued += 1
                    continue

                if mass_balanced:

                    res_result, s_mat_c = consistency_check([reactant1, reactant2], [product])
                    if not res_result:
                        pick_continued += 1
                        continue
                    else:
                        s_matrix = s_mat_c

                mod_species = []
                reg_signs = []
                reg_type = []

                if mod_reg:
                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                if gma_reg:
                    mod_species = random.sample(nodes_list, gma_num)
                    reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                    reg_type = ['gma' for _ in mod_species]

                if sc_reg:
                    mod_species = random.sample(nodes_list, sc_num)
                    reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                    reg_type = ['sc' for _ in mod_species]

                reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                reaction_list2.append([[reactant1, reactant2], [product]])

                if edge_type == 'generic':
                    edge_list.append((reactant1, product))
                    edge_list.append((reactant2, product))
                    node_set.add(reactant1)
                    node_set.add(reactant2)
                    node_set.add(product)
                    node_set.update(mod_species)

                if edge_type == 'metabolic':
                    if reactant1 != reactant2 and reactant1 != product and reactant2 != product:
                        edge_list.append((reactant1, product))
                        edge_list.append((reactant2, product))
                        node_set.add(reactant1)
                        node_set.add(reactant2)
                        node_set.add(product)
                    if reactant1 == reactant2 and reactant1 != product:
                        edge_list.append((reactant1, product))
                        node_set.add(reactant1)
                        node_set.add(product)
                    if reactant1 != reactant2 and reactant1 == product:
                        edge_list.append((reactant2, 'deg'))
                    if reactant1 != reactant2 and reactant2 == product:
                        edge_list.append((reactant1, 'deg'))
                    if reactant1 == reactant2 and reactant1 == product:
                        edge_list.append((reactant1, 'deg'))

                if connected:
                    c_groups = connected_check([reactant1, reactant2], [product], mod_species)

            if rt == TReactionType.UNIBI:

                reactant = random.choice(nodes_list)
                product1 = random.choice(nodes_list)
                product2 = random.choice(nodes_list)

                if [[reactant], [product1, product2]] in reaction_list2:
                    pick_continued += 1
                    continue

                if [[reactant], [product2, product1]] in reaction_list2:
                    pick_continued += 1
                    continue

                if not mass_violating_reactions and reactant in {product1, product2}:
                    pick_continued += 1
                    continue

                if mass_balanced:

                    res_result, s_mat_c = consistency_check([reactant], [product1, product2])
                    if not res_result:
                        pick_continued += 1
                        continue
                    else:
                        s_matrix = s_mat_c

                mod_species = []
                reg_signs = []
                reg_type = []

                if mod_reg:
                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                if gma_reg:
                    mod_species = random.sample(nodes_list, gma_num)
                    reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                    reg_type = ['gma' for _ in mod_species]

                if sc_reg:
                    mod_species = random.sample(nodes_list, sc_num)
                    reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                    reg_type = ['sc' for _ in mod_species]

                reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                reaction_list2.append([[reactant], [product1, product2]])

                if edge_type == 'generic':
                    edge_list.append((reactant, product1))
                    edge_list.append((reactant, product2))
                    node_set.add(reactant)
                    node_set.add(product1)
                    node_set.add(product2)
                    node_set.update(mod_species)

                if edge_type == 'metabolic':
                    if reactant != product1 and reactant != product2 and product1 != product2:
                        edge_list.append((reactant, product1))
                        edge_list.append((reactant, product2))
                        node_set.add(reactant)
                        node_set.add(product1)
                        node_set.add(product2)
                    if reactant != product1 and product1 == product2:
                        edge_list.append((reactant, product1))
                        node_set.add(reactant)
                        node_set.add(product1)
                    if reactant == product1 and product1 != product2:
                        edge_list.append(('syn', product2))
                    if reactant == product2 and product1 != product2:
                        edge_list.append(('syn', product1))
                    if reactant == product1 and product1 == product2:
                        edge_list.append(('syn', reactant))

                if connected:
                    c_groups = connected_check([reactant], [product1, product2], mod_species)

            if rt == TReactionType.BIBI:

                product1 = random.choice(nodes_list)
                product2 = random.choice(nodes_list)
                reactant1 = random.choice(nodes_list)
                reactant2 = random.choice(nodes_list)

                if [[reactant1, reactant2], [product1, product2]] in reaction_list2:
                    pick_continued += 1
                    continue

                if [[reactant2, reactant1], [product1, product2]] in reaction_list2:
                    pick_continued += 1
                    continue

                if [[reactant1, reactant2], [product2, product1]] in reaction_list2:
                    pick_continued += 1
                    continue

                if [[reactant2, reactant1], [product2, product1]] in reaction_list2:
                    pick_continued += 1
                    continue

                if {reactant1, reactant2} == {product1, product2}:
                    pick_continued += 1
                    continue

                if not unaffected_nodes:
                    intersect = {reactant1, reactant2}.intersection({product1, product2})
                    if len(intersect) > 0:
                        pick_continued += 1
                        continue

                if mass_balanced:
                    res_result, s_mat_c = consistency_check([reactant1, reactant2], [product1, product2])
                    if not res_result:
                        pick_continued += 1
                        continue
                    else:
                        s_matrix = s_mat_c

                mod_species = []
                reg_signs = []
                reg_type = []

                if mod_reg:
                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                if gma_reg:
                    mod_species = random.sample(nodes_list, gma_num)
                    reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                    reg_type = ['gma' for _ in mod_species]

                if sc_reg:
                    mod_species = random.sample(nodes_list, sc_num)
                    reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                    reg_type = ['sc' for _ in mod_species]

                reaction_list.append(
                    [rt, [reactant1, reactant2], [product1, product2], mod_species, reg_signs, reg_type])
                reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                if edge_type == 'generic':

                    edge_list.append((reactant1, product1))
                    edge_list.append((reactant2, product1))
                    edge_list.append((reactant1, product2))
                    edge_list.append((reactant2, product2))
                    node_set.add(reactant1)
                    node_set.add(reactant2)
                    node_set.add(product1)
                    node_set.add(product2)
                    node_set.update(mod_species)

                if edge_type == 'metabolic':

                    if len({reactant1, reactant2, product1, product2}) \
                            == len([reactant1, reactant2, product1, product2]):
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant1, product2))
                        edge_list.append((reactant2, product1))
                        edge_list.append((reactant2, product2))
                        node_set.add(reactant1)
                        node_set.add(reactant2)
                        node_set.add(product1)
                        node_set.add(product2)

                    if reactant1 == reactant2 and \
                            len({reactant1, product1, product2}) == len([reactant1, product1, product2]):
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant1, product2))
                        node_set.add(reactant1)
                        node_set.add(product1)
                        node_set.add(product2)

                    if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                        edge_list.append((reactant1, product1))
                        node_set.add(reactant1)
                        node_set.add(product1)

                    if product1 == product2 and \
                            len({reactant1, reactant2, product1}) == len([reactant1, reactant2, product1]):
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant2, product1))
                        node_set.add(reactant1)
                        node_set.add(reactant2)
                        node_set.add(product1)

                    if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant2, product2))
                        node_set.add(reactant2)
                        node_set.add(product2)

                    if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant2, product1))
                        node_set.add(reactant2)
                        node_set.add(product1)

                    if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant1, product2))
                        node_set.add(reactant1)
                        node_set.add(product2)

                    if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant1, product1))
                        node_set.add(reactant1)
                        node_set.add(product1)

                    if reactant1 != reactant2 and len({reactant1, product1, product2}) == 1:
                        edge_list.append((reactant2, product2))
                        node_set.add(reactant2)
                        node_set.add(product2)
                    if reactant1 != reactant2 and len({reactant2, product1, product2}) == 1:
                        edge_list.append((reactant1, product1))
                        node_set.add(reactant1)
                        node_set.add(product1)

                    if product1 != product2 and len({reactant1, reactant2, product1}) == 1:
                        edge_list.append((reactant2, product2))
                        node_set.add(reactant2)
                        node_set.add(product2)
                    if product1 != product2 and len({reactant1, reactant2, product2}) == 1:
                        edge_list.append((reactant1, product1))
                        node_set.add(reactant1)
                        node_set.add(product1)

                if connected:
                    c_groups = connected_check([reactant1, reactant2], [product1, product2], mod_species)

            if n_reactions:
                if len(node_set) >= n_species and len(reaction_list) >= n_reactions:
                    break
            else:
                if len(node_set) == n_species:
                    break

    # -----------------------------------------------------------------

    if not bool(out_samples) and bool(in_samples):

        pick_continued = 0

        while True:

            if pick_continued == n_species:
                return [None], [out_samples, in_samples, joint_samples]

            if rxn_prob:
                rt = _pick_reaction_type(rxn_prob)
            else:
                rt = _pick_reaction_type()

            mod_num = 0
            if mod_reg:
                mod_num = random.choices([0, 1, 2, 3], mod_reg[0])[0]

            gma_num = 0
            if gma_reg:
                gma_num = random.choices([0, 1, 2, 3], gma_reg[0])[0]

            sc_num = 0
            if sc_reg:
                sc_num = random.choices([0, 1, 2, 3], sc_reg[0])[0]

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIUNI:

                if edge_type == 'generic':

                    if max(in_nodes_count) < (1 + mod_num + gma_num + sc_num):
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product] < (1 + mod_num + gma_num + sc_num):
                        product = random.choices(in_nodes_list, prob_in)[0]

                    reactant = random.choice(in_nodes_list)

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]
    
                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    in_nodes_count[product] -= (1 + mod_num + gma_num + sc_num)
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                    edge_list.append((reactant, product))

                    if connected:
                        c_groups = connected_check([reactant], [product], mod_species)

                if edge_type == 'metabolic':

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]

                    reactant = random.choice(in_nodes_list)

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]
    
                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    in_nodes_count[product] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                    edge_list.append((reactant, product))

                    if connected:
                        c_groups = connected_check([reactant], [product], mod_species)

            # -----------------------------------------------------------------

            if rt == TReactionType.BIUNI:

                if edge_type == 'generic':

                    if max(in_nodes_count) < (2 + mod_num + gma_num + sc_num):
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product] < (2 + mod_num + gma_num + sc_num):
                        product = random.choices(in_nodes_list, prob_in)[0]

                    reactant1 = random.choice(in_nodes_list)
                    reactant2 = random.choice(in_nodes_list)

                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]
    
                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    in_nodes_count[product] -= (2 + mod_num + gma_num + sc_num)
                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

                    edge_list.append((reactant1, product))
                    edge_list.append((reactant2, product))

                    if connected:
                        c_groups = connected_check([reactant1, reactant2], [product], mod_species)

                if edge_type == 'metabolic':

                    if max(in_nodes_count) < 2:
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product] < 2:
                        product = random.choices(in_nodes_list, prob_in)[0]

                    reactant1 = random.choice(in_nodes_list)
                    reactant2 = random.choice(in_nodes_list)

                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]
    
                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if reactant1 != reactant2 and reactant1 != product and reactant2 != product:
                        edge_list.append((reactant1, product))
                        edge_list.append((reactant2, product))
                        in_nodes_count[product] -= 2
                    if reactant1 == reactant2 and reactant1 != product:
                        edge_list.append((reactant1, product))
                        in_nodes_count[product] -= 1
                    if reactant1 != reactant2 and reactant1 == product:
                        edge_list.append((reactant2, 'deg'))
                    if reactant1 != reactant2 and reactant2 == product:
                        edge_list.append((reactant1, 'deg'))
                    if reactant1 == reactant2 and reactant1 == product:
                        edge_list.append((reactant1, 'deg'))

                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

                    if connected:
                        c_groups = connected_check([reactant1, reactant2], [product], mod_species)

            if rt == TReactionType.UNIBI:

                if edge_type == 'generic':

                    if sum(1 for each in in_nodes_count if each >= (1 + mod_num + gma_num + sc_num)) < 2 \
                            and max(in_nodes_count) < (2 + 2 * mod_num + 2 * gma_num + 2 * sc_num):
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < (1 + mod_num + gma_num + sc_num):
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= (1 + mod_num + gma_num + sc_num)
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < (1 + mod_num + gma_num + sc_num):
                        product2 = random.choices(in_nodes_list, prob_in_copy)[0]

                    reactant = random.choice(in_nodes_list)

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]
    
                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    in_nodes_count[product1] -= (1 + mod_num + gma_num + sc_num)
                    in_nodes_count[product2] -= (1 + mod_num + gma_num + sc_num)
                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

                    edge_list.append((reactant, product1))
                    edge_list.append((reactant, product2))

                    if connected:
                        c_groups = connected_check([reactant], [product1, product2], mod_species)

                if edge_type == 'metabolic':

                    if sum(1 for each in in_nodes_count if each >= 1) < 2 \
                            and max(in_nodes_count) < 2:
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < 1:
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= 1
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < 1:
                        product2 = random.choices(in_nodes_list, prob_in_copy)[0]

                    reactant = random.choice(in_nodes_list)

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]
    
                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if reactant != product1 and reactant != product2 and product1 != product2:
                        edge_list.append((reactant, product1))
                        edge_list.append((reactant, product2))
                        in_nodes_count[product1] -= 1
                        in_nodes_count[product2] -= 1
                    if reactant != product1 and product1 == product2:
                        edge_list.append((reactant, product1))
                        in_nodes_count[product1] -= 1
                    if reactant == product1 and product1 != product2:
                        edge_list.append(('syn', product2))
                    if reactant == product2 and product1 != product2:
                        edge_list.append(('syn', product1))
                    if reactant == product1 and product1 == product2:
                        edge_list.append(('syn', reactant))

                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

                    if connected:
                        c_groups = connected_check([reactant], [product1, product2], mod_species)

            if rt == TReactionType.BIBI:

                if edge_type == 'generic':

                    if sum(1 for each in in_nodes_count if each >= (2 + mod_num + gma_num + sc_num)) < 2 \
                            and max(in_nodes_count) < (4 + 2 * mod_num + 2 * gma_num + 2 * sc_num):
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < (2 + mod_num + gma_num + sc_num):
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= (2 + mod_num + gma_num + sc_num)
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < (2 + mod_num + gma_num + sc_num):
                        product2 = random.choices(in_nodes_list, prob_in)[0]

                    reactant1 = random.choice(in_nodes_list)
                    reactant2 = random.choice(in_nodes_list)

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant1, reactant2], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if not unaffected_nodes:
                        intersect = {reactant1, reactant2}.intersection({product1, product2})
                        if len(intersect) > 0:
                            pick_continued += 1
                            continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]
    
                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    in_nodes_count[product1] -= (2 + mod_num + gma_num + sc_num)
                    in_nodes_count[product2] -= (2 + mod_num + gma_num + sc_num)
                    reaction_list.append([rt, [reactant1, reactant2], [product1, product2],
                                          mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                    edge_list.append((reactant1, product1))
                    edge_list.append((reactant2, product1))
                    edge_list.append((reactant1, product2))
                    edge_list.append((reactant2, product2))

                    if connected:
                        c_groups = connected_check([reactant1, reactant2], [product1, product2], mod_species)

                if edge_type == 'metabolic':

                    if sum(1 for each in in_nodes_count if each >= 2) < 2 \
                            and max(in_nodes_count) < 4:
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < 2:
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= 2
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < 2:
                        product2 = random.choices(in_nodes_list, prob_in)[0]

                    reactant1 = random.choice(in_nodes_list)
                    reactant2 = random.choice(in_nodes_list)

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant1, reactant2], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if not unaffected_nodes:
                        intersect = {reactant1, reactant2}.intersection({product1, product2})
                        if len(intersect) > 0:
                            pick_continued += 1
                            continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]
    
                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if len({reactant1, reactant2, product1, product2}) == 4:
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant1, product2))
                        edge_list.append((reactant2, product1))
                        edge_list.append((reactant2, product2))
                        in_nodes_count[product1] -= 2
                        in_nodes_count[product2] -= 2

                    if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant1, product2))
                        in_nodes_count[product1] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                        edge_list.append((reactant1, product1))
                        in_nodes_count[product1] -= 1

                    if product1 == product2 and \
                            len({reactant1, reactant2, product1}) == len([reactant1, reactant2, product1]):
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant2, product1))
                        in_nodes_count[product1] -= 2

                    if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant2, product2))
                        in_nodes_count[product2] -= 1

                    if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant2, product1))
                        in_nodes_count[product1] -= 1

                    if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant1, product2))
                        in_nodes_count[product2] -= 1

                    if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant1, product1))
                        in_nodes_count[product1] -= 1

                    if reactant1 != reactant2 and len({reactant1, product1, product2}) == 1:
                        edge_list.append((reactant2, product2))
                        in_nodes_count[product2] -= 1

                    if reactant1 != reactant2 and len({reactant2, product1, product2}) == 1:
                        edge_list.append((reactant1, product1))
                        in_nodes_count[product1] -= 1

                    if product1 != product2 and len({reactant1, reactant2, product1}) == 1:
                        edge_list.append((reactant2, product2))
                        in_nodes_count[product2] -= 1

                    if product1 != product2 and len({reactant1, reactant2, product2}) == 1:
                        edge_list.append((reactant1, product1))
                        in_nodes_count[product1] -= 1

                    reaction_list.append([rt, [reactant1, reactant2], [product1, product2],
                                          mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                    if connected:
                        c_groups = connected_check([reactant1, reactant2], [product1, product2], mod_species)

            if sum(in_nodes_count) == 0:
                break

    if bool(out_samples) and not bool(in_samples):

        pick_continued = 0
        while True:
            if pick_continued == n_species:
                return [None], [out_samples, in_samples, joint_samples]

            if rxn_prob:
                rt = _pick_reaction_type(rxn_prob)
            else:
                rt = _pick_reaction_type()

            mod_num = 0
            if mod_reg:
                mod_num = random.choices([0, 1, 2, 3], mod_reg[0])[0]

            gma_num = 0
            if gma_reg:
                gma_num = random.choices([0, 1, 2, 3], gma_reg[0])[0]

            sc_num = 0
            if sc_reg:
                sc_num = random.choices([0, 1, 2, 3], sc_reg[0])[0]

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIUNI:

                if edge_type == 'generic':

                    if sum(out_nodes_count) < (1 + mod_num + gma_num + sc_num):
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]

                    product = random.choice(out_nodes_list)

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        if mod_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < mod_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < mod_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        if gma_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < gma_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < gma_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        if sc_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < sc_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < sc_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    out_nodes_count[reactant] -= 1
                    for each in mod_species:
                        out_nodes_count[each] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                    edge_list.append((reactant, product))

                    if connected:
                        c_groups = connected_check([reactant], [product], mod_species)

                if edge_type == 'metabolic':

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]

                    product = random.choice(out_nodes_list)

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    out_nodes_count[reactant] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                    if reactant != product:
                        edge_list.append((reactant, product))

                    if connected:
                        c_groups = connected_check([reactant], [product], mod_species)

            if rt == TReactionType.BIUNI:

                if edge_type == 'generic':

                    if sum(out_nodes_count) < (2 + mod_num + gma_num + sc_num):
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 1
                    sum_out_copy = sum(out_nodes_count_copy)
                    prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                    reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    product = random.choice(out_nodes_list)

                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        if mod_num > 0:
                            out_nodes_count_copy[reactant2] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < mod_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < mod_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        if gma_num > 0:
                            out_nodes_count_copy[reactant2] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < gma_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < gma_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        if sc_num > 0:
                            out_nodes_count_copy[reactant2] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < sc_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < sc_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    out_nodes_count[reactant1] -= 1
                    out_nodes_count[reactant2] -= 1
                    for each in mod_species:
                        out_nodes_count[each] -= 1
                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

                    edge_list.append((reactant1, product))
                    edge_list.append((reactant2, product))

                    if connected:
                        c_groups = connected_check([reactant1, reactant2], [product], mod_species)

                if edge_type == 'metabolic':

                    if sum(out_nodes_count) < 2:
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 1
                    sum_out_copy = sum(out_nodes_count_copy)
                    prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                    reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    product = random.choice(out_nodes_list)

                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if reactant1 != reactant2 and reactant1 != product and reactant2 != product:
                        edge_list.append((reactant1, product))
                        edge_list.append((reactant2, product))
                        out_nodes_count[reactant1] -= 1
                        out_nodes_count[reactant2] -= 1
                    if reactant1 == reactant2 and reactant1 != product:
                        edge_list.append((reactant1, product))
                        out_nodes_count[reactant1] -= 1
                    if reactant1 != reactant2 and reactant1 == product:
                        edge_list.append((reactant2, 'deg'))
                    if reactant1 != reactant2 and reactant2 == product:
                        edge_list.append((reactant1, 'deg'))
                    if reactant1 == reactant2 and reactant1 == product:
                        edge_list.append((reactant1, 'deg'))

                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

                    if connected:
                        c_groups = connected_check([reactant1, reactant2], [product], mod_species)

            if rt == TReactionType.UNIBI:

                if edge_type == 'generic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= (1 + mod_num + gma_num + sc_num):
                        cont = True
                    if sum(1 for each in out_nodes_count if each >= 2) >= (mod_num + gma_num + sc_num - 1) \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True
                    if not cont:
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]
                    while out_nodes_count[reactant] < 2:
                        reactant = random.choices(out_nodes_list, prob_out)[0]

                    product1 = random.choice(out_nodes_list)
                    product2 = random.choice(out_nodes_list)

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        if mod_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < mod_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < mod_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        if gma_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < gma_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < gma_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        if sc_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < sc_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < sc_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    out_nodes_count[reactant] -= 2
                    for each in mod_species:
                        out_nodes_count[each] -= 2
                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

                    edge_list.append((reactant, product1))
                    edge_list.append((reactant, product2))

                    if connected:
                        c_groups = connected_check([reactant], [product1, product2], mod_species)

                if edge_type == 'metabolic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= 1:
                        cont = True
                    if sum(1 for each in out_nodes_count if each >= 2) >= -1 \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True
                    if not cont:
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]
                    while out_nodes_count[reactant] < 2:
                        reactant = random.choices(out_nodes_list, prob_out)[0]

                    product1 = random.choice(out_nodes_list)
                    product2 = random.choice(out_nodes_list)

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if reactant != product1 and reactant != product2 and product1 != product2:
                        edge_list.append((reactant, product1))
                        edge_list.append((reactant, product2))
                        out_nodes_count[reactant] -= 2
                    if reactant != product1 and product1 == product2:
                        edge_list.append((reactant, product1))
                        out_nodes_count[reactant] -= 1
                    if reactant == product1 and product1 != product2:
                        edge_list.append(('syn', product2))
                    if reactant == product2 and product1 != product2:
                        edge_list.append(('syn', product1))
                    if reactant == product1 and product1 == product2:
                        edge_list.append(('syn', reactant))

                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

                    if connected:
                        c_groups = connected_check([reactant], [product1, product2], mod_species)

            if rt == TReactionType.BIBI:

                if edge_type == 'generic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= (2 + mod_num + gma_num + sc_num):
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= mod_num + gma_num + sc_num \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= (mod_num + gma_num + sc_num - 2) \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 2:
                        cont = True

                    if not cont:
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]
                    while out_nodes_count[reactant1] < 2:
                        reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 2
                    sum_out_copy = sum(out_nodes_count_copy)
                    prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                    reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]
                    while out_nodes_count_copy[reactant2] < 2:
                        reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    product1 = random.choice(out_nodes_list)
                    product2 = random.choice(out_nodes_list)

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant1, reactant2], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if not unaffected_nodes:
                        intersect = {reactant1, reactant2}.intersection({product1, product2})
                        if len(intersect) > 0:
                            pick_continued += 1
                            continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        if mod_num > 0:
                            out_nodes_count_copy[reactant2] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < mod_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < mod_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        if gma_num > 0:
                            out_nodes_count_copy[reactant2] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < gma_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < gma_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        if sc_num > 0:
                            out_nodes_count_copy[reactant2] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < sc_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < sc_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    out_nodes_count[reactant1] -= 2
                    out_nodes_count[reactant2] -= 2
                    for each in mod_species:
                        out_nodes_count[each] -= 2
                    reaction_list.append(
                        [rt, [reactant1, reactant2], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                    edge_list.append((reactant1, product1))
                    edge_list.append((reactant2, product1))
                    edge_list.append((reactant1, product2))
                    edge_list.append((reactant2, product2))

                    if connected:
                        c_groups = connected_check([reactant1, reactant2], [product1, product2], mod_species)

                if edge_type == 'metabolic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= 2:
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= 0 \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= -2 \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 2:
                        cont = True

                    if not cont:
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]
                    while out_nodes_count[reactant1] < 2:
                        reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 2
                    sum_out_copy = sum(out_nodes_count_copy)
                    prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                    reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]
                    while out_nodes_count_copy[reactant2] < 2:
                        reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    product1 = random.choice(out_nodes_list)
                    product2 = random.choice(out_nodes_list)

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant1, reactant2], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if not unaffected_nodes:
                        intersect = {reactant1, reactant2}.intersection({product1, product2})
                        if len(intersect) > 0:
                            pick_continued += 1
                            continue

                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        else:
                            s_matrix = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    # ================================================

                    if len({reactant1, reactant2, product1, product2}) == 4:
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant1, product2))
                        edge_list.append((reactant2, product1))
                        edge_list.append((reactant2, product2))
                        out_nodes_count[reactant1] -= 2
                        out_nodes_count[reactant2] -= 2

                    if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant1, product2))
                        out_nodes_count[reactant1] -= 2

                    if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                        edge_list.append((reactant1, product1))
                        out_nodes_count[reactant1] -= 1

                    if product1 == product2 and \
                            len({reactant1, reactant2, product1}) == len([reactant1, reactant2, product1]):
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant2, product1))
                        out_nodes_count[reactant1] -= 1
                        out_nodes_count[reactant2] -= 1

                    if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant2, product2))
                        out_nodes_count[reactant2] -= 1

                    if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant2, product1))
                        out_nodes_count[reactant2] -= 1

                    if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant1, product2))
                        out_nodes_count[reactant1] -= 1

                    if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant1, product1))
                        out_nodes_count[reactant1] -= 1

                    if reactant1 != reactant2 and len({reactant1, product1, product2}) == 1:
                        edge_list.append((reactant2, product2))
                        out_nodes_count[reactant2] -= 1

                    if reactant1 != reactant2 and len({reactant2, product1, product2}) == 1:
                        edge_list.append((reactant1, product1))
                        out_nodes_count[reactant1] -= 1

                    if product1 != product2 and len({reactant1, reactant2, product1}) == 1:
                        edge_list.append((reactant2, product2))
                        out_nodes_count[reactant2] -= 1

                    if product1 != product2 and len({reactant1, reactant2, product2}) == 1:
                        edge_list.append((reactant1, product1))
                        out_nodes_count[reactant1] -= 1

                    reaction_list.append([rt, [reactant1, reactant2], [product1, product2],
                                          mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                    if connected:
                        c_groups = connected_check([reactant1, reactant2], [product1, product2], mod_species)

            if sum(out_nodes_count) == 0:
                break

    if (bool(out_samples) and bool(in_samples)) or bool(joint_samples):

        pick_continued = 0
        while True:

            if pick_continued == n_species:
                return [None], [out_samples, in_samples, joint_samples]

            if rxn_prob:
                rt = _pick_reaction_type(rxn_prob)
            else:
                rt = _pick_reaction_type()

            mod_num = 0
            if mod_reg:
                mod_num = random.choices([0, 1, 2, 3], mod_reg[0])[0]

            gma_num = 0
            if gma_reg:
                gma_num = random.choices([0, 1, 2, 3], gma_reg[0])[0]

            sc_num = 0
            if sc_reg:
                sc_num = random.choices([0, 1, 2, 3], sc_reg[0])[0]

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIUNI:

                if edge_type == 'generic':

                    if sum(out_nodes_count) < (1 + mod_num + gma_num + sc_num):
                        pick_continued += 1
                        continue

                    if max(in_nodes_count) < (1 + mod_num + gma_num + sc_num):
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product] < (1 + mod_num + gma_num + sc_num):
                        product = random.choices(in_nodes_list, prob_in)[0]

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    s_matrix_temp = None
                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        s_matrix_temp = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        if mod_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < mod_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < mod_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        if gma_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < gma_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < gma_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        if sc_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < sc_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < sc_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if connected:

                        onc = deepcopy(out_nodes_count)
                        inc = deepcopy(in_nodes_count)
                        onc[reactant] -= 1
                        for each in mod_species:
                            onc[each] -= 1
                        inc[product] -= (1 + mod_num + gma_num + sc_num)

                        closed_groups, c_groups_temp = iterative_connected_check([reactant], [product], mod_species,
                                                                                 onc, inc)

                        if closed_groups:
                            pick_continued += 1
                            continue

                        else:
                            c_groups = c_groups_temp

                    if mass_balanced:
                        s_matrix = s_matrix_temp

                    in_nodes_count[product] -= (1 + mod_num + gma_num + sc_num)
                    out_nodes_count[reactant] -= 1
                    for each in mod_species:
                        out_nodes_count[each] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])

                    reaction_list2.append([[reactant], [product]])

                    edge_list.append((reactant, product))

                if edge_type == 'metabolic':

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    s_matrix_temp = None
                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        s_matrix_temp = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if connected:

                        onc = deepcopy(out_nodes_count)
                        inc = deepcopy(in_nodes_count)
                        onc[reactant] -= 1
                        inc[product] -= 1

                        closed_groups, c_groups_temp = iterative_connected_check([reactant], [product], mod_species,
                                                                                 onc, inc)

                        if closed_groups:
                            pick_continued += 1
                            continue

                        else:
                            c_groups = c_groups_temp

                    if mass_balanced:
                        s_matrix = s_matrix_temp

                    out_nodes_count[reactant] -= 1
                    in_nodes_count[product] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                    edge_list.append((reactant, product))

            # -----------------------------------------------------------------

            if rt == TReactionType.BIUNI:

                if edge_type == 'generic':

                    if sum(out_nodes_count) < (2 + mod_num + gma_num + sc_num):
                        pick_continued += 1
                        continue

                    if max(in_nodes_count) < (2 + mod_num + gma_num + sc_num):
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 1
                    sum_out_copy = sum(out_nodes_count_copy)
                    prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                    reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product] < (2 + mod_num + gma_num + sc_num):
                        product = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    s_matrix_temp = None
                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        s_matrix_temp = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        if mod_num > 0:
                            out_nodes_count_copy[reactant2] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < mod_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < mod_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        if gma_num > 0:
                            out_nodes_count_copy[reactant2] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < gma_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < gma_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        if sc_num > 0:
                            out_nodes_count_copy[reactant2] -= 1
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < sc_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < sc_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 1
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if connected:

                        onc = deepcopy(out_nodes_count)
                        inc = deepcopy(in_nodes_count)
                        onc[reactant1] -= 1
                        onc[reactant2] -= 1
                        for each in mod_species:
                            onc[each] -= 1
                        inc[product] -= (2 + mod_num + gma_num + sc_num)

                        closed_groups, c_groups_temp = iterative_connected_check([reactant1, reactant2], [product],
                                                                                 mod_species, onc, inc)

                        if closed_groups:
                            pick_continued += 1
                            continue

                        else:
                            c_groups = c_groups_temp

                    if mass_balanced:
                        s_matrix = s_matrix_temp

                    in_nodes_count[product] -= (2 + mod_num + gma_num + sc_num)
                    out_nodes_count[reactant1] -= 1
                    out_nodes_count[reactant2] -= 1
                    for each in mod_species:
                        out_nodes_count[each] -= 1
                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

                    edge_list.append((reactant1, product))
                    edge_list.append((reactant2, product))

                if edge_type == 'metabolic':

                    if sum(out_nodes_count) < 2:
                        pick_continued += 1
                        continue

                    if max(in_nodes_count) < 2:
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 1
                    sum_out_copy = sum(out_nodes_count_copy)
                    prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                    reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product] < 2:
                        product = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    s_matrix_temp = None
                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product])
                        if not res_result:
                            pick_continued += 1
                            continue
                        s_matrix_temp = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if connected:

                        onc = deepcopy(out_nodes_count)
                        inc = deepcopy(in_nodes_count)

                        if reactant1 != reactant2 and reactant1 != product and reactant2 != product:
                            onc[reactant1] -= 1
                            onc[reactant2] -= 1
                            inc[product] -= 2
                        if reactant1 == reactant2 and reactant1 != product:
                            onc[reactant1] -= 1
                            inc[product] -= 1

                        closed_groups, c_groups_temp = iterative_connected_check([reactant1, reactant2], [product],
                                                                                 mod_species, onc, inc)

                        if closed_groups:
                            pick_continued += 1
                            continue
                        else:
                            c_groups = c_groups_temp

                    if mass_balanced:
                        s_matrix = s_matrix_temp

                    if reactant1 != reactant2 and reactant1 != product and reactant2 != product:
                        edge_list.append((reactant1, product))
                        edge_list.append((reactant2, product))
                        out_nodes_count[reactant1] -= 1
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product] -= 2
                    if reactant1 == reactant2 and reactant1 != product:
                        edge_list.append((reactant1, product))
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product] -= 1
                    if reactant1 != reactant2 and reactant1 == product:
                        edge_list.append((reactant2, 'deg'))
                    if reactant1 != reactant2 and reactant2 == product:
                        edge_list.append((reactant1, 'deg'))
                    if reactant1 == reactant2 and reactant1 == product:
                        edge_list.append((reactant1, 'deg'))

                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIBI:

                if edge_type == 'generic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= (1 + mod_num + gma_num + sc_num):
                        cont = True
                    if sum(1 for each in out_nodes_count if each >= 2) >= (mod_num + gma_num + sc_num - 1) \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True
                    if not cont:
                        pick_continued += 1
                        continue

                    if sum(1 for each in in_nodes_count if each >= (1 + mod_num + gma_num + sc_num)) < 2 \
                            and max(in_nodes_count) < (2 + 2 * mod_num + 2 * gma_num + 2 * sc_num):
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]
                    while out_nodes_count[reactant] < 2:
                        reactant = random.choices(out_nodes_list, prob_out)[0]

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < (1 + mod_num + gma_num + sc_num):
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= (1 + mod_num + gma_num + sc_num)
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < (1 + mod_num + gma_num + sc_num):
                        product2 = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    s_matrix_temp = None
                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        s_matrix_temp = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        if mod_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < mod_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < mod_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        if gma_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < gma_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < gma_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        if sc_num > 0:
                            out_nodes_count_copy = deepcopy(out_nodes_count)
                            out_nodes_count_copy[reactant] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < sc_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < sc_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if connected:

                        onc = deepcopy(out_nodes_count)
                        inc = deepcopy(in_nodes_count)
                        onc[reactant] -= 2
                        inc[product1] -= (1 + mod_num + gma_num + sc_num)
                        inc[product2] -= (1 + mod_num + gma_num + sc_num)
                        for each in mod_species:
                            onc[each] -= 2

                        closed_groups, c_groups_temp = iterative_connected_check([reactant], [product1, product2],
                                                                                 mod_species, onc, inc)

                        if closed_groups:
                            pick_continued += 1
                            continue
                        else:
                            c_groups = c_groups_temp

                    if mass_balanced:
                        s_matrix = s_matrix_temp

                    out_nodes_count[reactant] -= 2
                    in_nodes_count[product1] -= (1 + mod_num + gma_num + sc_num)
                    in_nodes_count[product2] -= (1 + mod_num + gma_num + sc_num)
                    for each in mod_species:
                        out_nodes_count[each] -= 2
                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

                    edge_list.append((reactant, product1))
                    edge_list.append((reactant, product2))

                if edge_type == 'metabolic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= 1:
                        cont = True
                    if sum(1 for each in out_nodes_count if each >= 2) >= -1 \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True
                    if not cont:
                        pick_continued += 1
                        continue

                    if sum(1 for each in in_nodes_count if each >= 1) < 2 \
                            and max(in_nodes_count) < 2:
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]
                    while out_nodes_count[reactant] < 2:
                        reactant = random.choices(out_nodes_list, prob_out)[0]

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < 1:
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= 1
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < 1:
                        product2 = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    s_matrix_temp = None
                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        s_matrix_temp = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if connected:

                        onc = deepcopy(out_nodes_count)
                        inc = deepcopy(in_nodes_count)

                        if reactant != product1 and reactant != product2 and product1 != product2:
                            onc[reactant] -= 2
                            inc[product1] -= 1
                            inc[product2] -= 1
                        if reactant != product1 and product1 == product2:
                            onc[reactant] -= 1
                            inc[product1] -= 1

                        closed_groups, c_groups_temp = iterative_connected_check([reactant], [product1, product2],
                                                                                 mod_species, onc, inc)

                        if closed_groups:
                            pick_continued += 1
                            continue
                        else:
                            c_groups = c_groups_temp

                    if mass_balanced:
                        s_matrix = s_matrix_temp

                    if reactant != product1 and reactant != product2 and product1 != product2:
                        edge_list.append((reactant, product1))
                        edge_list.append((reactant, product2))
                        out_nodes_count[reactant] -= 2
                        in_nodes_count[product1] -= 1
                        in_nodes_count[product2] -= 1
                    if reactant != product1 and product1 == product2:
                        edge_list.append((reactant, product1))
                        out_nodes_count[reactant] -= 1
                        in_nodes_count[product1] -= 1
                    if reactant == product1 and product1 != product2:
                        edge_list.append(('syn', product2))
                    if reactant == product2 and product1 != product2:
                        edge_list.append(('syn', product1))
                    if reactant == product1 and product1 == product2:
                        edge_list.append(('syn', reactant))

                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

            # -----------------------------------------------------------------

            if rt == TReactionType.BIBI:

                if edge_type == 'generic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= (2 + mod_num + gma_num + sc_num):
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= mod_num + gma_num + sc_num \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= (mod_num + gma_num + sc_num - 2) \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 2:
                        cont = True

                    if not cont:
                        pick_continued += 1
                        continue

                    if sum(1 for each in in_nodes_count if each >= (2 + mod_num + gma_num + sc_num)) < 2 \
                            and max(in_nodes_count) < (4 + 2 * mod_num + 2 * gma_num + 2 * sc_num):
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]
                    while out_nodes_count[reactant1] < 2:
                        reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 2
                    sum_out_copy = sum(out_nodes_count_copy)
                    prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                    reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]
                    while out_nodes_count_copy[reactant2] < 2:
                        reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < (2 + mod_num + gma_num + sc_num):
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= (2 + mod_num + gma_num + sc_num)
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < (2 + mod_num + gma_num + sc_num):
                        product2 = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant1, reactant2], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if not unaffected_nodes:
                        intersect = {reactant1, reactant2}.intersection({product1, product2})
                        if len(intersect) > 0:
                            pick_continued += 1
                            continue

                    s_matrix_temp = None
                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        s_matrix_temp = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        if mod_num > 0:
                            out_nodes_count_copy[reactant2] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < mod_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < mod_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        if gma_num > 0:
                            out_nodes_count_copy[reactant2] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < gma_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < gma_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        if sc_num > 0:
                            out_nodes_count_copy[reactant2] -= 2
                            sum_out_copy = sum(out_nodes_count_copy)
                            prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                            while len(mod_species) < sc_num:
                                new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                while out_nodes_count_copy[new_mod] < 2:
                                    new_mod = random.choices(out_nodes_list, prob_out_copy)[0]
                                if new_mod not in mod_species:
                                    mod_species.append(new_mod)
                                    if len(mod_species) < sc_num:
                                        out_nodes_count_copy[mod_species[-1]] -= 2
                                        sum_out_copy = sum(out_nodes_count_copy)
                                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]

                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if connected:

                        onc = deepcopy(out_nodes_count)
                        inc = deepcopy(in_nodes_count)
                        onc[reactant1] -= 2
                        onc[reactant2] -= 2
                        inc[product1] -= (2 + mod_num + gma_num + sc_num)
                        inc[product2] -= (2 + mod_num + gma_num + sc_num)
                        for each in mod_species:
                            onc[each] -= 2

                        closed_groups, c_groups_temp = iterative_connected_check([reactant1, reactant2],
                                                                                 [product1, product2], mod_species,
                                                                                 onc, inc)

                        if closed_groups:
                            pick_continued += 1
                            continue
                        else:
                            c_groups = c_groups_temp

                    if mass_balanced:
                        s_matrix = s_matrix_temp

                    out_nodes_count[reactant1] -= 2
                    out_nodes_count[reactant2] -= 2
                    in_nodes_count[product1] -= (2 + mod_num + gma_num + sc_num)
                    in_nodes_count[product2] -= (2 + mod_num + gma_num + sc_num)
                    for each in mod_species:
                        out_nodes_count[each] -= 2
                    reaction_list.append(
                        [rt, [reactant1, reactant2], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                    edge_list.append((reactant1, product1))
                    edge_list.append((reactant2, product1))
                    edge_list.append((reactant1, product2))
                    edge_list.append((reactant2, product2))

                if edge_type == 'metabolic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= 2:
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= 0 \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= -2 \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 2:
                        cont = True

                    if not cont:
                        pick_continued += 1
                        continue

                    if sum(1 for each in in_nodes_count if each >= 2) < 2 \
                            and max(in_nodes_count) < 4:
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]
                    while out_nodes_count[reactant1] < 2:
                        reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 2
                    sum_out_copy = sum(out_nodes_count_copy)
                    prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                    reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]
                    while out_nodes_count_copy[reactant2] < 2:
                        reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < 2:
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= 2
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < 2:
                        product2 = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant1, reactant2], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if [[reactant2, reactant1], [product2, product1]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if not unaffected_nodes:
                        intersect = {reactant1, reactant2}.intersection({product1, product2})
                        if len(intersect) > 0:
                            pick_continued += 1
                            continue

                    s_matrix_temp = None
                    if mass_balanced:
                        res_result, s_mat_c = consistency_check([reactant1, reactant2], [product1, product2])
                        if not res_result:
                            pick_continued += 1
                            continue
                        s_matrix_temp = s_mat_c

                    mod_species = []
                    reg_signs = []
                    reg_type = []

                    if mod_reg:
                        mod_species = random.sample(nodes_list, mod_num)
                        reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                        reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if gma_reg:
                        mod_species = random.sample(nodes_list, gma_num)
                        reg_signs = [random.choices([1, -1], [gma_reg[1], 1 - gma_reg[1]])[0] for _ in mod_species]
                        reg_type = ['gma' for _ in mod_species]

                    if sc_reg:
                        mod_species = random.sample(nodes_list, sc_num)
                        reg_signs = [random.choices([1, -1], [sc_reg[1], 1 - sc_reg[1]])[0] for _ in mod_species]
                        reg_type = ['sc' for _ in mod_species]

                    if connected:

                        onc = deepcopy(out_nodes_count)
                        inc = deepcopy(in_nodes_count)

                        if len({reactant1, reactant2, product1, product2}) == 4:
                            onc[reactant1] -= 2
                            onc[reactant2] -= 2
                            inc[product1] -= 2
                            inc[product2] -= 2

                        if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                            onc[reactant1] -= 2
                            inc[product1] -= 1
                            inc[product2] -= 1

                        if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                            onc[reactant1] -= 1
                            inc[product1] -= 1

                        if product1 == product2 and \
                                len({reactant1, reactant2, product1}) == len([reactant1, reactant2, product1]):
                            onc[reactant1] -= 1
                            onc[reactant2] -= 1
                            inc[product1] -= 2

                        if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                            onc[reactant2] -= 1
                            inc[product2] -= 1

                        if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                            onc[reactant2] -= 1
                            inc[product1] -= 1

                        if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                            onc[reactant1] -= 1
                            inc[product2] -= 1

                        if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                            onc[reactant1] -= 1
                            inc[product1] -= 1

                        if reactant1 != reactant2 and len({reactant1, product1, product2}) == 1:
                            onc[reactant2] -= 1
                            inc[product2] -= 1

                        if reactant1 != reactant2 and len({reactant2, product1, product2}) == 1:
                            onc[reactant1] -= 1
                            inc[product1] -= 1

                        if product1 != product2 and len({reactant1, reactant2, product1}) == 1:
                            onc[reactant2] -= 1
                            inc[product2] -= 1

                        if product1 != product2 and len({reactant1, reactant2, product2}) == 1:
                            onc[reactant1] -= 1
                            inc[product1] -= 1

                        closed_groups, c_groups_temp = iterative_connected_check([reactant1, reactant2],
                                                                                 [product1, product2], mod_species,
                                                                                 onc, inc)

                        if closed_groups:
                            pick_continued += 1
                            continue
                        else:
                            c_groups = c_groups_temp

                    if mass_balanced:
                        s_matrix = s_matrix_temp

                    if len({reactant1, reactant2, product1, product2}) == 4:
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant1, product2))
                        edge_list.append((reactant2, product1))
                        edge_list.append((reactant2, product2))
                        out_nodes_count[reactant1] -= 2
                        out_nodes_count[reactant2] -= 2
                        in_nodes_count[product1] -= 2
                        in_nodes_count[product2] -= 2

                    if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant1, product2))
                        out_nodes_count[reactant1] -= 2
                        in_nodes_count[product1] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                        edge_list.append((reactant1, product1))
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product1] -= 1

                    if product1 == product2 and \
                            len({reactant1, reactant2, product1}) == len([reactant1, reactant2, product1]):
                        edge_list.append((reactant1, product1))
                        edge_list.append((reactant2, product1))
                        out_nodes_count[reactant1] -= 1
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product1] -= 2

                    if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant2, product2))
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant2, product1))
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product1] -= 1

                    if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant1, product2))
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        edge_list.append((reactant1, product1))
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product1] -= 1

                    if reactant1 != reactant2 and len({reactant1, product1, product2}) == 1:
                        edge_list.append((reactant2, product2))
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant1 != reactant2 and len({reactant2, product1, product2}) == 1:
                        edge_list.append((reactant1, product1))
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product1] -= 1

                    if product1 != product2 and len({reactant1, reactant2, product1}) == 1:
                        edge_list.append((reactant2, product2))
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product2] -= 1

                    if product1 != product2 and len({reactant1, reactant2, product2}) == 1:
                        edge_list.append((reactant1, product1))
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product1] -= 1

                    reaction_list.append([rt, [reactant1, reactant2], [product1, product2],
                                          mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

            if sum(in_nodes_count) == 0:
                break

    if len(c_groups) > 1:
        return [None], [out_samples, in_samples, joint_samples]
    else:
        reaction_list.insert(0, n_species)
        return reaction_list, edge_list


# Includes boundary and floating species
# Returns a list:
# [New Stoichiometry matrix, list of floatingIds, list of boundaryIds]
# On entry, reaction_list has the structure:
# reaction_list = [numSpecies, reaction, reaction, ....]
# reaction = [reactionType, [list of reactants], [list of products], rateConstant]

# todo: allow this to read networks from network directory
def get_full_stoichiometry_matrix(reaction_list):
    n_species = reaction_list[0]
    reaction_list_copy = deepcopy(reaction_list)

    # Remove the first entry in the list which is the number of species
    reaction_list_copy.pop(0)
    st = np.zeros((n_species, len(reaction_list_copy)))

    for index, r in enumerate(reaction_list_copy):
        if r[0] == TReactionType.UNIUNI:
            # UniUni
            reactant = reaction_list_copy[index][1][0]
            st[reactant, index] = -1
            product = reaction_list_copy[index][2][0]
            st[product, index] = 1

        if r[0] == TReactionType.BIUNI:
            # BiUni
            reactant1 = reaction_list_copy[index][1][0]
            st[reactant1, index] += -1
            reactant2 = reaction_list_copy[index][1][1]
            st[reactant2, index] += -1
            product = reaction_list_copy[index][2][0]
            st[product, index] = 1

        if r[0] == TReactionType.UNIBI:
            # UniBi
            reactant1 = reaction_list_copy[index][1][0]
            st[reactant1, index] = -1
            product1 = reaction_list_copy[index][2][0]
            st[product1, index] += 1
            product2 = reaction_list_copy[index][2][1]
            st[product2, index] += 1

        if r[0] == TReactionType.BIBI:
            # BiBi
            reactant1 = reaction_list_copy[index][1][0]
            st[reactant1, index] += -1
            reactant2 = reaction_list_copy[index][1][1]
            st[reactant2, index] += -1
            product1 = reaction_list_copy[index][2][0]
            st[product1, index] += 1
            product2 = reaction_list_copy[index][2][1]
            st[product2, index] += 1

    return st


# Removes boundary or orphan species from stoichiometry matrix
def remove_boundary_nodes(st):
    dims = st.shape

    n_species = dims[0]
    n_reactions = dims[1]

    species_ids = np.arange(n_species)
    indexes = []
    orphan_species = []
    count_boundary_species = 0
    for r in range(n_species):
        # Scan across the columns, count + and - coefficients
        plus_coeff = 0
        minus_coeff = 0
        for c in range(n_reactions):
            if st[r, c] < 0:
                minus_coeff = minus_coeff + 1
            if st[r, c] > 0:
                plus_coeff = plus_coeff + 1
        if plus_coeff == 0 and minus_coeff == 0:
            # No reaction attached to this species
            orphan_species.append(r)
        if plus_coeff == 0 and minus_coeff != 0:
            # Species is a source
            indexes.append(r)
            count_boundary_species = count_boundary_species + 1
        if minus_coeff == 0 and plus_coeff != 0:
            # Species is a sink
            indexes.append(r)
            count_boundary_species = count_boundary_species + 1

    floating_ids = np.delete(species_ids, indexes + orphan_species, axis=0)

    boundary_ids = indexes
    return [np.delete(st, indexes + orphan_species, axis=0), floating_ids, boundary_ids]


def get_antimony_script(reaction_list, ic_params, kinetics, rev_prob, add_enzyme, constants, source, sink):

    n_species = reaction_list[0]
    reaction_list_copy = deepcopy(reaction_list)

    # Remove the first entry in the list which is the number of species
    reaction_list_copy.pop(0)
    st = np.zeros((n_species, len(reaction_list_copy)))
    # mass_violators = []

    for index, r in enumerate(reaction_list_copy):
        if r[0] == TReactionType.UNIUNI:
            # UniUni
            reactant = reaction_list_copy[index][1][0]
            st[reactant, index] = -1
            product = reaction_list_copy[index][2][0]
            st[product, index] = 1
            # if reactant == product:
            #     mass_violators.append(reactant)

        if r[0] == TReactionType.BIUNI:
            # BiUni
            reactant1 = reaction_list_copy[index][1][0]
            st[reactant1, index] += -1
            reactant2 = reaction_list_copy[index][1][1]
            st[reactant2, index] += -1
            product = reaction_list_copy[index][2][0]
            st[product, index] += 1
            # if reactant1 == product:
            #     mass_violators.append(reactant1)
            # if reactant2 == product:
            #     mass_violators.append(reactant2)

        if r[0] == TReactionType.UNIBI:
            # UniBi
            reactant = reaction_list_copy[index][1][0]
            st[reactant, index] += -1
            product1 = reaction_list_copy[index][2][0]
            st[product1, index] += 1
            product2 = reaction_list_copy[index][2][1]
            st[product2, index] += 1
            # if reactant == product1:
            #     mass_violators.append(reactant)
            # if reactant == product2:
            #     mass_violators.append(reactant)

        if r[0] == TReactionType.BIBI:
            # BiBi
            reactant1 = reaction_list_copy[index][1][0]
            st[reactant1, index] += -1
            reactant2 = reaction_list_copy[index][1][1]
            st[reactant2, index] += -1
            product1 = reaction_list_copy[index][2][0]
            st[product1, index] += 1
            product2 = reaction_list_copy[index][2][1]
            st[product2, index] += 1
            # if reactant1 == product1:
            #     mass_violators.append(reactant1)
            # if reactant1 == product2:
            #     mass_violators.append(reactant1)
            # if reactant2 == product1:
            #     mass_violators.append(reactant2)
            # if reactant2 == product2:
            #     mass_violators.append(reactant2)

    dims = st.shape

    n_reactions = dims[1]

    species_ids = np.arange(n_species)
    indexes = []
    orphan_species = []

    original_source_nodes = []
    original_sink_nodes = []
    count_boundary_species = 0

    for r in range(n_species):
        # Scan across the columns, count + and - coefficients
        plus_coeff = 0
        minus_coeff = 0
        for c in range(n_reactions):
            if st[r, c] < 0:
                minus_coeff = minus_coeff + 1
            if st[r, c] > 0:
                plus_coeff = plus_coeff + 1
        if plus_coeff == 0 and minus_coeff == 0:  # and r not in mass_violators:
            # No reaction attached to this species
            orphan_species.append(r)
            original_source_nodes.append(r)
            original_sink_nodes.append(r)
        if plus_coeff == 0 and minus_coeff != 0:
            # Species is a source

            original_source_nodes.append(r)

            indexes.append(r)
            count_boundary_species = count_boundary_species + 1
        if minus_coeff == 0 and plus_coeff != 0:
            # Species is a sink

            original_sink_nodes.append(r)

            indexes.append(r)
            count_boundary_species = count_boundary_species + 1

    floating_ids = np.delete(species_ids, indexes + orphan_species, axis=0)
    boundary_ids = indexes + orphan_species

    if 'modular' in kinetics[0] or kinetics[0] == 'gma' or kinetics[0] == 'saturating_cooperative':
        for each in reaction_list_copy:
            for item in each[3]:
                if item not in boundary_ids and item not in floating_ids:
                    boundary_ids.append(item)

    source_nodes = None
    sink_nodes = None

    if constants == False:
        if len(original_source_nodes) >= source[0]:
            source_nodes = original_source_nodes
        else:
            source_nodes = original_source_nodes + random.sample(list(floating_ids),
                                                                 source[0] - len(original_source_nodes))

        if len(original_sink_nodes) >= sink[0]:
            sink_nodes = original_sink_nodes
        else:
            sink_nodes = original_sink_nodes + random.sample(list(floating_ids), sink[0] - len(original_sink_nodes))

        source_nodes.sort()
        sink_nodes.sort()

        boundary_ids = []
        floating_ids = [i for i in range(n_species)]

    enzyme = ''
    enzyme_end = ''

    # Remove the first element which is the n_species
    reaction_list_copy = deepcopy(reaction_list)
    reaction_list_copy.pop(0)

    rxn_str = ''
    param_str = ''
    ic_str = ''

    if constants == True:
        if len(original_source_nodes) >= source[0]:
            source_nodes = original_source_nodes
        else:
            source_nodes = original_source_nodes + random.sample(list(floating_ids),
                                                                 source[0] - len(original_source_nodes))

        if len(original_sink_nodes) >= sink[0]:
            sink_nodes = original_sink_nodes
        else:
            sink_nodes = original_sink_nodes + random.sample(list(floating_ids), sink[0] - len(original_sink_nodes))

        source_nodes.sort()
        sink_nodes.sort()

        boundary_ids = deepcopy(source_nodes)
        boundary_ids.extend(deepcopy(sink_nodes))
        boundary_ids.sort()
        floating_ids = [i for i in range(n_species)]

        if len(floating_ids) > 0:
            rxn_str += 'var ' + 'S' + str(floating_ids[0])
            for index in floating_ids[1:]:
                rxn_str += ', ' + 'S' + str(index)
            rxn_str += '\n'

        if len(boundary_ids) > 0:
            rxn_str += 'ext ' + 'B' + str(boundary_ids[0])
            for index in boundary_ids[1:]:
                rxn_str += ', ' + 'B' + str(index)
            rxn_str += '\n'
        rxn_str += '\n'

    if constants == None:
        if len(floating_ids) > 0:
            rxn_str += 'var ' + 'S' + str(floating_ids[0])
            for index in floating_ids[1:]:
                rxn_str += ', ' + 'S' + str(index)
            rxn_str += '\n'

        if 'modular' in kinetics[0]:
            for each in reaction_list_copy:
                for item in each[3]:
                    if item not in boundary_ids and item not in floating_ids:
                        boundary_ids.append(item)

        if len(boundary_ids) > 0:
            rxn_str += 'ext ' + 'S' + str(boundary_ids[0])
            for index in boundary_ids[1:]:
                rxn_str += ', ' + 'S' + str(index)
            rxn_str += '\n'
        rxn_str += '\n'


    def reversibility():

        rev1 = False
        if rev_prob == 0:
            pass
        elif rev_prob == 1:
            rev1 = True
        else:
            rev1 = random.choices([True, False], [rev_prob, 1 - rev_prob])[0]

        return rev1

    reaction_index = None
    
    # ================================================================================

    if kinetics[0] == 'saturating_cooperative':

        v = []
        k = []
        n = []
        nr = []
        nr_sign = []

        for reaction_index, r in enumerate(reaction_list_copy):

            if add_enzyme:
                enzyme = 'E' + str(reaction_index) + '*('
                enzyme_end = ')'

            rxn_str += 'J' + str(reaction_index) + ': '

            if r[0] == TReactionType.UNIUNI:
                # UniUni
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + ' * S' + str(r[1][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][0])
                    
                    v.append('v' + str(reaction_index))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][0]))
                    
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^nr_' + str(reaction_index) + '_' + str(reg)
                        nr.append('nr_' + str(reaction_index) + '_' + str(reg))
                        nr_sign.append(r[4][i])
                    
                    rxn_str += '/((k' + str(reaction_index) + '_' + str(r[1][0]) + ' + ' + 'S' + str(r[1][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][0]) + ')'
                    k.append('k' + str(reaction_index) + '_' + str(r[1][0]))
                    
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * (k' + str(reaction_index) + '_' + str(reg) + ' + ' + 'S' + str(reg) \
                                   + '^nr_' + str(reaction_index) + '_' + str(reg) + ')'
                        k.append('k' + str(reaction_index) + '_' + str(reg))
                    
                    rxn_str += ')'

                else:
                    rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + ' * (S' + str(r[1][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' - S' + str(r[2][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[2][0]) + ')'
                    
                    v.append('v' + str(reaction_index))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][0]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[2][0]))
                    
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^nr_' + str(reaction_index) + '_' + str(reg)
                        nr.append('nr_' + str(reaction_index) + '_' + str(reg))
                        nr_sign.append(r[4][i])
                    
                    rxn_str += '/((k' + str(reaction_index) + '_' + str(r[1][0]) + ' + ' + 'S' + str(r[1][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + ' * ' + '(k' \
                               + str(reaction_index) + '_' + str(r[2][0]) + ' + ' + 'S' + str(r[2][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[2][0]) + ')'
                    k.append('k' + str(reaction_index) + '_' + str(r[1][0]))
                    k.append('k' + str(reaction_index) + '_' + str(r[2][0]))
                    
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * (k' + str(reaction_index) + '_' + str(reg) + ' + ' + 'S' + str(reg) \
                                   + '^nr_' + str(reaction_index) + '_' + str(reg) + ')'
                        k.append('k' + str(reaction_index) + '_' + str(reg))
                        
                    rxn_str += ')'

            if r[0] == TReactionType.BIUNI:
                # BiUni
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + ' * S' + str(r[1][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' * S' + str(r[1][1]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][1])

                    v.append('v' + str(reaction_index))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][0]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][1]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^nr_' + str(reaction_index) + '_' + str(reg)
                        nr.append('nr_' + str(reaction_index) + '_' + str(reg))
                        nr_sign.append(r[4][i])

                    rxn_str += '/((k' + str(reaction_index) + '_' + str(r[1][0]) + ' + ' + 'S' + str(r[1][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + ' * ' + '(k' \
                               + str(reaction_index) + '_' + str(r[1][1]) + ' + ' + 'S' + str(r[1][1]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][1]) + ')'
                    k.append('k' + str(reaction_index) + '_' + str(r[1][0]))
                    k.append('k' + str(reaction_index) + '_' + str(r[1][1]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * (k' + str(reaction_index) + '_' + str(reg) + ' + ' + 'S' + str(reg) \
                                   + '^nr_' + str(reaction_index) + '_' + str(reg) + ')'
                        k.append('k' + str(reaction_index) + '_' + str(reg))

                    rxn_str += ')'

                else:
                    rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + ' * (S' + str(r[1][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' * S' + str(r[1][1]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][1]) + ' - S' + str(r[2][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[2][0]) + ')'

                    v.append('v' + str(reaction_index))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][0]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][1]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[2][0]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^nr_' + str(reaction_index) + '_' + str(reg)
                        nr.append('nr_' + str(reaction_index) + '_' + str(reg))
                        nr_sign.append(r[4][i])

                    rxn_str += '/((k' + str(reaction_index) + '_' + str(r[1][0]) + ' + ' + 'S' + str(r[1][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + ' * ' + '(k' \
                               + str(reaction_index) + '_' + str(r[1][1]) + ' + ' + 'S' + str(r[1][1]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][1]) + ')' + ' * ' + '(k' \
                               + str(reaction_index) + '_' + str(r[2][0]) + ' + ' + 'S' + str(r[2][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[2][0]) + ')'
                    k.append('k' + str(reaction_index) + '_' + str(r[1][0]))
                    k.append('k' + str(reaction_index) + '_' + str(r[1][1]))
                    k.append('k' + str(reaction_index) + '_' + str(r[2][0]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * (k' + str(reaction_index) + '_' + str(reg) + ' + ' + 'S' + str(reg) \
                                   + '^nr_' + str(reaction_index) + '_' + str(reg) + ')'
                        k.append('k' + str(reaction_index) + '_' + str(reg))

                    rxn_str += ')'
            # -------------------------------------------------------------------------

            if r[0] == TReactionType.UNIBI:
                # UniBi
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[2][1])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + ' * S' + str(r[1][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][0])

                    v.append('v' + str(reaction_index))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][0]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^nr_' + str(reaction_index) + '_' + str(reg)
                        nr.append('nr_' + str(reaction_index) + '_' + str(reg))
                        nr_sign.append(r[4][i])

                    rxn_str += '/((k' + str(reaction_index) + '_' + str(r[1][0]) + ' + ' + 'S' + str(r[1][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][0]) + ')'
                    k.append('k' + str(reaction_index) + '_' + str(r[1][0]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * (k' + str(reaction_index) + '_' + str(reg) + ' + ' + 'S' + str(reg) \
                                   + '^nr_' + str(reaction_index) + '_' + str(reg) + ')'
                        k.append('k' + str(reaction_index) + '_' + str(reg))

                    rxn_str += ')'

                else:
                    rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + ' * (S' + str(r[1][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' - S' + str(r[2][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[2][0]) + ' * S' + str(r[2][1]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[2][1]) + ')'

                    v.append('v' + str(reaction_index))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][0]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[2][0]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[2][1]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^nr_' + str(reaction_index) + '_' + str(reg)
                        nr.append('nr_' + str(reaction_index) + '_' + str(reg))
                        nr_sign.append(r[4][i])

                    rxn_str += '/((k' + str(reaction_index) + '_' + str(r[1][0]) + ' + ' + 'S' + str(r[1][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + ' * ' + '(k' \
                               + str(reaction_index) + '_' + str(r[2][0]) + ' + ' + 'S' + str(r[2][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[2][0]) + ' * ' + '(k' \
                               + str(reaction_index) + '_' + str(r[2][1]) + ' + ' + 'S' + str(r[2][1]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[2][1]) + ')'
                    k.append('k' + str(reaction_index) + '_' + str(r[1][0]))
                    k.append('k' + str(reaction_index) + '_' + str(r[2][0]))
                    k.append('k' + str(reaction_index) + '_' + str(r[2][1]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * (k' + str(reaction_index) + '_' + str(reg) + ' + ' + 'S' + str(reg) \
                                   + '^nr_' + str(reaction_index) + '_' + str(reg) + ')'
                        k.append('k' + str(reaction_index) + '_' + str(reg))

                    rxn_str += ')'

            # -------------------------------------------------------------------------

            if r[0] == TReactionType.BIBI:
                # BiBi
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[2][1])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + ' * S' + str(r[1][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' * S' + str(r[1][1]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][1])

                    v.append('v' + str(reaction_index))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][0]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][1]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^nr_' + str(reaction_index) + '_' + str(reg)
                        nr.append('nr_' + str(reaction_index) + '_' + str(reg))
                        nr_sign.append(r[4][i])

                    rxn_str += '/((k' + str(reaction_index) + '_' + str(r[1][0]) + ' + ' + 'S' + str(r[1][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + ' * ' + '(k' \
                               + str(reaction_index) + '_' + str(r[1][1]) + ' + ' + 'S' + str(r[1][1]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][1]) + ')'
                    k.append('k' + str(reaction_index) + '_' + str(r[1][0]))
                    k.append('k' + str(reaction_index) + '_' + str(r[1][1]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * (k' + str(reaction_index) + '_' + str(reg) + ' + ' + 'S' + str(reg) \
                                   + '^nr_' + str(reaction_index) + '_' + str(reg) + ')'
                        k.append('k' + str(reaction_index) + '_' + str(reg))

                    rxn_str += ')'

                else:
                    rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + ' * (S' + str(r[1][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' * S' + str(r[1][1]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[1][1]) + ' - S' + str(r[2][0]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[2][0]) + ' * S' + str(r[2][1]) + '^n_' + \
                               str(reaction_index) + '_' + str(r[2][1]) + ')'

                    v.append('v' + str(reaction_index))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][0]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[1][1]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[2][0]))
                    n.append('n_' + str(reaction_index) + '_' + str(r[2][1]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^nr_' + str(reaction_index) + '_' + str(reg)
                        nr.append('nr_' + str(reaction_index) + '_' + str(reg))
                        nr_sign.append(r[4][i])

                    rxn_str += '/((k' + str(reaction_index) + '_' + str(r[1][0]) + ' + ' + 'S' + str(r[1][0]) \
                               + '^n_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + ' * ' + '(k' \
                               + str(reaction_index) + '_' + str(r[1][1]) + ' + ' + 'S' + str(r[1][1]) + '^n_' \
                               + str(reaction_index) + '_' + str(r[1][1]) + ')' + ' * ' + '(k' + str(reaction_index) \
                               + '_' + str(r[2][0]) + ' + ' + 'S' + str(r[2][0]) + '^n_' + str(reaction_index) + '_' \
                               + str(r[2][0]) + ')' + ' * ' + '(k' + str(reaction_index) + '_' + str(r[2][1]) + ' + ' \
                               + 'S' + str(r[2][1]) + '^n_' + str(reaction_index) + '_' + str(r[2][1]) + ')'

                    k.append('k' + str(reaction_index) + '_' + str(r[1][0]))
                    k.append('k' + str(reaction_index) + '_' + str(r[1][1]))
                    k.append('k' + str(reaction_index) + '_' + str(r[2][0]))
                    k.append('k' + str(reaction_index) + '_' + str(r[2][1]))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * (k' + str(reaction_index) + '_' + str(reg) + ' + ' + 'S' + str(reg) \
                                   + '^nr_' + str(reaction_index) + '_' + str(reg) + ')'
                        k.append('k' + str(reaction_index) + '_' + str(reg))

                    rxn_str += ')'

            # -------------------------------------------------------------------------

            rxn_str += '\n'
        rxn_str += '\n'

        for each in n:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('n')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('n')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('n')][0],
                                        scale=kinetics[3][kinetics[2].index('n')][1]
                                        - kinetics[3][kinetics[2].index('n')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('n')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('n')][0],
                                           kinetics[3][kinetics[2].index('n')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('n')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('n')][0],
                                         scale=kinetics[3][kinetics[2].index('n')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('n')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('n')][0],
                                        s=kinetics[3][kinetics[2].index('n')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('n')][0],
                                    scale=kinetics[3][kinetics[2].index('n')][1]
                                    - kinetics[3][kinetics[2].index('n')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('n')][0],
                                       kinetics[3][kinetics[2].index('n')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('n')][0],
                                     scale=kinetics[3][kinetics[2].index('n')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('n')][0],
                                    s=kinetics[3][kinetics[2].index('n')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if n:
            param_str += '\n'

        for i, each in enumerate(nr):

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('nr')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('nr')] == 'uniform':

                    if kinetics[3][kinetics[2].index('nr')][0] < 0:

                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('nr')][0],
                                            scale=kinetics[3][kinetics[2].index('nr')][1]
                                            - kinetics[3][kinetics[2].index('nr')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    else:
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('nr')][0],
                                            scale=kinetics[3][kinetics[2].index('nr')][1]
                                            - kinetics[3][kinetics[2].index('nr')][0])
                        if nr_sign[i] < 0:
                            param_str += each + ' = -' + str(const) + '\n'
                        else:
                            param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('nr')] == 'loguniform':

                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('nr')][0],
                                        scale=kinetics[3][kinetics[2].index('nr')][1]
                                        - kinetics[3][kinetics[2].index('nr')][0])
                    if nr_sign[i] < 0:
                        param_str += each + ' = -' + str(const) + '\n'
                    else:
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('nr')] == 'normal':

                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('nr')][0],
                                     scale=kinetics[3][kinetics[2].index('nr')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('nr')] == 'lognormal':

                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('nr')][0],
                                        s=kinetics[3][kinetics[2].index('nr')][1])
                    if nr_sign[i] < 0:
                        param_str += each + ' = -' + str(const) + '\n'
                    else:
                        param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':

                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':

                if kinetics[3][kinetics[2].index('nr')][0] < 0:

                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('nr')][0],
                                        scale=kinetics[3][kinetics[2].index('nr')][1]
                                        - kinetics[3][kinetics[2].index('nr')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                else:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('nr')][0],
                                        scale=kinetics[3][kinetics[2].index('nr')][1]
                                        - kinetics[3][kinetics[2].index('nr')][0])
                    if nr_sign[i] < 0:
                        param_str += each + ' = -' + str(const) + '\n'
                    else:
                        param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':

                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('nr')][0],
                                    scale=kinetics[3][kinetics[2].index('nr')][1]
                                    - kinetics[3][kinetics[2].index('nr')][0])
                if nr_sign[i] < 0:
                    param_str += each + ' = -' + str(const) + '\n'
                else:
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':

                const = norm.rvs(loc=kinetics[3][kinetics[2].index('nr')][0],
                                 scale=kinetics[3][kinetics[2].index('nr')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'lognormal':

                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('nr')][0],
                                    s=kinetics[3][kinetics[2].index('nr')][1])
                if nr_sign[i] < 0:
                    param_str += each + ' = -' + str(const) + '\n'
                else:
                    param_str += each + ' = ' + str(const) + '\n'

        if nr:
            param_str += '\n'

        for each in v:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('v')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('v')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                        scale=kinetics[3][kinetics[2].index('v')][1]
                                        - kinetics[3][kinetics[2].index('v')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('v')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0],
                                           kinetics[3][kinetics[2].index('v')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('v')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                         scale=kinetics[3][kinetics[2].index('v')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('v')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0],
                                        s=kinetics[3][kinetics[2].index('v')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                    scale=kinetics[3][kinetics[2].index('v')][1]
                                    - kinetics[3][kinetics[2].index('v')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0],
                                       kinetics[3][kinetics[2].index('v')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                     scale=kinetics[3][kinetics[2].index('v')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0],
                                    s=kinetics[3][kinetics[2].index('v')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if v:
            param_str += '\n'

        for each in k:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('k')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('k')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                        scale=kinetics[3][kinetics[2].index('k')][1]
                                        - kinetics[3][kinetics[2].index('k')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('k')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('k')][0],
                                           kinetics[3][kinetics[2].index('k')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('k')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                         scale=kinetics[3][kinetics[2].index('k')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('k')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('k')][0],
                                        s=kinetics[3][kinetics[2].index('k')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                    scale=kinetics[3][kinetics[2].index('k')][1]
                                    - kinetics[3][kinetics[2].index('k')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('k')][0],
                                       kinetics[3][kinetics[2].index('k')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                     scale=kinetics[3][kinetics[2].index('k')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('k')][0],
                                    s=kinetics[3][kinetics[2].index('k')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if k:
            param_str += '\n'

    if kinetics[0] == 'gma':

        kf = []
        kr = []
        kc = []
        ko = []
        kor = []
        kor_sign = []

        for reaction_index, r in enumerate(reaction_list_copy):

            if add_enzyme:
                enzyme = 'E' + str(reaction_index) + '*('
                enzyme_end = ')'

            rxn_str += 'J' + str(reaction_index) + ': '

            if r[0] == TReactionType.UNIUNI:
                # UniUni
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme + 'kc' + str(reaction_index) + ' * S' + str(r[1][0]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][0])
                    kc.append('kc' + str(reaction_index))
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^kor_' + str(reaction_index) + '_' + str(reg)
                        kor.append('kor_' + str(reaction_index) + '_' + str(reg))
                        kor_sign.append(r[4][i])
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][0]))
                else:
                    rxn_str += '; ' + enzyme + '(kf' + str(reaction_index) + ' * S' + str(r[1][0]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' - ' + 'kr' + str(reaction_index) + ' * S' \
                               + str(r[2][0]) + '^ko_' + str(reaction_index) + '_' + str(r[2][0]) + ')'
                    kf.append('kf' + str(reaction_index))
                    kr.append('kr' + str(reaction_index))
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^kor_' + str(reaction_index) + '_' + str(reg)
                        kor.append('kor_' + str(reaction_index) + '_' + str(reg))
                        kor_sign.append(r[4][i])
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][0]))
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[2][0]))

            # -------------------------------------------------------------------------

            if r[0] == TReactionType.BIUNI:
                # BiUni
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme + 'kc' + str(reaction_index) + ' * S' + str(r[1][0]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' * S' + str(r[1][1]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][1])
                    kc.append('kc' + str(reaction_index))

                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^kor_' + str(reaction_index) + '_' + str(reg)
                        kor.append('kor_' + str(reaction_index) + '_' + str(reg))
                        kor_sign.append(r[4][i])

                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][0]))
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][1]))

                else:
                    rxn_str += '; ' + enzyme + '(kf' + str(reaction_index) + ' * S' + str(r[1][0]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' * S' + str(r[1][1]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][1]) + ' - ' + 'kr' + str(reaction_index) + ' * S' \
                               + str(r[2][0]) + '^ko_' + str(reaction_index) + '_' + str(r[2][0]) + ')'
                    kf.append('kf' + str(reaction_index))
                    kr.append('kr' + str(reaction_index))
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^kor_' + str(reaction_index) + '_' + str(reg)
                        kor.append('kor_' + str(reaction_index) + '_' + str(reg))
                        kor_sign.append(r[4][i])

                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][0]))
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][1]))
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[2][0]))

            # -------------------------------------------------------------------------

            if r[0] == TReactionType.UNIBI:
                # UniBi
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[2][1])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme + 'kc' + str(reaction_index) + ' * S' + str(r[1][0]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][0])
                    kc.append('kc' + str(reaction_index))
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^kor_' + str(reaction_index) + '_' + str(reg)
                        kor.append('kor_' + str(reaction_index) + '_' + str(reg))
                        kor_sign.append(r[4][i])
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][0]))
                else:
                    rxn_str += '; ' + enzyme + '(kf' + str(reaction_index) + ' * S' + str(r[1][0]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' - ' + 'kr' + str(reaction_index) + ' * S' \
                               + str(r[2][0]) + '^ko_' + str(reaction_index) + '_' + str(r[2][0]) + ' * S' \
                               + str(r[2][1]) + '^ko_' + str(reaction_index) + '_' + str(r[2][1]) + ')'
                    kf.append('kf' + str(reaction_index))
                    kr.append('kr' + str(reaction_index))
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^kor_' + str(reaction_index) + '_' + str(reg)
                        kor.append('kor_' + str(reaction_index) + '_' + str(reg))
                        kor_sign.append(r[4][i])
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][0]))
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[2][0]))
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[2][1]))

            # -------------------------------------------------------------------------

            if r[0] == TReactionType.BIBI:
                # BiBi
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[2][1])

                rev = reversibility()

                if not rev:
                    if not rev:
                        rxn_str += '; ' + enzyme + 'kc' + str(reaction_index) + ' * S' + str(r[1][0]) + '^ko_' + \
                                   str(reaction_index) + '_' + str(r[1][0]) + ' * S' + str(r[1][1]) + '^ko_' + \
                                   str(reaction_index) + '_' + str(r[1][1])
                        kc.append('kc' + str(reaction_index))

                        for i, reg in enumerate(r[3]):
                            rxn_str += ' * S' + str(reg) + '^kor_' + str(reaction_index) + '_' + str(reg)
                            kor.append('kor_' + str(reaction_index) + '_' + str(reg))
                            kor_sign.append(r[4][i])

                else:
                    rxn_str += '; ' + enzyme + '(kf' + str(reaction_index) + ' * S' + str(r[1][0]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][0]) + ' * S' + str(r[1][1]) + '^ko_' + \
                               str(reaction_index) + '_' + str(r[1][1]) + ' - ' + 'kr' + str(reaction_index) + ' * S' \
                               + str(r[2][0]) + '^ko_' + str(reaction_index) + '_' + str(r[2][0]) + ' * S' \
                               + str(r[2][1]) + '^ko_' + str(reaction_index) + '_' + str(r[2][1]) + ')'
                    kf.append('kf' + str(reaction_index))
                    kr.append('kr' + str(reaction_index))
                    for i, reg in enumerate(r[3]):
                        rxn_str += ' * S' + str(reg) + '^kor_' + str(reaction_index) + '_' + str(reg)
                        kor.append('kor_' + str(reaction_index) + '_' + str(reg))
                        kor_sign.append(r[4][i])

                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][0]))
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[1][1]))
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[2][0]))
                    ko.append('ko_' + str(reaction_index) + '_' + str(r[2][1]))

            # -------------------------------------------------------------------------

            rxn_str += '\n'
        rxn_str += '\n'

        for each in ko:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('ko')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('ko')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ko')][0],
                                        scale=kinetics[3][kinetics[2].index('ko')][1]
                                        - kinetics[3][kinetics[2].index('ko')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('ko')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('ko')][0],
                                           kinetics[3][kinetics[2].index('ko')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('ko')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('ko')][0],
                                         scale=kinetics[3][kinetics[2].index('ko')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('ko')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ko')][0],
                                        s=kinetics[3][kinetics[2].index('ko')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ko')][0],
                                    scale=kinetics[3][kinetics[2].index('ko')][1]
                                    - kinetics[3][kinetics[2].index('ko')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('ko')][0],
                                       kinetics[3][kinetics[2].index('ko')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('ko')][0],
                                     scale=kinetics[3][kinetics[2].index('ko')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ko')][0],
                                    s=kinetics[3][kinetics[2].index('ko')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if ko:
            param_str += '\n'

        for i, each in enumerate(kor):

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('kor')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('kor')] == 'uniform':

                    if kinetics[3][kinetics[2].index('kor')][0] < 0:

                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kor')][0],
                                            scale=kinetics[3][kinetics[2].index('kor')][1]
                                            - kinetics[3][kinetics[2].index('kor')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    else:
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kor')][0],
                                            scale=kinetics[3][kinetics[2].index('kor')][1]
                                            - kinetics[3][kinetics[2].index('kor')][0])
                        if kor_sign[i] < 0:
                            param_str += each + ' = -' + str(const) + '\n'
                        else:
                            param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kor')] == 'loguniform':

                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kor')][0],
                                        scale=kinetics[3][kinetics[2].index('kor')][1]
                                        - kinetics[3][kinetics[2].index('kor')][0])
                    if kor_sign[i] < 0:
                        param_str += each + ' = -' + str(const) + '\n'
                    else:
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kor')] == 'normal':

                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kor')][0],
                                     scale=kinetics[3][kinetics[2].index('kor')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kor')] == 'lognormal':

                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kor')][0],
                                        s=kinetics[3][kinetics[2].index('kor')][1])
                    if kor_sign[i] < 0:
                        param_str += each + ' = -' + str(const) + '\n'
                    else:
                        param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':

                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':

                if kinetics[3][kinetics[2].index('kor')][0] < 0:

                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kor')][0],
                                        scale=kinetics[3][kinetics[2].index('kor')][1]
                                        - kinetics[3][kinetics[2].index('kor')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                else:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kor')][0],
                                        scale=kinetics[3][kinetics[2].index('kor')][1]
                                        - kinetics[3][kinetics[2].index('kor')][0])
                    if kor_sign[i] < 0:
                        param_str += each + ' = -' + str(const) + '\n'
                    else:
                        param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':

                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kor')][0],
                                    scale=kinetics[3][kinetics[2].index('kor')][1]
                                    - kinetics[3][kinetics[2].index('kor')][0])
                if kor_sign[i] < 0:
                    param_str += each + ' = -' + str(const) + '\n'
                else:
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':

                const = norm.rvs(loc=kinetics[3][kinetics[2].index('kor')][0],
                                 scale=kinetics[3][kinetics[2].index('kor')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'lognormal':

                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kor')][0],
                                    s=kinetics[3][kinetics[2].index('kor')][1])
                if kor_sign[i] < 0:
                    param_str += each + ' = -' + str(const) + '\n'
                else:
                    param_str += each + ' = ' + str(const) + '\n'

        if kor:
            param_str += '\n'

        for each in kf:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('kf')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('kf')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                        scale=kinetics[3][kinetics[2].index('kf')][1]
                                        - kinetics[3][kinetics[2].index('kf')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kf')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0],
                                           kinetics[3][kinetics[2].index('kf')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kf')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                         scale=kinetics[3][kinetics[2].index('kf')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('kf')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0],
                                        s=kinetics[3][kinetics[2].index('kf')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                    scale=kinetics[3][kinetics[2].index('kf')][1]
                                    - kinetics[3][kinetics[2].index('kf')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0],
                                       kinetics[3][kinetics[2].index('kf')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                     scale=kinetics[3][kinetics[2].index('kf')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0],
                                    s=kinetics[3][kinetics[2].index('kf')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if kf:
            param_str += '\n'

        for each in kr:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('kr')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('kr')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                        scale=kinetics[3][kinetics[2].index('kr')][1]
                                        - kinetics[3][kinetics[2].index('kr')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kr')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0],
                                           kinetics[3][kinetics[2].index('kr')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kr')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                         scale=kinetics[3][kinetics[2].index('kr')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('kr')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0],
                                        s=kinetics[3][kinetics[2].index('kr')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                    scale=kinetics[3][kinetics[2].index('kr')][1]
                                    - kinetics[3][kinetics[2].index('kr')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0],
                                       kinetics[3][kinetics[2].index('kr')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                     scale=kinetics[3][kinetics[2].index('kr')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0],
                                    s=kinetics[3][kinetics[2].index('kr')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if kr:
            param_str += '\n'

        for each in kc:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('kc')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('kc')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                        scale=kinetics[3][kinetics[2].index('kc')][1]
                                        - kinetics[3][kinetics[2].index('kc')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kc')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc')][0],
                                           kinetics[3][kinetics[2].index('kc')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kc')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                         scale=kinetics[3][kinetics[2].index('kc')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('kc')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc')][0],
                                        s=kinetics[3][kinetics[2].index('kc')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                    scale=kinetics[3][kinetics[2].index('kc')][1]
                                    - kinetics[3][kinetics[2].index('kc')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kc')][0],
                                       kinetics[3][kinetics[2].index('kc')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                     scale=kinetics[3][kinetics[2].index('kc')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc')][0],
                                    s=kinetics[3][kinetics[2].index('kc')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if kc:
            param_str += '\n'

    if kinetics[0] == 'mass_action':

        if len(kinetics[2]) == 3 or len(kinetics[2]) == 4:

            kf = []
            kr = []
            kc = []

            for reaction_index, r in enumerate(reaction_list_copy):

                if add_enzyme:
                    enzyme = 'E' + str(reaction_index) + '*('
                    enzyme_end = ')'

                rxn_str += 'J' + str(reaction_index) + ': '
                if r[0] == TReactionType.UNIUNI:
                    # UniUni
                    rxn_str += 'S' + str(r[1][0])
                    rxn_str += ' -> '
                    rxn_str += 'S' + str(r[2][0])

                    rev = reversibility()
                    if not rev:
                        rxn_str += '; ' + enzyme + 'kc' + str(reaction_index) + '*S' + str(r[1][0]) + enzyme_end
                        kc.append('kc' + str(reaction_index))

                    else:
                        rxn_str += '; ' + enzyme + 'kf' + str(reaction_index) + '*S' + str(r[1][0]) \
                                   + ' - kr' + str(reaction_index) + '*S' + str(r[2][0]) + enzyme_end
                        kf.append('kf' + str(reaction_index))
                        kr.append('kr' + str(reaction_index))

                if r[0] == TReactionType.BIUNI:
                    # BiUni
                    rxn_str += 'S' + str(r[1][0])
                    rxn_str += ' + '
                    rxn_str += 'S' + str(r[1][1])
                    rxn_str += ' -> '
                    rxn_str += 'S' + str(r[2][0])

                    rev = reversibility()

                    if not rev:
                        rxn_str += '; ' + enzyme + 'kc' + str(reaction_index) + '*S' + str(r[1][0]) \
                                   + '*S' + str(r[1][1]) + enzyme_end
                        kc.append('kc' + str(reaction_index))

                    else:
                        rxn_str += '; ' + enzyme + 'kf' + str(reaction_index) + '*S' + str(r[1][0]) \
                                   + '*S' + str(r[1][1]) + ' - kr' + str(reaction_index) + '*S' \
                                   + str(r[2][0]) + enzyme_end
                        kf.append('kf' + str(reaction_index))
                        kr.append('kr' + str(reaction_index))

                if r[0] == TReactionType.UNIBI:
                    # UniBi
                    rxn_str += 'S' + str(r[1][0])
                    rxn_str += ' -> '
                    rxn_str += 'S' + str(r[2][0])
                    rxn_str += ' + '
                    rxn_str += 'S' + str(r[2][1])

                    rev = reversibility()
                    if not rev:
                        rxn_str += '; ' + enzyme + 'kc' + str(reaction_index) + '*S' + str(r[1][0]) + enzyme_end
                        kc.append('kc' + str(reaction_index))

                    else:
                        rxn_str += '; ' + enzyme + 'kf' + str(reaction_index) + '*S' + str(r[1][0]) \
                                   + ' - kr' + str(reaction_index) + '*S' + str(r[2][0]) \
                                   + '*S' + str(r[2][1]) + enzyme_end
                        kf.append('kf' + str(reaction_index))
                        kr.append('kr' + str(reaction_index))

                if r[0] == TReactionType.BIBI:
                    # BiBi
                    rxn_str += 'S' + str(r[1][0])
                    rxn_str += ' + '
                    rxn_str += 'S' + str(r[1][1])
                    rxn_str += ' -> '
                    rxn_str += 'S' + str(r[2][0])
                    rxn_str += ' + '
                    rxn_str += 'S' + str(r[2][1])

                    rev = reversibility()
                    if not rev:
                        rxn_str += '; ' + enzyme + 'kc' + str(reaction_index) + '*S' + str(r[1][0]) \
                                   + '*S' + str(r[1][1]) + enzyme_end
                        kc.append('kc' + str(reaction_index))

                    else:
                        rxn_str += '; ' + enzyme + 'kf' + str(reaction_index) + '*S' + str(r[1][0]) \
                                   + '*S' + str(r[1][1]) + ' - kr' + str(reaction_index) + '*S' \
                                   + str(r[2][0]) + '*S' + str(r[2][1]) + enzyme_end
                        kf.append('kf' + str(reaction_index))
                        kr.append('kr' + str(reaction_index))

                rxn_str += '\n'
            rxn_str += '\n'

            for each in kf:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kf')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kf')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                            scale=kinetics[3][kinetics[2].index('kf')][1]
                                            - kinetics[3][kinetics[2].index('kf')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0],
                                               kinetics[3][kinetics[2].index('kf')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                             scale=kinetics[3][kinetics[2].index('kf')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kf')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0],
                                            s=kinetics[3][kinetics[2].index('kf')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                        scale=kinetics[3][kinetics[2].index('kf')][1]
                                        - kinetics[3][kinetics[2].index('kf')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0],
                                           kinetics[3][kinetics[2].index('kf')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                         scale=kinetics[3][kinetics[2].index('kf')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0],
                                        s=kinetics[3][kinetics[2].index('kf')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kf:
                param_str += '\n'

            for each in kr:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kr')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kr')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                            scale=kinetics[3][kinetics[2].index('kr')][1]
                                            - kinetics[3][kinetics[2].index('kr')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0],
                                               kinetics[3][kinetics[2].index('kr')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                             scale=kinetics[3][kinetics[2].index('kr')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kr')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0],
                                            s=kinetics[3][kinetics[2].index('kr')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                        scale=kinetics[3][kinetics[2].index('kr')][1]
                                        - kinetics[3][kinetics[2].index('kr')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0],
                                           kinetics[3][kinetics[2].index('kr')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                         scale=kinetics[3][kinetics[2].index('kr')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0],
                                        s=kinetics[3][kinetics[2].index('kr')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kr:
                param_str += '\n'

            for each in kc:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kc')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kc')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                            scale=kinetics[3][kinetics[2].index('kc')][1]
                                            - kinetics[3][kinetics[2].index('kc')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kc')][0],
                                               kinetics[3][kinetics[2].index('kc')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                             scale=kinetics[3][kinetics[2].index('kc')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kc')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc')][0],
                                            s=kinetics[3][kinetics[2].index('kc')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                        scale=kinetics[3][kinetics[2].index('kc')][1]
                                        - kinetics[3][kinetics[2].index('kc')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc')][0],
                                           kinetics[3][kinetics[2].index('kc')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                         scale=kinetics[3][kinetics[2].index('kc')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc')][0],
                                        s=kinetics[3][kinetics[2].index('kc')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kc:
                param_str += '\n'

        if len(kinetics[2]) == 12 or len(kinetics[2]) == 13:

            kf0 = []
            kr0 = []
            kc0 = []
            kf1 = []
            kr1 = []
            kc1 = []
            kf2 = []
            kr2 = []
            kc2 = []
            kf3 = []
            kr3 = []
            kc3 = []

            for reaction_index, r in enumerate(reaction_list_copy):

                if add_enzyme:
                    enzyme = 'E' + str(reaction_index) + '*('
                    enzyme_end = ')'

                rxn_str += 'J' + str(reaction_index) + ': '
                if r[0] == TReactionType.UNIUNI:
                    # UniUni
                    rxn_str += 'S' + str(r[1][0])
                    rxn_str += ' -> '
                    rxn_str += 'S' + str(r[2][0])

                    rev = reversibility()
                    if not rev:
                        rxn_str += '; ' + enzyme + 'kc0_' + str(reaction_index) + '*S' + str(r[1][0]) + enzyme_end
                        kc0.append('kc0_' + str(reaction_index))

                    else:
                        rxn_str += '; ' + enzyme + 'kf0_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + ' - kr0_' + str(reaction_index) + '*S' + str(r[2][0]) + enzyme_end
                        kf0.append('kf0_' + str(reaction_index))
                        kr0.append('kr0_' + str(reaction_index))

                if r[0] == TReactionType.BIUNI:
                    # BiUni
                    rxn_str += 'S' + str(r[1][0])
                    rxn_str += ' + '
                    rxn_str += 'S' + str(r[1][1])
                    rxn_str += ' -> '
                    rxn_str += 'S' + str(r[2][0])

                    rev = reversibility()
                    if not rev:
                        rxn_str += '; ' + enzyme + 'kc1_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + enzyme_end
                        kc1.append('kc1_' + str(reaction_index))

                    else:
                        rxn_str += '; ' + enzyme + 'kf1_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr1_' + str(reaction_index) + '*S' \
                                 + str(r[2][0]) + enzyme_end
                        kf1.append('kf1_' + str(reaction_index))
                        kr1.append('kr1_' + str(reaction_index))

                if r[0] == TReactionType.UNIBI:
                    # UniBi
                    rxn_str += 'S' + str(r[1][0])
                    rxn_str += ' -> '
                    rxn_str += 'S' + str(r[2][0])
                    rxn_str += ' + '
                    rxn_str += 'S' + str(r[2][1])

                    rev = reversibility()
                    if not rev:
                        rxn_str += '; ' + enzyme + 'kc2_' + str(reaction_index) + '*S' + str(r[1][0]) + enzyme_end
                        kc2.append('kc2_' + str(reaction_index))

                    else:
                        rxn_str += '; ' + enzyme + 'kf2_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + ' - kr2_' + str(reaction_index) + '*S' + str(r[2][0]) \
                                 + '*S' + str(r[2][1]) + enzyme_end
                        kf2.append('kf2_' + str(reaction_index))
                        kr2.append('kr2_' + str(reaction_index))

                if r[0] == TReactionType.BIBI:
                    # BiBi
                    rxn_str += 'S' + str(r[1][0])
                    rxn_str += ' + '
                    rxn_str += 'S' + str(r[1][1])
                    rxn_str += ' -> '
                    rxn_str += 'S' + str(r[2][0])
                    rxn_str += ' + '
                    rxn_str += 'S' + str(r[2][1])

                    rev = reversibility()
                    if not rev:
                        rxn_str += '; ' + enzyme + 'kc3_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + enzyme_end
                        kc3.append('kc3_' + str(reaction_index))

                    else:
                        rxn_str += '; ' + enzyme + 'kf3_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr3_' + str(reaction_index) + '*S' \
                                 + str(r[2][0]) + '*S' + str(r[2][1]) + enzyme_end
                        kf3.append('kf3_' + str(reaction_index))
                        kr3.append('kr3_' + str(reaction_index))

                rxn_str += '\n'
            rxn_str += '\n'

            for each in kf0:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kf0')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kf0')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf0')][0],
                                            scale=kinetics[3][kinetics[2].index('kf0')][1]
                                            - kinetics[3][kinetics[2].index('kf0')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf0')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kf0')][0],
                                               kinetics[3][kinetics[2].index('kf0')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf0')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf0')][0],
                                             scale=kinetics[3][kinetics[2].index('kf0')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kf0')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf0')][0],
                                            s=kinetics[3][kinetics[2].index('kf0')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf0')][0],
                                        scale=kinetics[3][kinetics[2].index('kf0')][1]
                                        - kinetics[3][kinetics[2].index('kf0')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf0')][0],
                                           kinetics[3][kinetics[2].index('kf0')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf0')][0],
                                         scale=kinetics[3][kinetics[2].index('kf0')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf0')][0],
                                        s=kinetics[3][kinetics[2].index('kf0')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kf0:
                param_str += '\n'

            for each in kf1:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kf1')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kf1')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf1')][0],
                                            scale=kinetics[3][kinetics[2].index('kf1')][1]
                                            - kinetics[3][kinetics[2].index('kf1')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf1')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kf1')][0],
                                               kinetics[3][kinetics[2].index('kf1')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf1')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf1')][0],
                                             scale=kinetics[3][kinetics[2].index('kf1')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kf1')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf1')][0],
                                            s=kinetics[3][kinetics[2].index('kf1')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf1')][0],
                                        scale=kinetics[3][kinetics[2].index('kf1')][1]
                                        - kinetics[3][kinetics[2].index('kf1')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf1')][0],
                                           kinetics[3][kinetics[2].index('kf1')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf1')][0],
                                         scale=kinetics[3][kinetics[2].index('kf1')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf1')][0],
                                        s=kinetics[3][kinetics[2].index('kf1')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kf1:
                param_str += '\n'

            for each in kf2:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kf2')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kf2')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf2')][0],
                                            scale=kinetics[3][kinetics[2].index('kf2')][1]
                                            - kinetics[3][kinetics[2].index('kf2')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf2')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kf2')][0],
                                               kinetics[3][kinetics[2].index('kf2')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf2')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf2')][0],
                                             scale=kinetics[3][kinetics[2].index('kf2')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kf2')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf2')][0],
                                            s=kinetics[3][kinetics[2].index('kf2')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf2')][0],
                                        scale=kinetics[3][kinetics[2].index('kf2')][1]
                                        - kinetics[3][kinetics[2].index('kf2')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf2')][0],
                                           kinetics[3][kinetics[2].index('kf2')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf2')][0],
                                         scale=kinetics[3][kinetics[2].index('kf2')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf2')][0],
                                        s=kinetics[3][kinetics[2].index('kf2')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kf2:
                param_str += '\n'

            for each in kf3:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kf3')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kf3')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf3')][0],
                                            scale=kinetics[3][kinetics[2].index('kf3')][1]
                                            - kinetics[3][kinetics[2].index('kf3')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf3')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kf3')][0],
                                               kinetics[3][kinetics[2].index('kf3')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kf3')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf3')][0],
                                             scale=kinetics[3][kinetics[2].index('kf3')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kf3')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf3')][0],
                                            s=kinetics[3][kinetics[2].index('kf3')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf3')][0],
                                        scale=kinetics[3][kinetics[2].index('kf3')][1]
                                        - kinetics[3][kinetics[2].index('kf3')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf3')][0],
                                           kinetics[3][kinetics[2].index('kf3')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf3')][0],
                                         scale=kinetics[3][kinetics[2].index('kf3')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf3')][0],
                                        s=kinetics[3][kinetics[2].index('kf3')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kf3:
                param_str += '\n'

            for each in kr0:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kr0')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kr0')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr0')][0],
                                            scale=kinetics[3][kinetics[2].index('kr0')][1]
                                            - kinetics[3][kinetics[2].index('kr0')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr0')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kr0')][0],
                                               kinetics[3][kinetics[2].index('kr0')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr0')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr0')][0],
                                             scale=kinetics[3][kinetics[2].index('kr0')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kr0')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr0')][0],
                                            s=kinetics[3][kinetics[2].index('kr0')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr0')][0],
                                        scale=kinetics[3][kinetics[2].index('kr0')][1]
                                        - kinetics[3][kinetics[2].index('kr0')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr0')][0],
                                           kinetics[3][kinetics[2].index('kr0')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr0')][0],
                                         scale=kinetics[3][kinetics[2].index('kr0')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr0')][0],
                                        s=kinetics[3][kinetics[2].index('kr0')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kr0:
                param_str += '\n'

            for each in kr1:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kr1')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kr1')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr1')][0],
                                            scale=kinetics[3][kinetics[2].index('kr1')][1]
                                            - kinetics[3][kinetics[2].index('kr1')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr1')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kr1')][0],
                                               kinetics[3][kinetics[2].index('kr1')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr1')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr1')][0],
                                             scale=kinetics[3][kinetics[2].index('kr1')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kr1')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr1')][0],
                                            s=kinetics[3][kinetics[2].index('kr1')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr1')][0],
                                        scale=kinetics[3][kinetics[2].index('kr1')][1]
                                        - kinetics[3][kinetics[2].index('kr1')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr1')][0],
                                           kinetics[3][kinetics[2].index('kr1')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr1')][0],
                                         scale=kinetics[3][kinetics[2].index('kr1')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr1')][0],
                                        s=kinetics[3][kinetics[2].index('kr1')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kr1:
                param_str += '\n'

            for each in kr2:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kr2')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kr2')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr2')][0],
                                            scale=kinetics[3][kinetics[2].index('kr2')][1]
                                            - kinetics[3][kinetics[2].index('kr2')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr2')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kr2')][0],
                                               kinetics[3][kinetics[2].index('kr2')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr2')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr2')][0],
                                             scale=kinetics[3][kinetics[2].index('kr2')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kr2')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr2')][0],
                                            s=kinetics[3][kinetics[2].index('kr2')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr2')][0],
                                        scale=kinetics[3][kinetics[2].index('kr2')][1]
                                        - kinetics[3][kinetics[2].index('kr2')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr2')][0],
                                           kinetics[3][kinetics[2].index('kr2')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr2')][0],
                                         scale=kinetics[3][kinetics[2].index('kr2')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr2')][0],
                                        s=kinetics[3][kinetics[2].index('kr2')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kr2:
                param_str += '\n'

            for each in kr3:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kr3')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kr3')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr3')][0],
                                            scale=kinetics[3][kinetics[2].index('kr3')][1]
                                            - kinetics[3][kinetics[2].index('kr3')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr3')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kr3')][0],
                                               kinetics[3][kinetics[2].index('kr3')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kr3')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr3')][0],
                                             scale=kinetics[3][kinetics[2].index('kr3')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kr3')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr3')][0],
                                            s=kinetics[3][kinetics[2].index('kr3')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr3')][0],
                                        scale=kinetics[3][kinetics[2].index('kr3')][1]
                                        - kinetics[3][kinetics[2].index('kr3')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr3')][0],
                                           kinetics[3][kinetics[2].index('kr3')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr3')][0],
                                         scale=kinetics[3][kinetics[2].index('kr3')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr3')][0],
                                        s=kinetics[3][kinetics[2].index('kr3')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kr3:
                param_str += '\n'

            for each in kc0:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kc0')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kc0')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc0')][0],
                                            scale=kinetics[3][kinetics[2].index('kc0')][1]
                                            - kinetics[3][kinetics[2].index('kc0')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc0')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kc0')][0],
                                               kinetics[3][kinetics[2].index('kc0')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc0')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc0')][0],
                                             scale=kinetics[3][kinetics[2].index('kc0')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kc0')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc0')][0],
                                            s=kinetics[3][kinetics[2].index('kc0')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc0')][0],
                                        scale=kinetics[3][kinetics[2].index('kc0')][1]
                                        - kinetics[3][kinetics[2].index('kc0')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc0')][0],
                                           kinetics[3][kinetics[2].index('kc0')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc0')][0],
                                         scale=kinetics[3][kinetics[2].index('kc0')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc0')][0],
                                        s=kinetics[3][kinetics[2].index('kc0')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kc0:
                param_str += '\n'

            for each in kc1:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kc1')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kc1')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc1')][0],
                                            scale=kinetics[3][kinetics[2].index('kc1')][1]
                                            - kinetics[3][kinetics[2].index('kc1')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc1')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kc1')][0],
                                               kinetics[3][kinetics[2].index('kc1')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc1')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc1')][0],
                                             scale=kinetics[3][kinetics[2].index('kc1')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kc1')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc1')][0],
                                            s=kinetics[3][kinetics[2].index('kc1')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc1')][0],
                                        scale=kinetics[3][kinetics[2].index('kc1')][1]
                                        - kinetics[3][kinetics[2].index('kc1')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc1')][0],
                                           kinetics[3][kinetics[2].index('kc1')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc1')][0],
                                         scale=kinetics[3][kinetics[2].index('kc1')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc1')][0],
                                        s=kinetics[3][kinetics[2].index('kc1')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kc1:
                param_str += '\n'

            for each in kc2:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kc2')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kc2')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc2')][0],
                                            scale=kinetics[3][kinetics[2].index('kc2')][1]
                                            - kinetics[3][kinetics[2].index('kc2')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc2')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kc2')][0],
                                               kinetics[3][kinetics[2].index('kc2')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc2')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc2')][0],
                                             scale=kinetics[3][kinetics[2].index('kc2')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kc2')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc2')][0],
                                            s=kinetics[3][kinetics[2].index('kc2')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc2')][0],
                                        scale=kinetics[3][kinetics[2].index('kc2')][1]
                                        - kinetics[3][kinetics[2].index('kc2')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc2')][0],
                                           kinetics[3][kinetics[2].index('kc2')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc2')][0],
                                         scale=kinetics[3][kinetics[2].index('kc2')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc2')][0],
                                        s=kinetics[3][kinetics[2].index('kc2')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kc2:
                param_str += '\n'

            for each in kc3:

                if type(kinetics[1]) is list:

                    if kinetics[1][kinetics[2].index('kc3')] == 'trivial':
                        param_str += each + ' = 1\n'

                    if kinetics[1][kinetics[2].index('kc3')] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc3')][0],
                                            scale=kinetics[3][kinetics[2].index('kc3')][1]
                                            - kinetics[3][kinetics[2].index('kc3')][0])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc3')] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('kc3')][0],
                                               kinetics[3][kinetics[2].index('kc3')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                    if kinetics[1][kinetics[2].index('kc3')] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc3')][0],
                                             scale=kinetics[3][kinetics[2].index('kc3')][1])
                            if const >= 0:
                                param_str += each + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1][kinetics[2].index('kc3')] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc3')][0],
                                            s=kinetics[3][kinetics[2].index('kc3')][1])
                        param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc3')][0],
                                        scale=kinetics[3][kinetics[2].index('kc3')][1]
                                        - kinetics[3][kinetics[2].index('kc3')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc3')][0],
                                           kinetics[3][kinetics[2].index('kc3')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc3')][0],
                                         scale=kinetics[3][kinetics[2].index('kc3')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc3')][0],
                                        s=kinetics[3][kinetics[2].index('kc3')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kc3:
                param_str += '\n'

    if kinetics[0] == 'hanekom':

        v = []
        keq = []
        k = []
        ks = []
        kp = []

        for reaction_index, r in enumerate(reaction_list_copy):

            if add_enzyme:
                enzyme = 'E' + str(reaction_index) + '*('
                enzyme_end = ')'

            v.append('v' + str(reaction_index))
            keq.append('keq' + str(reaction_index))

            rxn_str += 'J' + str(reaction_index) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])

                rev = reversibility()

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')/(1 + S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))

                    if 'k' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ')/(1 + S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')*(1-(S' \
                            + str(r[2][0]) + '/S' + str(r[1][0]) \
                            + ')/keq' + str(reaction_index) + ')/(1 + S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ' + S' \
                            + str(r[2][0]) + '/kp_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ')' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))
                        kp.append('kp_' + str(reaction_index) + '_' + str(r[2][0]))

                    if 'k' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ')*(1-(S' \
                            + str(r[2][0]) + '/S' + str(r[1][0]) \
                            + ')/keq' + str(reaction_index) + ')/(1 + S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ' + S' \
                            + str(r[2][0]) + '/k_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ')' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[2][0]))

            if r[0] == TReactionType.BIUNI:
                # BiUni
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])

                rev = reversibility()

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][1]) + ')/(1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[1][1]) + '/ks_' + str(reaction_index) + '_' + str(r[1][1]) \
                            + ' + (S' + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')*(S' + str(r[1][1]) + '/ks_' + str(reaction_index) + '_' + str(r[1][1]) \
                            + '))' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][1]))

                    if 'k' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][1]) + ')/(1 + S' \
                            + str(r[1][0]) + '/k_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[1][1]) + '/k_' + str(reaction_index) + '_' + str(r[1][1]) \
                            + ' + (S' + str(r[1][0]) + '/k_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')*(S' + str(r[1][1]) + '/k_' + str(reaction_index) + '_' + str(r[1][1]) \
                            + '))' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][1]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/ks_' \
                                   + str(reaction_index) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][1]) + ')' + '*(1-(S' \
                            + str(r[2][0]) + '/(S' + str(r[1][0]) \
                            + '*S' + str(r[1][1]) + '))/keq' + str(reaction_index) + ')/(1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[1][1]) + '/ks_' + str(reaction_index) + '_' + str(r[1][1]) \
                            + ' + (S' + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')*(S' + str(r[1][1]) + '/ks_' + str(reaction_index) + '_' + str(r[1][1]) \
                            + ') + S' + str(r[2][0]) + '/kp_' + str(reaction_index) + '_' + str(r[2][0]) \
                            + ')' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][1]))
                        kp.append('kp_' + str(reaction_index) + '_' + str(r[2][0]))

                    if 'k' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/k_' \
                                   + str(reaction_index) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][1]) + ')' + '*(1-(S' \
                            + str(r[2][0]) + '/(S' + str(r[1][0]) \
                            + '*S' + str(r[1][1]) + '))/keq' + str(reaction_index) + ')/(1 + S' \
                            + str(r[1][0]) + '/k_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[1][1]) + '/k_' + str(reaction_index) + '_' + str(r[1][1]) \
                            + ' + (S' + str(r[1][0]) + '/k_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')*(S' + str(r[1][1]) + '/k_' + str(reaction_index) + '_' + str(r[1][1]) \
                            + ') + S' + str(r[2][0]) + '/k_' + str(reaction_index) + '_' + str(r[2][0]) \
                            + ')' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][1]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[2][0]))

            if r[0] == TReactionType.UNIBI:
                # UniBi
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[2][1])

                rev = reversibility()

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')/(1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))

                    if 'k' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ')/(1 + S' \
                            + str(r[1][0]) + '/k_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')*(1-(S' \
                            + str(r[2][0]) + '*S' + str(r[2][1]) \
                            + '/S' + str(r[1][0]) + ')/keq' + str(reaction_index) + ')/(1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[2][0]) + '/kp_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ' + S' + str(r[2][1]) \
                            + '/kp_' + str(reaction_index) + '_' + str(r[2][1]) + ' + (S' \
                            + str(r[2][0]) + '/kp_' + str(reaction_index) + '_' + str(r[2][0]) \
                            + ')*(S' + str(r[2][1]) + '/kp_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ')' + ')' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))
                        kp.append('kp_' + str(reaction_index) + '_' + str(r[2][0]))
                        kp.append('kp_' + str(reaction_index) + '_' + str(r[2][1]))

                    if 'k' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ')*(1-(S' \
                            + str(r[2][0]) + '*S' + str(r[2][1]) \
                            + '/S' + str(r[1][0]) + ')/keq' + str(reaction_index) + ')/(1 + S' \
                            + str(r[1][0]) + '/k_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[2][0]) + '/k_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ' + S' + str(r[2][1]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[2][1]) + ' + (S' \
                            + str(r[2][0]) + '/k_' + str(reaction_index) + '_' + str(r[2][0]) \
                            + ')*(S' + str(r[2][1]) + '/k_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ')' + ')' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[2][0]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[2][1]))

            if r[0] == TReactionType.BIBI:
                # BiBi
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[2][1])

                rev = reversibility()

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/ks_' \
                                   + str(reaction_index) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][1]) + ')/((1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')*(1 + S' + str(r[1][1]) + '/ks_' + str(reaction_index) + '_' \
                            + str(r[1][1]) + '))' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][1]))

                    if 'k' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/k_' \
                                   + str(reaction_index) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][1]) + ')/((1 + S' \
                            + str(r[1][0]) + '/k_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')*(1 + S' + str(r[1][1]) + '/k_' + str(reaction_index) + '_' \
                            + str(r[1][1]) + '))' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][1]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/ks_' \
                                   + str(reaction_index) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][1]) + ')' + '*(1-(S' \
                            + str(r[2][0]) + '*S' + str(r[2][1]) \
                            + '/(S' + str(r[1][0]) + '*S' \
                            + str(r[1][1]) + '))/keq' + str(reaction_index) + ')/((1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[2][0]) + '/kp_' + str(reaction_index) + '_' + str(r[2][0]) \
                            + ')*(1 + S' + str(r[1][1]) + '/ks_' + str(reaction_index) + '_' \
                            + str(r[1][1]) + ' + S' + str(r[2][1]) \
                            + '/kp_' + str(reaction_index) + '_' + str(r[2][1]) + '))' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][1]))
                        kp.append('kp_' + str(reaction_index) + '_' + str(r[2][0]))
                        kp.append('kp_' + str(reaction_index) + '_' + str(r[2][1]))

                    if 'k' in kinetics[2]:
                        rxn_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/k_' \
                                   + str(reaction_index) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][1]) + ')' + '*(1-(S' \
                            + str(r[2][0]) + '*S' + str(r[2][1]) \
                            + '/(S' + str(r[1][0]) + '*S' \
                            + str(r[1][1]) + '))/keq' + str(reaction_index) + ')/((1 + S' \
                            + str(r[1][0]) + '/k_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[2][0]) + '/k_' + str(reaction_index) + '_' + str(r[2][0]) \
                            + ')*(1 + S' + str(r[1][1]) + '/k_' + str(reaction_index) + '_' \
                            + str(r[1][1]) + ' + S' + str(r[2][1]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[2][1]) + '))' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][1]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[2][0]))
                        k.append('k_' + str(reaction_index) + '_' + str(r[2][1]))

            rxn_str += '\n'
        rxn_str += '\n'

        for each in v:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('v')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('v')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                        scale=kinetics[3][kinetics[2].index('v')][1]
                                        - kinetics[3][kinetics[2].index('v')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('v')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0],
                                           kinetics[3][kinetics[2].index('v')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('v')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                         scale=kinetics[3][kinetics[2].index('v')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('v')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0],
                                        s=kinetics[3][kinetics[2].index('v')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                    scale=kinetics[3][kinetics[2].index('v')][1]
                                    - kinetics[3][kinetics[2].index('v')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0],
                                       kinetics[3][kinetics[2].index('v')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                     scale=kinetics[3][kinetics[2].index('v')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0],
                                    s=kinetics[3][kinetics[2].index('v')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if v:
            param_str += '\n'

        for each in keq:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('keq')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('keq')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('keq')][0],
                                        scale=kinetics[3][kinetics[2].index('keq')][1]
                                        - kinetics[3][kinetics[2].index('keq')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('keq')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('keq')][0],
                                           kinetics[3][kinetics[2].index('keq')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('keq')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('keq')][0],
                                         scale=kinetics[3][kinetics[2].index('keq')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('keq')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('keq')][0],
                                        s=kinetics[3][kinetics[2].index('keq')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('keq')][0],
                                    scale=kinetics[3][kinetics[2].index('keq')][1]
                                    - kinetics[3][kinetics[2].index('keq')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('keq')][0],
                                       kinetics[3][kinetics[2].index('keq')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('keq')][0],
                                     scale=kinetics[3][kinetics[2].index('keq')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('keq')][0],
                                    s=kinetics[3][kinetics[2].index('keq')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if keq:
            param_str += '\n'

        for each in k:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('k')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('k')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                        scale=kinetics[3][kinetics[2].index('k')][1]
                                        - kinetics[3][kinetics[2].index('k')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('k')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('k')][0],
                                           kinetics[3][kinetics[2].index('k')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('k')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                         scale=kinetics[3][kinetics[2].index('k')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('k')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('k')][0],
                                        s=kinetics[3][kinetics[2].index('k')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                    scale=kinetics[3][kinetics[2].index('k')][1]
                                    - kinetics[3][kinetics[2].index('k')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('k')][0],
                                       kinetics[3][kinetics[2].index('k')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                     scale=kinetics[3][kinetics[2].index('k')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('k')][0],
                                    s=kinetics[3][kinetics[2].index('k')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if k:
            param_str += '\n'

        for each in ks:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('ks')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('ks')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ks')][0],
                                        scale=kinetics[3][kinetics[2].index('ks')][1]
                                        - kinetics[3][kinetics[2].index('ks')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('ks')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('ks')][0],
                                           kinetics[3][kinetics[2].index('ks')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('ks')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('ks')][0],
                                         scale=kinetics[3][kinetics[2].index('ks')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('ks')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ks')][0],
                                        s=kinetics[3][kinetics[2].index('ks')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ks')][0],
                                    scale=kinetics[3][kinetics[2].index('ks')][1]
                                    - kinetics[3][kinetics[2].index('ks')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('ks')][0],
                                       kinetics[3][kinetics[2].index('ks')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('ks')][0],
                                     scale=kinetics[3][kinetics[2].index('ks')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ks')][0],
                                    s=kinetics[3][kinetics[2].index('ks')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if ks:
            param_str += '\n'
            
        for each in kp:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('kp')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('kp')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kp')][0],
                                        scale=kinetics[3][kinetics[2].index('kp')][1]
                                        - kinetics[3][kinetics[2].index('kp')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kp')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kp')][0],
                                           kinetics[3][kinetics[2].index('kp')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kp')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kp')][0],
                                         scale=kinetics[3][kinetics[2].index('kp')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('kp')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kp')][0],
                                        s=kinetics[3][kinetics[2].index('kp')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kp')][0],
                                    scale=kinetics[3][kinetics[2].index('kp')][1]
                                    - kinetics[3][kinetics[2].index('kp')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kp')][0],
                                       kinetics[3][kinetics[2].index('kp')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kp')][0],
                                     scale=kinetics[3][kinetics[2].index('kp')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kp')][0],
                                    s=kinetics[3][kinetics[2].index('kp')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if kp:
            param_str += '\n'

    if kinetics[0] == 'lin_log':

        rc = []

        for reaction_index, r in enumerate(reaction_list_copy):

            if add_enzyme:
                enzyme = 'E' + str(reaction_index) + '*('
                enzyme_end = ')'

            rev_stoic = defaultdict(int)
            for each in r[1]:
                if each in rev_stoic:
                    rev_stoic[each] += 1
                else:
                    rev_stoic[each] = 1
            irr_stoic = deepcopy(rev_stoic)
            for each in r[2]:
                if each in rev_stoic:
                    rev_stoic[each] -= 1
                else:
                    rev_stoic[each] = -1

            rxn_str += 'J' + str(reaction_index) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][2][0])
                rxn_str += '; ' + enzyme[0:2] + 'v' + str(reaction_index) + '*(1'

                rev = reversibility()
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            rxn_str += ' + ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            rxn_str += ' + ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -1:
                            rxn_str += ' - ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))

                rxn_str += ')'

            if r[0] == TReactionType.BIUNI:
                # BiUni
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][2][0])
                rxn_str += '; ' + enzyme[0:2] + 'v' + str(reaction_index) + '*(1'

                rev = reversibility()
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            rxn_str += ' + ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if irr_stoic[each] == 2:
                            rxn_str += ' + ' + '2*log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            rxn_str += ' + ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == 2:
                            rxn_str += ' + ' + '2*log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -1:
                            rxn_str += ' - ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))

                rxn_str += ')'

            if r[0] == TReactionType.UNIBI:
                # UniBi
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][2][1])
                rxn_str += '; ' + enzyme[0:2] + 'v' + str(reaction_index) + '*(1'

                rev = reversibility()
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            rxn_str += ' + ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if irr_stoic[each] == 2:
                            rxn_str += ' + ' + '2*log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            rxn_str += ' + ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -1:
                            rxn_str += ' - ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -2:
                            rxn_str += ' - ' + '2*log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))

                rxn_str += ')'

            if r[0] == TReactionType.BIBI:
                # BiBi
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(reaction_list_copy[reaction_index][2][1])
                rxn_str += '; ' + enzyme[0:2] + 'v' + str(reaction_index) + '*(1'

                rev = reversibility()
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            rxn_str += ' + ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if irr_stoic[each] == 2:
                            rxn_str += ' + ' + '2*log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            rxn_str += ' + ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == 2:
                            rxn_str += ' + ' + '2*log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -1:
                            rxn_str += ' - ' + 'log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -2:
                            rxn_str += ' - ' + '2*log(S' + str(each) + '/rc_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            rc.append('rc_' + str(each) + '_' + str(reaction_index))

                rxn_str += ')'
            rxn_str += '\n'
        rxn_str += '\n'

        for index, r in enumerate(reaction_list_copy):
            
            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('v')] == 'trivial':
                    param_str += 'v' + str(index) + ' = 1\n'

                if kinetics[1][kinetics[2].index('v')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                        scale=kinetics[3][kinetics[2].index('v')][1]
                                        - kinetics[3][kinetics[2].index('v')][0])
                    param_str += 'v' + str(index) + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('v')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0],
                                           kinetics[3][kinetics[2].index('v')][1])
                    param_str += 'v' + str(index) + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('v')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                         scale=kinetics[3][kinetics[2].index('v')][1])
                        if const >= 0:
                            param_str += 'v' + str(index) + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('v')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0],
                                        s=kinetics[3][kinetics[2].index('v')][1])
                    param_str += 'v' + str(index) + ' = ' + str(const) + '\n'
            
            if kinetics[1] == 'trivial':
                param_str += 'v' + str(index) + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                    scale=kinetics[3][kinetics[2].index('v')][1]
                                    - kinetics[3][kinetics[2].index('v')][0])
                param_str += 'v' + str(index) + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0],
                                       kinetics[3][kinetics[2].index('v')][1])
                param_str += 'v' + str(index) + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                     scale=kinetics[3][kinetics[2].index('v')][1])
                    if const >= 0:
                        param_str += 'v' + str(index) + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0],
                                    s=kinetics[3][kinetics[2].index('v')][1])
                param_str += 'v' + str(index) + ' = ' + str(const) + '\n'
        
        param_str += '\n'

        for each in rc:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('rc')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('rc')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('rc')][0],
                                        scale=kinetics[3][kinetics[2].index('rc')][1]
                                        - kinetics[3][kinetics[2].index('rc')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('rc')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('rc')][0],
                                           kinetics[3][kinetics[2].index('rc')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('rc')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('rc')][0],
                                         scale=kinetics[3][kinetics[2].index('rc')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('rc')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('rc')][0],
                                        s=kinetics[3][kinetics[2].index('rc')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('rc')][0],
                                    scale=kinetics[3][kinetics[2].index('rc')][1]
                                    - kinetics[3][kinetics[2].index('rc')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('rc')][0],
                                       kinetics[3][kinetics[2].index('rc')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('rc')][0],
                                     scale=kinetics[3][kinetics[2].index('rc')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('rc')][0],
                                    s=kinetics[3][kinetics[2].index('rc')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if rc:
            param_str += '\n'

    if 'modular' in kinetics[0]:

        ma = []
        kma = []
        ms = []
        kms = []
        ro = []
        kf = []
        kr = []
        m = []
        km = []

        for reaction_index, r in enumerate(reaction_list_copy):

            if add_enzyme:
                enzyme = 'E' + str(reaction_index) + '*('
                enzyme_end = ')'

            rxn_str += 'J' + str(reaction_index) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + ' * '
                        if r[5][i] == 'a' and r[4][i] == 1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.append('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.append('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.append('ro_' + str(reaction_index) + '_' + str(reg))

                    rxn_str += '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            rxn_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                rxn_str += ' + '

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end

                    km.append('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    kf.append('kf_' + str(reaction_index))

                else:
                    rxn_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + ' * '
                        if r[5][i] == 'a' and r[4][i] == 1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.append('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.append('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.append('ro_' + str(reaction_index) + '_' + str(reg))

                    rxn_str += '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' - kr_' \
                        + str(reaction_index) + '*(S' + str(r[2][0]) + '/km_' + str(reaction_index) + '_' \
                        + str(r[2][0]) \
                        + ')^m_' + str(reaction_index) + '_' + str(r[2][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + (1 + S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + (S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            rxn_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                rxn_str += ' + '

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end

                    km.append('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[2][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[2][0]))
                    kf.append('kf_' + str(reaction_index))
                    kr.append('kr_' + str(reaction_index))

            if r[0] == TReactionType.BIUNI:
                # BiUni
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + ' * '
                        if r[5][i] == 'a' and r[4][i] == 1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.append('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.append('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.append('ro_' + str(reaction_index) + '_' + str(reg))

                    rxn_str = rxn_str \
                        + '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reaction_index) + '_' + str(r[1][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            rxn_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                rxn_str += ' + '

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end

                    km.append('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[1][1]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][1]))
                    kf.append('kf_' + str(reaction_index))

                else:
                    rxn_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + ' * '
                        if r[5][i] == 'a' and r[4][i] == 1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.append('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.append('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.append('ro_' + str(reaction_index) + '_' + str(reg))

                    rxn_str = rxn_str \
                        + '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reaction_index) + '_' + str(r[1][1]) + ' - kr_' + str(reaction_index) + '*(S' \
                        + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) \
                        + ')^m_' + str(reaction_index) + '_' + str(r[2][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + (1 + S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + (S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + '*(1 + S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + '*(S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            rxn_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                rxn_str += ' + '

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end

                    km.append('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[1][1]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[2][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][1]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[2][0]))
                    kf.append('kf_' + str(reaction_index))
                    kr.append('kr_' + str(reaction_index))

            if r[0] == TReactionType.UNIBI:
                # UniBi
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[2][1])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a' and r[4][i] == 1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.append('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.append('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.append('ro_' + str(reaction_index) + '_' + str(reg))

                    rxn_str += '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            rxn_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                rxn_str += ' + '

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end

                    km.append('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    kf.append('kf_' + str(reaction_index))

                else:
                    rxn_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a' and r[4][i] == 1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.append('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.append('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.append('ro_' + str(reaction_index) + '_' + str(reg))

                    rxn_str += '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' - kr_' \
                        + str(reaction_index) + '*(S' + str(r[2][0]) + '/km_' + str(reaction_index) + '_' \
                        + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' + str(r[2][0]) + '*(S' + str(r[2][1]) \
                        + '/km_' + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                        + str(r[2][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + (1 + S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + (S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            rxn_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                rxn_str += ' + '

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end

                    km.append('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[2][0]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[2][1]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[2][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[2][1]))
                    kf.append('kf_' + str(reaction_index))
                    kr.append('kr_' + str(reaction_index))

            if r[0] == TReactionType.BIBI:
                # BiBi
                rxn_str += 'S' + str(r[1][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[1][1])
                rxn_str += ' -> '
                rxn_str += 'S' + str(r[2][0])
                rxn_str += ' + '
                rxn_str += 'S' + str(r[2][1])

                rev = reversibility()

                if not rev:
                    rxn_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a' and r[4][i] == 1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.append('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.append('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.append('ro_' + str(reaction_index) + '_' + str(reg))

                    rxn_str = rxn_str \
                        + '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reaction_index) + '_' + str(r[1][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            rxn_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                rxn_str += ' + '

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end

                    km.append('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[1][1]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][1]))
                    kf.append('kf_' + str(reaction_index))

                else:
                    rxn_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a' and r[4][i] == 1:
                            rxn_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.append('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.append('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.append('ro_' + str(reaction_index) + '_' + str(reg))

                    rxn_str = rxn_str \
                        + '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reaction_index) + '_' + str(r[1][1]) + ' - kr_' + str(reaction_index) + '*(S' \
                        + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) \
                        + ')^m_' + str(reaction_index) + '_' + str(r[2][0]) + '*(S' \
                        + str(r[2][1]) + '/km_' + str(reaction_index) + '_' + str(r[2][1]) \
                        + ')^m_' + str(reaction_index) + '_' + str(r[2][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + (1 + S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' + str(reaction_index) + '_' \
                            + str(r[2][1]) \
                            + ')^m_' + str(reaction_index) + '_' + str(r[2][1]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + (S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' + str(reaction_index) + '_' + str(r[2][1]) \
                            + ')^m_' + str(reaction_index) + '_' + str(r[2][1]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + '*(1 + S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' + str(reaction_index) + '_' \
                            + str(r[2][1]) \
                            + ')^m_' + str(reaction_index) + '_' + str(r[2][1]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            rxn_str += '/((('
                        else:
                            rxn_str += '/(('

                        rxn_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + '*(S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' + str(reaction_index) + '_' + str(r[2][1]) \
                            + ')^m_' + str(reaction_index) + '_' + str(r[2][1]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end
                        else:
                            rxn_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            rxn_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                rxn_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                rxn_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                rxn_str += ' + '

                            if r[5][i] == 's':
                                ms.append('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.append('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            rxn_str += ')' + enzyme_end

                    km.append('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[1][1]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[2][0]))
                    km.append('km_' + str(reaction_index) + '_' + str(r[2][1]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[1][1]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[2][0]))
                    m.append('m_' + str(reaction_index) + '_' + str(r[2][1]))
                    kf.append('kf_' + str(reaction_index))
                    kr.append('kr_' + str(reaction_index))

            rxn_str += '\n'
        rxn_str += '\n'

        for each in ro:

            if type(kinetics[1]) is list:

                if 'ro' in kinetics[2] and kinetics[1][kinetics[2].index('ro')] == 'trivial':
                    param_str += each + ' = 1\n'

                else:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ro')][0],
                                        scale=kinetics[3][kinetics[2].index('ro')][1]
                                        - kinetics[3][kinetics[2].index('ro')][0])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            else:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ro')][0],
                                    scale=kinetics[3][kinetics[2].index('ro')][1]
                                    - kinetics[3][kinetics[2].index('ro')][0])
                param_str += each + ' = ' + str(const) + '\n'

        for each in kf:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('kf')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('kf')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                        scale=kinetics[3][kinetics[2].index('kf')][1]
                                        - kinetics[3][kinetics[2].index('kf')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kf')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0],
                                           kinetics[3][kinetics[2].index('kf')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kf')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                         scale=kinetics[3][kinetics[2].index('kf')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('kf')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0],
                                        s=kinetics[3][kinetics[2].index('kf')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                    scale=kinetics[3][kinetics[2].index('kf')][1]
                                    - kinetics[3][kinetics[2].index('kf')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0],
                                       kinetics[3][kinetics[2].index('kf')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                     scale=kinetics[3][kinetics[2].index('kf')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0],
                                    s=kinetics[3][kinetics[2].index('kf')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if kf:
            param_str += '\n'

        for each in kr:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('kr')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('kr')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                        scale=kinetics[3][kinetics[2].index('kr')][1]
                                        - kinetics[3][kinetics[2].index('kr')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kr')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0],
                                           kinetics[3][kinetics[2].index('kr')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kr')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                         scale=kinetics[3][kinetics[2].index('kr')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('kr')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0],
                                        s=kinetics[3][kinetics[2].index('kr')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                    scale=kinetics[3][kinetics[2].index('kr')][1]
                                    - kinetics[3][kinetics[2].index('kr')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0],
                                       kinetics[3][kinetics[2].index('kr')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                     scale=kinetics[3][kinetics[2].index('kr')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0],
                                    s=kinetics[3][kinetics[2].index('kr')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if kr:
            param_str += '\n'

        for each in km:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('km')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('km')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                        scale=kinetics[3][kinetics[2].index('km')][1]
                                        - kinetics[3][kinetics[2].index('km')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('km')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('km')][0],
                                           kinetics[3][kinetics[2].index('km')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('km')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                         scale=kinetics[3][kinetics[2].index('km')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('km')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('km')][0],
                                        s=kinetics[3][kinetics[2].index('km')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                    scale=kinetics[3][kinetics[2].index('km')][1]
                                    - kinetics[3][kinetics[2].index('km')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('km')][0],
                                       kinetics[3][kinetics[2].index('km')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                     scale=kinetics[3][kinetics[2].index('km')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('km')][0],
                                    s=kinetics[3][kinetics[2].index('km')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if km:
            param_str += '\n'

        for each in kma:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('kma')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('kma')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kma')][0],
                                        scale=kinetics[3][kinetics[2].index('kma')][1]
                                        - kinetics[3][kinetics[2].index('kma')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kma')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kma')][0],
                                           kinetics[3][kinetics[2].index('kma')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kma')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kma')][0],
                                         scale=kinetics[3][kinetics[2].index('kma')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('kma')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kma')][0],
                                        s=kinetics[3][kinetics[2].index('kma')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kma')][0],
                                    scale=kinetics[3][kinetics[2].index('kma')][1]
                                    - kinetics[3][kinetics[2].index('kma')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kma')][0],
                                       kinetics[3][kinetics[2].index('kma')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kma')][0],
                                     scale=kinetics[3][kinetics[2].index('kma')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kma')][0],
                                    s=kinetics[3][kinetics[2].index('kma')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if kma:
            param_str += '\n'

        for each in kms:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('kms')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('kms')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kms')][0],
                                        scale=kinetics[3][kinetics[2].index('kms')][1]
                                        - kinetics[3][kinetics[2].index('kms')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kms')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kms')][0],
                                           kinetics[3][kinetics[2].index('kms')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('kms')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kms')][0],
                                         scale=kinetics[3][kinetics[2].index('kms')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('kms')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kms')][0],
                                        s=kinetics[3][kinetics[2].index('kms')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kms')][0],
                                    scale=kinetics[3][kinetics[2].index('kms')][1]
                                    - kinetics[3][kinetics[2].index('kms')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kms')][0],
                                       kinetics[3][kinetics[2].index('kms')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kms')][0],
                                     scale=kinetics[3][kinetics[2].index('kms')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kms')][0],
                                    s=kinetics[3][kinetics[2].index('kms')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if kms:
            param_str += '\n'

        for each in m:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('m')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('m')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('m')][0],
                                        scale=kinetics[3][kinetics[2].index('m')][1]
                                        - kinetics[3][kinetics[2].index('m')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('m')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('m')][0],
                                           kinetics[3][kinetics[2].index('m')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('m')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('m')][0],
                                         scale=kinetics[3][kinetics[2].index('m')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('m')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('m')][0],
                                        s=kinetics[3][kinetics[2].index('m')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('m')][0],
                                    scale=kinetics[3][kinetics[2].index('m')][1]
                                    - kinetics[3][kinetics[2].index('m')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('m')][0],
                                       kinetics[3][kinetics[2].index('m')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('m')][0],
                                     scale=kinetics[3][kinetics[2].index('m')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('m')][0],
                                    s=kinetics[3][kinetics[2].index('m')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if m:
            param_str += '\n'

        for each in ma:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('ma')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('ma')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ma')][0],
                                        scale=kinetics[3][kinetics[2].index('ma')][1]
                                        - kinetics[3][kinetics[2].index('ma')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('ma')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('ma')][0],
                                           kinetics[3][kinetics[2].index('ma')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('ma')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('ma')][0],
                                         scale=kinetics[3][kinetics[2].index('ma')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('ma')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ma')][0],
                                        s=kinetics[3][kinetics[2].index('ma')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ma')][0],
                                    scale=kinetics[3][kinetics[2].index('ma')][1]
                                    - kinetics[3][kinetics[2].index('ma')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('ma')][0],
                                       kinetics[3][kinetics[2].index('ma')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('ma')][0],
                                     scale=kinetics[3][kinetics[2].index('ma')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ma')][0],
                                    s=kinetics[3][kinetics[2].index('ma')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if ma:
            param_str += '\n'

        for each in ms:

            if type(kinetics[1]) is list:

                if kinetics[1][kinetics[2].index('ms')] == 'trivial':
                    param_str += each + ' = 1\n'

                if kinetics[1][kinetics[2].index('ms')] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ms')][0],
                                        scale=kinetics[3][kinetics[2].index('ms')][1]
                                        - kinetics[3][kinetics[2].index('ms')][0])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('ms')] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('ms')][0],
                                           kinetics[3][kinetics[2].index('ms')][1])
                    param_str += each + ' = ' + str(const) + '\n'

                if kinetics[1][kinetics[2].index('ms')] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('ms')][0],
                                         scale=kinetics[3][kinetics[2].index('ms')][1])
                        if const >= 0:
                            param_str += each + ' = ' + str(const) + '\n'
                            break

                if kinetics[1][kinetics[2].index('ms')] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ms')][0],
                                        s=kinetics[3][kinetics[2].index('ms')][1])
                    param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'trivial':
                param_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ms')][0],
                                    scale=kinetics[3][kinetics[2].index('ms')][1]
                                    - kinetics[3][kinetics[2].index('ms')][0])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('ms')][0],
                                       kinetics[3][kinetics[2].index('ms')][1])
                param_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('ms')][0],
                                     scale=kinetics[3][kinetics[2].index('ms')][1])
                    if const >= 0:
                        param_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ms')][0],
                                    s=kinetics[3][kinetics[2].index('ms')][1])
                param_str += each + ' = ' + str(const) + '\n'

        if ms:
            param_str += '\n'

    Bsource = []
    Bsink = []
    print(source)
    if source_nodes or sink_nodes:
        reaction_index += 1

    if constants == False and source_nodes:
        for each in source_nodes:
            rxn_str += 'J' + str(reaction_index) + ': -> S' + str(each) + '; syn' + str(each) + '\n'
            reaction_index += 1
        rxn_str += '\n'

    if constants == False and sink_nodes:
        for each in sink_nodes:
            rxn_str += 'J' + str(reaction_index) + ': S' + str(each) + ' -> ; ' + 'deg' + str(each) + '*' + 'S' \
                       + str(each) + '\n'
            reaction_index += 1
        rxn_str += '\n'

    if constants == True and source_nodes:
        for each in source_nodes:
            Bsource.append('B' + str(each))
            rxn_str += 'J' + str(reaction_index) + ': B' + str(each) + ' -> S' + str(each) + '; syn' + str(each) \
                       + ' * B' + str(each) + '\n'
            reaction_index += 1
        rxn_str += '\n'

    if constants == True and sink_nodes:
        for each in sink_nodes:
            Bsink.append('B' + str(each))
            rxn_str += 'J' + str(reaction_index) + ': S' + str(each) + ' -> B' + str(each) + '; deg' + str(each) \
                       + ' * S' + str(each) + '\n'
            reaction_index += 1
        rxn_str += '\n'

    for i in range(n_species):
        if ic_params == 'trivial':
            ic_str += 'S' + str(i) + ' = 1\n'
        if isinstance(ic_params, list) and ic_params[0] == 'uniform':
            ic = uniform.rvs(loc=ic_params[1], scale=ic_params[2]-ic_params[1])
            ic_str += 'S' + str(i) + ' = ' + str(ic) + '\n'
        if isinstance(ic_params, list) and ic_params[0] == 'loguniform':
            ic = loguniform.rvs(ic_params[1], ic_params[2])
            ic_str += 'S' + str(i) + ' = ' + str(ic) + '\n'
        if isinstance(ic_params, list) and ic_params[0] == 'normal':
            ic = norm.rvs(loc=ic_params[1], scale=ic_params[2])
            ic_str += 'S' + str(i) + ' = ' + str(ic) + '\n'
        if isinstance(ic_params, list) and ic_params[0] == 'lognormal':
            ic = lognorm.rvs(scale=ic_params[1], s=ic_params[2])
            ic_str += 'S' + str(i) + ' = ' + str(ic) + '\n'
        if isinstance(ic_params, list) and ic_params[0] == 'list':
            ic = ic_params[1][i]
            ic_str += 'S' + str(i) + ' = ' + str(ic) + '\n'
        if ic_params is None:
            ic = uniform.rvs(loc=0, scale=10)
            ic_str += 'S' + str(i) + ' = ' + str(ic) + '\n'

    if len(Bsource) > 0 or len(Bsink) > 0:
        ic_str += '\n'

    for each in Bsource:
        if ic_params == 'trivial':
            ic_str += each + ' = 1\n'
        if isinstance(source, list) and source[1] == 'uniform':
            ic = uniform.rvs(loc=source[2], scale=source[3]-source[3])
            ic_str += each + ' = ' + str(ic) + '\n'
        if isinstance(source, list) and source[1] == 'loguniform':
            ic = loguniform.rvs(source[2], source[3])
            ic_str += each + ' = ' + str(ic) + '\n'
        if isinstance(source, list) and source[1] == 'normal':
            ic = norm.rvs(loc=source[2], scale=source[3])
            ic_str += each + ' = ' + str(ic) + '\n'
        if isinstance(source, list) and source[1] == 'lognormal':
            ic = lognorm.rvs(scale=source[2], s=source[3])
            ic_str += each + ' = ' + str(ic) + '\n'
        if source is None:
            ic = uniform.rvs(loc=0, scale=10)
            ic_str += each + ' = ' + str(ic) + '\n'

    for each in Bsink:
        ic_str += each + ' = 0\n'

    # for each in Bsink:
    #     if sink == 'trivial':
    #         ic_str += each + ' = 1\n'
    #     if isinstance(sink, list) and sink[1] == 'uniform':
    #         ic = uniform.rvs(loc=sink[2], scale=sink[3]-sink[2])
    #         ic_str += each + ' = ' + str(ic) + '\n'
    #     if isinstance(sink, list) and sink[1] == 'loguniform':
    #         ic = loguniform.rvs(sink[2], sink[3])
    #         ic_str += each + ' = ' + str(ic) + '\n'
    #     if isinstance(sink, list) and sink[1] == 'normal':
    #         ic = norm.rvs(loc=sink[2], scale=sink[3])
    #         ic_str += each + ' = ' + str(ic) + '\n'
    #     if isinstance(sink, list) and sink[1] == 'lognormal':
    #         ic = lognorm.rvs(scale=sink[2], s=sink[3])
    #         ic_str += each + ' = ' + str(ic) + '\n'
    #     if sink is None:
    #         ic = uniform.rvs(loc=0, scale=10)
    #         ic_str += each + ' = ' + str(ic) + '\n'

    # if add_enzyme:
    #     ant_str += '\n'
    #     if isinstance(add_enzyme, bool) or add_enzyme == 'trivial':
    #         for index, r in enumerate(reaction_list_copy):
    #             ant_str += 'E' + str(index) + ' = 1\n'
    #
    #     if isinstance(add_enzyme, list):
    #         for index, r in enumerate(reaction_list_copy):
    #             if add_enzyme[0] == 'uniform':
    #                 enz = uniform.rvs(loc=add_enzyme[1], scale=add_enzyme[2]-add_enzyme[1])
    #                 ant_str += 'E' + str(index) + ' = ' + str(enz) + '\n'
    #             if add_enzyme[0] == 'loguniform':
    #                 enz = loguniform.rvs(add_enzyme[1], add_enzyme[2])
    #                 ant_str += 'E' + str(index) + ' = ' + str(enz) + '\n'
    #             if add_enzyme[0] == 'normal':
    #                 enz = norm.rvs(loc=add_enzyme[1], scale=add_enzyme[2])
    #                 ant_str += 'E' + str(index) + ' = ' + str(enz) + '\n'
    #             if add_enzyme[0] == 'lognormal':
    #                 enz = lognorm.rvs(scale=add_enzyme[1], s=add_enzyme[2])
    #                 ant_str += 'E' + str(index) + ' = ' + str(enz) + '\n'

    if add_enzyme:
        ic_str += '\n'
        if isinstance(add_enzyme, bool) or add_enzyme == 'trivial':
            for index, r in enumerate(reaction_list_copy):
                ic_str += 'E' + str(index) + ' = 1\n'

        if isinstance(add_enzyme, list):
            for index, r in enumerate(reaction_list_copy):
                if add_enzyme[0] == 'uniform':
                    enz = uniform.rvs(loc=add_enzyme[1], scale=add_enzyme[2]-add_enzyme[1])
                    ic_str += 'E' + str(index) + ' = ' + str(enz) + '\n'
                if add_enzyme[0] == 'loguniform':
                    enz = loguniform.rvs(add_enzyme[1], add_enzyme[2])
                    ic_str += 'E' + str(index) + ' = ' + str(enz) + '\n'
                if add_enzyme[0] == 'normal':
                    enz = norm.rvs(loc=add_enzyme[1], scale=add_enzyme[2])
                    ic_str += 'E' + str(index) + ' = ' + str(enz) + '\n'
                if add_enzyme[0] == 'lognormal':
                    enz = lognorm.rvs(scale=add_enzyme[1], s=add_enzyme[2])
                    ic_str += 'E' + str(index) + ' = ' + str(enz) + '\n'

    if source_nodes:

        if source == 'trivial':
            for each in source_nodes:
                param_str += 'deg' + str(each) + ' = 1\n'

        if isinstance(source, list):
            for each in source_nodes:
                if source[1] == 'uniform':
                    enz = uniform.rvs(loc=source[2], scale=source[3]-source[2])
                    param_str += 'syn' + str(each) + ' = ' + str(enz) + '\n'
                if source[1] == 'loguniform':
                    enz = loguniform.rvs(source[2], source[3])
                    param_str += 'syn' + str(each) + ' = ' + str(enz) + '\n'
                if source[1] == 'normal':
                    enz = norm.rvs(loc=source[2], scale=source[3])
                    param_str += 'syn' + str(each) + ' = ' + str(enz) + '\n'
                if source[1] == 'lognormal':
                    enz = lognorm.rvs(scale=source[2], s=source[3])
                    param_str += 'syn' + str(each) + ' = ' + str(enz) + '\n'

        param_str += '\n'

    if sink_nodes:

        if sink == 'trivial':
            for each in sink_nodes:
                param_str += 'deg' + str(each) + ' = 1\n'

        if isinstance(sink, list):
            for each in sink_nodes:
                if sink[1] == 'uniform':
                    enz = uniform.rvs(loc=sink[2], scale=sink[3]-sink[2])
                    param_str += 'deg' + str(each) + ' = ' + str(enz) + '\n'
                if sink[1] == 'loguniform':
                    enz = loguniform.rvs(sink[2], sink[3])
                    param_str += 'deg' + str(each) + ' = ' + str(enz) + '\n'
                if sink[1] == 'normal':
                    enz = norm.rvs(loc=sink[2], scale=sink[3])
                    param_str += 'deg' + str(each) + ' = ' + str(enz) + '\n'
                if sink[1] == 'lognormal':
                    enz = lognorm.rvs(scale=sink[2], s=sink[3])
                    param_str += 'deg' + str(each) + ' = ' + str(enz) + '\n'

        param_str += '\n'

    ant_str = rxn_str + param_str + ic_str

    return ant_str, source_nodes, sink_nodes


def generate_simple_linear(n_species):

    reaction_list = []
    edge_list = []
    node_set = set()
    last_products = None

    while True:

        if not node_set:
            reactant = 0
        else:
            reactant = max(last_products)

        product = reactant + 1
        last_products = {product}

        reaction_list.append([0, [reactant], [product], [], [], []])
        edge_list.append([reactant, product])

        node_set.add(reactant)
        node_set.add(product)

        if len(node_set) == n_species:

            break

    reaction_list.insert(0, n_species)
    return reaction_list, edge_list


def generate_simple_cyclic(min_species, max_species, n_cycles):

    reaction_list = []

    node_set = set()
    last_product = None
    cycle_lengths = []
    cycle_nodes = [set()]
    edge_list = []

    for _ in range(n_cycles):
        cycle_lengths.append(random.choice(range(min_species, max_species+1)))

    for i in range(cycle_lengths[0]-1):

        if not node_set:
            reactant = 0
        else:
            reactant = last_product

        product = reactant + 1
        last_product = product

        reaction_list.append([0, [reactant], [product], [], [], []])
        edge_list.append([reactant, product])

        node_set.add(reactant)
        node_set.add(product)
        cycle_nodes[-1].add(reactant)
        cycle_nodes[-1].add(product)

    reaction_list.append([0, [last_product], [0], [], [], []])
    edge_list.append([last_product, 0])

    if len(cycle_lengths) > 1:

        for cycle_length in cycle_lengths[1:]:

            link = random.choice(list(node_set))

            last_product += 1
            reaction_list.append([0, [link], [last_product], [], [], []])
            edge_list.append([link, last_product])

            for _ in range(cycle_length - 2):

                reactant = last_product
                product = reactant + 1
                last_product = product

                reaction_list.append([0, [reactant], [product], [], [], []])
                edge_list.append([reactant, product])

                node_set.add(reactant)
                node_set.add(product)
                cycle_nodes[-1].add(reactant)
                cycle_nodes[-1].add(product)

            reaction_list.append([0, [last_product], [link], [], [], []])
            edge_list.append([last_product, link])

    reaction_list.insert(0, len(node_set))
    return reaction_list, edge_list


def generate_simple_branched(n_species, seeds, path_probs, tips):

    reaction_list = []
    edge_list = []
    node_set = set()

    if not path_probs:
        path_probs = [.25, .5, .25]

    buds = []
    current = 0
    for i in range(seeds):

        buds.append(i)
        node_set.add(i)
        current = i + 1

    grow = True
    if len(node_set) >= n_species:
        grow = False

    if tips:
        stems_dict = defaultdict(int)
        while grow:

            stems_list = []
            for bud in buds:
                if bud in stems_dict and stems_dict[bud] not in stems_list:
                    stems_list.append(stems_dict[bud])

            route = random.choices([0, 1, 2], path_probs)[0]

            if route == 0:

                if len(stems_list) == 0:
                    continue

                reactant = random.choice(stems_list)
                product = current
                reaction_list.append([0, [reactant], [product], [], [], []])
                edge_list.append([reactant, product])
                node_set.add(product)
                buds.append(product)
                stems_dict[product] = reactant
                current += 1

            if route == 1:
                reactant = random.choice(buds)
                product = current
                reaction_list.append([0, [reactant], [product], [], [], []])
                edge_list.append([reactant, product])
                node_set.add(product)
                buds.remove(reactant)
                buds.append(product)
                stems_dict[product] = reactant
                current += 1

            if route == 2:

                if len(buds) == 1:
                    continue
                if len(stems_list) < 2:
                    continue

                reactant = random.choice(buds)
                stem_choices = [stem for stem in stems_list if stem != stems_dict[reactant]]
                product = random.choice(stem_choices)
                reaction_list.append([0, [reactant], [product], [], [], []])
                edge_list.append([reactant, product])
                buds.remove(reactant)

            if len(node_set) == n_species:
                grow = False

    else:
        stems = []
        while grow:

            route = random.choices([0, 1, 2], path_probs)[0]

            if route == 0:

                if len(stems) == 0:
                    continue

                reactant = random.choice(stems)
                product = current
                reaction_list.append([0, [reactant], [product], [], [], []])
                edge_list.append([reactant, product])
                node_set.add(product)
                buds.append(current)
                current += 1

            if route == 1:
                reactant = random.choice(buds)
                product = current
                reaction_list.append([0, [reactant], [product], [], [], []])
                edge_list.append([reactant, product])
                node_set.add(product)
                buds.remove(reactant)
                buds.append(product)
                stems.append(reactant)
                current += 1

            if route == 2:

                if len(buds) == 1:
                    continue

                reactant = random.choice(buds)
                product_selection = set(stems) | set(buds)
                product_selection = list(product_selection - {reactant})
                product = random.choice(product_selection)
                reaction_list.append([0, [reactant], [product], [], [], []])
                edge_list.append([reactant, product])
                buds.remove(reactant)
                stems.append(reactant)

            if len(node_set) == n_species:
                grow = False

    reaction_list.insert(0, n_species)
    return reaction_list, edge_list
