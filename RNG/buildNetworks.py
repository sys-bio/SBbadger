
import random
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
from scipy.stats import norm, lognorm, uniform, loguniform
from collections import defaultdict


# General settings for the package
def restore_default_probabilities():
    """Restore the default settings for the reaction mechanism probabilities"""
    Settings.ReactionProbabilities.UniUni = 0.35
    Settings.ReactionProbabilities.BiUni = 0.3
    Settings.ReactionProbabilities.UniBi = 0.3
    Settings.ReactionProbabilities.BiBI = 0.05


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
         BiBI  = 0.05
         """
        UniUni = 0.35
        BiUni = 0.3
        UniBi = 0.3
        BiBI = 0.05


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


def generate_samples(n_species, in_dist, out_dist, joint_dist, min_node_deg, in_range, out_range,
                     joint_range):

    # todo: expand kinetics?
    # todo: mass balance

    in_samples = []
    out_samples = []
    joint_samples = []

    def single_unbounded_pmf(sdist):
        """Assumes starting degree of 1 and extends until cutoff found"""

        deg = 1
        while True:
            dist = []
            for j in range(deg):
                dist.append(sdist(j + 1))
            distsum = sum(dist)
            dist_n = [x * n_species / distsum for x in dist]

            if dist_n[-1] < min_node_deg:
                pmf0 = dist[:-1]
                sum_dist_f = sum(pmf0)
                pmf0 = [x / sum_dist_f for x in pmf0]
                break
            else:
                deg += 1

        return pmf0

    # todo: generalize out the endpoint tests
    def single_bounded_pmf(sdist, drange):
        """Start with given degree range and trim until cutoffs found"""

        dist_ind = [j for j in range(drange[0], drange[1] + 1)]
        pmf0 = [sdist(j) for j in range(drange[0], drange[1] + 1)]
        dist_sum = min(sum(pmf0), 1)
        pmf0 = [x / dist_sum for x in pmf0]
        dist = [x * n_species / dist_sum for x in pmf0]

        while dist[0] < 1 or dist[-1] < 1:
            if dist[0] < dist[-1]:
                dist_ind.pop(0)
                pmf0.pop(0)
            else:
                dist_ind.pop(-1)
                pmf0.pop(-1)
            dist_sum = sum(pmf0)
            dist = [x * n_species / dist_sum for x in pmf0]
            pmf0 = [x / dist_sum for x in pmf0]
        startdeg = dist_ind[0]

        return pmf0, startdeg

    def sample_single_distribution(pmf0, start_deg0):

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
                samples.append((start_deg0+j, samplest[j]))

        return samples

    def sample_both_pmfs(pmf01, start_deg_01, pmf02, start_deg_02):

        # sample the first distribution
        samples1t = [0 for _ in pmf01]

        ind1 = [j for j in range(len(pmf01))]
        ind2 = [j for j in range(len(pmf02))]

        j = 0
        while j < n_species:
            ind = random.choices(ind1, pmf01)[0]
            samples1t[ind] += 1
            j += 1

        samples1 = []
        for j in range(len(pmf01)):
            if samples1t[j] > 0:
                samples1.append((start_deg_01 + j, samples1t[j]))

        # sample the second distribution so that the number of edges match
        edges1 = 0
        for item in samples1:
            edges1 += item[0] * item[1]
        num_tries = 0

        while True:
            num_tries += 1
            edges2 = 0
            nodes = 0
            samples2t = [0 for _ in pmf02]
            while edges2 < edges1 and nodes < n_species:
                ind = random.choices(ind2, pmf02)[0]
                samples2t[ind] += 1
                edges2 += ind + start_deg_02
                nodes += 1

            if edges2 == edges1:
                samples2 = []
                for j in range(len(pmf02)):
                    if samples2t[j] > 0:
                        samples2.append((start_deg_02 + j, samples2t[j]))
                break

            if num_tries == 10000:
                raise Exception("\nReconciliation of the input and output distributions was attempted 10000 times.\n"
                      "Consider revising these distributions.")

        return samples1, samples2

    def find_edge_count(dist):

        edge_count = 0
        for item in dist:
            edge_count += item[0] * item[1]

        return edge_count

    def find_edges_expected_value(x_dist, x_range):

        edge_ev = 0
        for j, item in enumerate(x_dist):
            if isinstance(x_range, list):
                edge_ev += item * x_range[j] * n_species
            elif isinstance(x_range, int):
                edge_ev += item * (j + x_range) * n_species
            else:
                edge_ev += item * (j+1) * n_species

        return edge_ev

    def trim_pmf(edges1, dist2):

        edges2 = 0
        m_deg = 0
        while edges2 < edges1:

            m_deg += 1
            dist = [dist2(j+1) for j in range(m_deg)]
            sum_dist = sum(dist)
            new_dist = [x/sum_dist for x in dist]
            edge_dist = [new_dist[j]/new_dist[-1] for j in range(len(new_dist))]
            edges2 = 0
            for j, item in enumerate(edge_dist):
                edges2 += item * (j+1)

        dist = [dist2(j+1) for j in range(m_deg-1)]
        sum_dist = sum(dist)
        new_dist = [x/sum_dist for x in dist]

        return new_dist

    def trim_pmf_2(edges_target, pmf0, start_deg0):

        deg_range = [j + start_deg0 for j in range(len(pmf0))]

        edges = 0
        for j, item in enumerate(pmf0):
            edges += item * n_species * deg_range[j]

        while edges > edges_target:

            pmf0.pop(-1)
            sum_pmf = sum(pmf)
            pmf0 = [x/sum_pmf for x in pmf0]
            edges = 0
            for j, item in enumerate(pmf0):
                edges += item * n_species * deg_range[j]

        return pmf0

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

            if any(x < min_node_deg for x in scaled_dscores):
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

            if count == 10000:
                raise Exception("\nYour joint distribution was sampled 10000 times.\n"
                                "Reconciliation of the outgoing and incoming edges was not achieved.\n"
                                "Consider revising this distribution.")

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
            input_case = 11  # todo (maybe): add this (unlikely edge) case
        if in_dist != out_dist and in_range is None and out_range is None:
            input_case = 12
        if in_dist != out_dist and in_range and in_range == out_range:
            input_case = 13
        if in_dist != out_dist and in_range != out_range:
            input_case = 14  # todo (maybe): add this (unlikely edge) case

    if isinstance(out_dist, list) and isinstance(in_dist, list):
        if all(isinstance(x[1], int) for x in out_dist) and all(isinstance(x[1], int) for x in in_dist):
            input_case = 15
        if all(isinstance(x[1], float) for x in out_dist) and all(isinstance(x[1], float) for x in in_dist):
            input_case = 16

    if callable(joint_dist):
        if not joint_range:
            input_case = 17
        if joint_range:
            # todo: include case defining different ranges for outgoing and incoming edges
            input_case = 18

    if isinstance(joint_dist, list):
        if all(isinstance(x[2], float) for x in joint_dist):
            input_case = 19
        if all(isinstance(x[2], int) for x in joint_dist):
            input_case = 20

    # ---------------------------------------------------------------------------

    # print('input_case', input_case)

    if input_case == 1:

        pmf = single_unbounded_pmf(out_dist)
        out_samples = sample_single_distribution(pmf, 1)

    # todo: generalize the ranges
    if input_case == 2:

        pmf, start_deg = single_bounded_pmf(out_dist, out_range)
        out_samples = sample_single_distribution(pmf, start_deg)

    if input_case == 3:

        pmf = [x[1] for x in out_dist]
        # todo: move warning to process file
        if sum(pmf) != 1:
            raise Exception("The PMF does not add to 1")

        start_deg = out_dist[0][0]
        out_samples = sample_single_distribution(pmf, start_deg)

    if input_case == 4:
        out_samples = out_dist

    if input_case == 5:

        pmf = single_unbounded_pmf(in_dist)
        in_samples = sample_single_distribution(pmf, 1)

    if input_case == 6:

        pmf, start_deg = single_bounded_pmf(in_dist, in_range)
        in_samples = sample_single_distribution(pmf, start_deg)

    if input_case == 7:

        pmf = [x[1] for x in in_dist]

        if sum(pmf) != 1:
            raise Exception("The PMF does not add to 1")

        start_deg = in_dist[0][0]
        in_samples = sample_single_distribution(pmf, start_deg)

    if input_case == 8:
        in_samples = in_dist

    if input_case == 9:

        pmf1 = single_unbounded_pmf(out_dist)
        pmf2 = single_unbounded_pmf(in_dist)
        in_or_out = random.randint(0, 1)  # choose which distribution is guaranteed n_species
        if in_or_out:
            in_samples, out_samples = sample_both_pmfs(pmf2, 1, pmf1, 1)
        else:
            out_samples, in_samples = sample_both_pmfs(pmf1, 1, pmf2, 1)

    if input_case == 10:

        pmf1, start_deg1 = single_bounded_pmf(out_dist, out_range)
        pmf2, start_deg2 = single_bounded_pmf(in_dist, in_range)
        in_or_out = random.randint(0, 1)  # choose which distribution is guaranteed n_species
        if in_or_out:
            in_samples, out_samples = sample_both_pmfs(pmf2, start_deg2, pmf1, start_deg1)
        else:
            out_samples, in_samples = sample_both_pmfs(pmf1, start_deg1, pmf2, start_deg2)

    if input_case == 11:

        pass  # todo: unlikely edge case

    if input_case == 12:

        pmf_out = single_unbounded_pmf(out_dist)
        pmf_in = single_unbounded_pmf(in_dist)

        edge_ev_out = find_edges_expected_value(pmf_out, out_range)
        edge_ev_in = find_edges_expected_value(pmf_in, in_range)

        if edge_ev_in < edge_ev_out:
            pmf_out = trim_pmf(edge_ev_in, out_dist)
            in_samples, out_samples = sample_both_pmfs(pmf_in, 1, pmf_out, 1)
        if edge_ev_in > edge_ev_out:
            pmf_in = trim_pmf(edge_ev_out, in_dist)
            out_samples, in_samples = sample_both_pmfs(pmf_out, 1, pmf_in, 1)
        if edge_ev_in == edge_ev_out:
            out_samples, in_samples = sample_both_pmfs(pmf_out, 1, pmf_in, 1)

    if input_case == 13:

        pmf_out, start_deg_out = single_bounded_pmf(out_dist, out_range)
        pmf_in, start_deg_in = single_bounded_pmf(in_dist, in_range)

        edge_ev_out = find_edges_expected_value(pmf_out, start_deg_out)
        edge_ev_in = find_edges_expected_value(pmf_in, start_deg_in)

        if edge_ev_in < edge_ev_out:
            pmf_out = trim_pmf_2(edge_ev_in, pmf_out, start_deg_out)
            in_samples, out_samples = sample_both_pmfs(pmf_in, start_deg_in, pmf_out, start_deg_out)
        if edge_ev_in > edge_ev_out:
            pmf_in = trim_pmf_2(edge_ev_out, pmf_in, start_deg_in)
            out_samples, in_samples = sample_both_pmfs(pmf_out, start_deg_out, pmf_in, start_deg_in)
        if edge_ev_in == edge_ev_out:
            out_samples, in_samples = sample_both_pmfs(pmf_out, start_deg_out, pmf_in, start_deg_in)

    if input_case == 14:

        pass  # todo: unlikely edge case

    if input_case == 15:

        if find_edge_count(out_dist) != find_edge_count(in_dist):
            raise Exception("The edges counts for the input and output distributions must match.")

        out_samples = out_dist
        in_samples = in_dist

    if input_case == 16:

        pmf1 = [x[1] for x in out_dist]
        pmf2 = [x[1] for x in in_dist]

        edge_ev_out = find_edges_expected_value(pmf1, out_dist[0][0])
        edge_ev_in = find_edges_expected_value(pmf2, in_dist[0][0])

        if edge_ev_in < edge_ev_out:
            pmf1 = trim_pmf_2(edge_ev_in, pmf1, out_dist[0][0])
            in_samples, out_samples = sample_both_pmfs(pmf2, in_dist[0][0], pmf1, out_dist[0][0])
        if edge_ev_in > edge_ev_out:
            pmf2 = trim_pmf_2(edge_ev_out, pmf2, in_dist[0][0])
            out_samples, in_samples = sample_both_pmfs(pmf1, out_dist[0][0], pmf2, in_dist[0][0])

    if input_case == 17:

        pmf = joint_unbounded_pmf(joint_dist)
        joint_samples = sample_joint(pmf)

    if input_case == 18:

        pmf = joint_bounded_pmf(joint_dist, joint_range)
        joint_samples = sample_joint(pmf)

    if input_case == 19:

        joint_samples = sample_joint(joint_dist)

    if input_case == 20:

        joint_samples = joint_dist

    # =======================================================================

    # print()
    # print('out_samples')
    # for each in out_samples:
    #     print(each)
    #
    # print()
    # print('in_samples')
    # for each in in_samples:
    #     print(each)
    #
    # print()
    # print('joint_samples')
    # for each in joint_samples:
    #     print(each)
    # print()
    # quit()

    # =======================================================================

    return in_samples, out_samples, joint_samples


def generate_reactions(in_samples, out_samples, joint_samples, n_species, n_reactions, rxn_prob, mod_reg,
                       mass_violating_reactions, edge_type, reaction_type):

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

    in_nodes_list = []
    for i, each in enumerate(in_nodes_count):
        in_nodes_list.append(i)

    out_nodes_list = []
    for i, each in enumerate(out_nodes_count):
        out_nodes_list.append(i)

    if not bool(joint_samples):
        random.shuffle(in_nodes_count)
        random.shuffle(out_nodes_count)

    reaction_list = []
    reaction_list2 = []
    metabolic_edge_list = []

    # todo: finish pick_continued
    # todo: adaptable probabilities
    # ---------------------------------------------------------------------------------------------------

    nodes_list = [i for i in range(n_species)]

    if not bool(out_samples) and not bool(in_samples):

        node_set = set()
        pick_continued = 0
        while True:

            # todo: This is an issue for larger networks: link cutoff with number of species
            if pick_continued == 10000:
                return None, [out_samples, in_samples, joint_samples]

            if rxn_prob:
                rt = _pick_reaction_type(rxn_prob)
            else:
                rt = _pick_reaction_type()

            mod_num = 0
            if mod_reg:
                mod_num = random.choices([0, 1, 2, 3], mod_reg[0])[0]

            # -----------------------------------------------------------------------------

            if rt == TReactionType.UNIUNI:

                product = random.choice(nodes_list)
                reactant = random.choice(nodes_list)

                if [[reactant], [product]] in reaction_list2 or reactant == product:
                    pick_continued += 1
                    continue

                mod_species = random.sample(nodes_list, mod_num)
                reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                reaction_list2.append([[reactant], [product]])

                node_set.add(reactant)
                node_set.add(product)
                node_set.update(mod_species)

                if edge_type == 'metabolic':
                    if reactant != product:
                        metabolic_edge_list.append([reactant, product])

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

                if reaction_type == 'metabolic' and product in {reactant1, reactant2}:
                    pick_continued += 1
                    continue

                mod_species = random.sample(nodes_list, mod_num)
                reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                reaction_list2.append([[reactant1, reactant2], [product]])

                node_set.add(reactant1)
                node_set.add(reactant2)
                node_set.add(product)
                node_set.update(mod_species)

                if edge_type == 'metabolic':
                    if reactant1 != reactant2 and reactant1 != product and reactant2 != product:
                        metabolic_edge_list.append([reactant1, product])
                        metabolic_edge_list.append([reactant2, product])
                    if reactant1 == reactant2 and reactant1 != product:
                        metabolic_edge_list.append([reactant1, product])
                    if reactant1 != reactant2 and reactant1 == product:
                        metabolic_edge_list.append([reactant2, 'deg'])
                    if reactant1 != reactant2 and reactant2 == product:
                        metabolic_edge_list.append([reactant1, 'deg'])
                    if reactant1 == reactant2 and reactant1 == product:
                        metabolic_edge_list.append([reactant1, 'deg'])

            if rt == TReactionType.UNIBI:

                reactant = random.choice(nodes_list)
                product1 = random.choice(nodes_list)
                product2 = random.choice(nodes_list)

                if [[reactant], [product1, product2]] in reaction_list2:
                    pick_continued += 1
                    continue

                if not mass_violating_reactions and reactant in {product1, product2}:
                    pick_continued += 1
                    continue

                if reaction_type == 'metabolic' and reactant in {product1, product2}:
                    pick_continued += 1
                    continue

                mod_species = random.sample(nodes_list, mod_num)
                reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                reaction_list2.append([[reactant], [product1, product2]])

                node_set.add(reactant)
                node_set.add(product1)
                node_set.add(product2)
                node_set.update(mod_species)

                if edge_type == 'metabolic':
                    if reactant != product1 and reactant != product2 and product1 != product2:
                        metabolic_edge_list.append([reactant, product1])
                        metabolic_edge_list.append([reactant, product2])
                    if reactant != product1 and product1 == product2:
                        metabolic_edge_list.append([reactant, product1])
                    if reactant == product1 and product1 != product2:
                        metabolic_edge_list.append(['syn', product2])
                    if reactant == product2 and product1 != product2:
                        metabolic_edge_list.append(['syn', product1])
                    if reactant == product1 and product1 == product2:
                        metabolic_edge_list.append(['syn', reactant])

            if rt == TReactionType.BIBI:

                product1 = random.choice(nodes_list)
                product2 = random.choice(nodes_list)
                reactant1 = random.choice(nodes_list)
                reactant2 = random.choice(nodes_list)

                if [[reactant1, reactant2], [product1, product2]] in reaction_list2 \
                        or {reactant1, reactant2} == {product1, product2}:
                    pick_continued += 1
                    continue

                if reaction_type == 'metabolic' and {reactant1, reactant2} & {product1, product2}:
                    pick_continued += 1
                    continue

                mod_species = random.sample(nodes_list, mod_num)
                reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                reaction_list.append(
                    [rt, [reactant1, reactant2], [product1, product2], mod_species, reg_signs, reg_type])
                reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                node_set.add(reactant1)
                node_set.add(reactant2)
                node_set.add(product1)
                node_set.add(product2)
                node_set.update(mod_species)

                if edge_type == 'metabolic':

                    if len({reactant1, reactant2, product1, product2}) \
                            == len([reactant1, reactant2, product1, product2]):
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant1, product2])
                        metabolic_edge_list.append([reactant2, product1])
                        metabolic_edge_list.append([reactant2, product2])

                    if reactant1 == reactant2 and \
                            len({reactant1, product1, product2}) == len([reactant1, product1, product2]):
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant1, product2])

                    if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                        metabolic_edge_list.append([reactant1, product1])

                    if product1 == product2 and \
                            len({reactant1, reactant2, product1}) == len([reactant1, reactant2, product1]):
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant2, product1])

                    # ------------------------

                    if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant2, product2])

                    if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant2, product1])

                    if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product2])

                    if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product1])

                    # ------------------------

                    if reactant1 != reactant2 and len({reactant1, product1, product2}) == 1:
                        metabolic_edge_list.append([reactant2, product2])
                    if reactant1 != reactant2 and len({reactant2, product1, product2}) == 1:
                        metabolic_edge_list.append([reactant1, product1])

                    # ------------------------

                    if product1 != product2 and len({reactant1, reactant2, product1}) == 1:
                        metabolic_edge_list.append([reactant2, product2])
                    if product1 != product2 and len({reactant1, reactant2, product2}) == 1:
                        metabolic_edge_list.append([reactant1, product1])

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

            if pick_continued == 1000:
                return None, [out_samples, in_samples, joint_samples]

            if rxn_prob:
                rt = _pick_reaction_type(rxn_prob)
            else:
                rt = _pick_reaction_type()

            mod_num = 0
            if mod_reg:
                mod_num = random.choices([0, 1, 2, 3], mod_reg[0])[0]

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIUNI:

                if edge_type == 'generic':

                    if max(in_nodes_count) < (1 + mod_num):
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product] < (1 + mod_num):
                        product = random.choices(in_nodes_list, prob_in)[0]

                    reactant = random.choice(in_nodes_list)

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    in_nodes_count[product] -= (1 + mod_num)
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                if edge_type == 'metabolic':

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]

                    reactant = random.choice(in_nodes_list)

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    in_nodes_count[product] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                    metabolic_edge_list.append([reactant, product])

            # -----------------------------------------------------------------

            if rt == TReactionType.BIUNI:

                if edge_type == 'generic':

                    if max(in_nodes_count) < (2 + mod_num):
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product] < (2 + mod_num):
                        product = random.choices(in_nodes_list, prob_in)[0]

                    reactant1 = random.choice(in_nodes_list)
                    reactant2 = random.choice(in_nodes_list)

                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    in_nodes_count[product] -= (2 + mod_num)
                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

                # if graph_type == 'hybrid':
                #
                #     if max(in_nodes_count) < (1 + mod_num):
                #         pick_continued += 1
                #         continue
                #
                #     sum_in = sum(in_nodes_count)
                #     prob_in = [x / sum_in for x in in_nodes_count]
                #     product = random.choices(in_nodes_list, prob_in)[0]
                #
                #     reactant1 = random.choice(in_nodes_list)
                #     reactant2 = random.choice(in_nodes_list)
                #
                #     if in_nodes_count[product] < (1 + mod_num):
                #         while len({reactant1, reactant2, product}) == len([reactant1, reactant2, product]):
                #             reactant1 = random.choice(in_nodes_list)
                #             reactant2 = random.choice(in_nodes_list)
                #
                #     if [[reactant1, reactant2], [product]] in reaction_list2:
                #         pick_continued += 1
                #         continue
                #
                #     if not mass_violating_reactions and product in {reactant1, reactant2}:
                #         pick_continued += 1
                #         continue
                #
                #     mod_species = random.sample(nodes_list, mod_num)
                #     reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                #     reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]
                #
                #     if len({reactant1, reactant2, product}) == len([reactant1, reactant2, product]):
                #         in_nodes_count[product] -= (2 + mod_num)
                #     if reactant1 == reactant2 and reactant1 != product:
                #         in_nodes_count[product] -= (1 + mod_num)
                #     if reactant1 != reactant2 and reactant1 == product:
                #         in_nodes_count[reactant2] -= (1 + mod_num)
                #     if reactant1 != reactant2 and reactant2 == product:
                #         in_nodes_count[reactant1] -= (1 + mod_num)
                #     if len({reactant1, reactant2, product}) == 1:
                #         in_nodes_count[reactant1] -= (1 + mod_num)
                #
                #     reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                #     reaction_list2.append([[reactant1, reactant2], [product]])
                        
                if edge_type == 'metabolic':

                    reactant1 = random.choice(in_nodes_list)
                    reactant2 = random.choice(in_nodes_list)

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]

                    if in_nodes_count[product] == 1:
                        reactant2 = deepcopy(reactant1)
                    
                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue
                    
                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if reactant1 != reactant2 and reactant1 != product and reactant2 != product:
                        metabolic_edge_list.append([reactant1, product])
                        metabolic_edge_list.append([reactant2, product])
                        in_nodes_count[product] -= 2
                    if reactant1 == reactant2 and reactant1 != product:
                        metabolic_edge_list.append([reactant1, product])
                        in_nodes_count[product] -= 1
                    if reactant1 != reactant2 and reactant1 == product:
                        metabolic_edge_list.append([reactant2, 'deg'])
                    if reactant1 != reactant2 and reactant2 == product:
                        metabolic_edge_list.append([reactant1, 'deg'])
                    if reactant1 == reactant2 and reactant1 == product:
                        metabolic_edge_list.append([reactant1, 'deg'])

                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIBI:

                if edge_type == 'generic':
                    
                    if sum(1 for each in in_nodes_count if each >= (1 + mod_num)) < 2 \
                            and max(in_nodes_count) < (2 + 2 * mod_num):
                        pick_continued += 1
                        continue
    
                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < (1 + mod_num):
                        product1 = random.choices(in_nodes_list, prob_in)[0]
    
                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= (1 + mod_num)
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]
    
                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < (1 + mod_num):
                        product2 = random.choices(in_nodes_list, prob_in_copy)[0]
    
                    reactant = random.choice(in_nodes_list)
    
                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue
    
                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]
    
                    in_nodes_count[product1] -= (1 + mod_num)
                    in_nodes_count[product2] -= (1 + mod_num)
                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

                # if graph_type == 'hybrid':
                #
                #     if max(in_nodes_count) < (1 + mod_num):
                #         pick_continued += 1
                #         continue
                #
                #     sum_in = sum(in_nodes_count)
                #     prob_in = [x / sum_in for x in in_nodes_count]
                #     product1 = random.choices(in_nodes_list, prob_in)[0]
                #
                #     in_nodes_count_copy = deepcopy(in_nodes_count)
                #     in_nodes_count_copy[product1] -= (1 + mod_num)
                #     if max(in_nodes_count_copy[product1]) < (1 + mod_num):
                #         product2 = deepcopy(product1)
                #     else:

                if edge_type == 'metabolic':

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= 1
                    sum_in_copy = sum(in_nodes_count_copy)

                    if sum_in_copy == 0:
                        product2 = deepcopy(product1)
                    else:
                        prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]
                        product2 = random.choices(in_nodes_list, prob_in_copy)[0]

                    reactant = random.choice(nodes_list)

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if reactant != product1 and reactant != product2 and product1 != product2:
                        metabolic_edge_list.append([reactant, product1])
                        metabolic_edge_list.append([reactant, product2])
                        in_nodes_count[product1] -= 1
                        in_nodes_count[product2] -= 1
                    if reactant != product1 and product1 == product2:
                        metabolic_edge_list.append([reactant, product1])
                        in_nodes_count[product1] -= 1
                    if reactant == product1 and product1 != product2:
                        metabolic_edge_list.append(['syn', product2])
                    if reactant == product2 and product1 != product2:
                        metabolic_edge_list.append(['syn', product1])
                    if reactant == product1 and product1 == product2:
                        metabolic_edge_list.append(['syn', reactant])

                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

            # -----------------------------------------------------------------

            if rt == TReactionType.BIBI:

                if edge_type == 'generic':

                    if sum(1 for each in in_nodes_count if each >= (2 + mod_num)) < 2 \
                            and max(in_nodes_count) < (4 + 2 * mod_num):
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product1] < (2 + mod_num):
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= (2 + mod_num)
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < (2 + mod_num):
                        product2 = random.choices(in_nodes_list, prob_in)[0]

                    reactant1 = random.choice(in_nodes_list)
                    reactant2 = random.choice(in_nodes_list)

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and {reactant1, reactant2} & {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    in_nodes_count[product1] -= (2 + mod_num)
                    in_nodes_count[product2] -= (2 + mod_num)
                    reaction_list.append([rt, [reactant1, reactant2], [product1, product2],
                                          mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                if edge_type == 'metabolic':

                    product1 = random.choice(in_nodes_list)
                    product2 = random.choice(in_nodes_list)

                    while (in_nodes_count[product1] + in_nodes_count[product2]) == 0:

                        product1 = random.choice(in_nodes_list)
                        product2 = random.choice(in_nodes_list)

                    reactant1 = random.choice(in_nodes_list)
                    reactant2 = random.choice(in_nodes_list)

                    while True:

                        p1_count = None
                        p2_count = None

                        if len({reactant1, reactant2, product1, product2}) == 4:
                            p1_count = 2
                            p2_count = 2
                        if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                            p1_count = 1
                            p2_count = 1
                        if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                            p1_count = 1
                            p2_count = 1
                        if product1 == product2 and len({reactant1, reactant2, product1}) == 3:
                            p1_count = 2
                            p2_count = 2
                        if reactant1 == product1 or reactant2 == product1 and \
                                len({reactant1, reactant2, product1, product2}) == 3:
                            p1_count = 0
                            p2_count = 1
                        if reactant1 == product2 or reactant2 == product2 and \
                                len({reactant1, reactant2, product1, product2}) == 3:
                            p1_count = 1
                            p2_count = 0
                        if product1 == product2 and len({reactant1, reactant2, product1, product2}) == 2:
                            p1_count = 1
                            p2_count = 1
                        if reactant1 == reactant2 and reactant1 == product1 and reactant1 != product2:
                            p1_count = 0
                            p2_count = 1
                        if reactant1 == reactant2 and reactant1 == product2 and reactant1 != product1:
                            p1_count = 1
                            p2_count = 0

                        if p1_count <= in_nodes_count[product1] and p2_count <= in_nodes_count[product2]:
                            break

                        reactant1 = random.choice(in_nodes_list)
                        reactant2 = random.choice(in_nodes_list)

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and {reactant1, reactant2} & {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if len({reactant1, reactant2, product1, product2}) == 4:
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant1, product2])
                        metabolic_edge_list.append([reactant2, product1])
                        metabolic_edge_list.append([reactant2, product2])
                        in_nodes_count[product1] -= 2
                        in_nodes_count[product2] -= 2

                    if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant1, product2])
                        in_nodes_count[product1] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                        metabolic_edge_list.append([reactant1, product1])
                        in_nodes_count[product1] -= 1

                    if product1 == product2 and \
                            len({reactant1, reactant2, product1}) == len([reactant1, reactant2, product1]):
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant2, product1])
                        in_nodes_count[product1] -= 2

                    # ------------------------

                    if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant2, product2])
                        in_nodes_count[product2] -= 1

                    if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant2, product1])
                        in_nodes_count[product1] -= 1

                    if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product2])
                        in_nodes_count[product2] -= 1

                    if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product1])
                        in_nodes_count[product1] -= 1

                    # ------------------------

                    if reactant1 != reactant2 and len({reactant1, product1, product2}) == 1:
                        metabolic_edge_list.append([reactant2, product2])
                        in_nodes_count[product2] -= 1

                    if reactant1 != reactant2 and len({reactant2, product1, product2}) == 1:
                        metabolic_edge_list.append([reactant1, product1])
                        in_nodes_count[product1] -= 1

                    # ------------------------

                    if product1 != product2 and len({reactant1, reactant2, product1}) == 1:
                        metabolic_edge_list.append([reactant2, product2])
                        in_nodes_count[product2] -= 1

                    if product1 != product2 and len({reactant1, reactant2, product2}) == 1:
                        metabolic_edge_list.append([reactant1, product1])
                        in_nodes_count[product1] -= 1

                    reaction_list.append([rt, [reactant1, reactant2], [product1, product2],
                                          mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

            if sum(in_nodes_count) == 0:
                break

    # -----------------------------------------------------------------

    if bool(out_samples) and not bool(in_samples):

        pick_continued = 0
        while True:
            if pick_continued == 1000:
                return None, [out_samples, in_samples, joint_samples]

            if rxn_prob:
                rt = _pick_reaction_type(rxn_prob)
            else:
                rt = _pick_reaction_type()

            mod_num = 0
            if mod_reg:
                mod_num = random.choices([0, 1, 2, 3], mod_reg[0])[0]

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIUNI:

                if edge_type == 'generic':

                    if sum(out_nodes_count) < (1 + mod_num):
                        pick_continued += 1
                        continue

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]

                    product = random.choice(out_nodes_list)

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    mod_species = []
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

                    out_nodes_count[reactant] -= 1
                    for each in mod_species:
                        out_nodes_count[each] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                if edge_type == 'metabolic':

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]

                    product = random.choice(out_nodes_list)

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    out_nodes_count[product] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                    if reactant != product:
                        metabolic_edge_list.append([reactant, product])

            # -----------------------------------------------------------------

            if rt == TReactionType.BIUNI:

                if edge_type == 'generic':

                    if sum(out_nodes_count) < (2 + mod_num):
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

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    mod_species = []
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

                    out_nodes_count[reactant1] -= 1
                    out_nodes_count[reactant2] -= 1
                    for each in mod_species:
                        out_nodes_count[each] -= 1
                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

                if edge_type == 'metabolic':
                    
                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 1
                    sum_out_copy = sum(out_nodes_count_copy)
                    
                    if sum_out_copy == 0:
                        reactant2 = deepcopy(reactant1)
                    else:
                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                        reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    product = random.choice(nodes_list)
                    
                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]
                    
                    if reactant1 != reactant2 and reactant1 != product and reactant2 != product:
                        metabolic_edge_list.append([reactant1, product])
                        metabolic_edge_list.append([reactant2, product])
                        out_nodes_count[reactant1] -= 1
                        out_nodes_count[reactant2] -= 1
                    if reactant1 == reactant2 and reactant1 != product:
                        metabolic_edge_list.append([reactant1, product])
                        out_nodes_count[reactant1] -= 1
                    if reactant1 != reactant2 and reactant1 == product:
                        metabolic_edge_list.append([reactant2, 'deg'])
                    if reactant1 != reactant2 and reactant2 == product:
                        metabolic_edge_list.append([reactant1, 'deg'])
                    if reactant1 == reactant2 and reactant1 == product:
                        metabolic_edge_list.append([reactant1, 'deg'])

                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIBI:

                if edge_type == 'generic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= (1 + mod_num):
                        cont = True
                    if sum(1 for each in out_nodes_count if each >= 2) >= (mod_num - 1) \
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

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = []
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

                    out_nodes_count[reactant] -= 2
                    for each in mod_species:
                        out_nodes_count[each] -= 2
                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

                if edge_type == 'metabolic':
                    
                    product1 = random.choice(out_nodes_list)
                    product2 = random.choice(out_nodes_list)

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]

                    if out_nodes_count[reactant] == 1:
                        product2 = deepcopy(product1)

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if reactant != product1 and reactant != product2 and product1 != product2:
                        metabolic_edge_list.append([reactant, product1])
                        metabolic_edge_list.append([reactant, product2])
                        out_nodes_count[reactant] -= 2
                    if reactant != product1 and product1 == product2:
                        metabolic_edge_list.append([reactant, product1])
                        out_nodes_count[reactant] -= 1
                    if reactant == product1 and product1 != product2:
                        metabolic_edge_list.append(['syn', product2])
                    if reactant == product2 and product1 != product2:
                        metabolic_edge_list.append(['syn', product1])
                    if reactant == product1 and product1 == product2:
                        metabolic_edge_list.append(['syn', reactant])

                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])
                    
            # -----------------------------------------------------------------

            if rt == TReactionType.BIBI:

                if edge_type == 'generic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= (2 + mod_num):
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= mod_num \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= (mod_num - 2) \
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

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic':

                        if {reactant1, reactant2} & {product1, product2}:
                            pick_continued += 1
                            continue
                        if len({reactant1, reactant2}) == 1 and len({product1, product2}) == 1:
                            pick_continued += 1
                            continue

                    mod_species = []
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

                    out_nodes_count[reactant1] -= 2
                    out_nodes_count[reactant2] -= 2
                    for each in mod_species:
                        out_nodes_count[each] -= 2
                    reaction_list.append(
                        [rt, [reactant1, reactant2], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                if edge_type == 'metabolic':
                    
                    reactant1 = random.choice(out_nodes_list)
                    reactant2 = random.choice(out_nodes_list)

                    while (out_nodes_count[reactant1] + out_nodes_count[reactant2]) == 0:

                        reactant1 = random.choice(out_nodes_list)
                        reactant2 = random.choice(out_nodes_list)

                    product1 = random.choice(out_nodes_list)
                    product2 = random.choice(out_nodes_list)

                    while True:

                        r1_count = None
                        r2_count = None

                        if len({reactant1, reactant2, product1, product2}) == 4:
                            r1_count = 2
                            r2_count = 2
                        if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                            r1_count = 2
                            r2_count = 2
                        if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                            r1_count = 1
                            r2_count = 1
                        if product1 == product2 and len({reactant1, reactant2, product1}) == 3:
                            r1_count = 1
                            r2_count = 1
                        if reactant1 == product1 or reactant1 == product2 and \
                                len({reactant1, reactant2, product1, product2}) == 3:
                            r1_count = 0
                            r2_count = 1
                        if reactant2 == product1 or reactant2 == product2 and \
                                len({reactant1, reactant2, product1, product2}) == 3:
                            r1_count = 1
                            r2_count = 0
                        if product1 == product2 and product1 == reactant1 and product1 != reactant2:
                            r1_count = 0
                            r2_count = 1
                        if product1 == product2 and product1 != reactant1 and product1 == reactant2:
                            r1_count = 1
                            r2_count = 0
                        if reactant1 == reactant2 and len({reactant1, reactant2, product1, product2}) == 2:
                            r1_count = 1
                            r2_count = 1

                        if r1_count <= out_nodes_count[reactant1] and r2_count <= out_nodes_count[reactant2]:
                            break

                        product1 = random.choice(out_nodes_list)
                        product2 = random.choice(out_nodes_list)

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic':

                        if {reactant1, reactant2} & {product1, product2}:
                            pick_continued += 1
                            continue
                        if len({reactant1, reactant2}) == 1 and len({product1, product2}) == 1:
                            pick_continued += 1
                            continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    # ================================================

                    if len({reactant1, reactant2, product1, product2}) == 4:
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant1, product2])
                        metabolic_edge_list.append([reactant2, product1])
                        metabolic_edge_list.append([reactant2, product2])
                        out_nodes_count[reactant1] -= 2
                        out_nodes_count[reactant2] -= 2

                    if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant1, product2])
                        out_nodes_count[reactant1] -= 2

                    if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                        metabolic_edge_list.append([reactant1, product1])
                        out_nodes_count[reactant1] -= 1

                    if product1 == product2 and \
                            len({reactant1, reactant2, product1}) == len([reactant1, reactant2, product1]):
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant2, product1])
                        out_nodes_count[reactant1] -= 1
                        out_nodes_count[reactant2] -= 1

                    # ------------------------

                    if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant2, product2])
                        out_nodes_count[reactant2] -= 1

                    if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant2, product1])
                        out_nodes_count[reactant2] -= 1

                    if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product2])
                        out_nodes_count[reactant1] -= 1

                    if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product1])
                        out_nodes_count[reactant1] -= 1

                    # ------------------------

                    if reactant1 != reactant2 and len({reactant1, product1, product2}) == 1:
                        metabolic_edge_list.append([reactant2, product2])
                        out_nodes_count[reactant2] -= 1

                    if reactant1 != reactant2 and len({reactant2, product1, product2}) == 1:
                        metabolic_edge_list.append([reactant1, product1])
                        out_nodes_count[reactant1] -= 1

                    # ------------------------

                    if product1 != product2 and len({reactant1, reactant2, product1}) == 1:
                        metabolic_edge_list.append([reactant2, product2])
                        out_nodes_count[reactant2] -= 1

                    if product1 != product2 and len({reactant1, reactant2, product2}) == 1:
                        metabolic_edge_list.append([reactant1, product1])
                        out_nodes_count[reactant1] -= 1

                    reaction_list.append([rt, [reactant1, reactant2], [product1, product2],
                                          mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

            if sum(out_nodes_count) == 0:
                break

    # -----------------------------------------------------------------

    if (bool(out_samples) and bool(in_samples)) or bool(joint_samples):
        pick_continued = 0
        while True:

            if pick_continued == 1000:
                return None, [out_samples, in_samples, joint_samples]

            if rxn_prob:
                rt = _pick_reaction_type(rxn_prob)
            else:
                rt = _pick_reaction_type()

            mod_num = 0
            if mod_reg:
                mod_num = random.choices([0, 1, 2, 3], mod_reg[0])[0]

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIUNI:

                if edge_type == 'generic':

                    if sum(out_nodes_count) < (1 + mod_num):
                        pick_continued += 1
                        continue

                    if max(in_nodes_count) < (1 + mod_num):
                        pick_continued += 1
                        continue

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]
                    while in_nodes_count[product] < (1 + mod_num):
                        product = random.choices(in_nodes_list, prob_in)[0]

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]

                    if [[reactant], [product]] in reaction_list2 or reactant == product:
                        pick_continued += 1
                        continue

                    mod_species = []
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

                    in_nodes_count[product] -= (1 + mod_num)
                    out_nodes_count[reactant] -= 1
                    for each in mod_species:
                        out_nodes_count[each] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

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

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    out_nodes_count[reactant] -= 1
                    in_nodes_count[product] -= 1
                    reaction_list.append([rt, [reactant], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product]])

                    metabolic_edge_list.append([reactant, product])

            # -----------------------------------------------------------------

            if rt == TReactionType.BIUNI:

                if edge_type == 'generic':

                    if sum(out_nodes_count) < (2 + mod_num):
                        pick_continued += 1
                        continue

                    if max(in_nodes_count) < (2 + mod_num):
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
                    while in_nodes_count[product] < (2 + mod_num):
                        product = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    mod_species = []
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

                    in_nodes_count[product] -= (2 + mod_num)
                    out_nodes_count[reactant1] -= 1
                    out_nodes_count[reactant2] -= 1
                    for each in mod_species:
                        out_nodes_count[each] -= 1
                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

                if edge_type == 'metabolic':

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant1 = random.choices(out_nodes_list, prob_out)[0]

                    out_nodes_count_copy = deepcopy(out_nodes_count)
                    out_nodes_count_copy[reactant1] -= 1
                    sum_out_copy = sum(out_nodes_count_copy)

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product = random.choices(in_nodes_list, prob_in)[0]

                    if sum_out_copy == 0 or in_nodes_count[product] == 1:
                        reactant2 = deepcopy(reactant1)
                    else:
                        prob_out_copy = [x / sum_out_copy for x in out_nodes_count_copy]
                        reactant2 = random.choices(out_nodes_list, prob_out_copy)[0]

                    if [[reactant1, reactant2], [product]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if reactant1 != reactant2 and reactant1 != product and reactant2 != product:
                        metabolic_edge_list.append([reactant1, product])
                        metabolic_edge_list.append([reactant2, product])
                        out_nodes_count[reactant1] -= 1
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product] -= 2
                    if reactant1 == reactant2 and reactant1 != product:
                        metabolic_edge_list.append([reactant1, product])
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product] -= 1
                    if reactant1 != reactant2 and reactant1 == product:
                        metabolic_edge_list.append([reactant2, 'deg'])
                    if reactant1 != reactant2 and reactant2 == product:
                        metabolic_edge_list.append([reactant1, 'deg'])
                    if reactant1 == reactant2 and reactant1 == product:
                        metabolic_edge_list.append([reactant1, 'deg'])

                    reaction_list.append([rt, [reactant1, reactant2], [product], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product]])

            # -----------------------------------------------------------------

            if rt == TReactionType.UNIBI:

                if edge_type == 'generic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= (1 + mod_num):
                        cont = True
                    if sum(1 for each in out_nodes_count if each >= 2) >= (mod_num - 1) \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True
                    if not cont:
                        pick_continued += 1
                        continue

                    if sum(1 for each in in_nodes_count if each >= (1 + mod_num)) < 2 \
                            and max(in_nodes_count) < (2 + 2 * mod_num):
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
                    while in_nodes_count[product1] < (1 + mod_num):
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= (1 + mod_num)
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < (1 + mod_num):
                        product2 = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = []
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

                    out_nodes_count[reactant] -= 2
                    in_nodes_count[product1] -= (1 + mod_num)
                    in_nodes_count[product2] -= (1 + mod_num)
                    for each in mod_species:
                        out_nodes_count[each] -= 2
                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

                if edge_type == 'metabolic':

                    sum_out = sum(out_nodes_count)
                    prob_out = [x / sum_out for x in out_nodes_count]
                    reactant = random.choices(out_nodes_list, prob_out)[0]

                    sum_in = sum(in_nodes_count)
                    prob_in = [x / sum_in for x in in_nodes_count]
                    product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= 1
                    sum_in_copy = sum(in_nodes_count_copy)

                    if sum_in_copy == 0 or out_nodes_count[reactant] == 1:
                        product2 = deepcopy(product1)
                    else:
                        prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]
                        product2 = random.choices(in_nodes_list, prob_in_copy)[0]

                    if [[reactant], [product1, product2]] in reaction_list2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    if reactant != product1 and reactant != product2 and product1 != product2:
                        metabolic_edge_list.append([reactant, product1])
                        metabolic_edge_list.append([reactant, product2])
                        out_nodes_count[reactant] -= 2
                        in_nodes_count[product1] -= 1
                        in_nodes_count[product2] -= 1
                    if reactant != product1 and product1 == product2:
                        metabolic_edge_list.append([reactant, product1])
                        out_nodes_count[reactant] -= 1
                        in_nodes_count[product1] -= 1
                    if reactant == product1 and product1 != product2:
                        metabolic_edge_list.append(['syn', product2])
                    if reactant == product2 and product1 != product2:
                        metabolic_edge_list.append(['syn', product1])
                    if reactant == product1 and product1 == product2:
                        metabolic_edge_list.append(['syn', reactant])

                    reaction_list.append([rt, [reactant], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant], [product1, product2]])

            # -----------------------------------------------------------------

            if rt == TReactionType.BIBI:

                if edge_type == 'generic':

                    cont = False
                    if sum(1 for each in out_nodes_count if each >= 2) >= (2 + mod_num):
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= mod_num \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 1:
                        cont = True

                    if sum(1 for each in out_nodes_count if each >= 2) >= (mod_num - 2) \
                            and sum(1 for each in out_nodes_count if each >= 4) >= 2:
                        cont = True

                    if not cont:
                        pick_continued += 1
                        continue

                    if sum(1 for each in in_nodes_count if each >= (2 + mod_num)) < 2 \
                            and max(in_nodes_count) < (4 + 2 * mod_num):
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
                    while in_nodes_count[product1] < (2 + mod_num):
                        product1 = random.choices(in_nodes_list, prob_in)[0]

                    in_nodes_count_copy = deepcopy(in_nodes_count)
                    in_nodes_count_copy[product1] -= (2 + mod_num)
                    sum_in_copy = sum(in_nodes_count_copy)
                    prob_in_copy = [x / sum_in_copy for x in in_nodes_count_copy]

                    product2 = random.choices(in_nodes_list, prob_in_copy)[0]
                    while in_nodes_count_copy[product2] < (2 + mod_num):
                        product2 = random.choices(in_nodes_list, prob_in)[0]

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and {reactant1, reactant2} & {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = []
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

                    out_nodes_count[reactant1] -= 2
                    out_nodes_count[reactant2] -= 2
                    in_nodes_count[product1] -= (2 + mod_num)
                    in_nodes_count[product2] -= (2 + mod_num)
                    for each in mod_species:
                        out_nodes_count[each] -= 2
                    reaction_list.append(
                        [rt, [reactant1, reactant2], [product1, product2], mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

                if edge_type == 'metabolic':

                    reactant1 = random.choice(out_nodes_list)
                    reactant2 = random.choice(out_nodes_list)

                    product1 = random.choice(in_nodes_list)
                    product2 = random.choice(in_nodes_list)

                    # todo: must be a more efficient way to do this
                    while (out_nodes_count[reactant1] + out_nodes_count[reactant2]) == 0:

                        reactant1 = random.choice(out_nodes_list)
                        reactant2 = random.choice(out_nodes_list)

                    while (in_nodes_count[product1] + in_nodes_count[product2]) == 0:

                        product1 = random.choice(in_nodes_list)
                        product2 = random.choice(in_nodes_list)

                    while True:

                        r1_count = None
                        r2_count = None

                        p1_count = None
                        p2_count = None

                        if len({reactant1, reactant2, product1, product2}) == 4:
                            r1_count = 2
                            r2_count = 2
                            p1_count = 2
                            p2_count = 2
                        if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                            r1_count = 2
                            r2_count = 2
                            p1_count = 1
                            p2_count = 1
                        if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                            r1_count = 1
                            r2_count = 1
                            p1_count = 1
                            p2_count = 1
                        if product1 == product2 and len({reactant1, reactant2, product1}) == 3:
                            r1_count = 1
                            r2_count = 1
                            p1_count = 2
                            p2_count = 2
                        if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                            r1_count = 0
                            r2_count = 1
                            p1_count = 0
                            p2_count = 1
                        if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                            r1_count = 0
                            r2_count = 1
                            p1_count = 1
                            p2_count = 0
                        if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                            r1_count = 1
                            r2_count = 0
                            p1_count = 0
                            p2_count = 1
                        if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                            r1_count = 1
                            r2_count = 0
                            p1_count = 1
                            p2_count = 0
                        if product1 == product2 and product1 == reactant1 and product1 != reactant2:
                            r1_count = 0
                            r2_count = 1
                            p1_count = 1
                            p2_count = 1
                        if product1 == product2 and product1 != reactant1 and product1 == reactant2:
                            r1_count = 1
                            r2_count = 0
                            p1_count = 1
                            p2_count = 1
                        if reactant1 == reactant2 and reactant1 == product1 and reactant1 != product2:
                            r1_count = 1
                            r2_count = 1
                            p1_count = 0
                            p2_count = 1
                        if reactant1 == reactant2 and reactant1 != product1 and reactant1 == product2:
                            r1_count = 1
                            r2_count = 1
                            p1_count = 1
                            p2_count = 0

                        if r1_count <= out_nodes_count[reactant1] and r2_count <= out_nodes_count[reactant2] and \
                                p1_count <= in_nodes_count[product1] and p2_count <= in_nodes_count[product2]:
                            break

                        while (out_nodes_list[reactant1] + out_nodes_list[reactant2]) == 0:
                            reactant1 = random.choice(out_nodes_list)
                            reactant2 = random.choice(out_nodes_list)

                        while (in_nodes_count[product1] + in_nodes_count[product2]) == 0:
                            product1 = random.choice(in_nodes_list)
                            product2 = random.choice(in_nodes_list)

                    if [[reactant1, reactant2], [product1, product2]] in reaction_list2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    if reaction_type == 'metabolic' and {reactant1, reactant2} & {product1, product2}:
                        pick_continued += 1
                        continue

                    mod_species = random.sample(nodes_list, mod_num)
                    reg_signs = [random.choices([1, -1], [mod_reg[1], 1 - mod_reg[1]])[0] for _ in mod_species]
                    reg_type = [random.choices(['a', 's'], [mod_reg[2], 1 - mod_reg[2]])[0] for _ in mod_species]

                    # ================================================

                    if len({reactant1, reactant2, product1, product2}) == 4:
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant1, product2])
                        metabolic_edge_list.append([reactant2, product1])
                        metabolic_edge_list.append([reactant2, product2])
                        out_nodes_count[reactant1] -= 2
                        out_nodes_count[reactant2] -= 2
                        in_nodes_count[product1] -= 2
                        in_nodes_count[product2] -= 2

                    if reactant1 == reactant2 and len({reactant1, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant1, product2])
                        out_nodes_count[reactant1] -= 2
                        in_nodes_count[product1] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant1 == reactant2 and product1 == product2 and reactant1 != product1:
                        metabolic_edge_list.append([reactant1, product1])
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product1] -= 1

                    if product1 == product2 and \
                            len({reactant1, reactant2, product1}) == len([reactant1, reactant2, product1]):
                        metabolic_edge_list.append([reactant1, product1])
                        metabolic_edge_list.append([reactant2, product1])
                        out_nodes_count[reactant1] -= 1
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product1] -= 2

                    # ------------------------

                    if reactant1 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant2, product2])
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant1 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant2, product1])
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product1] -= 1

                    if reactant2 == product1 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product2])
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant2 == product2 and len({reactant1, reactant2, product1, product2}) == 3:
                        metabolic_edge_list.append([reactant1, product1])
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product1] -= 1

                    # ------------------------

                    if reactant1 != reactant2 and len({reactant1, product1, product2}) == 1:
                        metabolic_edge_list.append([reactant2, product2])
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product2] -= 1

                    if reactant1 != reactant2 and len({reactant2, product1, product2}) == 1:
                        metabolic_edge_list.append([reactant1, product1])
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product1] -= 1

                    # ------------------------

                    if product1 != product2 and len({reactant1, reactant2, product1}) == 1:
                        metabolic_edge_list.append([reactant2, product2])
                        out_nodes_count[reactant2] -= 1
                        in_nodes_count[product2] -= 1

                    if product1 != product2 and len({reactant1, reactant2, product2}) == 1:
                        metabolic_edge_list.append([reactant1, product1])
                        out_nodes_count[reactant1] -= 1
                        in_nodes_count[product1] -= 1

                    reaction_list.append([rt, [reactant1, reactant2], [product1, product2],
                                          mod_species, reg_signs, reg_type])
                    reaction_list2.append([[reactant1, reactant2], [product1, product2]])

            if sum(in_nodes_count) == 0:
                break

    reaction_list.insert(0, n_species)
    return reaction_list


# Includes boundary and floating species
# Returns a list:
# [New Stoichiometry matrix, list of floatingIds, list of boundaryIds]
# On entry, reaction_list has the structure:
# reaction_list = [numSpecies, reaction, reaction, ....]
# reaction = [reactionType, [list of reactants], [list of products], rateConstant]


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


def get_antimony_script(reaction_list, ic_params, kinetics, rev_prob, add_enzyme):

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

    dims = st.shape

    # n_species = dims[0]
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

    enzyme = ''
    enzyme_end = ''
    if add_enzyme:
        enzyme = 'E*('
        enzyme_end = ')'

    # Remove the first element which is the n_species
    reaction_list_copy = deepcopy(reaction_list)
    reaction_list_copy.pop(0)

    ant_str = ''
    if len(floating_ids) > 0:
        ant_str += 'var ' + 'S' + str(floating_ids[0])
        for index in floating_ids[1:]:
            ant_str += ', ' + 'S' + str(index)
        ant_str += '\n'

    if 'modular' in kinetics[0]:
        for each in reaction_list_copy:
            for item in each[3]:
                if item not in boundary_ids and item not in floating_ids:
                    boundary_ids.append(item)

    if len(boundary_ids) > 0:
        ant_str += 'ext ' + 'S' + str(boundary_ids[0])
        for index in boundary_ids[1:]:
            ant_str += ', ' + 'S' + str(index)
        ant_str += '\n'
    ant_str += '\n'

    def reversibility(rxn_type):

        rev1 = False
        if rev_prob and isinstance(rev_prob, list):
            rev1 = random.choices([True, False], [rev_prob[rxn_type], 1.0 - rev_prob[rxn_type]])[0]
        if isinstance(rev_prob, float) or isinstance(rev_prob, int):
            rev1 = random.choices([True, False], [rev_prob, 1 - rev_prob])[0]

        return rev1

    if kinetics[0] == 'mass_action':

        if len(kinetics[2]) == 3 or len(kinetics[2]) == 4:

            kf = []
            kr = []
            kc = []

            reaction_index = None
            for reaction_index, r in enumerate(reaction_list_copy):

                ant_str += 'J' + str(reaction_index) + ': '
                if r[0] == TReactionType.UNIUNI:
                    # UniUni
                    ant_str += 'S' + str(r[1][0])
                    ant_str += ' -> '
                    ant_str += 'S' + str(r[2][0])

                    rev = reversibility(0)
                    if not rev:
                        ant_str += '; ' + enzyme + 'kc' + str(reaction_index) + '*S' + str(r[1][0]) + enzyme_end
                        kc.append('kc' + str(reaction_index))

                    else:
                        ant_str += '; ' + enzyme + 'kf' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + ' - kr' + str(reaction_index) + '*S' + str(r[2][0]) + enzyme_end
                        kf.append('kf' + str(reaction_index))
                        kr.append('kr' + str(reaction_index))

                if r[0] == TReactionType.BIUNI:
                    # BiUni
                    ant_str += 'S' + str(r[1][0])
                    ant_str += ' + '
                    ant_str += 'S' + str(r[1][1])
                    ant_str += ' -> '
                    ant_str += 'S' + str(r[2][0])

                    rev = reversibility(1)
                    if not rev:
                        ant_str += '; ' + enzyme + 'kc' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + enzyme_end
                        kc.append('kc' + str(reaction_index))

                    else:
                        ant_str += '; ' + enzyme + 'kf' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr' + str(reaction_index) + '*S' \
                                 + str(r[2][0]) + enzyme_end
                        kf.append('kf' + str(reaction_index))
                        kr.append('kr' + str(reaction_index))

                if r[0] == TReactionType.UNIBI:
                    # UniBi
                    ant_str += 'S' + str(r[1][0])
                    ant_str += ' -> '
                    ant_str += 'S' + str(r[2][0])
                    ant_str += ' + '
                    ant_str += 'S' + str(r[2][1])

                    rev = reversibility(2)
                    if not rev:
                        ant_str += '; ' + enzyme + 'kc' + str(reaction_index) + '*S' + str(r[1][0]) + enzyme_end
                        kc.append('kc' + str(reaction_index))

                    else:
                        ant_str += '; ' + enzyme + 'kf' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + ' - kr' + str(reaction_index) + '*S' + str(r[2][0]) \
                                 + '*S' + str(r[2][1]) + enzyme_end
                        kf.append('kf' + str(reaction_index))
                        kr.append('kr' + str(reaction_index))

                if r[0] == TReactionType.BIBI:
                    # BiBi
                    ant_str += 'S' + str(r[1][0])
                    ant_str += ' + '
                    ant_str += 'S' + str(r[1][1])
                    ant_str += ' -> '
                    ant_str += 'S' + str(r[2][0])
                    ant_str += ' + '
                    ant_str += 'S' + str(r[2][1])

                    rev = reversibility(3)
                    if not rev:
                        ant_str += '; ' + enzyme + 'kc' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + enzyme_end
                        kc.append('kc' + str(reaction_index))

                    else:
                        ant_str += '; ' + enzyme + 'kf' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr' + str(reaction_index) + '*S' \
                                 + str(r[2][0]) + '*S' + str(r[2][1]) + enzyme_end
                        kf.append('kf' + str(reaction_index))
                        kr.append('kr' + str(reaction_index))

                ant_str += '\n'

            parameter_index = None
            if 'deg' in kinetics[2]:
                reaction_index += 1
                parameter_index = reaction_index
                for sp in floating_ids:
                    ant_str += 'J' + str(reaction_index) + ': S' + str(sp) + ' ->; ' + 'k' \
                              + str(reaction_index) + '*' + 'S' + str(sp) + '\n'
                    reaction_index += 1
            ant_str += '\n'

            if kinetics[1] == 'trivial':

                for each in kf:
                    ant_str += each + ' = 1\n'
                for each in kr:
                    ant_str += each + ' = 1\n'
                for each in kc:
                    ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':

                for each in kf:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0], 
                                        scale=kinetics[3][kinetics[2].index('kf')][1] 
                                        - kinetics[3][kinetics[2].index('kf')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                        scale=kinetics[3][kinetics[2].index('kr')][1]
                                        - kinetics[3][kinetics[2].index('kr')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                        scale=kinetics[3][kinetics[2].index('kc')][1]
                                        - kinetics[3][kinetics[2].index('kc')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':

                for each in kf:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0],
                                           kinetics[3][kinetics[2].index('kf')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0],
                                           kinetics[3][kinetics[2].index('kr')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc')][0],
                                           kinetics[3][kinetics[2].index('kc')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':

                for each in kf:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                         scale=kinetics[3][kinetics[2].index('kf')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kr:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                         scale=kinetics[3][kinetics[2].index('kr')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kc:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc')][0],
                                         scale=kinetics[3][kinetics[2].index('kc')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

            if kinetics[1] == 'lognormal':

                for each in kf:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0],
                                        s=kinetics[3][kinetics[2].index('kf')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0],
                                        s=kinetics[3][kinetics[2].index('kr')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc')][0],
                                        s=kinetics[3][kinetics[2].index('kc')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

            if 'deg' in kinetics[2]:
                for _ in floating_ids:

                    if kinetics[1] == 'trivial':
                        ant_str += 'k' + str(parameter_index) + ' = 1\n'

                    if kinetics[1] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                            scale=kinetics[3][kinetics[2].index('deg')][1]
                                            - kinetics[3][kinetics[2].index('deg')][0])
                        ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                    if kinetics[1] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('deg')][0],
                                               kinetics[3][kinetics[2].index('deg')][1])
                        ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                    if kinetics[1] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                             scale=kinetics[3][kinetics[2].index('deg')][1])
                            if const >= 0:
                                ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('deg')][0],
                                            s=kinetics[3][kinetics[2].index('deg')][1])
                        ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                    parameter_index += 1

            ant_str += '\n'

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

            reaction_index = None
            for reaction_index, r in enumerate(reaction_list_copy):

                ant_str += 'J' + str(reaction_index) + ': '
                if r[0] == TReactionType.UNIUNI:
                    # UniUni
                    ant_str += 'S' + str(r[1][0])
                    ant_str += ' -> '
                    ant_str += 'S' + str(r[2][0])

                    rev = reversibility(0)
                    if not rev:
                        ant_str += '; ' + enzyme + 'kc0_' + str(reaction_index) + '*S' + str(r[1][0]) + enzyme_end
                        kc0.append('kc0_' + str(reaction_index))

                    else:
                        ant_str += '; ' + enzyme + 'kf0_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + ' - kr0_' + str(reaction_index) + '*S' + str(r[2][0]) + enzyme_end
                        kf0.append('kf0_' + str(reaction_index))
                        kr0.append('kr0_' + str(reaction_index))

                if r[0] == TReactionType.BIUNI:
                    # BiUni
                    ant_str += 'S' + str(r[1][0])
                    ant_str += ' + '
                    ant_str += 'S' + str(r[1][1])
                    ant_str += ' -> '
                    ant_str += 'S' + str(r[2][0])

                    rev = reversibility(1)
                    if not rev:
                        ant_str += '; ' + enzyme + 'kc1_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + enzyme_end
                        kc1.append('kc1_' + str(reaction_index))

                    else:
                        ant_str += '; ' + enzyme + 'kf1_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr1_' + str(reaction_index) + '*S' \
                                 + str(r[2][0]) + enzyme_end
                        kf1.append('kf1_' + str(reaction_index))
                        kr1.append('kr1_' + str(reaction_index))

                if r[0] == TReactionType.UNIBI:
                    # UniBi
                    ant_str += 'S' + str(r[1][0])
                    ant_str += ' -> '
                    ant_str += 'S' + str(r[2][0])
                    ant_str += ' + '
                    ant_str += 'S' + str(r[2][1])

                    rev = reversibility(2)
                    if not rev:
                        ant_str += '; ' + enzyme + 'kc2_' + str(reaction_index) + '*S' + str(r[1][0]) + enzyme_end
                        kc2.append('kc2_' + str(reaction_index))

                    else:
                        ant_str += '; ' + enzyme + 'kf2_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + ' - kr2_' + str(reaction_index) + '*S' + str(r[2][0]) \
                                 + '*S' + str(r[2][1]) + enzyme_end
                        kf2.append('kf2_' + str(reaction_index))
                        kr2.append('kr2_' + str(reaction_index))

                if r[0] == TReactionType.BIBI:
                    # BiBi
                    ant_str += 'S' + str(r[1][0])
                    ant_str += ' + '
                    ant_str += 'S' + str(r[1][1])
                    ant_str += ' -> '
                    ant_str += 'S' + str(r[2][0])
                    ant_str += ' + '
                    ant_str += 'S' + str(r[2][1])

                    rev = reversibility(3)
                    if not rev:
                        ant_str += '; ' + enzyme + 'kc3_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + enzyme_end
                        kc3.append('kc3_' + str(reaction_index))

                    else:
                        ant_str += '; ' + enzyme + 'kf3_' + str(reaction_index) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr3_' + str(reaction_index) + '*S' \
                                 + str(r[2][0]) + '*S' + str(r[2][1]) + enzyme_end
                        kf3.append('kf3_' + str(reaction_index))
                        kr3.append('kr3_' + str(reaction_index))

                ant_str += '\n'
            ant_str += '\n'

            parameter_index = None
            if 'deg' in kinetics[2]:
                reaction_index += 1
                parameter_index = reaction_index
                for sp in floating_ids:
                    ant_str += 'J' + str(reaction_index) + ': S' + str(sp) + ' ->; ' + 'k' + str(reaction_index) \
                               + '*' + 'S' + str(sp) + '\n'
                    reaction_index += 1
            ant_str += '\n'

            # todo: fix this?
            if kinetics[1] == 'trivial':

                for each in kf0:
                    ant_str += each + ' = 1\n'
                for each in kf1:
                    ant_str += each + ' = 1\n'
                for each in kf2:
                    ant_str += each + ' = 1\n'
                for each in kf3:
                    ant_str += each + ' = 1\n'
                for each in kr0:
                    ant_str += each + ' = 1\n'
                for each in kr1:
                    ant_str += each + ' = 1\n'
                for each in kr2:
                    ant_str += each + ' = 1\n'
                for each in kr3:
                    ant_str += each + ' = 1\n'
                for each in kc0:
                    ant_str += each + ' = 1\n'
                for each in kc1:
                    ant_str += each + ' = 1\n'
                for each in kc2:
                    ant_str += each + ' = 1\n'
                for each in kc3:
                    ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':

                for each in kf0:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf0')][0],
                                        scale=kinetics[3][kinetics[2].index('kf0')][1]
                                        - kinetics[3][kinetics[2].index('kf0')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kf1:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf1')][0],
                                        scale=kinetics[3][kinetics[2].index('kf1')][1]
                                        - kinetics[3][kinetics[2].index('kf1')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kf2:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf2')][0],
                                        scale=kinetics[3][kinetics[2].index('kf2')][1]
                                        - kinetics[3][kinetics[2].index('kf2')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kf3:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf3')][0],
                                        scale=kinetics[3][kinetics[2].index('kf3')][1]
                                        - kinetics[3][kinetics[2].index('kf3')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr0:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr0')][0],
                                        scale=kinetics[3][kinetics[2].index('kr0')][1]
                                        - kinetics[3][kinetics[2].index('kr0')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr1:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr1')][0],
                                        scale=kinetics[3][kinetics[2].index('kr1')][1]
                                        - kinetics[3][kinetics[2].index('kr1')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr2:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr2')][0],
                                        scale=kinetics[3][kinetics[2].index('kr2')][1]
                                        - kinetics[3][kinetics[2].index('kr2')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr3:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr3')][0],
                                        scale=kinetics[3][kinetics[2].index('kr3')][1]
                                        - kinetics[3][kinetics[2].index('kr3')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc0:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc0')][0],
                                        scale=kinetics[3][kinetics[2].index('kc0')][1]
                                        - kinetics[3][kinetics[2].index('kc0')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc1:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc1')][0],
                                        scale=kinetics[3][kinetics[2].index('kc1')][1]
                                        - kinetics[3][kinetics[2].index('kc1')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc2:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc2')][0],
                                        scale=kinetics[3][kinetics[2].index('kc2')][1]
                                        - kinetics[3][kinetics[2].index('kc2')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc3:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc3')][0],
                                        scale=kinetics[3][kinetics[2].index('kc3')][1]
                                        - kinetics[3][kinetics[2].index('kc3')][0])
                    ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':

                for each in kf0:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf0')][0],
                                           kinetics[3][kinetics[2].index('kf0')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kf1:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf1')][0],
                                           kinetics[3][kinetics[2].index('kf1')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kf2:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf2')][0],
                                           kinetics[3][kinetics[2].index('kf2')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kf3:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf3')][0],
                                           kinetics[3][kinetics[2].index('kf3')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr0:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr0')][0],
                                           kinetics[3][kinetics[2].index('kr0')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr1:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr1')][0],
                                           kinetics[3][kinetics[2].index('kr1')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr2:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr2')][0],
                                           kinetics[3][kinetics[2].index('kr2')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr3:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr3')][0],
                                           kinetics[3][kinetics[2].index('kr3')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc0:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc0')][0],
                                           kinetics[3][kinetics[2].index('kc0')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc1:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc1')][0],
                                           kinetics[3][kinetics[2].index('kc1')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc2:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc2')][0],
                                           kinetics[3][kinetics[2].index('kc2')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc3:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc3')][0],
                                           kinetics[3][kinetics[2].index('kc3')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':

                for each in kf0:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf0')][0],
                                         scale=kinetics[3][kinetics[2].index('kf0')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kf1:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf1')][0],
                                         scale=kinetics[3][kinetics[2].index('kf1')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kf2:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf2')][0],
                                         scale=kinetics[3][kinetics[2].index('kf2')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kf3:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf3')][0],
                                         scale=kinetics[3][kinetics[2].index('kf3')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kr0:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr0')][0],
                                         scale=kinetics[3][kinetics[2].index('kr0')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kr1:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr1')][0],
                                         scale=kinetics[3][kinetics[2].index('kr1')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kr2:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr2')][0],
                                         scale=kinetics[3][kinetics[2].index('kr2')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kr3:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr3')][0],
                                         scale=kinetics[3][kinetics[2].index('kr3')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kc0:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc0')][0],
                                         scale=kinetics[3][kinetics[2].index('kc0')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kc1:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc1')][0],
                                         scale=kinetics[3][kinetics[2].index('kc1')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kc2:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc2')][0],
                                         scale=kinetics[3][kinetics[2].index('kc2')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

                for each in kc3:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc3')][0],
                                         scale=kinetics[3][kinetics[2].index('kc3')][1])
                        if const >= 0:
                            ant_str += each + ' = ' + str(const) + '\n'
                            break

            if kinetics[1] == 'lognormal':

                for each in kf0:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf0')][0],
                                        s=kinetics[3][kinetics[2].index('kf0')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kf1:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf1')][0],
                                        s=kinetics[3][kinetics[2].index('kf1')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kf2:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf2')][0],
                                        s=kinetics[3][kinetics[2].index('kf2')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kf3:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf3')][0],
                                        s=kinetics[3][kinetics[2].index('kf3')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr0:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr0')][0],
                                        s=kinetics[3][kinetics[2].index('kr0')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr1:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr1')][0],
                                        s=kinetics[3][kinetics[2].index('kr1')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr2:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr2')][0],
                                        s=kinetics[3][kinetics[2].index('kr2')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kr3:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr3')][0],
                                        s=kinetics[3][kinetics[2].index('kr3')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc0:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc0')][0],
                                        s=kinetics[3][kinetics[2].index('kc0')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc1:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc1')][0],
                                        s=kinetics[3][kinetics[2].index('kc1')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc2:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc2')][0],
                                        s=kinetics[3][kinetics[2].index('kc2')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

                for each in kc3:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc3')][0],
                                        s=kinetics[3][kinetics[2].index('kc3')][1])
                    ant_str += each + ' = ' + str(const) + '\n'

            if 'deg' in kinetics[2]:
                for _ in floating_ids:

                    if kinetics[1] == 'trivial':
                        ant_str += 'k' + str(parameter_index) + ' = 1\n'

                    if kinetics[1] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                            scale=kinetics[3][kinetics[2].index('deg')][1]
                                            - kinetics[3][kinetics[2].index('deg')][0])
                        ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                    if kinetics[1] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('deg')][0],
                                               kinetics[3][kinetics[2].index('deg')][1])
                        ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                    if kinetics[1] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                             scale=kinetics[3][kinetics[2].index('deg')][1])
                            if const >= 0:
                                ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('deg')][0],
                                            s=kinetics[3][kinetics[2].index('deg')][1])
                        ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                    parameter_index += 1

            ant_str += '\n'

    if kinetics[0] == 'hanekom':

        v = []
        keq = []
        k = []
        ks = []
        kp = []

        reaction_index = None
        for reaction_index, r in enumerate(reaction_list_copy):

            v.append('v' + str(reaction_index))
            keq.append('keq' + str(reaction_index))

            ant_str += 'J' + str(reaction_index) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                ant_str += 'S' + str(r[1][0])
                ant_str += ' -> '
                ant_str += 'S' + str(r[2][0])

                rev = reversibility(0)

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')/(1 + S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))

                    if 'k' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ')/(1 + S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ')' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')*(1-(S' \
                            + str(r[2][0]) + '/S' + str(r[1][0]) \
                            + ')/keq' + str(reaction_index) + ')/(1 + S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ' + S' \
                            + str(r[2][0]) + '/kp_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ')' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))
                        kp.append('kp_' + str(reaction_index) + '_' + str(r[2][0]))

                    if 'k' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
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
                ant_str += 'S' + str(r[1][0])
                ant_str += ' + '
                ant_str += 'S' + str(r[1][1])
                ant_str += ' -> '
                ant_str += 'S' + str(r[2][0])

                rev = reversibility(1)

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
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
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
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
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/ks_' \
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
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/k_' \
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
                ant_str += 'S' + str(r[1][0])
                ant_str += ' -> '
                ant_str += 'S' + str(r[2][0])
                ant_str += ' + '
                ant_str += 'S' + str(r[2][1])

                rev = reversibility(2)

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) + ')/(1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))

                    if 'k' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reaction_index) + '_' + str(r[1][0]) + ')/(1 + S' \
                            + str(r[1][0]) + '/k_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')' + enzyme_end
                        k.append('k_' + str(reaction_index) + '_' + str(r[1][0]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
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
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) \
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
                ant_str += 'S' + str(r[1][0])
                ant_str += ' + '
                ant_str += 'S' + str(r[1][1])
                ant_str += ' -> '
                ant_str += 'S' + str(r[2][0])
                ant_str += ' + '
                ant_str += 'S' + str(r[2][1])

                rev = reversibility(3)

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/ks_' \
                                   + str(reaction_index) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/ks_' + str(reaction_index) + '_' + str(r[1][1]) + ')/((1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reaction_index) + '_' + str(r[1][0]) \
                            + ')*(1 + S' + str(r[1][1]) + '/ks_' + str(reaction_index) + '_' \
                            + str(r[1][1]) + '))' + enzyme_end
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][0]))
                        ks.append('ks_' + str(reaction_index) + '_' + str(r[1][1]))

                    if 'k' in kinetics[2]:
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/k_' \
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
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/ks_' \
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
                        ant_str += '; ' + enzyme + 'v' + str(reaction_index) + '*(S' + str(r[1][0]) + '/k_' \
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

            ant_str += '\n'
        ant_str += '\n'

        parameter_index = None
        if 'deg' in kinetics[2]:
            reaction_index += 1
            parameter_index = reaction_index
            for sp in floating_ids:
                ant_str += 'J' + str(reaction_index) + ': S' + str(sp) + ' ->; ' + 'k' + str(reaction_index) \
                           + '*' + 'S' + str(sp) + '\n'
                reaction_index += 1
        ant_str += '\n'

        if kinetics[1] == 'trivial':

            for each in v:
                ant_str += each + ' = 1\n'
            if v:
                ant_str += '\n'
            for each in keq:
                ant_str += each + ' = 1\n'
            if keq:
                ant_str += '\n'
            for each in k:
                ant_str += each + ' = 1\n'
            if k:
                ant_str += '\n'
            for each in ks:
                ant_str += each + ' = 1\n'
            if ks:
                ant_str += '\n'
            for each in kp:
                ant_str += each + ' = 1\n'
            if kp:
                ant_str += '\n'

        if kinetics[1] == 'uniform':

            for each in v:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                    scale=kinetics[3][kinetics[2].index('v')][1]
                                    - kinetics[3][kinetics[2].index('v')][0])
                ant_str += each + ' = ' + str(const) + '\n'
            if v:
                ant_str += '\n'

            for each in keq:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('keq')][0],
                                    scale=kinetics[3][kinetics[2].index('keq')][1]
                                    - kinetics[3][kinetics[2].index('keq')][0])
                ant_str += each + ' = ' + str(const) + '\n'
            if keq:
                ant_str += '\n'

            for each in k:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                    scale=kinetics[3][kinetics[2].index('k')][1]
                                    - kinetics[3][kinetics[2].index('k')][0])
                ant_str += each + ' = ' + str(const) + '\n'
            if k:
                ant_str += '\n'

            for each in ks:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ks')][0],
                                    scale=kinetics[3][kinetics[2].index('ks')][1]
                                    - kinetics[3][kinetics[2].index('ks')][0])
                ant_str += each + ' = ' + str(const) + '\n'
            if ks:
                ant_str += '\n'

            for each in kp:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kp')][0],
                                    scale=kinetics[3][kinetics[2].index('kp')][1]
                                    - kinetics[3][kinetics[2].index('kp')][0])
                ant_str += each + ' = ' + str(const) + '\n'
            if kp:
                ant_str += '\n'

        if kinetics[1] == 'loguniform':

            for each in v:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0],
                                       kinetics[3][kinetics[2].index('v')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if v:
                ant_str += '\n'

            for each in keq:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('keq')][0],
                                       kinetics[3][kinetics[2].index('keq')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if keq:
                ant_str += '\n'

            for each in k:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('k')][0],
                                       kinetics[3][kinetics[2].index('k')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if k:
                ant_str += '\n'

            for each in ks:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('ks')][0],
                                       kinetics[3][kinetics[2].index('ks')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if ks:
                ant_str += '\n'

            for each in kp:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kp')][0],
                                       kinetics[3][kinetics[2].index('kp')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if kp:
                ant_str += '\n'

        if kinetics[1] == 'normal':

            for each in v:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                     scale=kinetics[3][kinetics[2].index('v')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break
            if v:
                ant_str += '\n'

            for each in keq:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('keq')][0],
                                     scale=kinetics[3][kinetics[2].index('keq')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break
            if keq:
                ant_str += '\n'

            for each in k:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('k')][0],
                                     scale=kinetics[3][kinetics[2].index('k')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break
            if k:
                ant_str += '\n'

            for each in ks:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('ks')][0],
                                     scale=kinetics[3][kinetics[2].index('ks')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break
            if ks:
                ant_str += '\n'

            for each in kp:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kp')][0],
                                     scale=kinetics[3][kinetics[2].index('kp')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break
            if kp:
                ant_str += '\n'

        if kinetics[1] == 'lognormal':

            for each in v:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0],
                                    s=kinetics[3][kinetics[2].index('v')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if v:
                ant_str += '\n'

            for each in keq:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('keq')][0],
                                    s=kinetics[3][kinetics[2].index('keq')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if keq:
                ant_str += '\n'

            for each in k:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('k')][0],
                                    s=kinetics[3][kinetics[2].index('k')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if k:
                ant_str += '\n'

            for each in ks:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ks')][0],
                                    s=kinetics[3][kinetics[2].index('ks')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if ks:
                ant_str += '\n'

            for each in kp:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kp')][0],
                                    s=kinetics[3][kinetics[2].index('kp')][1])
                ant_str += each + ' = ' + str(const) + '\n'
            if kp:
                ant_str += '\n'

        if 'deg' in kinetics[2]:
            for _ in floating_ids:

                if kinetics[1] == 'trivial':
                    ant_str += 'k' + str(parameter_index) + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                        scale=kinetics[3][kinetics[2].index('deg')][1]
                                        - kinetics[3][kinetics[2].index('deg')][0])
                    ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('deg')][0],
                                           kinetics[3][kinetics[2].index('deg')][1])
                    ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                         scale=kinetics[3][kinetics[2].index('deg')][1])
                        if const >= 0:
                            ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('deg')][0],
                                        s=kinetics[3][kinetics[2].index('deg')][1])
                    ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                parameter_index += 1

    if kinetics[0] == 'lin_log':

        hs = []

        reaction_index = None
        for reaction_index, r in enumerate(reaction_list_copy):

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

            ant_str += 'J' + str(reaction_index) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                ant_str += 'S' + str(reaction_list_copy[reaction_index][1][0])
                ant_str += ' -> '
                ant_str += 'S' + str(reaction_list_copy[reaction_index][2][0])
                ant_str += '; ' + enzyme[0:2] + 'v' + str(reaction_index) + '*(1'

                rev = reversibility(0)
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            ant_str += ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            ant_str += ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -1:
                            ant_str += ' - ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))

                ant_str += ')'

            if r[0] == TReactionType.BIUNI:
                # BiUni
                ant_str += 'S' + str(reaction_list_copy[reaction_index][1][0])
                ant_str += ' + '
                ant_str += 'S' + str(reaction_list_copy[reaction_index][1][1])
                ant_str += ' -> '
                ant_str += 'S' + str(reaction_list_copy[reaction_index][2][0])
                ant_str += '; ' + enzyme[0:2] + 'v' + str(reaction_index) + '*(1'

                rev = reversibility(1)
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            ant_str += ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if irr_stoic[each] == 2:
                            ant_str += ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            ant_str += ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == 2:
                            ant_str += ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -1:
                            ant_str += ' - ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))

                ant_str += ')'

            if r[0] == TReactionType.UNIBI:
                # UniBi
                ant_str += 'S' + str(reaction_list_copy[reaction_index][1][0])
                ant_str += ' -> '
                ant_str += 'S' + str(reaction_list_copy[reaction_index][2][0])
                ant_str += ' + '
                ant_str += 'S' + str(reaction_list_copy[reaction_index][2][1])
                ant_str += '; ' + enzyme[0:2] + 'v' + str(reaction_index) + '*(1'

                rev = reversibility(2)
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            ant_str += ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if irr_stoic[each] == 2:
                            ant_str += ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            ant_str += ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -1:
                            ant_str += ' - ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -2:
                            ant_str += ' - ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))

                ant_str += ')'

            if r[0] == TReactionType.BIBI:
                # BiBi
                ant_str += 'S' + str(reaction_list_copy[reaction_index][1][0])
                ant_str += ' + '
                ant_str += 'S' + str(reaction_list_copy[reaction_index][1][1])
                ant_str += ' -> '
                ant_str += 'S' + str(reaction_list_copy[reaction_index][2][0])
                ant_str += ' + '
                ant_str += 'S' + str(reaction_list_copy[reaction_index][2][1])
                ant_str += '; ' + enzyme[0:2] + 'v' + str(reaction_index) + '*(1'

                rev = reversibility(3)
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            ant_str += ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if irr_stoic[each] == 2:
                            ant_str += ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            ant_str += ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == 2:
                            ant_str += ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -1:
                            ant_str += ' - ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))
                        if rev_stoic[each] == -2:
                            ant_str += ' - ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reaction_index) \
                                       + ')'
                            hs.append('hs_' + str(each) + '_' + str(reaction_index))

                ant_str += ')'
            ant_str += '\n'
        ant_str += '\n'

        parameter_index = None
        if 'deg' in kinetics[2]:
            reaction_index += 1
            parameter_index = reaction_index
            for sp in floating_ids:
                ant_str += 'J' + str(reaction_index) + ': S' + str(sp) + ' ->; ' + 'k' + str(reaction_index) + '*' \
                           + 'S' + str(sp) + '\n'
                reaction_index += 1
        ant_str += '\n'

        for index, r in enumerate(reaction_list_copy):
            if kinetics[1] == 'trivial':
                ant_str += 'v' + str(index) + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                    scale=kinetics[3][kinetics[2].index('v')][1]
                                    - kinetics[3][kinetics[2].index('v')][0])
                ant_str += 'v' + str(index) + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0],
                                       kinetics[3][kinetics[2].index('v')][1])
                ant_str += 'v' + str(index) + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
                                     scale=kinetics[3][kinetics[2].index('v')][1])
                    if const >= 0:
                        ant_str += 'v' + str(index) + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0],
                                    s=kinetics[3][kinetics[2].index('v')][1])
                ant_str += 'v' + str(index) + ' = ' + str(const) + '\n'

        ant_str += '\n'

        for each in hs:
            if kinetics[1] == 'trivial':
                ant_str += each + ' = 1\n'
            else:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('hs')][0],
                                    scale=kinetics[3][kinetics[2].index('hs')][1]
                                    - kinetics[3][kinetics[2].index('hs')][0])
                ant_str += each + ' = ' + str(const) + '\n'

        if 'deg' in kinetics[2]:
            for _ in floating_ids:

                if kinetics[1] == 'trivial':
                    ant_str += 'k' + str(parameter_index) + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                        scale=kinetics[3][kinetics[2].index('deg')][1]
                                        - kinetics[3][kinetics[2].index('deg')][0])
                    ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('deg')][0],
                                           kinetics[3][kinetics[2].index('deg')][1])
                    ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                         scale=kinetics[3][kinetics[2].index('deg')][1])
                        if const >= 0:
                            ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('deg')][0],
                                        s=kinetics[3][kinetics[2].index('deg')][1])
                    ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                parameter_index += 1
        ant_str += '\n'

        # todo: Save this later. Allow for different distributions types depending on parameter type
        # if kinetics[1] == 'uniform':
        #     const = uniform.rvs(loc=kinetics[3][kinetics[2].index('hs')][0],
        #                         scale=kinetics[3][kinetics[2].index('hs')][1]
        #                         - kinetics[3][kinetics[2].index('hs')][0])
        #     ant_str += each + ' = ' + str(const) + '\n'
        #
        # if kinetics[1] == 'loguniform':
        #     const = uniform.rvs(kinetics[3][kinetics[2].index('hs')][0],
        #                         kinetics[3][kinetics[2].index('hs')][1])
        #     ant_str += each + ' = ' + str(const) + '\n'
        #
        # if kinetics[1] == 'normal':
        #     const = uniform.rvs(loc=kinetics[3][kinetics[2].index('hs')][0],
        #                         scale=kinetics[3][kinetics[2].index('hs')][1])
        #     ant_str += each + ' = ' + str(const) + '\n'
        #
        # if kinetics[1] == 'lognormal':
        #     const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0],
        #                         scale=kinetics[3][kinetics[2].index('v')][1])
        #     ant_str += each + ' = ' + str(const) + '\n'

    if 'modular' in kinetics[0]:

        ma = set()
        kma = set()
        ms = set()
        kms = set()
        ro = set()
        kf = set()
        kr = set()
        m = set()
        km = set()

        reaction_index = None
        for reaction_index, r in enumerate(reaction_list_copy):

            ant_str += 'J' + str(reaction_index) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                ant_str += 'S' + str(r[1][0])
                ant_str += ' -> '
                ant_str += 'S' + str(r[2][0])

                rev = reversibility(0)

                if not rev:
                    ant_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + ' * '
                        if r[5][i] == 'a' and r[4][i] == 1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.add('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.add('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.add('ro_' + str(reaction_index) + '_' + str(reg))

                    ant_str += '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            ant_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                ant_str += ' + '

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end

                    km.add('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    kf.add('kf_' + str(reaction_index))

                else:
                    ant_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + ' * '
                        if r[5][i] == 'a' and r[4][i] == 1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.add('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.add('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.add('ro_' + str(reaction_index) + '_' + str(reg))

                    ant_str += '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' - kr_' \
                        + str(reaction_index) + '*(S' + str(r[2][0]) + '/km_' + str(reaction_index) + '_' \
                        + str(r[2][0]) \
                        + ')^m_' + str(reaction_index) + '_' + str(r[2][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + (1 + S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + (S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            ant_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                ant_str += ' + '

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end

                    km.add('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[2][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[2][0]))
                    kf.add('kf_' + str(reaction_index))
                    kr.add('kr_' + str(reaction_index))

            if r[0] == TReactionType.BIUNI:
                # BiUni
                ant_str += 'S' + str(r[1][0])
                ant_str += ' + '
                ant_str += 'S' + str(r[1][1])
                ant_str += ' -> '
                ant_str += 'S' + str(r[2][0])

                rev = reversibility(1)

                if not rev:
                    ant_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + ' * '
                        if r[5][i] == 'a' and r[4][i] == 1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.add('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.add('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.add('ro_' + str(reaction_index) + '_' + str(reg))

                    ant_str = ant_str \
                        + '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reaction_index) + '_' + str(r[1][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            ant_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                ant_str += ' + '

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end

                    km.add('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[1][1]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][1]))
                    kf.add('kf_' + str(reaction_index))

                else:
                    ant_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + ' * '
                        if r[5][i] == 'a' and r[4][i] == 1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.add('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.add('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.add('ro_' + str(reaction_index) + '_' + str(reg))

                    ant_str = ant_str \
                        + '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reaction_index) + '_' + str(r[1][1]) + ' - kr_' + str(reaction_index) + '*(S' \
                        + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) \
                        + ')^m_' + str(reaction_index) + '_' + str(r[2][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + (1 + S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + (S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + '*(1 + S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + '*(S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            ant_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                ant_str += ' + '

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end

                    km.add('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[1][1]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[2][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][1]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[2][0]))
                    kf.add('kf_' + str(reaction_index))
                    kr.add('kr_' + str(reaction_index))

            if r[0] == TReactionType.UNIBI:
                # UniBi
                ant_str += 'S' + str(r[1][0])
                ant_str += ' -> '
                ant_str += 'S' + str(r[2][0])
                ant_str += ' + '
                ant_str += 'S' + str(r[2][1])

                rev = reversibility(2)

                if not rev:
                    ant_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a' and r[4][i] == 1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.add('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.add('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.add('ro_' + str(reaction_index) + '_' + str(reg))

                    ant_str += '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            ant_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                ant_str += ' + '

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end

                    km.add('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    kf.add('kf_' + str(reaction_index))

                else:
                    ant_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a' and r[4][i] == 1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.add('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.add('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.add('ro_' + str(reaction_index) + '_' + str(reg))

                    ant_str += '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' - kr_' \
                        + str(reaction_index) + '*(S' + str(r[2][0]) + '/km_' + str(reaction_index) + '_' \
                        + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' + str(r[2][0]) + '*(S' + str(r[2][1]) \
                        + '/km_' + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                        + str(r[2][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + (1 + S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + ' + (S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[2][0]) + '/km_' + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][1]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][1]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            ant_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                ant_str += ' + '

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end

                    km.add('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[2][0]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[2][1]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[2][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[2][1]))
                    kf.add('kf_' + str(reaction_index))
                    kr.add('kr_' + str(reaction_index))

            if r[0] == TReactionType.BIBI:
                # BiBi
                ant_str += 'S' + str(r[1][0])
                ant_str += ' + '
                ant_str += 'S' + str(r[1][1])
                ant_str += ' -> '
                ant_str += 'S' + str(r[2][0])
                ant_str += ' + '
                ant_str += 'S' + str(r[2][1])

                rev = reversibility(3)

                if not rev:
                    ant_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a' and r[4][i] == 1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.add('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.add('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.add('ro_' + str(reaction_index) + '_' + str(reg))

                    ant_str = ant_str \
                        + '(kf_' + str(reaction_index) + '*(S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reaction_index) + '_' + str(r[1][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            ant_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                ant_str += ' + '

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end

                    km.add('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[1][1]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][1]))
                    kf.add('kf_' + str(reaction_index))

                else:
                    ant_str += '; ' + enzyme
                    for i, reg in enumerate(r[3]):
                        if r[5][i] == 'a' and r[4][i] == -1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a' and r[4][i] == 1:
                            ant_str += '(' + 'ro_' + str(reaction_index) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reaction_index) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' \
                                + str(reaction_index) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reaction_index) \
                                + '_' + str(reg) + '))^ma_' + str(reaction_index) + '_' + str(reg) + '*'
                        if r[5][i] == 'a':
                            ma.add('ma_' + str(reaction_index) + '_' + str(reg))
                            kma.add('kma_' + str(reaction_index) + '_' + str(reg))
                            ro.add('ro_' + str(reaction_index) + '_' + str(reg))

                    ant_str = ant_str \
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
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + (1 + S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' + str(reaction_index) + '_' \
                            + str(r[2][1]) \
                            + ')^m_' + str(reaction_index) + '_' + str(r[2][1]) + ' - 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'DM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + ' + (S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' + str(reaction_index) + '_' + str(r[2][1]) \
                            + ')^m_' + str(reaction_index) + '_' + str(r[2][1]) + ' + 1)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'SM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += '1 + S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + '*(1 + S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' + str(reaction_index) + '_' \
                            + str(r[2][1]) \
                            + ')^m_' + str(reaction_index) + '_' + str(r[2][1]) + ')'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'FM':

                        if 's' in r[5]:
                            ant_str += '/((('
                        else:
                            ant_str += '/(('

                        ant_str += 'S' + str(r[1][0]) + '/km_' + str(reaction_index) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reaction_index) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reaction_index) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reaction_index) + '_' + str(r[1][1]) + '*(S' + str(r[2][0]) + '/km_' \
                            + str(reaction_index) + '_' + str(r[2][0]) + ')^m_' + str(reaction_index) + '_' \
                            + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' + str(reaction_index) + '_' + str(r[2][1]) \
                            + ')^m_' + str(reaction_index) + '_' + str(r[2][1]) + ')^(1/2)'

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += ' + (S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += ' + (kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end
                        else:
                            ant_str += enzyme_end

                    if kinetics[0][8:10] == 'PM':

                        num_s = r[5].count('s')

                        if 's' in r[5]:
                            ant_str += '/('

                        for i, reg in enumerate(r[3]):

                            if r[5][i] == 's' and r[4][i] == -1:
                                ant_str += '(S' + str(reg) + '/kms_' + str(reaction_index) + '_' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if r[5][i] == 's' and r[4][i] == 1:
                                ant_str += '(kms_' + str(reaction_index) + '_' + str(reg) + '/S' + str(reg) \
                                         + ')^ms_' + str(reaction_index) + '_' + str(reg)

                            if (i + 1) < num_s:
                                ant_str += ' + '

                            if r[5][i] == 's':
                                ms.add('ms_' + str(reaction_index) + '_' + str(reg))
                                kms.add('kms_' + str(reaction_index) + '_' + str(reg))

                        if 's' in r[5]:
                            ant_str += ')' + enzyme_end

                    km.add('km_' + str(reaction_index) + '_' + str(r[1][0]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[1][1]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[2][0]))
                    km.add('km_' + str(reaction_index) + '_' + str(r[2][1]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[1][1]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[2][0]))
                    m.add('m_' + str(reaction_index) + '_' + str(r[2][1]))
                    kf.add('kf_' + str(reaction_index))
                    kr.add('kr_' + str(reaction_index))

            ant_str += '\n'
        ant_str += '\n'

        parameter_index = None
        if 'deg' in kinetics[2]:
            reaction_index += 1
            parameter_index = reaction_index
            for sp in floating_ids:
                ant_str += 'J' + str(reaction_index) + ': S' + str(sp) + ' ->; ' + 'k' + str(reaction_index) + '*' \
                           + 'S' + str(sp) + '\n'
                reaction_index += 1
            ant_str += '\n'

        ro = list(ro)
        ro.sort()
        if kinetics[1] == 'trivial':
            for each in ro:
                ant_str += each + ' = ' + str(1) + '\n'
        else:
            for each in ro:
                ant_str += each + ' = ' + str(uniform.rvs(loc=0, scale=1)) + '\n'
        ant_str += '\n'

        kf = list(kf)
        kf.sort()
        for each in kf:

            if kinetics[1] == 'trivial':
                ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                    scale=kinetics[3][kinetics[2].index('kf')][1]
                                    - kinetics[3][kinetics[2].index('kf')][0])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0],
                                       kinetics[3][kinetics[2].index('kf')][1])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0],
                                     scale=kinetics[3][kinetics[2].index('kf')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0],
                                    s=kinetics[3][kinetics[2].index('kf')][1])
                ant_str += each + ' = ' + str(const) + '\n'

        if kf:
            ant_str += '\n'

        kr = list(kr)
        kr.sort()
        for each in kr:

            if kinetics[1] == 'trivial':
                ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                    scale=kinetics[3][kinetics[2].index('kr')][1]
                                    - kinetics[3][kinetics[2].index('kr')][0])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0],
                                       kinetics[3][kinetics[2].index('kr')][1])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0],
                                     scale=kinetics[3][kinetics[2].index('kr')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0],
                                    s=kinetics[3][kinetics[2].index('kr')][1])
                ant_str += each + ' = ' + str(const) + '\n'

        if kr:
            ant_str += '\n'

        km = list(km)
        km.sort()
        for each in km:

            if kinetics[1] == 'trivial':
                ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                    scale=kinetics[3][kinetics[2].index('km')][1]
                                    - kinetics[3][kinetics[2].index('km')][0])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('km')][0],
                                       kinetics[3][kinetics[2].index('km')][1])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                     scale=kinetics[3][kinetics[2].index('km')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('km')][0],
                                    s=kinetics[3][kinetics[2].index('km')][1])
                ant_str += each + ' = ' + str(const) + '\n'

        if km:
            ant_str += '\n'

        kma = list(kma)
        kma.sort()
        for each in kma:

            if kinetics[1] == 'trivial':
                ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                    scale=kinetics[3][kinetics[2].index('km')][1]
                                    - kinetics[3][kinetics[2].index('km')][0])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('km')][0],
                                       kinetics[3][kinetics[2].index('km')][1])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                     scale=kinetics[3][kinetics[2].index('km')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('km')][0],
                                    s=kinetics[3][kinetics[2].index('km')][1])
                ant_str += each + ' = ' + str(const) + '\n'

        if kma:
            ant_str += '\n'

        kms = list(kms)
        kms.sort()
        for each in kms:

            if kinetics[1] == 'trivial':
                ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                    scale=kinetics[3][kinetics[2].index('km')][1]
                                    - kinetics[3][kinetics[2].index('km')][0])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('km')][0],
                                       kinetics[3][kinetics[2].index('km')][1])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('km')][0],
                                     scale=kinetics[3][kinetics[2].index('km')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('km')][0],
                                    s=kinetics[3][kinetics[2].index('km')][1])
                ant_str += each + ' = ' + str(const) + '\n'

        if kms:
            ant_str += '\n'

        m = list(m)
        m.sort()
        for each in m:

            if kinetics[1] == 'trivial':
                ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('mol')][0],
                                    scale=kinetics[3][kinetics[2].index('mol')][1]
                                    - kinetics[3][kinetics[2].index('mol')][0])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('mol')][0],
                                       kinetics[3][kinetics[2].index('mol')][1])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('mol')][0],
                                     scale=kinetics[3][kinetics[2].index('mol')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('mol')][0],
                                    s=kinetics[3][kinetics[2].index('mol')][1])
                ant_str += each + ' = ' + str(const) + '\n'

        if m:
            ant_str += '\n'

        ma = list(ma)
        ma.sort()
        for each in ma:

            if kinetics[1] == 'trivial':
                ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('mol')][0],
                                    scale=kinetics[3][kinetics[2].index('mol')][1]
                                    - kinetics[3][kinetics[2].index('mol')][0])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('mol')][0],
                                       kinetics[3][kinetics[2].index('mol')][1])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('mol')][0],
                                     scale=kinetics[3][kinetics[2].index('mol')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('mol')][0],
                                    s=kinetics[3][kinetics[2].index('mol')][1])
                ant_str += each + ' = ' + str(const) + '\n'

        if ma:
            ant_str += '\n'

        ms = list(ms)
        ms.sort()
        for each in ms:

            if kinetics[1] == 'trivial':
                ant_str += each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('mol')][0],
                                    scale=kinetics[3][kinetics[2].index('mol')][1]
                                    - kinetics[3][kinetics[2].index('mol')][0])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('mol')][0],
                                       kinetics[3][kinetics[2].index('mol')][1])
                ant_str += each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('mol')][0],
                                     scale=kinetics[3][kinetics[2].index('mol')][1])
                    if const >= 0:
                        ant_str += each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('mol')][0],
                                    s=kinetics[3][kinetics[2].index('mol')][1])
                ant_str += each + ' = ' + str(const) + '\n'

        if ms:
            ant_str += '\n'

        if 'deg' in kinetics[2]:
            for _ in floating_ids:

                if kinetics[1] == 'trivial':
                    ant_str += 'k' + str(parameter_index) + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                        scale=kinetics[3][kinetics[2].index('deg')][1]
                                        - kinetics[3][kinetics[2].index('deg')][0])
                    ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('deg')][0],
                                           kinetics[3][kinetics[2].index('deg')][1])
                    ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('deg')][0],
                                         scale=kinetics[3][kinetics[2].index('deg')][1])
                        if const >= 0:
                            ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('deg')][0],
                                        s=kinetics[3][kinetics[2].index('deg')][1])
                    ant_str += 'k' + str(parameter_index) + ' = ' + str(const) + '\n'

                parameter_index += 1

            ant_str += '\n'

    def get_i_cvalue(ic_ind):

        ic = None
        if ic_params == 'trivial':
            ic = 1
        if isinstance(ic_params, list) and ic_params[0] == 'uniform':
            ic = uniform.rvs(loc=ic_params[1], scale=ic_params[2]-ic_params[1])
        if isinstance(ic_params, list) and ic_params[0] == 'loguniform':
            ic = loguniform.rvs(ic_params[1], ic_params[2])
        if isinstance(ic_params, list) and ic_params[0] == 'normal':
            ic = norm.rvs(loc=ic_params[1], scale=ic_params[2])
        if isinstance(ic_params, list) and ic_params[0] == 'lognormal':
            ic = lognorm.rvs(scale=ic_params[1], s=ic_params[2])
        if isinstance(ic_params, list) and ic_params[0] == 'list':
            ic = ic_params[1][ic_ind]
        if ic_params is None:
            ic = uniform.rvs(loc=0, scale=10)

        return ic

    for index, b in enumerate(boundary_ids):
        i_cvalue = get_i_cvalue(b, )
        ant_str += 'S' + str(b) + ' = ' + str(i_cvalue) + '\n'

    ant_str += '\n'
    for index, b in enumerate(floating_ids):
        i_cvalue = get_i_cvalue(b)
        ant_str += 'S' + str(b) + ' = ' + str(i_cvalue) + '\n'

    if add_enzyme:
        ant_str += '\n'
        for index, r in enumerate(reaction_list_copy):
            ant_str += 'E' + str(index) + ' = 1\n'

    return ant_str
