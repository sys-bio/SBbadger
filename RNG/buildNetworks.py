# -*- coding: utf-8 -*-
""" A module for creating random network models """

import random
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
from scipy.stats import norm, lognorm, uniform, loguniform
from collections import defaultdict

# General settings for the package
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

    def restoreDefaultProbabilities(self):
        """Restore the default settings for the reaction mechanism probabilities"""
        Settings.ReactionProbabilities.UniUni = 0.35
        Settings.ReactionProbabilities.BiUni = 0.3
        Settings.ReactionProbabilities.UniBi = 0.3
        Settings.ReactionProbabilities.BiBI = 0.05


def _getMMRateLaw(k, s1, s2):
    return 'Vm' + str(k) + '/Km' + str(k) + '0*(' + s1 + '-' + s2 + '/Keq' + str(k) + \
           ')/(' + '1 + ' + s1 + '/' + 'Km' + str(k) + '0' + ' + ' \
           + s2 + '/' + 'Km' + str(k) + '1' + ')'


def _getMARateLaw(k, s1, s2):
    return 'k' + str(k) + '0*' + s1 + ' - k' + str(k) + '1' + '*' + s2


@dataclass
class TReactionType:
    UNIUNI = 0
    BIUNI = 1
    UNIBI = 2
    BIBI = 3


def _pickReactionType(prob=None):
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
        if rt < Settings.ReactionProbabilities.UniUni + Settings.ReactionProbabilities.BiUni + Settings.ReactionProbabilities.UniBi:
            return TReactionType.UNIBI
        return TReactionType.BIBI


def _generateReactionList(n_species, n_reactions, kinetics, in_dist, out_dist, joint_dist, min_node_deg,
                          in_range, out_range, joint_range, rxn_prob, allo_reg, spec_reg, mass_violating_reactions):

    # todo: expand kinetics?
    # todo: mass balance

    paramDists = defaultdict(list)
    if kinetics[1] != 'trivial':
        for i, each in enumerate(kinetics[2]):
            paramDists[each] = kinetics[3][i]
    else:
        for i, each in enumerate(kinetics[2]):
            paramDists[each] = [1]

    jointSamples = []
    inSamples = []
    outSamples = []

    def single_unbounded_pmf(sdist):
        """Assumes starting degree of 1 and extends until cutoff found"""

        deg = 1
        while True:
            dist = []
            for j in range(deg):
                dist.append(sdist(j + 1))
            distsum = sum(dist)
            distN = [x * n_species / distsum for x in dist]

            if distN[-1] < min_node_deg:
                pmf0 = dist[:-1]
                sumDistF = sum(pmf0)
                pmf0 = [x / sumDistF for x in pmf0]
                break
            else:
                deg += 1

        return pmf0

    # todo: generalize out the endpoint tests
    def single_bounded_pmf(sdist, drange):
        """Start with given degree range and trim until cutoffs found"""

        distInd = [j for j in range(drange[0], drange[1] + 1)]
        pmf0 = [sdist(j) for j in range(drange[0], drange[1] + 1)]
        distSum = min(sum(pmf0), 1)
        pmf0 = [x / distSum for x in pmf0]
        dist = [x * n_species / distSum for x in pmf0]

        while dist[0] < 1 or dist[-1] < 1:
            if dist[0] < dist[-1]:
                distInd.pop(0)
                pmf0.pop(0)
            else:
                distInd.pop(-1)
                pmf0.pop(-1)
            distSum = sum(pmf0)
            dist = [x * n_species / distSum for x in pmf0]
            pmf0 = [x / distSum for x in pmf0]
        startdeg = distInd[0]

        return pmf0, startdeg

    def sample_single_distribution(pmf0, startDeg0):

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
                samples.append((startDeg0+j, samplest[j]))

        return samples

    def sample_both_pmfs(pmf01, startDeg01, pmf02, startDeg02):

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
                samples1.append((startDeg01 + j, samples1t[j]))

        # sample the second distribution so that the number of edges match
        edges1 = 0
        for item in samples1:
            edges1 += item[0] * item[1]
        numTrys = 0

        while True:
            numTrys += 1
            edges2 = 0
            nodes = 0
            samples2t = [0 for _ in pmf02]
            while edges2 < edges1 and nodes < n_species:
                ind = random.choices(ind2, pmf02)[0]
                samples2t[ind] += 1
                edges2 += ind + startDeg02
                nodes += 1

            if edges2 == edges1:
                samples2 = []
                for j in range(len(pmf02)):
                    if samples2t[j] > 0:
                        samples2.append((startDeg02 + j, samples2t[j]))
                break

            if numTrys == 10000:
                raise Exception("\nReconciliation of the input and output distributions was attempted 10000 times.\n"
                      "Consider revising these distributions.")

        return samples1, samples2

    def find_edge_count(dist):

        edgeCount = 0
        for item in dist:
            edgeCount += item[0] * item[1]

        return edgeCount

    def find_edges_expected_value(xDist, xRange):

        edgeEV = 0
        for j, item in enumerate(xDist):
            if isinstance(xRange, list):
                edgeEV += item * xRange[j] * n_species
            elif isinstance(xRange, int):
                edgeEV += item * (j+xRange) * n_species
            else:
                edgeEV += item * (j+1) * n_species

        return edgeEV

    def trim_pmf(edges1, dist2):

        edges2 = 0
        mDeg = 0
        while edges2 < edges1:

            mDeg += 1
            dist = [dist2(j+1) for j in range(mDeg)]
            sumDist = sum(dist)
            newDist = [x/sumDist for x in dist]
            edgeDist = [newDist[j]/newDist[-1] for j in range(len(newDist))]
            edges2 = 0
            for j, item in enumerate(edgeDist):
                edges2 += item * (j+1)

        dist = [dist2(j+1) for j in range(mDeg-1)]
        sumDist = sum(dist)
        newDist = [x/sumDist for x in dist]

        return newDist

    def trim_pmf_2(edgesTarget, pmf0, startDeg0):

        degRange = [j + startDeg0 for j in range(len(pmf0))]

        edges = 0
        for j, item in enumerate(pmf0):
            edges += item * n_species * degRange[j]

        while edges > edgesTarget:

            pmf0.pop(-1)
            sumPmf = sum(pmf)
            pmf0 = [x/sumPmf for x in pmf0]
            edges = 0
            for j, item in enumerate(pmf0):
                edges += item * n_species * degRange[j]

        return pmf0

    def joint_unbounded_pmf(joint_dist1):

        dist = [(1, 1)]
        dscores = [joint_dist1(1, 1)]
        dsum = dscores[-1]
        edge = []
        edgeScores = []

        while True:

            for item in dist:
                item1 = (item[0] + 1, item[1])
                item2 = (item[0], item[1] + 1)
                item3 = (item[0] + 1, item[1] + 1)
                if item1 not in dist and item1 not in edge:
                    edge.append(item1)
                    edgeScores.append(joint_dist1(item1[0], item1[1]))
                if item2 not in dist and item2 not in edge:
                    edge.append(item2)
                    edgeScores.append(joint_dist1(item2[0], item2[1]))
                if item3 not in dist and item3 not in edge:
                    edge.append(item3)
                    edgeScores.append(joint_dist1(item3[0], item3[1]))

            tiles = []
            lowScore = 0
            for j, item in enumerate(edgeScores):
                if item == lowScore:
                    tiles.append(j)
                elif item > lowScore:
                    tiles = [j]
                    lowScore = item

            newDist = deepcopy(dist)
            newDscores = deepcopy(dscores)

            for j in tiles:
                newDist.append(edge[j])
                newDscores.append(joint_dist1(edge[j][0], edge[j][1]))
                dsum += joint_dist1(edge[j][0], edge[j][1])

            scaledDscores = []
            for item in newDscores:
                scaledDscores.append(n_species * item / dsum)

            if any(x < min_node_deg for x in scaledDscores):
                break

            dist = newDist
            dscores = newDscores

            newEdge = []
            newEdgeScores = []

            for j, item in enumerate(edge):
                if j not in tiles:
                    newEdge.append(item)
                    newEdgeScores.append(edgeScores[j])

            edge = newEdge
            edgeScores = newEdgeScores

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

            outEdges = 0
            inEdges = 0
            samples = []
            for j, item in enumerate(samplest):

                outEdges += item*cells[j][0]
                inEdges += item*cells[j][1]
                samples.append((cells[j][0], cells[j][1], item))

            if outEdges == inEdges:

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
        pmfSum = sum(joint_pmf[j][0] for j in range(len(joint_pmf)))
        joint_pmf = [[joint_pmf[j][0]/pmfSum, joint_pmf[j][0]*n_species/pmfSum, joint_pmf[j][2]] for j in range(len(joint_pmf))]
        joint_pmf.sort(key=lambda x: x[1])
        while joint_pmf[0][1] < min_node_deg:
            value = joint_pmf[0][1]
            joint_pmf = [x for x in joint_pmf if x[1] != value]
            pmfSum = sum(joint_pmf[j][0] for j in range(len(joint_pmf)))
            joint_pmf = [[joint_pmf[j][0]/pmfSum, joint_pmf[j][0]*n_species/pmfSum, joint_pmf[j][2]] for j in range(len(joint_pmf))]

        joint_pmf_temp = []
        for item in joint_pmf:
            joint_pmf_temp.append([item[2][0], item[2][1], item[0]])
        joint_pmf = joint_pmf_temp

        return joint_pmf

    inputCase = None

    if out_dist == 'random' and in_dist == 'random':
        inputCase = 0

    if callable(out_dist) and in_dist == 'random' and out_range is None:
        inputCase = 1

    if callable(out_dist) and in_dist == 'random' and isinstance(out_range, list):
        inputCase = 2

    if isinstance(out_dist, list) and in_dist == 'random' and all(isinstance(x[1], float) for x in out_dist):
        inputCase = 3

    if isinstance(out_dist, list) and in_dist == 'random' and all(isinstance(x[1], int) for x in out_dist):
        inputCase = 4

    if out_dist == 'random' and callable(in_dist) and in_range is None:
        inputCase = 5

    if out_dist == 'random' and callable(in_dist) and isinstance(in_range, list):
        inputCase = 6

    if out_dist == 'random' and isinstance(in_dist, list) and all(isinstance(x[1], float) for x in in_dist):
        inputCase = 7

    if out_dist == 'random' and isinstance(in_dist, list) and all(isinstance(x[1], int) for x in in_dist):
        inputCase = 8

    if callable(out_dist) and callable(in_dist):

        if in_dist == out_dist and in_range is None and out_range is None:
            inputCase = 9
        if in_dist == out_dist and in_range and in_range == out_range:
            inputCase = 10
        if in_dist == out_dist and in_range != out_range:
            inputCase = 11  # todo (maybe): add this (unlikely edge) case
        if in_dist != out_dist and in_range is None and out_range is None:
            inputCase = 12
        if in_dist != out_dist and in_range and in_range == out_range:
            inputCase = 13
        if in_dist != out_dist and in_range != out_range:
            inputCase = 14  # todo (maybe): add this (unlikely edge) case

    if isinstance(out_dist, list) and isinstance(in_dist, list):
        if all(isinstance(x[1], int) for x in out_dist) and all(isinstance(x[1], int) for x in in_dist):
            inputCase = 15
        if all(isinstance(x[1], float) for x in out_dist) and all(isinstance(x[1], float) for x in in_dist):
            inputCase = 16

    if callable(joint_dist):
        if not joint_range:
            inputCase = 17
        if joint_range:
            # todo: include case defining different ranges for outgoing and incoming edges
            inputCase = 18

    if isinstance(joint_dist, list):
        if all(isinstance(x[2], float) for x in joint_dist):
            inputCase = 19
        if all(isinstance(x[2], int) for x in joint_dist):
            inputCase = 20

    # ---------------------------------------------------------------------------

    # print('inputCase', inputCase)

    if inputCase == 1:

        pmf = single_unbounded_pmf(out_dist)
        outSamples = sample_single_distribution(pmf, 1)

    # todo: generalize the ranges
    if inputCase == 2:

        pmf, startDeg = single_bounded_pmf(out_dist, out_range)
        outSamples = sample_single_distribution(pmf, startDeg)

    if inputCase == 3:

        pmf = [x[1] for x in out_dist]
        if sum(pmf) != 1:
            raise Exception("The PMF does not add to 1")

        startDeg = out_dist[0][0]
        outSamples = sample_single_distribution(pmf, startDeg)

    if inputCase == 4:
        outSamples = out_dist

    if inputCase == 5:

        pmf = single_unbounded_pmf(in_dist)
        inSamples = sample_single_distribution(pmf, 1)

    if inputCase == 6:

        pmf, startDeg = single_bounded_pmf(in_dist, in_range)
        inSamples = sample_single_distribution(pmf, startDeg)

    if inputCase == 7:

        pmf = [x[1] for x in in_dist]

        if sum(pmf) != 1:
            raise Exception("The PMF does not add to 1")

        startDeg = in_dist[0][0]
        inSamples = sample_single_distribution(pmf, startDeg)

    if inputCase == 8:
        inSamples = in_dist

    if inputCase == 9:

        pmf1 = single_unbounded_pmf(out_dist)
        pmf2 = single_unbounded_pmf(in_dist)
        inOrOut = random.randint(0, 1)  # choose which distribution is guaranteed n_species
        if inOrOut:
            inSamples, outSamples = sample_both_pmfs(pmf2, 1, pmf1, 1)
        else:
            outSamples, inSamples = sample_both_pmfs(pmf1, 1, pmf2, 1)

    if inputCase == 10:

        pmf1, startDeg1 = single_bounded_pmf(out_dist, out_range)
        pmf2, startDeg2 = single_bounded_pmf(in_dist, in_range)
        inOrOut = random.randint(0, 1)  # choose which distribution is guaranteed n_species
        if inOrOut:
            inSamples, outSamples = sample_both_pmfs(pmf2, startDeg2, pmf1, startDeg1)
        else:
            outSamples, inSamples = sample_both_pmfs(pmf1, startDeg1, pmf2, startDeg2)

    if inputCase == 11:

        pass  # todo: unlikely edge case

    if inputCase == 12:

        pmfOut = single_unbounded_pmf(out_dist)
        pmfIn = single_unbounded_pmf(in_dist)

        edgeEVout = find_edges_expected_value(pmfOut, out_range)
        edgeEVin = find_edges_expected_value(pmfIn, in_range)

        if edgeEVin < edgeEVout:
            pmfOut = trim_pmf(edgeEVin, out_dist)
            inSamples, outSamples = sample_both_pmfs(pmfIn, 1, pmfOut, 1)
        if edgeEVin > edgeEVout:
            pmfIn = trim_pmf(edgeEVout, in_dist)
            outSamples, inSamples = sample_both_pmfs(pmfOut, 1, pmfIn, 1)
        if edgeEVin == edgeEVout:
            outSamples, inSamples = sample_both_pmfs(pmfOut, 1, pmfIn, 1)

    if inputCase == 13:

        pmfOut, StartDegOut = single_bounded_pmf(out_dist, out_range)
        pmfIn, StartDegIn = single_bounded_pmf(in_dist, in_range)

        edgeEVout = find_edges_expected_value(pmfOut, StartDegOut)
        edgeEVin = find_edges_expected_value(pmfIn, StartDegIn)

        if edgeEVin < edgeEVout:
            pmfOut = trim_pmf_2(edgeEVin, pmfOut, StartDegOut)
            inSamples, outSamples = sample_both_pmfs(pmfIn, StartDegIn, pmfOut, StartDegOut)
        if edgeEVin > edgeEVout:
            pmfIn = trim_pmf_2(edgeEVout, pmfIn, StartDegIn)
            outSamples, inSamples = sample_both_pmfs(pmfOut, StartDegOut, pmfIn, StartDegIn)
        if edgeEVin == edgeEVout:
            outSamples, inSamples = sample_both_pmfs(pmfOut, StartDegOut, pmfIn, StartDegIn)

    if inputCase == 14:

        pass  # todo: unlikely edge case

    if inputCase == 15:

        if find_edge_count(out_dist) != find_edge_count(in_dist):
            raise Exception("The edges counts for the input and output distributions must match.")

        outSamples = out_dist
        inSamples = in_dist

    if inputCase == 16:

        pmf1 = [x[1] for x in out_dist]
        pmf2 = [x[1] for x in in_dist]

        edgeEVout = find_edges_expected_value(pmf1, out_dist[0][0])
        edgeEVin = find_edges_expected_value(pmf2, in_dist[0][0])

        if edgeEVin < edgeEVout:
            pmf1 = trim_pmf_2(edgeEVin, pmf1, out_dist[0][0])
            inSamples, outSamples = sample_both_pmfs(pmf2, in_dist[0][0], pmf1, out_dist[0][0])
        if edgeEVin > edgeEVout:
            pmf2 = trim_pmf_2(edgeEVout, pmf2, in_dist[0][0])
            outSamples, inSamples = sample_both_pmfs(pmf1, out_dist[0][0], pmf2, in_dist[0][0])

    if inputCase == 17:

        pmf = joint_unbounded_pmf(joint_dist)
        jointSamples = sample_joint(pmf)

    if inputCase == 18:

        pmf = joint_bounded_pmf(joint_dist, joint_range)
        jointSamples = sample_joint(pmf)

    if inputCase == 19:

        jointSamples = sample_joint(joint_dist)

    if inputCase == 20:

        jointSamples = joint_dist

    # =======================================================================

    # print()
    #
    # print('inSamples')
    # for each in inSamples:
    #     print(each)
    #
    # print()
    # print('outSamples')
    # for each in outSamples:
    #     print(each)
    #
    # print()
    # print('jointSamples')
    # for each in jointSamples:
    #     print(each)

    # quit()

    # =======================================================================

    inNodesCount = []
    if bool(inSamples):
        for each in inSamples:
            for i in range(each[1]):
                inNodesCount.append(each[0])

    outNodesCount = []
    if bool(outSamples):
        for each in outSamples:
            for i in range(each[1]):
                outNodesCount.append(each[0])

    if bool(jointSamples):
        for each in jointSamples:
            for i in range(each[2]):
                outNodesCount.append(each[0])
                inNodesCount.append(each[1])

    inNodesList = []
    for i, each in enumerate(inNodesCount):
        inNodesList.append(i)

    outNodesList = []
    for i, each in enumerate(outNodesCount):
        outNodesList.append(i)

    if not bool(jointSamples):
        random.shuffle(inNodesCount)
        random.shuffle(outNodesCount)

    reactionList = []
    reactionList2 = []

    # todo: finish pick_continued
    # todo: adaptable probabilities
    # ---------------------------------------------------------------------------------------------------

    if 'modular' in kinetics[0]:

        nodesList = [i for i in range(n_species)]

        if not bool(outSamples) and not bool(inSamples):

            nodeSet = set()
            pick_continued = 0
            while True:

                # todo: This is an issue for larger networks: link cutoff with number of species
                if pick_continued == 10000:
                    return None, [outSamples, inSamples, jointSamples]

                if rxn_prob:
                    rt = _pickReactionType(rxn_prob)
                else:
                    rt = _pickReactionType()

                allo_num = 0
                if allo_reg:
                    allo_num = random.choices([0, 1, 2, 3], allo_reg[0])[0]

                # -----------------------------------------------------------------------------

                if rt == TReactionType.UNIUNI:

                    product = random.choice(nodesList)
                    reactant = random.choice(nodesList)

                    if [[reactant], [product]] in reactionList2 or reactant == product:
                        pick_continued += 1
                        continue

                    allo_species = random.sample(nodesList, allo_num)
                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    reactionList.append([rt, [reactant], [product], allo_species, reg_signs])
                    reactionList2.append([[reactant], [product]])

                    nodeSet.add(reactant)
                    nodeSet.add(product)
                    nodeSet.update(allo_species)

                if rt == TReactionType.BIUNI:

                    product = random.choice(nodesList)
                    reactant1 = random.choice(nodesList)
                    reactant2 = random.choice(nodesList)

                    if [[reactant1, reactant2], [product]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    allo_species = random.sample(nodesList, allo_num)
                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    reactionList.append([rt, [reactant1, reactant2], [product], allo_species, reg_signs])
                    reactionList2.append([[reactant1, reactant2], [product]])

                    nodeSet.add(reactant1)
                    nodeSet.add(reactant2)
                    nodeSet.add(product)
                    nodeSet.update(allo_species)

                if rt == TReactionType.UNIBI:

                    product1 = random.choice(nodesList)
                    product2 = random.choice(nodesList)
                    reactant = random.choice(nodesList)

                    if [[reactant], [product1, product2]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    allo_species = random.sample(nodesList, allo_num)
                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    reactionList.append([rt, [reactant], [product1, product2], allo_species, reg_signs])
                    reactionList2.append([[reactant], [product1, product2]])

                    nodeSet.add(reactant)
                    nodeSet.add(product1)
                    nodeSet.add(product2)
                    nodeSet.update(allo_species)

                if rt == TReactionType.BIBI:

                    product1 = random.choice(nodesList)
                    product2 = random.choice(nodesList)
                    reactant1 = random.choice(nodesList)
                    reactant2 = random.choice(nodesList)

                    if [[reactant1, reactant2], [product1, product2]] in reactionList2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    allo_species = random.sample(nodesList, allo_num)
                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    reactionList.append([rt, [reactant1, reactant2], [product1, product2], allo_species, reg_signs])
                    reactionList2.append([[reactant1, reactant2], [product1, product2]])

                    nodeSet.add(reactant1)
                    nodeSet.add(reactant2)
                    nodeSet.add(product1)
                    nodeSet.add(product2)
                    nodeSet.update(allo_species)

                if n_reactions:
                    if len(nodeSet) >= n_species and len(reactionList) >= n_reactions:
                        break
                else:
                    if len(nodeSet) == n_species:
                        break

        # -----------------------------------------------------------------

        if not bool(outSamples) and bool(inSamples):
            pick_continued = 0

            while True:

                if pick_continued == 1000:
                    return None, [outSamples, inSamples, jointSamples]

                if rxn_prob:
                    rt = _pickReactionType(rxn_prob)
                else:
                    rt = _pickReactionType()

                allo_num = 0
                if allo_reg:
                    allo_num = random.choices([0, 1, 2, 3], allo_reg[0])[0]

                # -----------------------------------------------------------------

                if rt == TReactionType.UNIUNI:

                    if max(inNodesCount) < (1 + allo_num):
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product] < (1 + allo_num):
                        product = random.choices(inNodesList, probIn)[0]

                    reactant = random.choice(inNodesList)

                    if [[reactant], [product]] in reactionList2 or reactant == product:
                        pick_continued += 1
                        continue

                    allo_species = random.sample(nodesList, allo_num)

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    inNodesCount[product] -= (1 + allo_num)
                    reactionList.append([rt, [reactant], [product], allo_species, reg_signs])
                    reactionList2.append([[reactant], [product]])

                # -----------------------------------------------------------------

                if rt == TReactionType.BIUNI:

                    if max(inNodesCount) < (2 + allo_num):
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product] < (2 + allo_num):
                        product = random.choices(inNodesList, probIn)[0]

                    reactant1 = random.choice(inNodesList)
                    reactant2 = random.choice(inNodesList)

                    if [[reactant1, reactant2], [product]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    allo_species = random.sample(nodesList, allo_num)

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    inNodesCount[product] -= (2 + allo_num)
                    reactionList.append([rt, [reactant1, reactant2], [product], allo_species, reg_signs])
                    reactionList2.append([[reactant1, reactant2], [product]])

                # -----------------------------------------------------------------

                if rt == TReactionType.UNIBI:

                    if sum(1 for each in inNodesCount if each >= (1 + allo_num)) < 2 \
                            and max(inNodesCount) < (2 + 2*allo_num):
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product1 = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product1] < (1 + allo_num):
                        product1 = random.choices(inNodesList, probIn)[0]

                    inNodesCountCopy = deepcopy(inNodesCount)
                    inNodesCountCopy[product1] -= (1 + allo_num)
                    sumInCopy = sum(inNodesCountCopy)
                    probInCopy = [x/sumInCopy for x in inNodesCountCopy]

                    product2 = random.choices(inNodesList, probInCopy)[0]
                    while inNodesCountCopy[product2] < (1 + allo_num):
                        product2 = random.choices(inNodesList, probIn)[0]

                    reactant = random.choice(inNodesList)

                    if [[reactant], [product1, product2]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    allo_species = random.sample(nodesList, allo_num)

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    inNodesCount[product1] -= (1 + allo_num)
                    inNodesCount[product2] -= (1 + allo_num)
                    reactionList.append([rt, [reactant], [product1, product2], allo_species, reg_signs])
                    reactionList2.append([[reactant], [product1, product2]])

                # -----------------------------------------------------------------

                if rt == TReactionType.BIBI:

                    if sum(1 for each in inNodesCount if each >= (2 + allo_num)) < 2 \
                            and max(inNodesCount) < (4 + 2*allo_num):
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product1 = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product1] < (2 + allo_num):
                        product1 = random.choices(inNodesList, probIn)[0]

                    inNodesCountCopy = deepcopy(inNodesCount)
                    inNodesCountCopy[product1] -= (2 + allo_num)
                    sumInCopy = sum(inNodesCountCopy)
                    probInCopy = [x/sumInCopy for x in inNodesCountCopy]

                    product2 = random.choices(inNodesList, probInCopy)[0]
                    while inNodesCountCopy[product2] < (2 + allo_num):
                        product2 = random.choices(inNodesList, probIn)[0]

                    reactant1 = random.choice(inNodesList)
                    reactant2 = random.choice(inNodesList)

                    if [[reactant1, reactant2], [product1, product2]] in reactionList2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    allo_species = random.sample(nodesList, allo_num)

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    inNodesCount[product1] -= (2 + allo_num)
                    inNodesCount[product2] -= (2 + allo_num)
                    reactionList.append([rt, [reactant1, reactant2], [product1, product2], allo_species, reg_signs])
                    reactionList2.append([[reactant1, reactant2], [product1, product2]])

                if sum(inNodesCount) == 0:
                    break

        # -----------------------------------------------------------------

        if bool(outSamples) and not bool(inSamples):

            pick_continued = 0
            while True:

                if pick_continued == 1000:
                    return None, [outSamples, inSamples, jointSamples]

                if rxn_prob:
                    rt = _pickReactionType(rxn_prob)
                else:
                    rt = _pickReactionType()

                allo_num = 0
                if allo_reg:
                    allo_num = random.choices([0, 1, 2, 3], allo_reg[0])[0]

                # -----------------------------------------------------------------

                if rt == TReactionType.UNIUNI:

                    if sum(outNodesCount) < (1 + allo_num):
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x/sumOut for x in outNodesCount]
                    reactant = random.choices(outNodesList, probOut)[0]

                    product = random.choice(outNodesList)

                    if [[reactant], [product]] in reactionList2 or reactant == product:
                        pick_continued += 1
                        continue

                    allo_species = []
                    if allo_num > 0:
                        outNodesCountCopy = deepcopy(outNodesCount)
                        outNodesCountCopy[reactant] -= 1
                        sumOutCopy = sum(outNodesCountCopy)
                        probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                        while len(allo_species) < allo_num:
                            new_allo = random.choices(outNodesList, probOutCopy)[0]
                            if new_allo not in allo_species:
                                allo_species.append(new_allo)
                                if len(allo_species) < allo_num:
                                    outNodesCountCopy[allo_species[-1]] -= 1
                                    sumOutCopy = sum(outNodesCountCopy)
                                    probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    outNodesCount[reactant] -= 1
                    for each in allo_species:
                        outNodesCount[each] -= 1
                    reactionList.append([rt, [reactant], [product], allo_species, reg_signs])
                    reactionList2.append([[reactant], [product]])

                # -----------------------------------------------------------------

                if rt == TReactionType.BIUNI:

                    if sum(outNodesCount) < (2 + allo_num):
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x/sumOut for x in outNodesCount]
                    reactant1 = random.choices(outNodesList, probOut)[0]

                    outNodesCountCopy = deepcopy(outNodesCount)
                    outNodesCountCopy[reactant1] -= 1
                    sumOutCopy = sum(outNodesCountCopy)
                    probOutCopy = [x/sumOutCopy for x in outNodesCountCopy]
                    reactant2 = random.choices(outNodesList, probOutCopy)[0]

                    product = random.choice(outNodesList)

                    if [[reactant1, reactant2], [product]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    allo_species = []
                    if allo_num > 0:
                        outNodesCountCopy[reactant2] -= 1
                        sumOutCopy = sum(outNodesCountCopy)
                        probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                        while len(allo_species) < allo_num:
                            new_allo = random.choices(outNodesList, probOutCopy)[0]
                            if new_allo not in allo_species:
                                allo_species.append(new_allo)
                                if len(allo_species) < allo_num:
                                    outNodesCountCopy[allo_species[-1]] -= 1
                                    sumOutCopy = sum(outNodesCountCopy)
                                    probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    outNodesCount[reactant1] -= 1
                    outNodesCount[reactant2] -= 1
                    for each in allo_species:
                        outNodesCount[each] -= 1
                    reactionList.append([rt, [reactant1, reactant2], [product], allo_species, reg_signs])
                    reactionList2.append([[reactant1, reactant2], [product]])

                # -----------------------------------------------------------------

                if rt == TReactionType.UNIBI:

                    cont = False
                    if sum(1 for each in outNodesCount if each >= 2) >= (1 + allo_num):
                        cont = True
                    if sum(1 for each in outNodesCount if each >= 2) >= (allo_num - 1) \
                            and sum(1 for each in outNodesCount if each >= 4) >= 1:
                        cont = True
                    if not cont:
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x/sumOut for x in outNodesCount]
                    reactant = random.choices(outNodesList, probOut)[0]
                    while outNodesCount[reactant] < 2:
                        reactant = random.choices(outNodesList, probOut)[0]

                    product1 = random.choice(outNodesList)
                    product2 = random.choice(outNodesList)

                    if [[reactant], [product1, product2]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    allo_species = []
                    if allo_num > 0:
                        outNodesCountCopy = deepcopy(outNodesCount)
                        outNodesCountCopy[reactant] -= 2
                        sumOutCopy = sum(outNodesCountCopy)
                        probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                        while len(allo_species) < allo_num:
                            new_allo = random.choices(outNodesList, probOutCopy)[0]
                            while outNodesCountCopy[new_allo] < 2:
                                new_allo = random.choices(outNodesList, probOutCopy)[0]
                            if new_allo not in allo_species:
                                allo_species.append(new_allo)
                                if len(allo_species) < allo_num:
                                    outNodesCountCopy[allo_species[-1]] -= 2
                                    sumOutCopy = sum(outNodesCountCopy)
                                    probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    outNodesCount[reactant] -= 2
                    for each in allo_species:
                        outNodesCount[each] -= 2
                    reactionList.append([rt, [reactant], [product1, product2], allo_species, reg_signs])
                    reactionList2.append([[reactant], [product1, product2]])

                # -----------------------------------------------------------------

                if rt == TReactionType.BIBI:

                    cont = False
                    if sum(1 for each in outNodesCount if each >= 2) >= (2 + allo_num):
                        cont = True

                    if sum(1 for each in outNodesCount if each >= 2) >= allo_num \
                            and sum(1 for each in outNodesCount if each >= 4) >= 1:
                        cont = True

                    if sum(1 for each in outNodesCount if each >= 2) >= (allo_num - 2) \
                            and sum(1 for each in outNodesCount if each >= 4) >= 2:
                        cont = True

                    if not cont:
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x / sumOut for x in outNodesCount]
                    reactant1 = random.choices(outNodesList, probOut)[0]
                    while outNodesCount[reactant1] < 2:
                        reactant1 = random.choices(outNodesList, probOut)[0]

                    outNodesCountCopy = deepcopy(outNodesCount)
                    outNodesCountCopy[reactant1] -= 2
                    sumOutCopy = sum(outNodesCountCopy)
                    probOutCopy = [x/sumOutCopy for x in outNodesCountCopy]
                    reactant2 = random.choices(outNodesList, probOutCopy)[0]
                    while outNodesCountCopy[reactant2] < 2:
                        reactant2 = random.choices(outNodesList, probOutCopy)[0]

                    product1 = random.choice(outNodesList)
                    product2 = random.choice(outNodesList)

                    if [[reactant1, reactant2], [product1, product2]] in reactionList2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    allo_species = []
                    if allo_num > 0:
                        outNodesCountCopy[reactant2] -= 2
                        sumOutCopy = sum(outNodesCountCopy)
                        probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                        while len(allo_species) < allo_num:
                            new_allo = random.choices(outNodesList, probOutCopy)[0]
                            while outNodesCountCopy[new_allo] < 2:
                                new_allo = random.choices(outNodesList, probOutCopy)[0]
                            if new_allo not in allo_species:
                                allo_species.append(new_allo)
                                if len(allo_species) < allo_num:
                                    outNodesCountCopy[allo_species[-1]] -= 2
                                    sumOutCopy = sum(outNodesCountCopy)
                                    probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    outNodesCount[reactant1] -= 2
                    outNodesCount[reactant2] -= 2
                    for each in allo_species:
                        outNodesCount[each] -= 2
                    reactionList.append([rt, [reactant1, reactant2], [product1, product2], allo_species, reg_signs])
                    reactionList2.append([[reactant1, reactant2], [product1, product2]])

                if sum(outNodesCount) == 0:
                    break

        # -----------------------------------------------------------------

        if (bool(outSamples) and bool(inSamples)) or bool(jointSamples):
            pick_continued = 0
            while True:

                if pick_continued == 1000:
                    return None, [outSamples, inSamples, jointSamples]

                if rxn_prob:
                    rt = _pickReactionType(rxn_prob)
                else:
                    rt = _pickReactionType()

                allo_num = 0
                if allo_reg:
                    allo_num = random.choices([0, 1, 2, 3], allo_reg[0])[0]

                # -----------------------------------------------------------------

                if rt == TReactionType.UNIUNI:

                    if sum(outNodesCount) < (1 + allo_num):
                        pick_continued += 1
                        continue

                    if max(inNodesCount) < (1 + allo_num):
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product] < (1 + allo_num):
                        product = random.choices(inNodesList, probIn)[0]

                    sumOut = sum(outNodesCount)
                    probOut = [x / sumOut for x in outNodesCount]
                    reactant = random.choices(outNodesList, probOut)[0]

                    if [[reactant], [product]] in reactionList2 or reactant == product:
                        pick_continued += 1
                        continue

                    allo_species = []
                    if allo_num > 0:
                        outNodesCountCopy = deepcopy(outNodesCount)
                        outNodesCountCopy[reactant] -= 1
                        sumOutCopy = sum(outNodesCountCopy)
                        probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                        while len(allo_species) < allo_num:
                            new_allo = random.choices(outNodesList, probOutCopy)[0]
                            if new_allo not in allo_species:
                                allo_species.append(new_allo)
                                if len(allo_species) < allo_num:
                                    outNodesCountCopy[allo_species[-1]] -= 1
                                    sumOutCopy = sum(outNodesCountCopy)
                                    probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    inNodesCount[product] -= (1 + allo_num)
                    outNodesCount[reactant] -= 1
                    for each in allo_species:
                        outNodesCount[each] -= 1
                    reactionList.append([rt, [reactant], [product], allo_species, reg_signs])
                    reactionList2.append([[reactant], [product]])

                # -----------------------------------------------------------------

                if rt == TReactionType.BIUNI:

                    if sum(outNodesCount) < (2 + allo_num):
                        pick_continued += 1
                        continue

                    if max(inNodesCount) < (2 + allo_num):
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x/sumOut for x in outNodesCount]
                    reactant1 = random.choices(outNodesList, probOut)[0]
                    
                    outNodesCountCopy = deepcopy(outNodesCount)
                    outNodesCountCopy[reactant1] -= 1
                    sumOutCopy = sum(outNodesCountCopy)
                    probOutCopy = [x/sumOutCopy for x in outNodesCountCopy]
                    reactant2 = random.choices(outNodesList, probOutCopy)[0]
                    
                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product] < (2 + allo_num):
                        product = random.choices(inNodesList, probIn)[0]

                    if [[reactant1, reactant2], [product]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    allo_species = []
                    if allo_num > 0:
                        outNodesCountCopy[reactant2] -= 1
                        sumOutCopy = sum(outNodesCountCopy)
                        probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                        while len(allo_species) < allo_num:
                            new_allo = random.choices(outNodesList, probOutCopy)[0]
                            if new_allo not in allo_species:
                                allo_species.append(new_allo)
                                if len(allo_species) < allo_num:
                                    outNodesCountCopy[allo_species[-1]] -= 1
                                    sumOutCopy = sum(outNodesCountCopy)
                                    probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    inNodesCount[product] -= (2 + allo_num)
                    outNodesCount[reactant1] -= 1
                    outNodesCount[reactant2] -= 1
                    for each in allo_species:
                        outNodesCount[each] -= 1
                    reactionList.append([rt, [reactant1, reactant2], [product], allo_species, reg_signs])
                    reactionList2.append([[reactant1, reactant2], [product]])

                # -----------------------------------------------------------------
                
                if rt == TReactionType.UNIBI:

                    cont = False
                    if sum(1 for each in outNodesCount if each >= 2) >= (1 + allo_num):
                        cont = True
                    if sum(1 for each in outNodesCount if each >= 2) >= (allo_num - 1) \
                            and sum(1 for each in outNodesCount if each >= 4) >= 1:
                        cont = True
                    if not cont:
                        pick_continued += 1
                        continue

                    if sum(1 for each in inNodesCount if each >= (1 + allo_num)) < 2 \
                            and max(inNodesCount) < (2 + 2*allo_num):
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x / sumOut for x in outNodesCount]
                    reactant = random.choices(outNodesList, probOut)[0]
                    while outNodesCount[reactant] < 2:
                        reactant = random.choices(outNodesList, probOut)[0]

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product1 = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product1] < (1 + allo_num):
                        product1 = random.choices(inNodesList, probIn)[0]

                    inNodesCountCopy = deepcopy(inNodesCount)
                    inNodesCountCopy[product1] -= (1 + allo_num)
                    sumInCopy = sum(inNodesCountCopy)
                    probInCopy = [x/sumInCopy for x in inNodesCountCopy]

                    product2 = random.choices(inNodesList, probInCopy)[0]
                    while inNodesCountCopy[product2] < (1 + allo_num):
                        product2 = random.choices(inNodesList, probIn)[0]

                    if [[reactant], [product1, product2]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    allo_species = []
                    if allo_num > 0:
                        outNodesCountCopy = deepcopy(outNodesCount)
                        outNodesCountCopy[reactant] -= 2
                        sumOutCopy = sum(outNodesCountCopy)
                        probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                        while len(allo_species) < allo_num:
                            new_allo = random.choices(outNodesList, probOutCopy)[0]
                            while outNodesCountCopy[new_allo] < 2:
                                new_allo = random.choices(outNodesList, probOutCopy)[0]
                            if new_allo not in allo_species:
                                allo_species.append(new_allo)
                                if len(allo_species) < allo_num:
                                    outNodesCountCopy[allo_species[-1]] -= 2
                                    sumOutCopy = sum(outNodesCountCopy)
                                    probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    outNodesCount[reactant] -= 2
                    inNodesCount[product1] -= (1 + allo_num)
                    inNodesCount[product2] -= (1 + allo_num)
                    for each in allo_species:
                        outNodesCount[each] -= 2
                    reactionList.append([rt, [reactant], [product1, product2], allo_species, reg_signs])
                    reactionList2.append([[reactant], [product1, product2]])

                # -----------------------------------------------------------------

                if rt == TReactionType.BIBI:

                    cont = False
                    if sum(1 for each in outNodesCount if each >= 2) >= (2 + allo_num):
                        cont = True

                    if sum(1 for each in outNodesCount if each >= 2) >= allo_num \
                            and sum(1 for each in outNodesCount if each >= 4) >= 1:
                        cont = True

                    if sum(1 for each in outNodesCount if each >= 2) >= (allo_num - 2) \
                            and sum(1 for each in outNodesCount if each >= 4) >= 2:
                        cont = True

                    if not cont:
                        pick_continued += 1
                        continue

                    if sum(1 for each in inNodesCount if each >= (2 + allo_num)) < 2 \
                            and max(inNodesCount) < (4 + 2*allo_num):
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x / sumOut for x in outNodesCount]
                    reactant1 = random.choices(outNodesList, probOut)[0]
                    while outNodesCount[reactant1] < 2:
                        reactant1 = random.choices(outNodesList, probOut)[0]

                    outNodesCountCopy = deepcopy(outNodesCount)
                    outNodesCountCopy[reactant1] -= 2
                    sumOutCopy = sum(outNodesCountCopy)
                    probOutCopy = [x/sumOutCopy for x in outNodesCountCopy]
                    reactant2 = random.choices(outNodesList, probOutCopy)[0]
                    while outNodesCountCopy[reactant2] < 2:
                        reactant2 = random.choices(outNodesList, probOutCopy)[0]

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product1 = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product1] < (2 + allo_num):
                        product1 = random.choices(inNodesList, probIn)[0]

                    inNodesCountCopy = deepcopy(inNodesCount)
                    inNodesCountCopy[product1] -= (2 + allo_num)
                    sumInCopy = sum(inNodesCountCopy)
                    probInCopy = [x/sumInCopy for x in inNodesCountCopy]

                    product2 = random.choices(inNodesList, probInCopy)[0]
                    while inNodesCountCopy[product2] < (2 + allo_num):
                        product2 = random.choices(inNodesList, probIn)[0]

                    if [[reactant1, reactant2], [product1, product2]] in reactionList2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    allo_species = []
                    if allo_num > 0:
                        outNodesCountCopy[reactant2] -= 2
                        sumOutCopy = sum(outNodesCountCopy)
                        probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                        while len(allo_species) < allo_num:
                            new_allo = random.choices(outNodesList, probOutCopy)[0]
                            while outNodesCountCopy[new_allo] < 2:
                                new_allo = random.choices(outNodesList, probOutCopy)[0]
                            if new_allo not in allo_species:
                                allo_species.append(new_allo)
                                if len(allo_species) < allo_num:
                                    outNodesCountCopy[allo_species[-1]] -= 2
                                    sumOutCopy = sum(outNodesCountCopy)
                                    probOutCopy = [x / sumOutCopy for x in outNodesCountCopy]

                    reg_signs = [random.choices([1, -1], [allo_reg[1], 1-allo_reg[1]])[0] for _ in allo_species]

                    outNodesCount[reactant1] -= 2
                    outNodesCount[reactant2] -= 2
                    inNodesCount[product1] -= (2 + allo_num)
                    inNodesCount[product2] -= (2 + allo_num)
                    for each in allo_species:
                        outNodesCount[each] -= 2
                    reactionList.append([rt, [reactant1, reactant2], [product1, product2], allo_species, reg_signs])
                    reactionList2.append([[reactant1, reactant2], [product1, product2]])

                if sum(inNodesCount) == 0:
                    break

        # -----------------------------------------------------------------

    else:

        if not bool(outSamples) and not bool(inSamples):

            nodesList = [i for i in range(n_species)]
            nodeSet = set()
            pick_continued = 0
            while True:

                if pick_continued == 1000:
                    return None, [outSamples, inSamples, jointSamples]

                if rxn_prob:
                    rt = _pickReactionType(rxn_prob)
                else:
                    rt = _pickReactionType()

                # -------------------------------------------------------------------

                if rt == TReactionType.UNIUNI:

                    product = random.choice(nodesList)
                    reactant = random.choice(nodesList)

                    if [[reactant], [product]] in reactionList2 or reactant == product:
                        pick_continued += 1
                        continue

                    reactionList.append([rt, [reactant], [product]])
                    reactionList2.append([[reactant], [product]])

                    nodeSet.add(reactant)
                    nodeSet.add(product)

                if rt == TReactionType.BIUNI:

                    product = random.choice(nodesList)
                    reactant1 = random.choice(nodesList)
                    reactant2 = random.choice(nodesList)

                    if [[reactant1, reactant2], [product]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    reactionList.append([rt, [reactant1, reactant2], [product]])
                    reactionList2.append([[reactant1, reactant2], [product]])

                    nodeSet.add(reactant1)
                    nodeSet.add(reactant2)
                    nodeSet.add(product)

                if rt == TReactionType.UNIBI:

                    product1 = random.choice(nodesList)
                    product2 = random.choice(nodesList)
                    reactant = random.choice(nodesList)

                    if [[reactant], [product1, product2]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    reactionList.append([rt, [reactant], [product1, product2]])
                    reactionList2.append([[reactant], [product1, product2]])

                    nodeSet.add(reactant)
                    nodeSet.add(product1)
                    nodeSet.add(product2)

                if rt == TReactionType.BIBI:

                    product1 = random.choice(nodesList)
                    product2 = random.choice(nodesList)
                    reactant1 = random.choice(nodesList)
                    reactant2 = random.choice(nodesList)

                    if [[reactant1, reactant2], [product1, product2]] in reactionList2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    reactionList.append([rt, [reactant1, reactant2], [product1, product2]])
                    reactionList2.append([[reactant1, reactant2], [product1, product2]])

                    nodeSet.add(reactant1)
                    nodeSet.add(reactant2)
                    nodeSet.add(product1)
                    nodeSet.add(product2)

                if n_reactions:
                    if len(nodeSet) >= n_species and len(reactionList) >= n_reactions:
                        break
                else:
                    if len(nodeSet) == n_species:
                        break

        # -----------------------------------------------------------------

        if not bool(outSamples) and bool(inSamples):
            pick_continued = 0
            while True:

                if pick_continued == 1000:
                    return None, [outSamples, inSamples, jointSamples]

                if rxn_prob:
                    rt = _pickReactionType(rxn_prob)
                else:
                    rt = _pickReactionType()

                if rt == TReactionType.UNIUNI:

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product = random.choices(inNodesList, probIn)[0]

                    reactant = random.choice(inNodesList)

                    if [[reactant], [product]] in reactionList2 or reactant == product:
                        pick_continued += 1
                        continue

                    inNodesCount[product] -= 1
                    reactionList.append([rt, [reactant], [product]])
                    reactionList2.append([[reactant], [product]])

                if rt == TReactionType.BIUNI:

                    if max(inNodesCount) < 2:
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product] < 2:
                        product = random.choices(inNodesList, probIn)[0]

                    reactant1 = random.choice(inNodesList)
                    reactant2 = random.choice(inNodesList)

                    if [[reactant1, reactant2], [product]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    inNodesCount[product] -= 2
                    reactionList.append([rt, [reactant1, reactant2], [product]])
                    reactionList2.append([[reactant1, reactant2], [product]])

                if rt == TReactionType.UNIBI:

                    if sum(inNodesCount) < 2:
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product1 = random.choices(inNodesList, probIn)[0]

                    inNodesCountCopy = deepcopy(inNodesCount)
                    inNodesCountCopy[product1] -= 1
                    sumInCopy = sum(inNodesCountCopy)
                    probInCopy = [x/sumInCopy for x in inNodesCountCopy]
                    product2 = random.choices(inNodesList, probInCopy)[0]

                    reactant = random.choice(inNodesList)

                    if [[reactant], [product1, product2]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    inNodesCount[product1] -= 1
                    inNodesCount[product2] -= 1
                    reactionList.append([rt, [reactant], [product1, product2]])
                    reactionList2.append([[reactant], [product1, product2]])

                if rt == TReactionType.BIBI:

                    if sum(1 for each in inNodesCount if each > 1) < 2 and max(inNodesCount) < 4:
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product1 = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product1] < 2:
                        product1 = random.choices(inNodesList, probIn)[0]

                    inNodesCountCopy = deepcopy(inNodesCount)
                    inNodesCountCopy[product1] -= 2
                    sumInCopy = sum(inNodesCountCopy)
                    probInCopy = [x/sumInCopy for x in inNodesCountCopy]
                    product2 = random.choices(inNodesList, probInCopy)[0]
                    while inNodesCountCopy[product2] < 2:
                        product2 = random.choices(inNodesList, probInCopy)[0]

                    reactant1 = random.choice(inNodesList)
                    reactant2 = random.choice(inNodesList)

                    if [[reactant1, reactant2], [product1, product2]] in reactionList2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    inNodesCount[product1] -= 2
                    inNodesCount[product2] -= 2
                    reactionList.append([rt, [reactant1, reactant2], [product1, product2]])
                    reactionList2.append([[reactant1, reactant2], [product1, product2]])

                if sum(inNodesCount) == 0:
                    break

        # -----------------------------------------------------------------

        if bool(outSamples) and not bool(inSamples):

            pick_continued = 0
            while True:

                if pick_continued == 1000:
                    return None, [outSamples, inSamples, jointSamples]

                if rxn_prob:
                    rt = _pickReactionType(rxn_prob)
                else:
                    rt = _pickReactionType()

                if rt == TReactionType.UNIUNI:

                    sumOut = sum(outNodesCount)
                    probOut = [x/sumOut for x in outNodesCount]
                    reactant = random.choices(outNodesList, probOut)[0]

                    product = random.choice(outNodesList)

                    if [[reactant], [product]] in reactionList2 or reactant == product:
                        pick_continued += 1
                        continue

                    outNodesCount[reactant] -= 1
                    reactionList.append([rt, [reactant], [product]])
                    reactionList2.append([[reactant], [product]])

                if rt == TReactionType.BIUNI:

                    if sum(outNodesCount) < 2:
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x/sumOut for x in outNodesCount]
                    reactant1 = random.choices(outNodesList, probOut)[0]

                    outNodesCountCopy = deepcopy(outNodesCount)
                    outNodesCountCopy[reactant1] -= 1
                    sumOutCopy = sum(outNodesCountCopy)
                    probOutCopy = [x/sumOutCopy for x in outNodesCountCopy]
                    reactant2 = random.choices(outNodesList, probOutCopy)[0]

                    product = random.choice(outNodesList)

                    if [[reactant1, reactant2], [product]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    outNodesCount[reactant1] -= 1
                    outNodesCount[reactant2] -= 1
                    reactionList.append([rt, [reactant1, reactant2], [product]])
                    reactionList2.append([[reactant1, reactant2], [product]])

                if rt == TReactionType.UNIBI:

                    if max(outNodesCount) < 2:
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x/sumOut for x in outNodesCount]
                    reactant = random.choices(outNodesList, probOut)[0]
                    while outNodesCount[reactant] < 2:
                        reactant = random.choices(outNodesList, probOut)[0]

                    product1 = random.choice(outNodesList)
                    product2 = random.choice(outNodesList)

                    if [[reactant], [product1, product2]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    outNodesCount[reactant] -= 2
                    reactionList.append([rt, [reactant], [product1, product2]])
                    reactionList2.append([[reactant], [product1, product2]])

                if rt == TReactionType.BIBI:

                    if sum(1 for each in outNodesCount if each > 1) < 2 \
                            and max(outNodesCount) < 4:
                        pick_continued += 1
                        continue

                    sumOut = sum(outNodesCount)
                    probOut = [x / sumOut for x in outNodesCount]
                    reactant1 = random.choices(outNodesList, probOut)[0]
                    while outNodesCount[reactant1] < 2:
                        reactant1 = random.choices(outNodesList, probOut)[0]

                    outNodesCountCopy = deepcopy(outNodesCount)
                    outNodesCountCopy[reactant1] -= 2
                    sumOutCopy = sum(outNodesCountCopy)
                    probOutCopy = [x/sumOutCopy for x in outNodesCountCopy]
                    reactant2 = random.choices(outNodesList, probOutCopy)[0]
                    while outNodesCountCopy[reactant2] < 2:
                        reactant2 = random.choices(outNodesList, probOutCopy)[0]

                    product1 = random.choice(outNodesList)
                    product2 = random.choice(outNodesList)

                    if [[reactant1, reactant2], [product1, product2]] in reactionList2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    outNodesCount[reactant1] -= 2
                    outNodesCount[reactant2] -= 2
                    reactionList.append([rt, [reactant1, reactant2], [product1, product2]])
                    reactionList2.append([[reactant1, reactant2], [product1, product2]])

                if sum(outNodesCount) == 0:
                    break

        # -----------------------------------------------------------------

        if (bool(outSamples) and bool(inSamples)) or bool(jointSamples):
            pick_continued = 0
            while True:

                if pick_continued == 1000:
                    return None, [outSamples, inSamples, jointSamples]

                if rxn_prob:
                    rt = _pickReactionType(rxn_prob)
                else:
                    rt = _pickReactionType()

                if rt == TReactionType.UNIUNI:

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product = random.choices(inNodesList, probIn)[0]

                    sumOut = sum(outNodesCount)
                    probOut = [x / sumOut for x in outNodesCount]
                    reactant = random.choices(outNodesList, probOut)[0]

                    if [[reactant], [product]] in reactionList2 or reactant == product:
                        pick_continued += 1
                        continue

                    inNodesCount[product] -= 1
                    outNodesCount[reactant] -= 1
                    reactionList.append([rt, [reactant], [product]])
                    reactionList2.append([[reactant], [product]])

                if rt == TReactionType.BIUNI:

                    if max(inNodesCount) < 2:
                        pick_continued += 1
                        continue

                    if sum(outNodesCount) < 2:
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product] < 2:
                        product = random.choices(inNodesList, probIn)[0]

                    sumOut = sum(outNodesCount)
                    probOut = [x/sumOut for x in outNodesCount]
                    reactant1 = random.choices(outNodesList, probOut)[0]

                    outNodesCountCopy = deepcopy(outNodesCount)
                    outNodesCountCopy[reactant1] -= 1
                    sumOutCopy = sum(outNodesCountCopy)
                    probOutCopy = [x/sumOutCopy for x in outNodesCountCopy]
                    reactant2 = random.choices(outNodesList, probOutCopy)[0]

                    if [[reactant1, reactant2], [product]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and product in {reactant1, reactant2}:
                        pick_continued += 1
                        continue

                    inNodesCount[product] -= 2
                    outNodesCount[reactant1] -= 1
                    outNodesCount[reactant2] -= 1
                    reactionList.append([rt, [reactant1, reactant2], [product]])
                    reactionList2.append([[reactant1, reactant2], [product]])

                if rt == TReactionType.UNIBI:

                    if sum(inNodesCount) < 2:
                        pick_continued += 1
                        continue

                    if max(outNodesCount) < 2:
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product1 = random.choices(inNodesList, probIn)[0]

                    inNodesCountCopy = deepcopy(inNodesCount)
                    inNodesCountCopy[product1] -= 1
                    sumInCopy = sum(inNodesCountCopy)
                    probInCopy = [x/sumInCopy for x in inNodesCountCopy]
                    product2 = random.choices(inNodesList, probInCopy)[0]

                    sumOut = sum(outNodesCount)
                    probOut = [x / sumOut for x in outNodesCount]
                    reactant = random.choices(outNodesList, probOut)[0]
                    while outNodesCount[reactant] < 2:
                        reactant = random.choices(outNodesList, probOut)[0]

                    if [[reactant], [product1, product2]] in reactionList2:
                        pick_continued += 1
                        continue

                    if not mass_violating_reactions and reactant in {product1, product2}:
                        pick_continued += 1
                        continue

                    inNodesCount[product1] -= 1
                    inNodesCount[product2] -= 1
                    outNodesCount[reactant] -= 2
                    reactionList.append([rt, [reactant], [product1, product2]])
                    reactionList2.append([[reactant], [product1, product2]])

                if rt == TReactionType.BIBI:

                    if sum(1 for each in inNodesCount if each > 1) < 2 and max(inNodesCount) < 4:
                        pick_continued += 1
                        continue

                    if sum(1 for each in outNodesCount if each > 1) < 2 and max(outNodesCount) < 4:
                        pick_continued += 1
                        continue

                    sumIn = sum(inNodesCount)
                    probIn = [x/sumIn for x in inNodesCount]
                    product1 = random.choices(inNodesList, probIn)[0]
                    while inNodesCount[product1] < 2:
                        product1 = random.choices(inNodesList, probIn)[0]

                    inNodesCountCopy = deepcopy(inNodesCount)
                    inNodesCountCopy[product1] -= 2
                    sumInCopy = sum(inNodesCountCopy)
                    probInCopy = [x/sumInCopy for x in inNodesCountCopy]
                    product2 = random.choices(inNodesList, probInCopy)[0]
                    while inNodesCountCopy[product2] < 2:
                        product2 = random.choices(inNodesList, probInCopy)[0]

                    sumOut = sum(outNodesCount)
                    probOut = [x / sumOut for x in outNodesCount]
                    reactant1 = random.choices(outNodesList, probOut)[0]
                    while outNodesCount[reactant1] < 2:
                        reactant1 = random.choices(outNodesList, probOut)[0]

                    outNodesCountCopy = deepcopy(outNodesCount)
                    outNodesCountCopy[reactant1] -= 2
                    sumOutCopy = sum(outNodesCountCopy)
                    probOutCopy = [x/sumOutCopy for x in outNodesCountCopy]
                    reactant2 = random.choices(outNodesList, probOutCopy)[0]
                    while outNodesCountCopy[reactant2] < 2:
                        reactant2 = random.choices(outNodesList, probOutCopy)[0]

                    if [[reactant1, reactant2], [product1, product2]] in reactionList2 \
                            or {reactant1, reactant2} == {product1, product2}:
                        pick_continued += 1
                        continue

                    inNodesCount[product1] -= 2
                    inNodesCount[product2] -= 2
                    outNodesCount[reactant1] -= 2
                    outNodesCount[reactant2] -= 2
                    reactionList.append([rt, [reactant1, reactant2], [product1, product2]])
                    reactionList2.append([[reactant1, reactant2], [product1, product2]])

                if sum(inNodesCount) == 0:
                    break

    reactionList.insert(0, n_species)
    return reactionList, [outSamples, inSamples, jointSamples]

# Includes boundary and floating species
# Returns a list:
# [New Stoichiometry matrix, list of floatingIds, list of boundaryIds]
# On entry, reactionList has the structure:
# reactionList = [numSpecies, reaction, reaction, ....]
# reaction = [reactionType, [list of reactants], [list of products], rateConstant]


def _getFullStoichiometryMatrix(reactionList):
    n_species = reactionList[0]
    reactionListCopy = deepcopy(reactionList)

    # Remove the first entry in the list which is the number of species
    reactionListCopy.pop(0)
    st = np.zeros((n_species, len(reactionListCopy)))

    for index, r in enumerate(reactionListCopy):
        if r[0] == TReactionType.UNIUNI:
            # UniUni
            reactant = reactionListCopy[index][1][0]
            st[reactant, index] = -1
            product = reactionListCopy[index][2][0]
            st[product, index] = 1

        if r[0] == TReactionType.BIUNI:
            # BiUni
            reactant1 = reactionListCopy[index][1][0]
            st[reactant1, index] += -1
            reactant2 = reactionListCopy[index][1][1]
            st[reactant2, index] += -1
            product = reactionListCopy[index][2][0]
            st[product, index] = 1

        if r[0] == TReactionType.UNIBI:
            # UniBi
            reactant1 = reactionListCopy[index][1][0]
            st[reactant1, index] = -1
            product1 = reactionListCopy[index][2][0]
            st[product1, index] += 1
            product2 = reactionListCopy[index][2][1]
            st[product2, index] += 1

        if r[0] == TReactionType.BIBI:
            # BiBi
            reactant1 = reactionListCopy[index][1][0]
            st[reactant1, index] += -1
            reactant2 = reactionListCopy[index][1][1]
            st[reactant2, index] += -1
            product1 = reactionListCopy[index][2][0]
            st[product1, index] += 1
            product2 = reactionListCopy[index][2][1]
            st[product2, index] += 1

    return st


# Removes boundary or orphan species from stoichiometry matrix
def _removeBoundaryNodes(st):
    dims = st.shape

    n_species = dims[0]
    nReactions = dims[1]

    speciesIds = np.arange(n_species)
    indexes = []
    orphan_species = []
    countBoundarySpecies = 0
    for r in range(n_species):
        # Scan across the columns, count + and - coefficients
        plusCoeff = 0
        minusCoeff = 0
        for c in range(nReactions):
            if st[r, c] < 0:
                minusCoeff = minusCoeff + 1
            if st[r, c] > 0:
                plusCoeff = plusCoeff + 1
        if plusCoeff == 0 and minusCoeff == 0:
            # No reaction attached to this species
            orphan_species.append(r)
        if plusCoeff == 0 and minusCoeff != 0:
            # Species is a source
            indexes.append(r)
            countBoundarySpecies = countBoundarySpecies + 1
        if minusCoeff == 0 and plusCoeff != 0:
            # Species is a sink
            indexes.append(r)
            countBoundarySpecies = countBoundarySpecies + 1

    floatingIds = np.delete(speciesIds, indexes + orphan_species, axis=0)

    boundaryIds = indexes
    return [np.delete(st, indexes + orphan_species, axis=0), floatingIds, boundaryIds]


# todo: fix inputs
def _getAntimonyScript(floatingIds, boundaryIds, reactionList, ic_params, kinetics, rev_prob, add_E, allo_reg):

    E = ''
    E_end = ''
    if add_E:
        E = 'E*('
        E_end = ')'

    # Remove the first element which is the n_species
    reactionListCopy = deepcopy(reactionList)
    reactionListCopy.pop(0)

    antStr = ''
    if len(floatingIds) > 0:
        antStr = antStr + 'var ' + 'S' + str(floatingIds[0])
        for index in floatingIds[1:]:
            antStr = antStr + ', ' + 'S' + str(index)
        antStr = antStr + '\n'

    if 'modular' in kinetics[0]:
        for each in reactionListCopy:
            for item in each[3]:
                if item not in boundaryIds and item not in floatingIds:
                    boundaryIds.append(item)

    if len(boundaryIds) > 0:
        antStr = antStr + 'ext ' + 'S' + str(boundaryIds[0])
        for index in boundaryIds[1:]:
            antStr = antStr + ', ' + 'S' + str(index)
        antStr = antStr + '\n'
    antStr = antStr + '\n'

    def reversibility(rxnType):

        rev1 = False
        if rev_prob and isinstance(rev_prob, list):
            rev1 = random.choices([True, False], [rev_prob[rxnType], 1.0 - rev_prob[rxnType]])[0]
        if isinstance(rev_prob, float) or isinstance(rev_prob, int):
            rev1 = random.choices([True, False], [rev_prob, 1 - rev_prob])[0]

        return rev1

    if kinetics[0] == 'mass_action':

        if len(kinetics[2]) == 3 or len(kinetics[2]) == 4:

            kf = []
            kr = []
            kc = []

            for reactionIndex, r in enumerate(reactionListCopy):

                antStr = antStr + 'J' + str(reactionIndex) + ': '
                if r[0] == TReactionType.UNIUNI:
                    # UniUni
                    antStr = antStr + 'S' + str(r[1][0])
                    antStr = antStr + ' -> '
                    antStr = antStr + 'S' + str(r[2][0])

                    rev = reversibility(0)
                    if not rev:
                        antStr = antStr + '; ' + E + 'kc' + str(reactionIndex) + '*S' + str(r[1][0]) + E_end
                        kc.append('kc' + str(reactionIndex))

                    else:
                        antStr = antStr + '; ' + E + 'kf' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + ' - kr' + str(reactionIndex) + '*S' + str(r[2][0]) + E_end
                        kf.append('kf' + str(reactionIndex))
                        kr.append('kr' + str(reactionIndex))

                if r[0] == TReactionType.BIUNI:
                    # BiUni
                    antStr = antStr + 'S' + str(r[1][0])
                    antStr = antStr + ' + '
                    antStr = antStr + 'S' + str(r[1][1])
                    antStr = antStr + ' -> '
                    antStr = antStr + 'S' + str(r[2][0])

                    rev = reversibility(1)
                    if not rev:
                        antStr = antStr + '; ' + E + 'kc' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + E_end
                        kc.append('kc' + str(reactionIndex))

                    else:
                        antStr = antStr + '; ' + E + 'kf' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr' + str(reactionIndex) + '*S' \
                                 + str(r[2][0]) + E_end
                        kf.append('kf' + str(reactionIndex))
                        kr.append('kr' + str(reactionIndex))

                if r[0] == TReactionType.UNIBI:
                    # UniBi
                    antStr = antStr + 'S' + str(r[1][0])
                    antStr = antStr + ' -> '
                    antStr = antStr + 'S' + str(r[2][0])
                    antStr = antStr + ' + '
                    antStr = antStr + 'S' + str(r[2][1])

                    rev = reversibility(2)
                    if not rev:
                        antStr = antStr + '; ' + E + 'kc' + str(reactionIndex) + '*S' + str(r[1][0]) + E_end
                        kc.append('kc' + str(reactionIndex))

                    else:
                        antStr = antStr + '; ' + E + 'kf' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + ' - kr' + str(reactionIndex) + '*S' + str(r[2][0]) \
                                 + '*S' + str(r[2][1]) + E_end
                        kf.append('kf' + str(reactionIndex))
                        kr.append('kr' + str(reactionIndex))

                if r[0] == TReactionType.BIBI:
                    # BiBi
                    antStr = antStr + 'S' + str(r[1][0])
                    antStr = antStr + ' + '
                    antStr = antStr + 'S' + str(r[1][1])
                    antStr = antStr + ' -> '
                    antStr = antStr + 'S' + str(r[2][0])
                    antStr = antStr + ' + '
                    antStr = antStr + 'S' + str(r[2][1])

                    rev = reversibility(3)
                    if not rev:
                        antStr = antStr + '; ' + E + 'kc' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + E_end
                        kc.append('kc' + str(reactionIndex))

                    else:
                        antStr = antStr + '; ' + E + 'kf' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr' + str(reactionIndex) + '*S' \
                                 + str(r[2][0]) + '*S' + str(r[2][1]) + E_end
                        kf.append('kf' + str(reactionIndex))
                        kr.append('kr' + str(reactionIndex))

                antStr = antStr + '\n'
            antStr = antStr + '\n'

            parameterIndex = None
            if 'deg' in kinetics[2]:
                reactionIndex += 1
                parameterIndex = reactionIndex
                for sp in floatingIds:
                    antStr = antStr + 'J' + str(reactionIndex) + ': S' + str(sp) + ' ->; ' + 'k' + str(reactionIndex) + '*' + 'S' + str(sp) + '\n'
                    reactionIndex += 1
            antStr = antStr + '\n'

            # for index, r in enumerate(reactionListCopy):

            if kinetics[1] == 'trivial':

                for each in kf:
                    antStr = antStr + each + ' = 1\n'
                for each in kr:
                    antStr = antStr + each + ' = 1\n'
                for each in kc:
                    antStr = antStr + each + ' = 1\n'

            if kinetics[1] == 'uniform':

                for each in kf:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0], scale=kinetics[3][kinetics[2].index('kf')][1]
                                        - kinetics[3][kinetics[2].index('kf')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0], scale=kinetics[3][kinetics[2].index('kr')][1]
                                        - kinetics[3][kinetics[2].index('kr')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc')][0], scale=kinetics[3][kinetics[2].index('kc')][1]
                                        - kinetics[3][kinetics[2].index('kc')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':

                for each in kf:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0], kinetics[3][kinetics[2].index('kf')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0], kinetics[3][kinetics[2].index('kr')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc')][0], kinetics[3][kinetics[2].index('kc')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':

                for each in kf:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0], scale=kinetics[3][kinetics[2].index('kf')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kr:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0], scale=kinetics[3][kinetics[2].index('kr')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kc:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc')][0], scale=kinetics[3][kinetics[2].index('kc')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

            if kinetics[1] == 'lognormal':

                for each in kf:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0], s=kinetics[3][kinetics[2].index('kf')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0], s=kinetics[3][kinetics[2].index('kr')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc')][0], s=kinetics[3][kinetics[2].index('kc')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

            if 'deg' in kinetics[2]:
                # Next the degradation rate constants
                for _ in floatingIds:

                    if kinetics[1] == 'trivial':
                        antStr = antStr + 'k' + str(parameterIndex) + ' = 1\n'

                    if kinetics[1] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('deg')][0], scale=kinetics[3][kinetics[2].index('deg')][1]
                                            - kinetics[3][kinetics[2].index('deg')][0])
                        antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                    if kinetics[1] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('deg')][0], kinetics[3][kinetics[2].index('deg')][1])
                        antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                    if kinetics[1] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('deg')][0], scale=kinetics[3][kinetics[2].index('deg')][1])
                            if const >= 0:
                                antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('deg')][0], s=kinetics[3][kinetics[2].index('deg')][1])
                        antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                    parameterIndex += 1

            antStr = antStr + '\n'

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

            for reactionIndex, r in enumerate(reactionListCopy):

                antStr = antStr + 'J' + str(reactionIndex) + ': '
                if r[0] == TReactionType.UNIUNI:
                    # UniUni
                    antStr = antStr + 'S' + str(r[1][0])
                    antStr = antStr + ' -> '
                    antStr = antStr + 'S' + str(r[2][0])

                    rev = reversibility(0)
                    if not rev:
                        antStr = antStr + '; ' + E + 'kc0_' + str(reactionIndex) + '*S' + str(r[1][0]) + E_end
                        kc0.append('kc0_' + str(reactionIndex))

                    else:
                        antStr = antStr + '; ' + E + 'kf0_' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + ' - kr0_' + str(reactionIndex) + '*S' + str(r[2][0]) + E_end
                        kf0.append('kf0_' + str(reactionIndex))
                        kr0.append('kr0_' + str(reactionIndex))

                if r[0] == TReactionType.BIUNI:
                    # BiUni
                    antStr = antStr + 'S' + str(r[1][0])
                    antStr = antStr + ' + '
                    antStr = antStr + 'S' + str(r[1][1])
                    antStr = antStr + ' -> '
                    antStr = antStr + 'S' + str(r[2][0])

                    rev = reversibility(1)
                    if not rev:
                        antStr = antStr + '; ' + E + 'kc1_' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + E_end
                        kc1.append('kc1_' + str(reactionIndex))

                    else:
                        antStr = antStr + '; ' + E + 'kf1_' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr1_' + str(reactionIndex) + '*S' \
                                 + str(r[2][0]) + E_end
                        kf1.append('kf1_' + str(reactionIndex))
                        kr1.append('kr1_' + str(reactionIndex))

                if r[0] == TReactionType.UNIBI:
                    # UniBi
                    antStr = antStr + 'S' + str(r[1][0])
                    antStr = antStr + ' -> '
                    antStr = antStr + 'S' + str(r[2][0])
                    antStr = antStr + ' + '
                    antStr = antStr + 'S' + str(r[2][1])

                    rev = reversibility(2)
                    if not rev:
                        antStr = antStr + '; ' + E + 'kc2_' + str(reactionIndex) + '*S' + str(r[1][0]) + E_end
                        kc2.append('kc2_' + str(reactionIndex))

                    else:
                        antStr = antStr + '; ' + E + 'kf2_' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + ' - kr2_' + str(reactionIndex) + '*S' + str(r[2][0]) \
                                 + '*S' + str(r[2][1]) + E_end
                        kf2.append('kf2_' + str(reactionIndex))
                        kr2.append('kr2_' + str(reactionIndex))

                if r[0] == TReactionType.BIBI:
                    # BiBi
                    antStr = antStr + 'S' + str(r[1][0])
                    antStr = antStr + ' + '
                    antStr = antStr + 'S' + str(r[1][1])
                    antStr = antStr + ' -> '
                    antStr = antStr + 'S' + str(r[2][0])
                    antStr = antStr + ' + '
                    antStr = antStr + 'S' + str(r[2][1])

                    rev = reversibility(3)
                    if not rev:
                        antStr = antStr + '; ' + E + 'kc3_' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + E_end
                        kc3.append('kc3_' + str(reactionIndex))

                    else:
                        antStr = antStr + '; ' + E + 'kf3_' + str(reactionIndex) + '*S' + str(r[1][0]) \
                                 + '*S' + str(r[1][1]) + ' - kr3_' + str(reactionIndex) + '*S' \
                                 + str(r[2][0]) + '*S' + str(r[2][1]) + E_end
                        kf3.append('kf3_' + str(reactionIndex))
                        kr3.append('kr3_' + str(reactionIndex))

                antStr = antStr + '\n'
            antStr = antStr + '\n'

            if 'deg' in kinetics[2]:
                reactionIndex += 1
                parameterIndex = reactionIndex
                for sp in floatingIds:
                    antStr = antStr + 'J' + str(reactionIndex) + ': S' + str(sp) + ' ->; ' + 'k' + str(reactionIndex) + '*' + 'S' + str(sp) + '\n'
                    reactionIndex += 1
            antStr = antStr + '\n'

            # for index, r in enumerate(reactionListCopy):
            # todo: fix this
            if kinetics[1] == 'trivial':

                for each in kf0:
                    antStr = antStr + each + ' = 1\n'
                for each in kf1:
                    antStr = antStr + each + ' = 1\n'
                for each in kf2:
                    antStr = antStr + each + ' = 1\n'
                for each in kf3:
                    antStr = antStr + each + ' = 1\n'
                for each in kr0:
                    antStr = antStr + each + ' = 1\n'
                for each in kr1:
                    antStr = antStr + each + ' = 1\n'
                for each in kr2:
                    antStr = antStr + each + ' = 1\n'
                for each in kr3:
                    antStr = antStr + each + ' = 1\n'
                for each in kc0:
                    antStr = antStr + each + ' = 1\n'
                for each in kc1:
                    antStr = antStr + each + ' = 1\n'
                for each in kc2:
                    antStr = antStr + each + ' = 1\n'
                for each in kc3:
                    antStr = antStr + each + ' = 1\n'

            if kinetics[1] == 'uniform':

                for each in kf0:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf0')][0], scale=kinetics[3][kinetics[2].index('kf0')][1]
                                        - kinetics[3][kinetics[2].index('kf0')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kf1:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf1')][0], scale=kinetics[3][kinetics[2].index('kf1')][1]
                                        - kinetics[3][kinetics[2].index('kf1')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kf2:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf2')][0], scale=kinetics[3][kinetics[2].index('kf2')][1]
                                        - kinetics[3][kinetics[2].index('kf2')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kf3:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf3')][0], scale=kinetics[3][kinetics[2].index('kf3')][1]
                                        - kinetics[3][kinetics[2].index('kf3')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr0:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr0')][0], scale=kinetics[3][kinetics[2].index('kr0')][1]
                                        - kinetics[3][kinetics[2].index('kr0')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr1:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr1')][0], scale=kinetics[3][kinetics[2].index('kr1')][1]
                                        - kinetics[3][kinetics[2].index('kr1')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr2:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr2')][0], scale=kinetics[3][kinetics[2].index('kr2')][1]
                                        - kinetics[3][kinetics[2].index('kr2')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr3:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr3')][0], scale=kinetics[3][kinetics[2].index('kr3')][1]
                                        - kinetics[3][kinetics[2].index('kr3')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc0:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc0')][0], scale=kinetics[3][kinetics[2].index('kc0')][1]
                                        - kinetics[3][kinetics[2].index('kc0')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc1:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc1')][0], scale=kinetics[3][kinetics[2].index('kc1')][1]
                                        - kinetics[3][kinetics[2].index('kc1')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc2:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc2')][0], scale=kinetics[3][kinetics[2].index('kc2')][1]
                                        - kinetics[3][kinetics[2].index('kc2')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc3:
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kc3')][0], scale=kinetics[3][kinetics[2].index('kc3')][1]
                                        - kinetics[3][kinetics[2].index('kc3')][0])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':

                for each in kf0:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf0')][0], kinetics[3][kinetics[2].index('kf0')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kf1:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf1')][0], kinetics[3][kinetics[2].index('kf1')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kf2:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf2')][0], kinetics[3][kinetics[2].index('kf2')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kf3:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kf3')][0], kinetics[3][kinetics[2].index('kf3')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr0:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr0')][0], kinetics[3][kinetics[2].index('kr0')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr1:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr1')][0], kinetics[3][kinetics[2].index('kr1')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr2:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr2')][0], kinetics[3][kinetics[2].index('kr2')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr3:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kr3')][0], kinetics[3][kinetics[2].index('kr3')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc0:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc0')][0], kinetics[3][kinetics[2].index('kc0')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc1:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc1')][0], kinetics[3][kinetics[2].index('kc1')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc2:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc2')][0], kinetics[3][kinetics[2].index('kc2')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc3:
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('kc3')][0], kinetics[3][kinetics[2].index('kc3')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':

                for each in kf0:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf0')][0], scale=kinetics[3][kinetics[2].index('kf0')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kf1:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf1')][0], scale=kinetics[3][kinetics[2].index('kf1')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kf2:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf2')][0], scale=kinetics[3][kinetics[2].index('kf2')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kf3:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf3')][0], scale=kinetics[3][kinetics[2].index('kf3')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kr0:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr0')][0], scale=kinetics[3][kinetics[2].index('kr0')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kr1:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr1')][0], scale=kinetics[3][kinetics[2].index('kr1')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kr2:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr2')][0], scale=kinetics[3][kinetics[2].index('kr2')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kr3:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr3')][0], scale=kinetics[3][kinetics[2].index('kr3')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kc0:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc0')][0], scale=kinetics[3][kinetics[2].index('kc0')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kc1:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc1')][0], scale=kinetics[3][kinetics[2].index('kc1')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kc2:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc2')][0], scale=kinetics[3][kinetics[2].index('kc2')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

                for each in kc3:
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('kc3')][0], scale=kinetics[3][kinetics[2].index('kc3')][1])
                        if const >= 0:
                            antStr = antStr + each + ' = ' + str(const) + '\n'
                            break

            if kinetics[1] == 'lognormal':

                for each in kf0:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf0')][0], s=kinetics[3][kinetics[2].index('kf0')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kf1:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf1')][0], s=kinetics[3][kinetics[2].index('kf1')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kf2:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf2')][0], s=kinetics[3][kinetics[2].index('kf2')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kf3:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf3')][0], s=kinetics[3][kinetics[2].index('kf3')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr0:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr0')][0], s=kinetics[3][kinetics[2].index('kr0')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr1:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr1')][0], s=kinetics[3][kinetics[2].index('kr1')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr2:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr2')][0], s=kinetics[3][kinetics[2].index('kr2')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kr3:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr3')][0], s=kinetics[3][kinetics[2].index('kr3')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc0:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc0')][0], s=kinetics[3][kinetics[2].index('kc0')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc1:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc1')][0], s=kinetics[3][kinetics[2].index('kc1')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc2:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc2')][0], s=kinetics[3][kinetics[2].index('kc2')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

                for each in kc3:
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kc3')][0], s=kinetics[3][kinetics[2].index('kc3')][1])
                    antStr = antStr + each + ' = ' + str(const) + '\n'

            if 'deg' in kinetics[2]:
                # Next the degradation rate constants
                for _ in floatingIds:

                    if kinetics[1] == 'trivial':
                        antStr = antStr + 'k' + str(parameterIndex) + ' = 1\n'

                    if kinetics[1] == 'uniform':
                        const = uniform.rvs(loc=kinetics[3][kinetics[2].index('deg')][0], scale=kinetics[3][kinetics[2].index('deg')][1]
                                            - kinetics[3][kinetics[2].index('deg')][0])
                        antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                    if kinetics[1] == 'loguniform':
                        const = loguniform.rvs(kinetics[3][kinetics[2].index('deg')][0], kinetics[3][kinetics[2].index('deg')][1])
                        antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                    if kinetics[1] == 'normal':
                        while True:
                            const = norm.rvs(loc=kinetics[3][kinetics[2].index('deg')][0], scale=kinetics[3][kinetics[2].index('deg')][1])
                            if const >= 0:
                                antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'
                                break

                    if kinetics[1] == 'lognormal':
                        const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('deg')][0], s=kinetics[3][kinetics[2].index('deg')][1])
                        antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                    parameterIndex += 1

            antStr = antStr + '\n'

    if kinetics[0] == 'hanekom':

        v = []
        keq = []
        k = []
        ks = []
        kp = []

        for reactionIndex, r in enumerate(reactionListCopy):

            v.append('v' + str(reactionIndex))
            keq.append('keq' + str(reactionIndex))

            antStr = antStr + 'J' + str(reactionIndex) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                antStr = antStr + 'S' + str(r[1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(r[2][0])

                rev = reversibility(0)

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) + ')/(1 + S' + str(r[1][0]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) + ')' + E_end
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][0]))

                    if 'k' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) + ')/(1 + S' + str(r[1][0]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) + ')' + E_end
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][0]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) + ')*(1-(S' \
                            + str(r[2][0]) + '/S' + str(r[1][0]) \
                            + ')/keq' + str(reactionIndex) + ')/(1 + S' + str(r[1][0]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) + ' + S' \
                            + str(r[2][0]) + '/kp_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + ')' + E_end
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][0]))
                        kp.append('kp_' + str(reactionIndex) + '_' + str(r[2][0]))

                    if 'k' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) + ')*(1-(S' \
                            + str(r[2][0]) + '/S' + str(r[1][0]) \
                            + ')/keq' + str(reactionIndex) + ')/(1 + S' + str(r[1][0]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) + ' + S' \
                            + str(r[2][0]) + '/k_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + ')' + E_end
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][0]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[2][0]))

            if r[0] == TReactionType.BIUNI:
                # BiUni
                antStr = antStr + 'S' + str(r[1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(r[1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(r[2][0])

                rev = reversibility(1)

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][1]) + ')/(1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[1][1]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][1]) \
                            + ' + (S' + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ')*(S' + str(r[1][1]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][1]) \
                            + '))' + E_end
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][0]))
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][1]))

                    if 'k' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][1]) + ')/(1 + S' \
                            + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[1][1]) + '/k_' + str(reactionIndex) + '_' + str(r[1][1]) \
                            + ' + (S' + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ')*(S' + str(r[1][1]) + '/k_' + str(reactionIndex) + '_' + str(r[1][1]) \
                            + '))' + E_end
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][0]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][1]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][1]) + ')' + '*(1-(S' \
                            + str(r[2][0]) + '/(S' + str(r[1][0]) \
                            + '*S' + str(r[1][1]) + '))/keq' + str(reactionIndex) + ')/(1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[1][1]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][1]) \
                            + ' + (S' + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ')*(S' + str(r[1][1]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][1]) \
                            + ') + S' + str(r[2][0]) + '/kp_' + str(reactionIndex) + '_' + str(r[2][0]) \
                            + ')' + E_end
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][0]))
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][1]))
                        kp.append('kp_' + str(reactionIndex) + '_' + str(r[2][0]))

                    if 'k' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][1]) + ')' + '*(1-(S' \
                            + str(r[2][0]) + '/(S' + str(r[1][0]) \
                            + '*S' + str(r[1][1]) + '))/keq' + str(reactionIndex) + ')/(1 + S' \
                            + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[1][1]) + '/k_' + str(reactionIndex) + '_' + str(r[1][1]) \
                            + ' + (S' + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ')*(S' + str(r[1][1]) + '/k_' + str(reactionIndex) + '_' + str(r[1][1]) \
                            + ') + S' + str(r[2][0]) + '/k_' + str(reactionIndex) + '_' + str(r[2][0]) \
                            + ')' + E_end
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][0]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][1]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[2][0]))

            if r[0] == TReactionType.UNIBI:
                # UniBi
                antStr = antStr + 'S' + str(r[1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(r[2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(r[2][1])

                rev = reversibility(2)

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) + ')/(1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ')' + E_end
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][0]))

                    if 'k' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) + ')/(1 + S' \
                            + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ')' + E_end
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][0]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) + ')*(1-(S' \
                            + str(r[2][0]) + '*S' + str(r[2][1]) \
                            + '/S' + str(r[1][0]) + ')/keq' + str(reactionIndex) + ')/(1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[2][0]) + '/kp_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + ' + S' + str(r[2][1]) \
                            + '/kp_' + str(reactionIndex) + '_' + str(r[2][1]) + ' + (S' \
                            + str(r[2][0]) + '/kp_' + str(reactionIndex) + '_' + str(r[2][0]) \
                            + ')*(S' + str(r[2][1]) + '/kp_' + str(reactionIndex) + '_' \
                            + str(r[2][1]) + ')' + ')' + E_end
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][0]))
                        kp.append('kp_' + str(reactionIndex) + '_' + str(r[2][0]))
                        kp.append('kp_' + str(reactionIndex) + '_' + str(r[2][1]))

                    if 'k' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) + ')*(1-(S' \
                            + str(r[2][0]) + '*S' + str(r[2][1]) \
                            + '/S' + str(r[1][0]) + ')/keq' + str(reactionIndex) + ')/(1 + S' \
                            + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[2][0]) + '/k_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + ' + S' + str(r[2][1]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[2][1]) + ' + (S' \
                            + str(r[2][0]) + '/k_' + str(reactionIndex) + '_' + str(r[2][0]) \
                            + ')*(S' + str(r[2][1]) + '/k_' + str(reactionIndex) + '_' \
                            + str(r[2][1]) + ')' + ')' + E_end
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][0]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[2][0]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[2][1]))

            if r[0] == TReactionType.BIBI:
                # BiBi
                antStr = antStr + 'S' + str(r[1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(r[1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(r[2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(r[2][1])

                rev = reversibility(3)

                if not rev:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][1]) + ')/((1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ')*(1 + S' + str(r[1][1]) + '/ks_' + str(reactionIndex) + '_' \
                            + str(r[1][1]) + '))' + E_end
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][0]))
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][1]))

                    if 'k' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][1]) + ')/((1 + S' \
                            + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ')*(1 + S' + str(r[1][1]) + '/k_' + str(reactionIndex) + '_' \
                            + str(r[1][1]) + '))' + E_end
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][0]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][1]))

                else:
                    if 'ks' in kinetics[2] and 'kp' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/ks_' + str(reactionIndex) + '_' + str(r[1][1]) + ')' + '*(1-(S' \
                            + str(r[2][0]) + '*S' + str(r[2][1]) \
                            + '/(S' + str(r[1][0]) + '*S' \
                            + str(r[1][1]) + '))/keq' + str(reactionIndex) + ')/((1 + S' \
                            + str(r[1][0]) + '/ks_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[2][0]) + '/kp_' + str(reactionIndex) + '_' + str(r[2][0]) \
                            + ')*(1 + S' + str(r[1][1]) + '/ks_' + str(reactionIndex) + '_' \
                            + str(r[1][1]) + ' + S' + str(r[2][1]) \
                            + '/kp_' + str(reactionIndex) + '_' + str(r[2][1]) + '))' + E_end
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][0]))
                        ks.append('ks_' + str(reactionIndex) + '_' + str(r[1][1]))
                        kp.append('kp_' + str(reactionIndex) + '_' + str(r[2][0]))
                        kp.append('kp_' + str(reactionIndex) + '_' + str(r[2][1]))

                    if 'k' in kinetics[2]:
                        antStr = antStr + '; ' + E + 'v' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' \
                            + str(r[1][0]) + ')*(S' + str(r[1][1]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[1][1]) + ')' + '*(1-(S' \
                            + str(r[2][0]) + '*S' + str(r[2][1]) \
                            + '/(S' + str(r[1][0]) + '*S' \
                            + str(r[1][1]) + '))/keq' + str(reactionIndex) + ')/((1 + S' \
                            + str(r[1][0]) + '/k_' + str(reactionIndex) + '_' + str(r[1][0]) \
                            + ' + S' + str(r[2][0]) + '/k_' + str(reactionIndex) + '_' + str(r[2][0]) \
                            + ')*(1 + S' + str(r[1][1]) + '/k_' + str(reactionIndex) + '_' \
                            + str(r[1][1]) + ' + S' + str(r[2][1]) \
                            + '/k_' + str(reactionIndex) + '_' + str(r[2][1]) + '))' + E_end
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][0]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[1][1]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[2][0]))
                        k.append('k_' + str(reactionIndex) + '_' + str(r[2][1]))

            antStr = antStr + '\n'
        antStr = antStr + '\n'

        if 'deg' in kinetics[2]:
            reactionIndex += 1
            parameterIndex = reactionIndex
            for sp in floatingIds:
                antStr = antStr + 'J' + str(reactionIndex) + ': S' + str(sp) + ' ->; ' + 'k' + str(reactionIndex) + '*' + 'S' + str(sp) + '\n'
                reactionIndex += 1
        antStr = antStr + '\n'

        if kinetics[1] == 'trivial':

            for each in v:
                antStr = antStr + each + ' = 1\n'
            if v:
                antStr = antStr + '\n'
            for each in keq:
                antStr = antStr + each + ' = 1\n'
            if keq:
                antStr = antStr + '\n'
            for each in k:
                antStr = antStr + each + ' = 1\n'
            if k:
                antStr = antStr + '\n'
            for each in ks:
                antStr = antStr + each + ' = 1\n'
            if ks:
                antStr = antStr + '\n'
            for each in kp:
                antStr = antStr + each + ' = 1\n'
            if kp:
                antStr = antStr + '\n'

        if kinetics[1] == 'uniform':

            for each in v:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0], scale=kinetics[3][kinetics[2].index('v')][1]
                                    - kinetics[3][kinetics[2].index('v')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if v:
                antStr = antStr + '\n'

            for each in keq:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('keq')][0], scale=kinetics[3][kinetics[2].index('keq')][1]
                                    - kinetics[3][kinetics[2].index('keq')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if keq:
                antStr = antStr + '\n'

            for each in k:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('k')][0], scale=kinetics[3][kinetics[2].index('k')][1]
                                    - kinetics[3][kinetics[2].index('k')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if k:
                antStr = antStr + '\n'

            for each in ks:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('ks')][0], scale=kinetics[3][kinetics[2].index('ks')][1]
                                    - kinetics[3][kinetics[2].index('ks')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if ks:
                antStr = antStr + '\n'

            for each in kp:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kp')][0], scale=kinetics[3][kinetics[2].index('kp')][1]
                                    - kinetics[3][kinetics[2].index('kp')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if kp:
                antStr = antStr + '\n'

        if kinetics[1] == 'loguniform':

            for each in v:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0], kinetics[3][kinetics[2].index('v')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if v:
                antStr = antStr + '\n'

            for each in keq:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('keq')][0], kinetics[3][kinetics[2].index('keq')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if keq:
                antStr = antStr + '\n'

            for each in k:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('k')][0], kinetics[3][kinetics[2].index('k')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if k:
                antStr = antStr + '\n'

            for each in ks:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('ks')][0], kinetics[3][kinetics[2].index('ks')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if ks:
                antStr = antStr + '\n'

            for each in kp:
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kp')][0], kinetics[3][kinetics[2].index('kp')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if kp:
                antStr = antStr + '\n'

        if kinetics[1] == 'normal':

            for each in v:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0], scale=kinetics[3][kinetics[2].index('v')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break
            if v:
                antStr = antStr + '\n'

            for each in keq:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('keq')][0], scale=kinetics[3][kinetics[2].index('keq')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break
            if keq:
                antStr = antStr + '\n'

            for each in k:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('k')][0], scale=kinetics[3][kinetics[2].index('k')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break
            if k:
                antStr = antStr + '\n'

            for each in ks:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('ks')][0], scale=kinetics[3][kinetics[2].index('ks')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break
            if ks:
                antStr = antStr + '\n'

            for each in kp:
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kp')][0], scale=kinetics[3][kinetics[2].index('kp')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break
            if kp:
                antStr = antStr + '\n'

        if kinetics[1] == 'lognormal':

            for each in v:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0], s=kinetics[3][kinetics[2].index('v')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if v:
                antStr = antStr + '\n'

            for each in keq:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('keq')][0], s=kinetics[3][kinetics[2].index('keq')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if keq:
                antStr = antStr + '\n'

            for each in k:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('k')][0], s=kinetics[3][kinetics[2].index('k')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if k:
                antStr = antStr + '\n'

            for each in ks:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('ks')][0], s=kinetics[3][kinetics[2].index('ks')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if ks:
                antStr = antStr + '\n'

            for each in kp:
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kp')][0], s=kinetics[3][kinetics[2].index('kp')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'
            if kp:
                antStr = antStr + '\n'

    if kinetics[0] == 'lin_log':

        hs = []

        for reactionIndex, r in enumerate(reactionListCopy):

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

            antStr = antStr + 'J' + str(reactionIndex) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])
                antStr = antStr + '; ' + E[0:2] + 'v' + str(reactionIndex) + '*(1'

                rev = reversibility(0)
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            antStr = antStr + ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            antStr = antStr + ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if rev_stoic[each] == -1:
                            antStr = antStr + ' - ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))

                antStr = antStr + ')'

            if r[0] == TReactionType.BIUNI:
                # BiUni
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])
                antStr = antStr + '; ' + E[0:2] + 'v' + str(reactionIndex) + '*(1'

                rev = reversibility(1)
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            antStr = antStr + ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if irr_stoic[each] == 2:
                            antStr = antStr + ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            antStr = antStr + ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if rev_stoic[each] == 2:
                            antStr = antStr + ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if rev_stoic[each] == -1:
                            antStr = antStr + ' - ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))

                antStr = antStr + ')'

            if r[0] == TReactionType.UNIBI:
                # UniBi
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][1])
                antStr = antStr + '; ' + E[0:2] + 'v' + str(reactionIndex) + '*(1'

                rev = reversibility(2)
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            antStr = antStr + ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if irr_stoic[each] == 2:
                            antStr = antStr + ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            antStr = antStr + ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if rev_stoic[each] == -1:
                            antStr = antStr + ' - ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if rev_stoic[each] == -2:
                            antStr = antStr + ' - ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))

                antStr = antStr + ')'

            if r[0] == TReactionType.BIBI:
                # BiBi
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][1])
                antStr = antStr + '; ' + E[0:2] + 'v' + str(reactionIndex) + '*(1'

                rev = reversibility(3)
                if not rev:
                    for each in irr_stoic:
                        if irr_stoic[each] == 1:
                            antStr = antStr + ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if irr_stoic[each] == 2:
                            antStr = antStr + ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                else:
                    for each in rev_stoic:
                        if rev_stoic[each] == 1:
                            antStr = antStr + ' + ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if rev_stoic[each] == 2:
                            antStr = antStr + ' + ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if rev_stoic[each] == -1:
                            antStr = antStr + ' - ' + 'log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))
                        if rev_stoic[each] == -2:
                            antStr = antStr + ' - ' + '2*log(S' + str(each) + '/hs_' + str(each) + '_' + str(reactionIndex) + ')'
                            hs.append('hs_' + str(each) + '_' + str(reactionIndex))

                antStr = antStr + ')'
            antStr = antStr + '\n'
        antStr = antStr + '\n'

        if 'deg' in kinetics[2]:
            reactionIndex += 1
            parameterIndex = reactionIndex
            for sp in floatingIds:
                antStr = antStr + 'J' + str(reactionIndex) + ': S' + str(sp) + ' ->; ' + 'k' + str(reactionIndex) + '*' + 'S' + str(sp) + '\n'
                reactionIndex += 1
        antStr = antStr + '\n'

        for index, r in enumerate(reactionListCopy):
            if kinetics[1] == 'trivial':
                antStr = antStr + 'v' + str(index) + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0], scale=kinetics[3][kinetics[2].index('v')][1]
                                    - kinetics[3][kinetics[2].index('v')][0])
                antStr = antStr + 'v' + str(index) + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0], kinetics[3][kinetics[2].index('v')][1])
                antStr = antStr + 'v' + str(index) + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0], scale=kinetics[3][kinetics[2].index('v')][1])
                    if const >= 0:
                        antStr = antStr + 'v' + str(index) + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0], s=kinetics[3][kinetics[2].index('v')][1])
                antStr = antStr + 'v' + str(index) + ' = ' + str(const) + '\n'

        antStr = antStr + '\n'

        for each in hs:
            if kinetics[1] == 'trivial':
                antStr = antStr + each + ' = 1\n'
            else:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('hs')][0], scale=kinetics[3][kinetics[2].index('hs')][1]
                                    - kinetics[3][kinetics[2].index('hs')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'

        if 'deg' in kinetics[2]:
            # Next the degradation rate constants
            for _ in floatingIds:

                if kinetics[1] == 'trivial':
                    antStr = antStr + 'k' + str(parameterIndex) + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('deg')][0], scale=kinetics[3][kinetics[2].index('deg')][1]
                                        - kinetics[3][kinetics[2].index('deg')][0])
                    antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('deg')][0], kinetics[3][kinetics[2].index('deg')][1])
                    antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('deg')][0], scale=kinetics[3][kinetics[2].index('deg')][1])
                        if const >= 0:
                            antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('deg')][0], s=kinetics[3][kinetics[2].index('deg')][1])
                    antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                parameterIndex += 1
        antStr = antStr + '\n'

        # todo: Save this later. Allow for different distributions types depending on parameter type
        # if kinetics[1] == 'uniform':
        #     const = uniform.rvs(loc=kinetics[3][kinetics[2].index('hs')][0], scale=kinetics[3][kinetics[2].index('hs')][1]
        #                         - kinetics[3][kinetics[2].index('hs')][0])
        #     antStr = antStr + each + ' = ' + str(const) + '\n'
        #
        # if kinetics[1] == 'loguniform':
        #     const = uniform.rvs(kinetics[3][kinetics[2].index('hs')][0], kinetics[3][kinetics[2].index('hs')][1])
        #     antStr = antStr + each + ' = ' + str(const) + '\n'
        #
        # if kinetics[1] == 'normal':
        #     const = uniform.rvs(loc=kinetics[3][kinetics[2].index('hs')][0], scale=kinetics[3][kinetics[2].index('hs')][1])
        #     antStr = antStr + each + ' = ' + str(const) + '\n'
        #
        # if kinetics[1] == 'lognormal':
        #     const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0], scale=kinetics[3][kinetics[2].index('v')][1])
        #     antStr = antStr + each + ' = ' + str(const) + '\n'

    if 'modular' in kinetics[0]:

        ma = set()
        kma = set()
        ro = set()
        kf = set()
        kr = set()
        m = set()
        km = set()

        for reactionIndex, r in enumerate(reactionListCopy):

            antStr = antStr + 'J' + str(reactionIndex) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                antStr = antStr + 'S' + str(r[1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(r[2][0])

                rev = reversibility(0)

                if not rev:
                    antStr = antStr + '; ' + E
                    for i, reg in enumerate(r[3]):
                        if r[4][i] == -1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + ' * '
                        if r[4][i] == 1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'

                        ma.add('ma_' + str(reactionIndex) + '_' + str(reg))
                        kma.add('kma_' + str(reactionIndex) + '_' + str(reg))
                        ro.add('ro_' + str(reactionIndex) + '_' + str(reg))

                    antStr = antStr + '(kf_' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' - 1)' + E_end

                    if kinetics[0][8:10] == 'DM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' + 1)' + E_end

                    if kinetics[0][8:10] == 'SM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ')' + E_end

                    if kinetics[0][8:10] == 'FM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ')^(1/2)' + E_end

                    if kinetics[0][8:10] == 'PM':
                        pass

                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][0]))
                    kf.add('kf_' + str(reactionIndex))

                else:
                    antStr = antStr + '; ' + E
                    for i, reg in enumerate(r[3]):
                        if r[4][i] == -1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + ' * '
                        if r[4][i] == 1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'

                        ma.add('ma_' + str(reactionIndex) + '_' + str(reg))
                        kma.add('kma_' + str(reactionIndex) + '_' + str(reg))
                        ro.add('ro_' + str(reactionIndex) + '_' + str(reg))

                    antStr = antStr + '(kf_' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' - kr_' \
                        + str(reactionIndex) + '*(S' + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) \
                        + ')^m_' + str(reactionIndex) + '_' + str(r[2][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' + (1 + S' \
                            + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ' - 1)' + E_end

                    if kinetics[0][8:10] == 'DM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' + (S' \
                            + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ' + 1)' + E_end

                    if kinetics[0][8:10] == 'SM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')' + E_end

                    if kinetics[0][8:10] == 'FM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')^(1/2)' + E_end

                    if kinetics[0][8:10] == 'PM':
                        pass

                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][0]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[2][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[2][0]))
                    kf.add('kf_' + str(reactionIndex))
                    kr.add('kr_' + str(reactionIndex))

            if r[0] == TReactionType.BIUNI:
                # BiUni
                antStr = antStr + 'S' + str(r[1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(r[1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(r[2][0])

                rev = reversibility(1)

                if not rev:
                    antStr = antStr + '; ' + E
                    for i, reg in enumerate(r[3]):
                        if r[4][i] == -1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + ' * '
                        if r[4][i] == 1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'

                        ma.add('ma_' + str(reactionIndex) + '_' + str(reg))
                        kma.add('kma_' + str(reactionIndex) + '_' + str(reg))
                        ro.add('ro_' + str(reactionIndex) + '_' + str(reg))

                    antStr = antStr \
                        + '(kf_' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reactionIndex) + '_' + str(r[1][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ' - 1)' + E_end

                    if kinetics[0][8:10] == 'DM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ' + 1)' + E_end

                    if kinetics[0][8:10] == 'SM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ')' + E_end

                    if kinetics[0][8:10] == 'FM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ')^(1/2)' + E_end

                    if kinetics[0][8:10] == 'PM':
                        pass

                    # print(antStr)
                    # quit()

                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][0]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][1]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][1]))
                    kf.add('kf_' + str(reactionIndex))

                else:
                    antStr = antStr + '; ' + E
                    for i, reg in enumerate(r[3]):
                        if r[4][i] == -1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + ' * '
                        if r[4][i] == 1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'

                        ma.add('ma_' + str(reactionIndex) + '_' + str(reg))
                        kma.add('kma_' + str(reactionIndex) + '_' + str(reg))
                        ro.add('ro_' + str(reactionIndex) + '_' + str(reg))

                    antStr = antStr \
                        + '(kf_' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reactionIndex) + '_' + str(r[1][1]) + ' - kr_' + str(reactionIndex) + '*(S' \
                        + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) \
                        + ')^m_' + str(reactionIndex) + '_' + str(r[2][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ' + (1 + S' + str(r[2][0]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + ' - 1)' + E_end

                    if kinetics[0][8:10] == 'DM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ' + (S' + str(r[2][0]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + ' + 1)' + E_end

                    if kinetics[0][8:10] == 'SM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*' + '(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + '*(1 + S' + str(r[2][0]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + ')' + E_end

                    if kinetics[0][8:10] == 'FM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*' + '(S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + '*(S' + str(r[2][0]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + ')^(1/2)' + E_end

                    if kinetics[0][8:10] == 'PM':
                        pass

                    # print(antStr)
                    # quit()

                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][0]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][1]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[2][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][1]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[2][0]))
                    kf.add('kf_' + str(reactionIndex))
                    kr.add('kr_' + str(reactionIndex))

            if r[0] == TReactionType.UNIBI:
                # UniBi
                antStr = antStr + 'S' + str(r[1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(r[2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(r[2][1])

                rev = reversibility(2)

                if not rev:
                    antStr = antStr + '; ' + E
                    for i, reg in enumerate(r[3]):
                        if r[4][i] == -1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'
                        if r[4][i] == 1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'
                        
                        ma.add('ma_' + str(reactionIndex) + '_' + str(reg))
                        kma.add('kma_' + str(reactionIndex) + '_' + str(reg))
                        ro.add('ro_' + str(reactionIndex) + '_' + str(reg))
                    
                    antStr = antStr + '(kf_' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' - 1)' + E_end

                    if kinetics[0][8:10] == 'DM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' + 1)' + E_end

                    if kinetics[0][8:10] == 'SM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ')' + E_end

                    if kinetics[0][8:10] == 'FM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ')^(1/2)' + E_end

                    if kinetics[0][8:10] == 'PM':
                        pass

                    # print(antStr)
                    # quit()

                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][0]))
                    kf.add('kf_' + str(reactionIndex))

                else:
                    antStr = antStr + '; ' + E
                    for i, reg in enumerate(r[3]):
                        if r[4][i] == -1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'
                        if r[4][i] == 1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'

                        ma.add('ma_' + str(reactionIndex) + '_' + str(reg))
                        kma.add('kma_' + str(reactionIndex) + '_' + str(reg))
                        ro.add('ro_' + str(reactionIndex) + '_' + str(reg))

                    antStr = antStr + '(kf_' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' - kr_' \
                        + str(reactionIndex) + '*(S' + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) \
                        + ')^m_' + str(reactionIndex) + '_' + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' \
                        + str(reactionIndex) + '_' + str(r[2][1]) + ')^m_' + str(reactionIndex) + '_' + str(r[2][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' + (1 + S' \
                            + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][1]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][1]) + ' - 1)' + E_end

                    if kinetics[0][8:10] == 'DM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + ' + (S' \
                            + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][1]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][1]) + ' + 1)' + E_end

                    if kinetics[0][8:10] == 'SM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][1]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][1]) + ')' + E_end

                    if kinetics[0][8:10] == 'FM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][1]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][1]) + ')^(1/2)' + E_end

                    if kinetics[0][8:10] == 'PM':
                        pass

                    # print(antStr)
                    # quit()

                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][0]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[2][0]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[2][1]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[2][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[2][1]))
                    kf.add('kf_' + str(reactionIndex))
                    kr.add('kr_' + str(reactionIndex))

            if r[0] == TReactionType.BIBI:
                # BiBi
                antStr = antStr + 'S' + str(r[1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(r[1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(r[2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(r[2][1])

                rev = reversibility(3)

                if not rev:
                    antStr = antStr + '; ' + E
                    for i, reg in enumerate(r[3]):
                        if r[4][i] == -1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'
                        if r[4][i] == 1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'

                        ma.add('ma_' + str(reactionIndex) + '_' + str(reg))
                        kma.add('kma_' + str(reactionIndex) + '_' + str(reg))
                        ro.add('ro_' + str(reactionIndex) + '_' + str(reg))

                    antStr = antStr \
                        + '(kf_' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reactionIndex) + '_' + str(r[1][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ' - 1)' + E_end

                    if kinetics[0][8:10] == 'DM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ' + 1)' + E_end

                    if kinetics[0][8:10] == 'SM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ')' + E_end

                    if kinetics[0][8:10] == 'FM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ')^(1/2)' + E_end

                    if kinetics[0][8:10] == 'PM':
                        pass

                    # print(antStr)
                    # quit()

                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][0]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][1]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][1]))
                    kf.add('kf_' + str(reactionIndex))

                else:
                    antStr = antStr + '; ' + E
                    for i, reg in enumerate(r[3]):
                        if r[4][i] == -1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'
                        if r[4][i] == 1:
                            antStr = antStr + '(' + 'ro_' + str(reactionIndex) + '_' + str(reg) + ' + (1 - ' + 'ro_' \
                                + str(reactionIndex) + '_' + str(reg) + ')*(S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + ')/(1 + S' + str(reg) + '/kma_' + str(reactionIndex) \
                                + '_' + str(reg) + '))^ma_' + str(reactionIndex) + '_' + str(reg) + '*'

                        ma.add('ma_' + str(reactionIndex) + '_' + str(reg))
                        kma.add('kma_' + str(reactionIndex) + '_' + str(reg))
                        ro.add('ro_' + str(reactionIndex) + '_' + str(reg))

                    antStr = antStr \
                        + '(kf_' + str(reactionIndex) + '*(S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                        + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(' + 'S' \
                        + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                        + str(reactionIndex) + '_' + str(r[1][1]) + ' - kr_' + str(reactionIndex) + '*(S' \
                        + str(r[2][0]) + '/km_' + str(reactionIndex) + '_' + str(r[2][0]) \
                        + ')^m_' + str(reactionIndex) + '_' + str(r[2][0]) + '*(S' \
                        + str(r[2][1]) + '/km_' + str(reactionIndex) + '_' + str(r[2][1]) \
                        + ')^m_' + str(reactionIndex) + '_' + str(r[2][1]) + ')'

                    if kinetics[0][8:10] == 'CM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ' + (1 + S' + str(r[2][0]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' + str(reactionIndex) + '_' + str(r[2][1]) \
                            + ')^m_' + str(reactionIndex) + '_' + str(r[2][1]) + ' - 1)' + E_end

                    if kinetics[0][8:10] == 'DM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + ' + (S' + str(r[2][0]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' + str(reactionIndex) + '_' + str(r[2][1]) \
                            + ')^m_' + str(reactionIndex) + '_' + str(r[2][1]) + ' + 1)' + E_end

                    if kinetics[0][8:10] == 'SM':

                        antStr = antStr + '/((1 + S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(1 + S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + '*(1 + S' + str(r[2][0]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + '*(1 + S' + str(r[2][1]) + '/km_' + str(reactionIndex) + '_' + str(r[2][1]) \
                            + ')^m_' + str(reactionIndex) + '_' + str(r[2][1]) + ')' + E_end

                    if kinetics[0][8:10] == 'FM':

                        antStr = antStr + '/((S' + str(r[1][0]) + '/km_' + str(reactionIndex) \
                            + '_' + str(r[1][0]) + ')^m_' + str(reactionIndex) + '_' + str(r[1][0]) + '*(S' \
                            + str(r[1][1]) + '/km_' + str(reactionIndex) + '_' + str(r[1][1]) + ')^m_' \
                            + str(reactionIndex) + '_' + str(r[1][1]) + '*(S' + str(r[2][0]) + '/km_' \
                            + str(reactionIndex) + '_' + str(r[2][0]) + ')^m_' + str(reactionIndex) + '_' \
                            + str(r[2][0]) + '*(S' + str(r[2][1]) + '/km_' + str(reactionIndex) + '_' + str(r[2][1]) \
                            + ')^m_' + str(reactionIndex) + '_' + str(r[2][1]) + ')^(1/2)' + E_end

                    if kinetics[0][8:10] == 'PM':
                        pass

                    # print(antStr)
                    # quit()

                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][0]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[1][1]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[2][0]))
                    km.add('km_' + str(reactionIndex) + '_' + str(r[2][1]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[1][1]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[2][0]))
                    m.add('m_' + str(reactionIndex) + '_' + str(r[2][1]))
                    kf.add('kf_' + str(reactionIndex))
                    kr.add('kr_' + str(reactionIndex))

            antStr = antStr + '\n'
        antStr = antStr + '\n'

        if 'deg' in kinetics[2]:
            reactionIndex += 1
            parameterIndex = reactionIndex
            for sp in floatingIds:
                antStr = antStr + 'J' + str(reactionIndex) + ': S' + str(sp) + ' ->; ' + 'k' + str(reactionIndex) + '*' + 'S' + str(sp) + '\n'
                reactionIndex += 1
            antStr = antStr + '\n'

        ro = list(ro)
        ro.sort()
        if kinetics[1] == 'trivial':
            for each in ro:
                antStr = antStr + each + ' = ' + str(1) + '\n'
        else:
            for each in ro:
                antStr = antStr + each + ' = ' + str(uniform.rvs(loc=0, scale=1)) + '\n'
        antStr = antStr + '\n'

        kf = list(kf)
        kf.sort()
        for each in kf:
        
            if kinetics[1] == 'trivial':
                antStr = antStr + each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kf')][0], scale=kinetics[3][kinetics[2].index('kf')][1]
                                    - kinetics[3][kinetics[2].index('kf')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kf')][0], kinetics[3][kinetics[2].index('kf')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kf')][0], scale=kinetics[3][kinetics[2].index('kf')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kf')][0], s=kinetics[3][kinetics[2].index('kf')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

        antStr = antStr + '\n'

        kr = list(kr)
        kr.sort()
        for each in kr:

            if kinetics[1] == 'trivial':
                antStr = antStr + each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('kr')][0], scale=kinetics[3][kinetics[2].index('kr')][1]
                                    - kinetics[3][kinetics[2].index('kr')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('kr')][0], kinetics[3][kinetics[2].index('kr')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('kr')][0], scale=kinetics[3][kinetics[2].index('kr')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('kr')][0], s=kinetics[3][kinetics[2].index('kr')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

        antStr = antStr + '\n'

        km = list(km)
        km.sort()
        for each in km:
        
            if kinetics[1] == 'trivial':
                antStr = antStr + each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('km')][0], scale=kinetics[3][kinetics[2].index('km')][1]
                                    - kinetics[3][kinetics[2].index('km')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('km')][0], kinetics[3][kinetics[2].index('km')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('km')][0], scale=kinetics[3][kinetics[2].index('km')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('km')][0], s=kinetics[3][kinetics[2].index('km')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

        antStr = antStr + '\n'

        kma = list(kma)
        kma.sort()
        for each in kma:

            if kinetics[1] == 'trivial':
                antStr = antStr + each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('km')][0], scale=kinetics[3][kinetics[2].index('km')][1]
                                    - kinetics[3][kinetics[2].index('km')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('km')][0], kinetics[3][kinetics[2].index('km')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('km')][0], scale=kinetics[3][kinetics[2].index('km')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('km')][0], s=kinetics[3][kinetics[2].index('km')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

        antStr = antStr + '\n'

        m = list(m)
        m.sort()
        for each in m:
        
            if kinetics[1] == 'trivial':
                antStr = antStr + each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('mol')][0], scale=kinetics[3][kinetics[2].index('mol')][1]
                                    - kinetics[3][kinetics[2].index('mol')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('mol')][0], kinetics[3][kinetics[2].index('mol')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('mol')][0], scale=kinetics[3][kinetics[2].index('mol')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('mol')][0], s=kinetics[3][kinetics[2].index('mol')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

        antStr = antStr + '\n'

        ma = list(ma)
        ma.sort()
        for each in ma:

            if kinetics[1] == 'trivial':
                antStr = antStr + each + ' = 1\n'

            if kinetics[1] == 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('mol')][0], scale=kinetics[3][kinetics[2].index('mol')][1]
                                    - kinetics[3][kinetics[2].index('mol')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('mol')][0], kinetics[3][kinetics[2].index('mol')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            if kinetics[1] == 'normal':
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('mol')][0], scale=kinetics[3][kinetics[2].index('mol')][1])
                    if const >= 0:
                        antStr = antStr + each + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] == 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('mol')][0], s=kinetics[3][kinetics[2].index('mol')][1])
                antStr = antStr + each + ' = ' + str(const) + '\n'

        antStr = antStr + '\n'
        
        if 'deg' in kinetics[2]:
            # Next the degradation rate constants
            for _ in floatingIds:

                if kinetics[1] == 'trivial':
                    antStr = antStr + 'k' + str(parameterIndex) + ' = 1\n'

                if kinetics[1] == 'uniform':
                    const = uniform.rvs(loc=kinetics[3][kinetics[2].index('deg')][0], scale=kinetics[3][kinetics[2].index('deg')][1]
                                        - kinetics[3][kinetics[2].index('deg')][0])
                    antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'loguniform':
                    const = loguniform.rvs(kinetics[3][kinetics[2].index('deg')][0], kinetics[3][kinetics[2].index('deg')][1])
                    antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                if kinetics[1] == 'normal':
                    while True:
                        const = norm.rvs(loc=kinetics[3][kinetics[2].index('deg')][0], scale=kinetics[3][kinetics[2].index('deg')][1])
                        if const >= 0:
                            antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'
                            break

                if kinetics[1] == 'lognormal':
                    const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('deg')][0], s=kinetics[3][kinetics[2].index('deg')][1])
                    antStr = antStr + 'k' + str(parameterIndex) + ' = ' + str(const) + '\n'

                parameterIndex += 1
        antStr = antStr + '\n'
        
    # quit()

    def getICvalue(ICind):

        # todo: add additional distributions (maybe, maybe not)

        IC = None
        if ic_params == 'trivial':
            IC = 1
        if isinstance(ic_params, list) and ic_params[0] == 'uniform':
            IC = uniform.rvs(loc=ic_params[1], scale=ic_params[2]-ic_params[1])
        if isinstance(ic_params, list) and ic_params[0] == 'loguniform':
            IC = loguniform.rvs(ic_params[1], ic_params[2])
        if isinstance(ic_params, list) and ic_params[0] == 'normal':
            IC = norm.rvs(loc=ic_params[1], scale=ic_params[2])
        if isinstance(ic_params, list) and ic_params[0] == 'lognormal':
            IC = lognorm.rvs(scale=ic_params[1], s=ic_params[2])
        if isinstance(ic_params, list) and ic_params[0] == 'list':
            IC = ic_params[1][ICind]
        if ic_params is None:
            IC = uniform.rvs(loc=0, scale=10)

        return IC

    for index, b in enumerate(boundaryIds):
        ICvalue = getICvalue(b, )
        antStr = antStr + 'S' + str(b) + ' = ' + str(ICvalue) + '\n'

    antStr = antStr + '\n'
    for index, b in enumerate(floatingIds):
        ICvalue = getICvalue(b)
        antStr = antStr + 'S' + str(b) + ' = ' + str(ICvalue) + '\n'

    if add_E:
        antStr = antStr + '\n'
        for index, r in enumerate(reactionListCopy):
            antStr = antStr + 'E' + str(index) + ' = 1\n'

    return antStr
