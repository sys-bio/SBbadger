# -*- coding: utf-8 -*-
""" A module for creating random network models """

import random
from dataclasses import dataclass
import numpy as np
import sys
from copy import deepcopy
from scipy.stats import norm, lognorm, uniform, loguniform
from collections import defaultdict

# General settings for the package
@dataclass
class Settings:
    """ Settings to control some properties of the network generation"""
    rateConstantScale = 1.0
    """ How much the rate cosntants are scaled by. By default rate constants ge values between 0 and 1.0"""
    allowMassViolatingReactions = False
    """ If set to true, reactions such as A + B -> A are allowed"""
    addDegradationSteps = False;
    """Set true if you want every floating node (not boundary nodes) to have a degradation step"""
    # todo: add degradation step for species with no outlet (downstream boundries)
    # todo: add function to as inputs (upstream boundries)
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
        """Restore the default settings for the reaction mechanism propabilities"""
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


def _generateReactionList(nSpecies, kinetics=None, rxn_prob=None, rev_prob=False, constDist=None,
                          constParams=None, inDist='random', outDist='random', jointDist=None, inRange=None,
                          outRange=None, jointRange=None, cutOff=1.0):

    # todo: expand kinetics?
    # todo: mass balance

    paramDists = defaultdict(list)
    if kinetics[1] is not 'trivial':
        for i, each in enumerate(kinetics[2]):
            paramDists[each] = kinetics[3][i]
    else:
        for i, each in enumerate(kinetics[2]):
            paramDists[each] = [1]


    # if jointDist and (inDist is not 'random' or outDist is not 'random'):
    #     print("You have provided both a joint distribution and onr or both of the input and output distributions")
    #     sys.exit(1)
    #
    # if rxn_prob:
    #     try:
    #         if round(sum(rxn_prob), 10) != 1:
    #             raise ValueError
    #     except ValueError:
    #         print('Your stated probabilities are', rxn_prob, 'and they do not add to 1.')
    #         sys.exit(1)
    #
    # if isinstance(rev_prob, list):
    #     try:
    #         if any(x < 0.0 for x in rev_prob) or any(x > 1.0 for x in rev_prob):
    #             raise ValueError
    #     except ValueError:
    #         print('One or more of your reversibility probabilities is not between 0 and 1')
    #         sys.exit(1)
    #
    # if isinstance(rev_prob, float):
    #     try:
    #         if rev_prob < 0.0 or rev_prob > 1.0:
    #             raise ValueError
    #     except ValueError:
    #         print('Your reversibility probability is not between 0 and 1')
    #         sys.exit(1)
    #
    # if isinstance(jointRange, list) and jointRange[0] < 1:
    #     print("Node degree cannot be less than 1.")
    #     sys.exit(1)
    # if isinstance(inRange, list) and inRange[0] < 1:
    #     print("Node degree cannot be less than 1.")
    #     sys.exit(1)
    # if isinstance(outRange, list) and outRange[0] < 1:
    #     print("Node degree cannot be less than 1.")
    #     sys.exit(1)

    jointSamples = []
    inSamples = []
    outSamples = []

    def single_unbounded_pmf(sdist):
        """Assumes starting degree of 1 and extends until cutoff found"""

        deg = 1
        while True:
            dist = []
            for i in range(deg):
                dist.append(sdist(i + 1))
            distsum = sum(dist)
            distN = [x * nSpecies / distsum for x in dist]

            if distN[-1] < cutOff:
                pmf = dist[:-1]
                sumDistF = sum(pmf)
                pmf = [x / sumDistF for x in pmf]
                break
            else:
                deg += 1

        return pmf

    def single_bounded_pmf(sdist, drange):
        """Start with given degree range and trim until cutoffs found"""

        distInd = [i for i in range(drange[0], drange[1] + 1)]
        pmf = [sdist(i) for i in range(drange[0], drange[1] + 1)]
        distSum = min(sum(pmf), 1)
        pmf = [x / distSum for x in pmf]
        dist = [x * nSpecies / distSum for x in pmf]

        while dist[0] < 1 or dist[-1] < 1:
            if dist[0] < dist[-1]:
                distInd.pop(0)
                pmf.pop(0)
            else:
                distInd.pop(-1)
                pmf.pop(-1)
            distSum = sum(pmf)
            dist = [x * nSpecies / distSum for x in pmf]
            pmf = [x / distSum for x in pmf]
        startdeg = distInd[0]

        return pmf, startdeg

    def sample_single_distribution(pmf, startDeg):

        samplest = [0 for _ in pmf]
        outind = [i for i in range(len(pmf))]

        i = 0
        while i < nSpecies:
            ind = random.choices(outind, pmf)[0]
            samplest[ind] += 1
            i += 1

        samples = []
        for i in range(len(pmf)):
            if samplest[i] > 0:
                samples.append((startDeg+i, samplest[i]))

        return samples

    def sample_both_pmfs(pmf1, startDeg1, pmf2, startDeg2):

        # sample the first distribution
        samples1t = [0 for _ in pmf1]

        ind1 = [i for i in range(len(pmf1))]
        ind2 = [i for i in range(len(pmf2))]

        i = 0
        while i < nSpecies:
            ind = random.choices(ind1, pmf1)[0]
            samples1t[ind] += 1
            i += 1

        samples1 = []
        for i in range(len(pmf1)):
            if samples1t[i] > 0:
                samples1.append((startDeg1+i, samples1t[i]))

        # sample the second distribution so that the number of edges match
        edges1 = 0
        for each in samples1:
            edges1 += each[0] * each[1]
        numTrys = 0
        while True:
            numTrys += 1

            edges2 = 0
            nodes = 0
            samples2t = [0 for _ in pmf2]

            while edges2 < edges1 and nodes < nSpecies:
                ind = random.choices(ind2, pmf2)[0]
                samples2t[ind] += 1
                edges2 += ind + startDeg2
                nodes += 1

            if edges2 == edges1:
                samples2 = []
                for i in range(len(pmf2)):
                    if samples2t[i] > 0:
                        samples2.append((startDeg2 + i, samples2t[i]))
                break

            if numTrys == 10000:
                print("\nReconciliation of the input and output distributions was attempted 10000 times.\n"
                      "Consider revising these distributions.")
                sys.exit(1)

        return samples1, samples2

    def find_edge_count(dist):

        edgeCount = 0
        for each in dist:
            edgeCount += each[0] * each[1]

        return edgeCount

    def find_edges_expected_value(xDist, xRange):

        edgeEV = 0
        for i, each in enumerate(xDist):
            if isinstance(xRange, list):
                edgeEV += each * xRange[i] * nSpecies
            elif isinstance(xRange, int):
                edgeEV += each * (i+xRange) * nSpecies
            else:
                edgeEV += each * (i+1) * nSpecies

        return edgeEV

    def trim_pmf(edges1, dist2):

        edges2 = 0
        mDeg = 0
        while edges2 < edges1:

            mDeg += 1
            dist = [dist2(i+1) for i in range(mDeg)]
            sumDist = sum(dist)
            newDist = [x/sumDist for x in dist]
            edgeDist = [newDist[i]/newDist[-1] for i in range(len(newDist))]
            edges2 = 0
            for i, each in enumerate(edgeDist):
                edges2 += each*(i+1)

        dist = [dist2(i+1) for i in range(mDeg-1)]
        sumDist = sum(dist)
        newDist = [x/sumDist for x in dist]

        return newDist

    def trim_pmf_2(edgesTarget, pmf, startDeg):

        degRange = [i + startDeg for i in range(len(pmf))]

        edges = 0
        for i, each in enumerate(pmf):
            edges += each * nSpecies * degRange[i]

        while edges > edgesTarget:

            pmf.pop(-1)
            sumPmf = sum(pmf)
            pmf = [x/sumPmf for x in pmf]
            edges = 0
            for i, each in enumerate(pmf):
                edges += each * nSpecies * degRange[i]

        return pmf

    def joint_unbounded_pmf(jointDist):

        dist = [(1, 1)]
        dscores = [jointDist(1, 1)]
        dsum = dscores[-1]
        edge = []
        edgeScores = []

        while True:

            for each in dist:
                each1 = (each[0] + 1, each[1])
                each2 = (each[0], each[1] + 1)
                each3 = (each[0] + 1, each[1] + 1)
                if each1 not in dist and each1 not in edge:
                    edge.append(each1)
                    edgeScores.append(jointDist(each1[0], each1[1]))
                if each2 not in dist and each2 not in edge:
                    edge.append(each2)
                    edgeScores.append(jointDist(each2[0], each2[1]))
                if each3 not in dist and each3 not in edge:
                    edge.append(each3)
                    edgeScores.append(jointDist(each3[0], each3[1]))

            tiles = []
            lowScore = 0
            for i, each in enumerate(edgeScores):
                if each == lowScore:
                    tiles.append(i)
                elif each > lowScore:
                    tiles = [i]
                    lowScore = each

            newDist = deepcopy(dist)
            newDscores = deepcopy(dscores)

            for i in tiles:
                newDist.append(edge[i])
                newDscores.append(jointDist(edge[i][0], edge[i][1]))
                dsum += jointDist(edge[i][0], edge[i][1])

            scaledDscores = []
            for each in newDscores:
                scaledDscores.append(nSpecies * each / dsum)

            if any(x < cutOff for x in scaledDscores):
                break

            dist = newDist
            dscores = newDscores

            newEdge = []
            newEdgeScores = []

            for i, each in enumerate(edge):
                if i not in tiles:
                    newEdge.append(each)
                    newEdgeScores.append(edgeScores[i])

            edge = newEdge
            edgeScores = newEdgeScores

        pmf = []
        dsum = sum(dscores)
        for i, each in enumerate(dist):
            pmf.append([each[0], each[1], dscores[i] / dsum])

        return pmf

    def sample_joint(pmf):

        cells, pmf = [[x[0], x[1]] for x in pmf], [x[2] for x in pmf]

        ind = [i for i, each in enumerate(pmf)]

        count = 0
        while True:
            count += 1
            samplest = [0 for _ in pmf]
            i = 0
            while i < nSpecies:
                sample = random.choices(ind, pmf)[0]
                samplest[sample] += 1
                i += 1

            outEdges = 0
            inEdges = 0
            samples = []
            for i, each in enumerate(samplest):

                outEdges += each*cells[i][0]
                inEdges += each*cells[i][1]
                samples.append((cells[i][0], cells[i][1], each))

            if outEdges == inEdges:

                return samples

            if count == 10000:
                print("\nYour joint distribution was sampled 10000 times.\n"
                      "Reconciliation of the outgoing and incoming edges was not acheived.\n"
                      "Consider revising this distribution.")
                sys.exit(1)

    def joint_bounded_pmf(jointDist, jointRange):

        pmf = []
        for i in range(jointRange[0], jointRange[1]+1):
            for j in range(jointRange[0], jointRange[1]+1):
                pmf.append([jointDist(i, j), 0., (i, j)])
        pmfSum = sum(pmf[i][0] for i in range(len(pmf)))
        pmf = [[pmf[i][0]/pmfSum, pmf[i][0]*nSpecies/pmfSum, pmf[i][2]] for i in range(len(pmf))]
        pmf.sort(key=lambda x: x[1])
        while pmf[0][1] < cutOff:
            value = pmf[0][1]
            pmf = [x for x in pmf if x[1] != value]
            pmfSum = sum(pmf[i][0] for i in range(len(pmf)))
            pmf = [[pmf[i][0]/pmfSum, pmf[i][0]*nSpecies/pmfSum, pmf[i][2]] for i in range(len(pmf))]

        pmfT = []
        for each in pmf:
            pmfT.append([each[2][0], each[2][1], each[0]])
        pmf = pmfT

        return pmf

    inputCase = None

    if outDist == 'Random' and inDist == 'Random':
        inputCase = 0

    if callable(outDist) and inDist == 'random' and outRange is None:
        inputCase = 1

    if callable(outDist) and inDist == 'random' and isinstance(outRange, list):
        inputCase = 2

    if isinstance(outDist, list) and inDist == 'random' and all(isinstance(x[1], float) for x in outDist):
        inputCase = 3

    if isinstance(outDist, list) and inDist == 'random' and all(isinstance(x[1], int) for x in outDist):
        inputCase = 4

    if outDist == 'random' and callable(inDist) and inRange is None:
        inputCase = 5

    if outDist == 'random' and callable(inDist) and isinstance(inRange, list):
        inputCase = 6

    if outDist == 'random' and isinstance(inDist, list) and all(isinstance(x[1], float) for x in inDist):
        inputCase = 7

    if outDist == 'random' and isinstance(inDist, list) and all(isinstance(x[1], int) for x in inDist):
        inputCase = 8

    if callable(outDist) and callable(inDist):
        if inDist == outDist and inRange is None and outRange is None:
            inputCase = 9
        if inDist == outDist and inRange and inRange == outRange:
            inputCase = 10
        if inDist == outDist and inRange != outRange:
            inputCase = 11  # todo (maybe): add this (unlikely edge) case
        if inDist != outDist and inRange is None and outRange is None:
            inputCase = 12
        if inDist != outDist and inRange and inRange == outRange:
            inputCase = 13
        if inDist != outDist and inRange != outRange:
            inputCase = 14  # todo (maybe): add this (unlikely edge) case

    if isinstance(outDist, list) and isinstance(inDist, list):
        if all(isinstance(x[1], int) for x in outDist) and all(isinstance(x[1], int) for x in inDist):
            inputCase = 15
        if all(isinstance(x[1], float) for x in outDist) and all(isinstance(x[1], float) for x in inDist):
            inputCase = 16

    if callable(jointDist):
        if not jointRange:
            inputCase = 17
        if jointRange:
            # todo: include case defining different ranges for outgoing and incoming edges
            inputCase = 18

    if isinstance(jointDist, list):
        if all(isinstance(x[2], float) for x in jointDist):
            inputCase = 19
        if all(isinstance(x[2], int) for x in jointDist):
            inputCase = 20

    # ---------------------------------------------------------------------------

    if inputCase == 1:

        pmf = single_unbounded_pmf(outDist)
        outSamples = sample_single_distribution(pmf, 1)

    if inputCase == 2:

        pmf, startDeg = single_bounded_pmf(outDist, outRange)
        outSamples = sample_single_distribution(pmf, startDeg)

    if inputCase == 3:

        pmf = [x[1] for x in outDist]
        if sum(pmf) != 1:
            print("The PMF does not add to 1")
            sys.exit(1)

        startDeg = outDist[0][0]
        outSamples = sample_single_distribution(pmf, startDeg)

    if inputCase == 4:
        outSamples = outDist

    if inputCase == 5:

        pmf = single_unbounded_pmf(inDist)
        inSamples = sample_single_distribution(pmf, 1)

    if inputCase == 6:

        pmf, startDeg = single_bounded_pmf(inDist, inRange)
        inSamples = sample_single_distribution(pmf, startDeg)

    if inputCase == 7:

        pmf = [x[1] for x in inDist]

        if sum(pmf) != 1:
            print("The PMF does not add to 1")
            sys.exit(1)
        startDeg = inDist[0][0]
        inSamples = sample_single_distribution(pmf, startDeg)

    if inputCase == 8:
        inSamples = inDist

    if inputCase == 9:

        pmf1 = single_unbounded_pmf(outDist)
        pmf2 = single_unbounded_pmf(inDist)
        inOrOut = random.randint(0, 1)  # choose which distribution is guaranteed nSpecies
        if inOrOut:
            inSamples, outSamples = sample_both_pmfs(pmf2, 1, pmf1, 1)
        else:
            outSamples, inSamples = sample_both_pmfs(pmf1, 1, pmf2, 1)

    if inputCase == 10:

        pmf1, startDeg1 = single_bounded_pmf(outDist, outRange)
        pmf2, startDeg2 = single_bounded_pmf(inDist, inRange)
        inOrOut = random.randint(0, 1)  # choose which distribution is guaranteed nSpecies
        if inOrOut:
            inSamples, outSamples = sample_both_pmfs(pmf2, startDeg2, pmf1, startDeg1)
        else:
            outSamples, inSamples = sample_both_pmfs(pmf1, startDeg1, pmf2, startDeg2)

    if inputCase == 11:

        pass  # todo: unlikely edges case

    if inputCase == 12:

        pmfOut = single_unbounded_pmf(outDist)
        pmfIn = single_unbounded_pmf(inDist)

        edgeEVout = find_edges_expected_value(pmfOut, outRange)
        edgeEVin = find_edges_expected_value(pmfIn, inRange)

        if edgeEVin < edgeEVout:
            pmfOut = trim_pmf(edgeEVin, outDist)
            inSamples, outSamples = sample_both_pmfs(pmfIn, 1, pmfOut, 1)
        if edgeEVin > edgeEVout:
            pmfIn = trim_pmf(edgeEVout, inDist)
            outSamples, inSamples = sample_both_pmfs(pmfOut, 1, pmfIn, 1)

    if inputCase == 13:

        pmf1, startDeg1 = single_bounded_pmf(outDist, outRange)
        pmf2, startDeg2 = single_bounded_pmf(inDist, inRange)

        edgeEVout = find_edges_expected_value(pmf1, startDeg1)
        edgeEVin = find_edges_expected_value(pmf2, startDeg2)

        if edgeEVin < edgeEVout:
            pmf1 = trim_pmf_2(edgeEVin, pmf1, startDeg1)
            inSamples, outSamples = sample_both_pmfs(pmf2, startDeg2, pmf1, startDeg1)
        if edgeEVin > edgeEVout:
            pmf2 = trim_pmf_2(edgeEVout, pmf2, startDeg2)
            outSamples, inSamples = sample_both_pmfs(pmf1, startDeg1, pmf2, startDeg2)

    if inputCase == 14:

        pass  # todo: unlikely edges case

    if inputCase == 15:

        if find_edge_count(outDist) != find_edge_count(inDist):
            print("The edges counts for the input and output distributions must match.")
            sys.exit(1)

        outSamples = outDist
        inSamples = inDist

    if inputCase == 16:

        pmf1 = [x[1] for x in outDist]
        pmf2 = [x[1] for x in inDist]

        edgeEVout = find_edges_expected_value(pmf1, outDist[0][0])
        edgeEVin = find_edges_expected_value(pmf2, inDist[0][0])

        if edgeEVin < edgeEVout:
            pmf1 = trim_pmf_2(edgeEVin, pmf1, outDist[0][0])
            inSamples, outSamples = sample_both_pmfs(pmf2, inDist[0][0], pmf1, outDist[0][0])
        if edgeEVin > edgeEVout:
            pmf2 = trim_pmf_2(edgeEVout, pmf2, inDist[0][0])
            outSamples, inSamples = sample_both_pmfs(pmf1, outDist[0][0], pmf2, inDist[0][0])

    if inputCase == 17:

        pmf = joint_unbounded_pmf(jointDist)
        jointSamples = sample_joint(pmf)

    if inputCase == 18:

        pmf = joint_bounded_pmf(jointDist, jointRange)
        jointSamples = sample_joint(pmf)

    if inputCase == 19:

        jointSamples = sample_joint(jointDist)

    if inputCase == 20:

        jointSamples = jointDist

    # =======================================================================

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

    def reversibility(rxnType):

        rev = False
        if rev_prob and isinstance(rev_prob, list):
            rev = random.choices([True, False], [rev_prob[rxnType], 1.0 - rev_prob[rxnType]])[0]
        if isinstance(rev_prob, float) or isinstance(rev_prob, int):
            rev = random.choices([True, False], [rev_prob, 1 - rev_prob])[0]

        # todo: add straight Boolean case

        return rev

    def getMassAction3RateConstants(rxnType, rev):

        constants = defaultdict(list)

        # todo: make the trivial case a stand-alone function
        if kinetics[1] == 'trivial':
            
            if rev:
                constants['kf'].append(1)
                constants['kr'].append(1)
            else:
                constants['kc'].append(1)

        if kinetics[1] == 'uniform':
            
            if rev:
                constants['kf'].append(uniform.rvs(loc=paramDists['kf'][0], scale=paramDists['kf'][1]-paramDists['kf'][0]))
                constants['kr'].append(uniform.rvs(loc=paramDists['kr'][0], scale=paramDists['kr'][1]-paramDists['kr'][0]))
            else:
                constants['kc'].append(uniform.rvs(loc=paramDists['kc'][0], scale=paramDists['kc'][1]-paramDists['kc'][0]))
                
        if kinetics[1] == 'loguniform':
            
            if rev:
                constants['kf'].append(loguniform.rvs(paramDists['kf'][0], paramDists['kf'][1]))
                constants['kr'].append(loguniform.rvs(paramDists['kr'][0], paramDists['kr'][1]))
            else:
                constants['kc'].append(loguniform.rvs(paramDists['kc'][0], paramDists['kc'][1]))

        if kinetics[1] == 'norm':

            if rev:
                while True:
                    constant = norm.rvs(loc=paramDists['kf'][0], scale=paramDists['kf'][1])
                    if constant > 0:
                        constants['kf'].append(constant)
                        break
                while True:
                    constant = norm.rvs(loc=paramDists['kr'][0], scale=paramDists['kr'][1])
                    if constant > 0:
                        constants['kr'].append(constant)
                        break
            else:
                while True:
                    constant = norm.rvs(loc=paramDists['kc'][0], scale=paramDists['kc'][1])
                    if constant > 0:
                        constants['kc'].append(constant)
                        break

        if kinetics[1] == 'lognorm':

            if rev:
                constants['kf'].append(lognorm.rvs(scale=paramDists['kf'][0], s=paramDists['kf'][1]))
                constants['kr'].append(lognorm.rvs(scale=paramDists['kr'][0], s=paramDists['kr'][1]))
            else:
                constants['kc'].append(lognorm.rvs(scale=paramDists['kc'][0], s=paramDists['kc'][1]))

        return constants

    def getMassAction12RateConstants(rxnType, rev):

        constants = defaultdict(list)

        if kinetics[1] == 'uniform':

            if rev:
                constants['kf'].append(uniform.rvs(loc=paramDists['kf' + str(rxnType)][0], scale=paramDists['kf' + str(rxnType)][1] - paramDists['kf' + str(rxnType)][0]))
                constants['kr'].append(uniform.rvs(loc=paramDists['kr' + str(rxnType)][0], scale=paramDists['kr' + str(rxnType)][1] - paramDists['kr' + str(rxnType)][0]))
            else:
                constants['kc'].append(uniform.rvs(loc=paramDists['kc' + str(rxnType)][0], scale=paramDists['kc' + str(rxnType)][1] - paramDists['kc' + str(rxnType)][0]))

        if kinetics[1] == 'loguniform':

            if rev:
                constants['kf'].append(loguniform.rvs(paramDists['kf' + str(rxnType)][0], paramDists['kf' + str(rxnType)][1]))
                constants['kr'].append(loguniform.rvs(paramDists['kr' + str(rxnType)][0], paramDists['kr' + str(rxnType)][1]))
            else:
                constants['kc'].append(loguniform.rvs(paramDists['kc' + str(rxnType)][0], paramDists['kc' + str(rxnType)][1]))

        if kinetics[1] == 'norm':

            if rev:
                while True:
                    constant = norm.rvs(loc=paramDists['kf' + str(rxnType)][0], scale=paramDists['kf' + str(rxnType)][1])
                    if constant > 0:
                        constants['kf'].append(constant)
                        break
                while True:
                    constant = norm.rvs(loc=paramDists['kr' + str(rxnType)][0], scale=paramDists['kr' + str(rxnType)][1])
                    if constant > 0:
                        constants['kr'].append(constant)
                        break
            else:
                while True:
                    constant = norm.rvs(loc=paramDists['kc' + str(rxnType)][0], scale=paramDists['kc' + str(rxnType)][1])
                    if constant > 0:
                        constants['kc'].append(constant)
                        break

        if kinetics[1] == 'lognorm':

            if rev:
                constants['kf'].append(lognorm.rvs(scale=paramDists['kf' + str(rxnType)][0], s=paramDists['kf' + str(rxnType)][1]))
                constants['kr'].append(lognorm.rvs(scale=paramDists['kr' + str(rxnType)][0], s=paramDists['kr' + str(rxnType)][1]))
            else:
                constants['kc'].append(lognorm.rvs(scale=paramDists['kc' + str(rxnType)][0], s=paramDists['kc' + str(rxnType)][1]))

        return constants

    def getHanekom3RateConstants(rxnType):

        constants = defaultdict(list)

        if kinetics[1] == 'trivial':

            constants['v'].append(1)
            constants['keq'].append(1)
            constants['k'].extend([1, 1])
            if rxnType == 1 or rxnType == 2:
                constants['k'].append(1)
            if rxnType == 3:
                constants['k'].append(1)
                constants['k'].append(1)

        if kinetics[1] == 'uniform':

            constants['v'].append(uniform.rvs(loc=paramDists['v'][0], scale=paramDists['v'][1]-paramDists['v'][0]))
            constants['keq'].append(uniform.rvs(loc=paramDists['keq'][0], scale=paramDists['keq'][1]-paramDists['keq'][0]))
            constants['k'].append(uniform.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1]-paramDists['k'][0]))
            constants['k'].append(uniform.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1]-paramDists['k'][0]))
            if rxnType == 1 or rxnType == 2:
                constants['k'].append(uniform.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1]-paramDists['k'][0]))
            if rxnType == 3:
                constants['k'].append(uniform.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1]-paramDists['k'][0]))
                constants['k'].append(uniform.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1]-paramDists['k'][0]))

        if kinetics[1] == 'loguniform':

            constants['v'].append(loguniform.rvs(paramDists['v'][0], paramDists['v'][1]))
            constants['keq'].append(loguniform.rvs(paramDists['keq'][0], paramDists['keq'][1]))
            constants['k'].append(loguniform.rvs(paramDists['k'][0], paramDists['k'][1]))
            constants['k'].append(loguniform.rvs(paramDists['k'][0], paramDists['k'][1]))
            if rxnType == 1 or rxnType == 2:
                constants['k'].append(loguniform.rvs(paramDists['k'][0], paramDists['k'][1]))
            if rxnType == 3:
                constants['k'].append(loguniform.rvs(paramDists['k'][0], paramDists['k'][1]))
                constants['k'].append(loguniform.rvs(paramDists['k'][0], paramDists['k'][1]))

        if kinetics[1] == 'normal':

            while True:
                constant = norm.rvs(loc=paramDists['v'][0], scale=paramDists['v'][1])
                if constant > 0:
                    constants['v'].append(constant)
                    break
            while True:
                constant = norm.rvs(loc=paramDists['keq'][0], scale=paramDists['keq'][1])
                if constant > 0:
                    constants['keq'].append(constant)
                    break
            while True:
                constant = norm.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1])
                if constant > 0:
                    constants['k'].append(constant)
                    break
            while True:
                constant = norm.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1])
                if constant > 0:
                    constants['k'].append(constant)
                    break
            if rxnType == 1 or rxnType == 2:
                while True:
                    constant = norm.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1])
                    if constant > 0:
                        constants['k'].append(constant)
                        break
            if rxnType == 3:
                while True:
                    constant = norm.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1])
                    if constant > 0:
                        constants['k'].append(constant)
                        break
                while True:
                    constant = norm.rvs(loc=paramDists['k'][0], scale=paramDists['k'][1])
                    if constant > 0:
                        constants['k'].append(constant)
                        break

        if kinetics[1] == 'lognormal':

            constants['v'].append(lognorm.rvs(scale=paramDists['v'][0], s=paramDists['v'][1]))
            constants['keq'].append(lognorm.rvs(scale=paramDists['keq'][0], s=paramDists['keq'][1]))
            constants['k'].append(lognorm.rvs(scale=paramDists['k'][0], s=paramDists['k'][1]))
            constants['k'].append(lognorm.rvs(scale=paramDists['k'][0], s=paramDists['k'][1]))
            if rxnType == 1 or rxnType == 2:
                constants['k'].append(lognorm.rvs(scale=paramDists['k'][0], s=paramDists['k'][1]))
            if rxnType == 3:
                constants['k'].append(lognorm.rvs(scale=paramDists['k'][0], s=paramDists['k'][1]))
                constants['k'].append(lognorm.rvs(scale=paramDists['k'][0], s=paramDists['k'][1]))

        return constants

    def getHanekom4RateConstants(rxnType):

        constants = defaultdict(list)

        if kinetics[1] == 'trivial':

            constants['v'].append(1)
            constants['keq'].append(1)
            constants['ks'].append(1)
            constants['kp'].append(1)
            if rxnType == 1 or rxnType == 3:
                constants['ks'].append(1)
            if rxnType == 2 or rxnType == 3:
                constants['kp'].append(1)

        if kinetics[1] == 'uniform':

            constants['v'].append(uniform.rvs(loc=paramDists['v'][0], scale=paramDists['v'][1]-paramDists['v'][0]))
            constants['keq'].append(uniform.rvs(loc=paramDists['keq'][0], scale=paramDists['keq'][1]-paramDists['keq'][0]))
            constants['ks'].append(uniform.rvs(loc=paramDists['ks'][0], scale=paramDists['ks'][1]-paramDists['ks'][0]))
            constants['kp'].append(uniform.rvs(loc=paramDists['kp'][0], scale=paramDists['kp'][1]-paramDists['kp'][0]))
            if rxnType == 1 or rxnType == 3:
                constants['ks'].append(uniform.rvs(loc=paramDists['ks'][0], scale=paramDists['ks'][1]-paramDists['ks'][0]))
            if rxnType == 2 or rxnType == 3:
                constants['kp'].append(uniform.rvs(loc=paramDists['kp'][0], scale=paramDists['kp'][1]-paramDists['kp'][0]))

        if kinetics[1] == 'loguniform':

            constants['v'].append(loguniform.rvs(paramDists['v'][0], paramDists['v'][1]))
            constants['keq'].append(loguniform.rvs(paramDists['keq'][0], paramDists['keq'][1]))
            constants['ks'].append(loguniform.rvs(paramDists['ks'][0], paramDists['ks'][1]))
            constants['kp'].append(loguniform.rvs(paramDists['kp'][0], paramDists['kp'][1]))
            if rxnType == 1 or rxnType == 3:
                constants['ks'].append(loguniform.rvs(paramDists['ks'][0], paramDists['ks'][1]))
            if rxnType == 2 or rxnType == 3:
                constants['kp'].append(loguniform.rvs(paramDists['kp'][0], paramDists['kp'][1]))

        if kinetics[1] == 'normal':

            while True:
                constant = norm.rvs(loc=paramDists['v'][0], scale=paramDists['v'][1])
                if constant > 0:
                    constants['v'].append(constant)
                    break
            while True:
                constant = norm.rvs(loc=paramDists['keq'][0], scale=paramDists['keq'][1])
                if constant > 0:
                    constants['keq'].append(constant)
                    break
            while True:
                constant = norm.rvs(loc=paramDists['ks'][0], scale=paramDists['ks'][1])
                if constant > 0:
                    constants['ks'].append(constant)
                    break
            while True:
                constant = norm.rvs(loc=paramDists['kp'][0], scale=paramDists['kp'][1])
                if constant > 0:
                    constants['kp'].append(constant)
                    break
            if rxnType == 1 or rxnType == 3:
                while True:
                    constant = norm.rvs(loc=paramDists['ks'][0], scale=paramDists['ks'][1])
                    if constant > 0:
                        constants['ks'].append(constant)
                        break
            if rxnType == 2 or rxnType == 3:
                while True:
                    constant = norm.rvs(loc=paramDists['kp'][0], scale=paramDists['kp'][1])
                    if constant > 0:
                        constants['kp'].append(constant)
                        break

        if kinetics[1] == 'lognormal':

            constants['v'].append(lognorm.rvs(scale=paramDists['v'][0], s=paramDists['v'][1]))
            constants['keq'].append(lognorm.rvs(scale=paramDists['keq'][0], s=paramDists['keq'][1]))
            constants['ks'].append(lognorm.rvs(scale=paramDists['ks'][0], s=paramDists['ks'][1]))
            constants['kp'].append(lognorm.rvs(scale=paramDists['kp'][0], s=paramDists['kp'][1]))
            if rxnType == 1 or rxnType == 3:
                constants['ks'].append(lognorm.rvs(scale=paramDists['ks'][0], s=paramDists['ks'][1]))
            if rxnType == 2 or rxnType == 3:
                constants['kp'].append(lognorm.rvs(scale=paramDists['kp'][0], s=paramDists['kp'][1]))

        return constants

    # ---------------------------------------------------------------------------------------------------
    
    if not bool(outSamples) and not bool(inSamples):

        nodesList = [i for i in range(nSpecies)]
        nodeSet = set()

        while True:

            # todo: make parameter selection a function

            if rxn_prob:
                rt = _pickReactionType(rxn_prob)
            else:
                rt = _pickReactionType()

            # -------------------------------------------------------------------

            if rt == TReactionType.UNIUNI:

                product = random.choice(nodesList)
                reactant = random.choice(nodesList)

                if [[reactant], [product]] in reactionList2:
                    continue

                reversible = reversibility(0)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(0, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(0, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(0)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(0)



                reactionList.append([rt, [reactant], [product], rateConstants])
                reactionList2.append([[reactant], [product]])

                nodeSet.add(reactant)
                nodeSet.add(product)

            if rt == TReactionType.BIUNI:

                product = random.choice(nodesList)
                reactant1 = random.choice(nodesList)
                reactant2 = random.choice(nodesList)

                if [[reactant1, reactant2], [product]] in reactionList2:
                    continue

                reversible = reversibility(1)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(1, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(1, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(1)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(1)

                reactionList.append([rt, [reactant1, reactant2], [product], rateConstants])
                reactionList2.append([[reactant1, reactant2], [product]])

                nodeSet.add(reactant1)
                nodeSet.add(reactant2)
                nodeSet.add(product)

            if rt == TReactionType.UNIBI:

                product1 = random.choice(nodesList)
                product2 = random.choice(nodesList)
                reactant = random.choice(nodesList)

                if [[reactant], [product1, product2]] in reactionList2:
                    continue

                reversible = reversibility(2)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(2, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(2, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(2)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(2)

                reactionList.append([rt, [reactant], [product1, product2], rateConstants])
                reactionList2.append([[reactant], [product1, product2]])

                nodeSet.add(reactant)
                nodeSet.add(product1)
                nodeSet.add(product2)

            if rt == TReactionType.BIBI:

                product1 = random.choice(nodesList)
                product2 = random.choice(nodesList)
                reactant1 = random.choice(nodesList)
                reactant2 = random.choice(nodesList)

                if [[reactant1, reactant2], [product1, product2]] in reactionList2:
                    continue

                reversible = reversibility(3)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(3, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(3, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(3)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(3)

                reactionList.append([rt, [reactant1, reactant2], [product1, product2], rateConstants])
                reactionList2.append([[reactant1, reactant2], [product1, product2]])

                nodeSet.add(reactant1)
                nodeSet.add(reactant2)
                nodeSet.add(product1)
                nodeSet.add(product2)

            if len(nodeSet) == nSpecies:
                break

    # -----------------------------------------------------------------

    if not bool(outSamples) and bool(inSamples):

        while True:

            if rxn_prob:
                rt = _pickReactionType(rxn_prob)
            else:
                rt = _pickReactionType()

            if rt == TReactionType.UNIUNI:

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product = random.choices(inNodesList, probIn)[0]

                reactant = random.choice(inNodesList)

                if [[reactant], [product]] in reactionList2:
                    continue

                reversible = reversibility(0)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(0, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(0, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(0)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(0)

                inNodesCount[product] -= 1
                reactionList.append([rt, [reactant], [product], rateConstants])
                reactionList2.append([[reactant], [product]])

            if rt == TReactionType.BIUNI:

                if max(inNodesCount) < 2:
                    continue

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product = random.choices(inNodesList, probIn)[0]
                while inNodesCount[product] < 2:
                    product = random.choices(inNodesList, probIn)[0]

                reactant1 = random.choice(inNodesList)
                reactant2 = random.choice(inNodesList)

                if [[reactant1, reactant2], [product]] in reactionList2:
                    continue

                reversible = reversibility(1)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(1, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(1, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(1)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(1)

                inNodesCount[product] -= 2
                reactionList.append([rt, [reactant1, reactant2], [product], rateConstants])
                reactionList2.append([[reactant1, reactant2], [product]])

            if rt == TReactionType.UNIBI:

                if sum(inNodesCount) < 2:
                    continue

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product1 = random.choices(inNodesList, probIn)[0]

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product2 = random.choices(inNodesList, probIn)[0]

                reactant = random.choice(inNodesList)

                if [[reactant], [product1, product2]] in reactionList2:
                    continue

                reversible = reversibility(2)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(2, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(2, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(2)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(2)

                inNodesCount[product1] -= 1
                inNodesCount[product2] -= 1
                reactionList.append([rt, [reactant], [product1, product2], rateConstants])
                reactionList2.append([[reactant], [product1, product2]])

            if rt == TReactionType.BIBI:

                if max(inNodesCount) < 2:
                    continue

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product1 = random.choices(inNodesList, probIn)[0]

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product2 = random.choices(inNodesList, probIn)[0]

                reactant1 = random.choice(inNodesList)
                reactant2 = random.choice(inNodesList)

                if [[reactant1, reactant2], [product1, product2]] in reactionList2:
                    continue

                reversible = reversibility(3)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(3, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(3, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(3)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(3)

                inNodesCount[product1] -= 1
                inNodesCount[product2] -= 1
                reactionList.append([rt, [reactant1, reactant2], [product1, product2], rateConstants])
                reactionList2.append([[reactant1, reactant2], [product1, product2]])

            if sum(inNodesCount) == 0:
                break

    # -----------------------------------------------------------------

    if bool(outSamples) and not bool(inSamples):

        while True:

            if rxn_prob:
                rt = _pickReactionType(rxn_prob)
            else:
                rt = _pickReactionType()

            if rt == TReactionType.UNIUNI:

                sumOut = sum(outNodesCount)
                probOut = [x/sumOut for x in outNodesCount]
                reactant = random.choices(outNodesList, probOut)[0]

                product = random.choice(outNodesList)

                if [[reactant], [product]] in reactionList2:
                    continue

                reversible = reversibility(0)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(0, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(0, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(0)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(0)

                outNodesCount[reactant] -= 1
                reactionList.append([rt, [reactant], [product], rateConstants])
                reactionList2.append([[reactant], [product]])

            if rt == TReactionType.BIUNI:

                if sum(outNodesCount) < 2:
                    continue

                sumOut = sum(outNodesCount)
                probOut = [x/sumOut for x in outNodesCount]
                reactant1 = random.choices(outNodesList, probOut)[0]

                sumOut = sum(outNodesCount)
                probOut = [x/sumOut for x in outNodesCount]
                reactant2 = random.choices(outNodesList, probOut)[0]

                product = random.choice(outNodesList)

                if [[reactant1, reactant2], [product]] in reactionList2:
                    continue

                reversible = reversibility(1)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(1, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(1, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(1)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(1)

                outNodesCount[reactant1] -= 1
                outNodesCount[reactant2] -= 1
                reactionList.append([rt, [reactant1, reactant2], [product], rateConstants])
                reactionList2.append([[reactant1, reactant2], [product]])

            if rt == TReactionType.UNIBI:

                if max(outNodesCount) < 2:
                    continue

                sumOut = sum(outNodesCount)
                probOut = [x/sumOut for x in outNodesCount]
                reactant = random.choices(outNodesList, probOut)[0]
                while outNodesCount[reactant] < 2:
                    reactant = random.choices(outNodesList, probOut)[0]

                product1 = random.choice(outNodesList)
                product2 = random.choice(outNodesList)

                if [[reactant], [product1, product2]] in reactionList2:
                    continue

                reversible = reversibility(2)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(2, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(2, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(2)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(2)

                outNodesCount[reactant] -= 2
                reactionList.append([rt, [reactant], [product1, product2], rateConstants])
                reactionList2.append([[reactant], [product1, product2]])

            if rt == TReactionType.BIBI:

                if max(outNodesCount) < 2:
                    continue

                sumOut = sum(outNodesCount)
                probOut = [x / sumOut for x in outNodesCount]
                reactant1 = random.choices(outNodesList, probOut)[0]

                sumOut = sum(outNodesCount)
                probOut = [x / sumOut for x in outNodesCount]
                reactant2 = random.choices(outNodesList, probOut)[0]

                product1 = random.choice(outNodesList)
                product2 = random.choice(outNodesList)

                if [[reactant1, reactant2], [product1, product2]] in reactionList2:
                    continue

                reversible = reversibility(3)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(3, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(3, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(3)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(3)

                outNodesCount[reactant1] -= 1
                outNodesCount[reactant2] -= 1
                reactionList.append([rt, [reactant1, reactant2], [product1, product2], rateConstants])
                reactionList2.append([[reactant1, reactant2], [product1, product2]])

            if sum(outNodesCount) == 0:
                break

    # -----------------------------------------------------------------

    if (bool(outSamples) and bool(inSamples)) or bool(jointSamples):

        while True:

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

                if [[reactant], [product]] in reactionList2:
                    continue

                reversible = reversibility(0)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(0, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(0, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(0)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(0)

                inNodesCount[product] -= 1
                outNodesCount[reactant] -= 1
                reactionList.append([rt, [reactant], [product], rateConstants])
                reactionList2.append([[reactant], [product]])

            if rt == TReactionType.BIUNI:

                if max(inNodesCount) < 2:
                    continue

                if sum(outNodesCount) < 2:
                    continue

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product = random.choices(inNodesList, probIn)[0]
                while inNodesCount[product] < 2:
                    product = random.choices(inNodesList, probIn)[0]

                sumOut = sum(outNodesCount)
                probOut = [x/sumOut for x in outNodesCount]
                reactant1 = random.choices(outNodesList, probOut)[0]

                sumOut = sum(outNodesCount)
                probOut = [x/sumOut for x in outNodesCount]
                reactant2 = random.choices(outNodesList, probOut)[0]

                if [[reactant1, reactant2], [product]] in reactionList2:
                    continue

                reversible = reversibility(1)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(1, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(1, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(1)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(1)

                inNodesCount[product] -= 2
                outNodesCount[reactant1] -= 1
                outNodesCount[reactant2] -= 1
                reactionList.append([rt, [reactant1, reactant2], [product], rateConstants])
                reactionList2.append([[reactant1, reactant2], [product]])

            if rt == TReactionType.UNIBI:

                if sum(inNodesCount) < 2:
                    continue

                if max(outNodesCount) < 2:
                    continue

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product1 = random.choices(inNodesList, probIn)[0]

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product2 = random.choices(inNodesList, probIn)[0]

                sumOut = sum(outNodesCount)
                probOut = [x / sumOut for x in outNodesCount]
                reactant = random.choices(outNodesList, probOut)[0]
                while outNodesCount[reactant] < 2:
                    reactant = random.choices(outNodesList, probOut)[0]

                if [[reactant], [product1, product2]] in reactionList2:
                    continue

                reversible = reversibility(2)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(2, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(2, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(2)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(2)

                inNodesCount[product1] -= 1
                inNodesCount[product2] -= 1
                outNodesCount[reactant] -= 2
                reactionList.append([rt, [reactant], [product1, product2], rateConstants])
                reactionList2.append([[reactant], [product1, product2]])

            if rt == TReactionType.BIBI:

                if max(inNodesCount) < 2:
                    continue

                if max(outNodesCount) < 2:
                    continue

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product1 = random.choices(inNodesList, probIn)[0]

                sumIn = sum(inNodesCount)
                probIn = [x/sumIn for x in inNodesCount]
                product2 = random.choices(inNodesList, probIn)[0]

                sumOut = sum(outNodesCount)
                probOut = [x / sumOut for x in outNodesCount]
                reactant1 = random.choices(outNodesList, probOut)[0]

                sumOut = sum(outNodesCount)
                probOut = [x / sumOut for x in outNodesCount]
                reactant2 = random.choices(outNodesList, probOut)[0]

                if [[reactant1, reactant2], [product1, product2]] in reactionList2:
                    continue

                reversible = reversibility(3)

                rateConstants = None
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 3:
                    rateConstants = getMassAction3RateConstants(3, reversible)
                if kinetics[0] == 'mass_action' and len(kinetics[2]) == 12:
                    rateConstants = getMassAction12RateConstants(3, reversible)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 3:
                    rateConstants = getHanekom3RateConstants(3)
                if kinetics[0] == 'hanekom' and len(kinetics[2]) == 4:
                    rateConstants = getHanekom4RateConstants(3)

                inNodesCount[product1] -= 1
                inNodesCount[product2] -= 1
                outNodesCount[reactant1] -= 1
                outNodesCount[reactant2] -= 1
                reactionList.append([rt, [reactant1, reactant2], [product1, product2], rateConstants])
                reactionList2.append([[reactant1, reactant2], [product1, product2]])

            if sum(inNodesCount) == 0:
                break

    reactionList.insert(0, nSpecies)
    return reactionList, [outSamples, inSamples, jointSamples]

# Includes boundary and floating species
# Returns a list:
# [New Stoichiometry matrix, list of floatingIds, list of boundaryIds]
# On entry, reactionList has the structure:
# reactionList = [numSpecies, reaction, reaction, ....]
# reaction = [reactionType, [list of reactants], [list of products], rateConstant]

def _getFullStoichiometryMatrix(reactionList):
    nSpecies = reactionList[0]
    reactionListCopy = deepcopy(reactionList)

    # Remove the first entry in the list which is the number of species
    reactionListCopy.pop(0)
    st = np.zeros((nSpecies, len(reactionListCopy)))

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

    nSpecies = dims[0]
    nReactions = dims[1]

    speciesIds = np.arange(nSpecies)
    indexes = []
    orphanSpecies = []
    countBoundarySpecies = 0
    for r in range(nSpecies):
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
            orphanSpecies.append(r)
        if plusCoeff == 0 and minusCoeff != 0:
            # Species is a source
            indexes.append(r)
            countBoundarySpecies = countBoundarySpecies + 1
        if minusCoeff == 0 and plusCoeff != 0:
            # Species is a sink
            indexes.append(r)
            countBoundarySpecies = countBoundarySpecies + 1

    floatingIds = np.delete(speciesIds, indexes + orphanSpecies, axis=0)

    boundaryIds = indexes
    return [np.delete(st, indexes + orphanSpecies, axis=0), floatingIds, boundaryIds]

def _getAntimonyScript(floatingIds, boundaryIds, reactionList, ICparams=None, kinetics='mass_action', rev_prob=False):

    # Remove the first element which is the nSpecies
    reactionListCopy = deepcopy(reactionList)
    reactionListCopy.pop(0)

    antStr = ''
    if len(floatingIds) > 0:
        antStr = antStr + 'var ' + 'S' + str(floatingIds[0])
        for index in floatingIds[1:]:
            antStr = antStr + ', ' + 'S' + str(index)
        antStr = antStr + '\n'

    if len(boundaryIds) > 0:
        antStr = antStr + 'ext ' + 'S' + str(boundaryIds[0])
        for index in boundaryIds[1:]:
            antStr = antStr + ', ' + 'S' + str(index)
        antStr = antStr + ';\n\n'

    if kinetics[0] is 'mass_action':

        for reactionIndex, r in enumerate(reactionListCopy):
            antStr = antStr + 'J' + str(reactionIndex) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])

                if len(r[3]) == 1:
                    antStr = antStr + '; ' + 'kc' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][1][0])
                if len(r[3]) == 2:
                    antStr = antStr + '; ' + 'kf' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][1][0]) \
                             + ' - kr' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][2][0])

            if r[0] == TReactionType.BIUNI:
                # BiUni
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])

                if len(r[3]) == 1:
                    antStr = antStr + '; ' + 'kc' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][1][0]) \
                             + '*S' + str(reactionListCopy[reactionIndex][1][1])
                if len(r[3]) == 2:
                    antStr = antStr + '; ' + 'kf' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][1][0]) \
                             + '*S' + str(reactionListCopy[reactionIndex][1][1]) + ' - kr' + str(reactionIndex) + '*S' \
                             + str(reactionListCopy[reactionIndex][2][0])

            if r[0] == TReactionType.UNIBI:
                # UniBi
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][1])

                if len(r[3]) == 1:
                    antStr = antStr + '; ' + 'kc' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][1][0])
                if len(r[3]) == 2:
                    antStr = antStr + '; ' + 'kf' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][1][0]) \
                             + ' - kr' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][2][0]) \
                             + '*S' + str(reactionListCopy[reactionIndex][2][1])

            if r[0] == TReactionType.BIBI:
                # BiBi
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][1])

                if len(r[3]) == 1:
                    antStr = antStr + '; ' + 'kc' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][1][0]) \
                             + '*S' + str(reactionListCopy[reactionIndex][1][1])
                if len(r[3]) == 2:
                    antStr = antStr + '; ' + 'kf' + str(reactionIndex) + '*S' + str(reactionListCopy[reactionIndex][1][0]) \
                             + '*S' + str(reactionListCopy[reactionIndex][1][1]) + ' - kr' + str(reactionIndex) + '*S' \
                             + str(reactionListCopy[reactionIndex][2][0]) + '*S' + str(reactionListCopy[reactionIndex][2][1])

            antStr = antStr + ';\n'
        antStr = antStr + '\n'

        for index, r in enumerate(reactionListCopy):

            for param in r[3]:
                antStr = antStr + param + str(index) + ' = ' + str(r[3][param][0]) + '\n'

        antStr = antStr + '\n'

    if kinetics[0] is 'hanekom':

        for reactionIndex, r in enumerate(reactionListCopy):

            antStr = antStr + 'J' + str(reactionIndex) + ': '
            if r[0] == TReactionType.UNIUNI:
                # UniUni
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])

                if len(kinetics[2]) == 4:
                    antStr = antStr + '; v' + str(reactionIndex) + '*(S' + str(reactionListCopy[reactionIndex][1][0]) \
                             + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) + ')*(1-(S' \
                             + str(reactionListCopy[reactionIndex][2][0]) + '/S' + str(reactionListCopy[reactionIndex][1][0]) \
                             + ')/keq' + str(reactionIndex) + ')/(1 + S' + str(reactionListCopy[reactionIndex][1][0]) \
                             + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) + ' + S' \
                             + str(reactionListCopy[reactionIndex][2][0]) + '/kp_' + str(reactionIndex) + '_' + str(
                        reactionListCopy[reactionIndex][2][0]) + ')'

                if len(kinetics[2]) == 3:
                    antStr = antStr + '; v' + str(reactionIndex) + '*(S' + str(reactionListCopy[reactionIndex][1][0]) \
                        + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) + ')*(1-(S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '/S' + str(reactionListCopy[reactionIndex][1][0]) \
                        + ')/keq' + str(reactionIndex) + ')/(1 + S' + str(reactionListCopy[reactionIndex][1][0]) \
                        + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) + ' + S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][0]) + ')'

            if r[0] == TReactionType.BIUNI:
                # BiUni
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])

                if len(kinetics[2]) == 4:
                    antStr = antStr + '; v' + str(reactionIndex) + '*(S' + str(reactionListCopy[reactionIndex][1][0]) + '/ks_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][1][0]) + ')*(S' + str(reactionListCopy[reactionIndex][1][1]) \
                        + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][1]) + ')' + '*(1-(S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '/(S' + str(reactionListCopy[reactionIndex][1][0]) \
                        + '*S' + str(reactionListCopy[reactionIndex][1][1]) + '))/keq' + str(reactionIndex) + ')/(1 + S' \
                        + str(reactionListCopy[reactionIndex][1][0]) + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) \
                        + ' + S' + str(reactionListCopy[reactionIndex][1][1]) + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][1]) \
                        + ' + (S' + str(reactionListCopy[reactionIndex][1][0]) + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) \
                        + ')*(S' + str(reactionListCopy[reactionIndex][1][1]) + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][1]) \
                        + ') + S' + str(reactionListCopy[reactionIndex][2][0]) + '/kp_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][0]) \
                        + ')'

                if len(kinetics[2]) == 3:
                    antStr = antStr + '; v' + str(reactionIndex) + '*(S' + str(reactionListCopy[reactionIndex][1][0]) + '/k_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][1][0]) + ')*(S' + str(reactionListCopy[reactionIndex][1][1]) \
                        + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][1]) + ')' + '*(1-(S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '/(S' + str(reactionListCopy[reactionIndex][1][0]) \
                        + '*S' + str(reactionListCopy[reactionIndex][1][1]) + '))/keq' + str(reactionIndex) + ')/(1 + S' \
                        + str(reactionListCopy[reactionIndex][1][0]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) \
                        + ' + S' + str(reactionListCopy[reactionIndex][1][1]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][1]) \
                        + ' + (S' + str(reactionListCopy[reactionIndex][1][0]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) \
                        + ')*(S' + str(reactionListCopy[reactionIndex][1][1]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][1]) \
                        + ') + S' + str(reactionListCopy[reactionIndex][2][0]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][0]) \
                        + ')'

            if r[0] == TReactionType.UNIBI:
                # UniBi
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][1])

                if len(kinetics[2]) == 4:
                    antStr = antStr + '; v' + str(reactionIndex) + '*(S' + str(reactionListCopy[reactionIndex][1][0]) \
                        + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) + ')*(1-(S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '*S' + str(reactionListCopy[reactionIndex][2][1]) \
                        + '/S' + str(reactionListCopy[reactionIndex][1][0]) + ')/keq' + str(reactionIndex) + ')/(1 + S' \
                        + str(reactionListCopy[reactionIndex][1][0]) + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) \
                        + ' + S' + str(reactionListCopy[reactionIndex][2][0]) + '/kp_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][2][0]) + ' + S' + str(reactionListCopy[reactionIndex][2][1]) \
                        + '/kp_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][1]) + ' + (S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '/kp_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][0]) \
                        + ')*(S' + str(reactionListCopy[reactionIndex][2][1]) + '/kp_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][2][1]) + ')' + ')'

                if len(kinetics[2]) == 3:
                    antStr = antStr + '; v' + str(reactionIndex) + '*(S' + str(reactionListCopy[reactionIndex][1][0]) \
                        + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) + ')*(1-(S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '*S' + str(reactionListCopy[reactionIndex][2][1]) \
                        + '/S' + str(reactionListCopy[reactionIndex][1][0]) + ')/keq' + str(reactionIndex) + ')/(1 + S' \
                        + str(reactionListCopy[reactionIndex][1][0]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) \
                        + ' + S' + str(reactionListCopy[reactionIndex][2][0]) + '/k_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][2][0]) + ' + S' + str(reactionListCopy[reactionIndex][2][1]) \
                        + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][1]) + ' + (S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][0]) \
                        + ')*(S' + str(reactionListCopy[reactionIndex][2][1]) + '/k_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][2][1]) + ')' + ')'

            if r[0] == TReactionType.BIBI:
                # BiBi
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][1][1])
                antStr = antStr + ' -> '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][0])
                antStr = antStr + ' + '
                antStr = antStr + 'S' + str(reactionListCopy[reactionIndex][2][1])

                if len(kinetics[2]) == 4:
                    antStr = antStr + '; v' + str(reactionIndex) + '*(S' + str(reactionListCopy[reactionIndex][1][0]) + '/ks_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][1][0]) + ')*(S' + str(reactionListCopy[reactionIndex][1][1]) \
                        + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][1]) + ')' + '*(1-(S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '*S' + str(reactionListCopy[reactionIndex][2][1]) \
                        + '/(S' + str(reactionListCopy[reactionIndex][1][0]) + '*S' \
                        + str(reactionListCopy[reactionIndex][1][1]) + '))/keq' + str(reactionIndex) + ')/((1 + S' \
                        + str(reactionListCopy[reactionIndex][1][0]) + '/ks_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) \
                        + ' + S' + str(reactionListCopy[reactionIndex][2][0]) + '/kp_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][0]) \
                        + ')*(1 + S' + str(reactionListCopy[reactionIndex][1][1]) + '/ks_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][1][1]) + ' + S' + str(reactionListCopy[reactionIndex][2][1]) \
                        + '/kp_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][1]) + '))'

                if len(kinetics[2]) == 3:
                    antStr = antStr + '; v' + str(reactionIndex) + '*(S' + str(reactionListCopy[reactionIndex][1][0]) + '/k_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][1][0]) + ')*(S' + str(reactionListCopy[reactionIndex][1][1]) \
                        + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][1]) + ')' + '*(1-(S' \
                        + str(reactionListCopy[reactionIndex][2][0]) + '*S' + str(reactionListCopy[reactionIndex][2][1]) \
                        + '/(S' + str(reactionListCopy[reactionIndex][1][0]) + '*S' \
                        + str(reactionListCopy[reactionIndex][1][1]) + '))/keq' + str(reactionIndex) + ')/((1 + S' \
                        + str(reactionListCopy[reactionIndex][1][0]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][1][0]) \
                        + ' + S' + str(reactionListCopy[reactionIndex][2][0]) + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][0]) \
                        + ')*(1 + S' + str(reactionListCopy[reactionIndex][1][1]) + '/k_' + str(reactionIndex) + '_' \
                        + str(reactionListCopy[reactionIndex][1][1]) + ' + S' + str(reactionListCopy[reactionIndex][2][1]) \
                        + '/k_' + str(reactionIndex) + '_' + str(reactionListCopy[reactionIndex][2][1]) + '))'

            antStr = antStr + ';\n'
        antStr = antStr + '\n'

        V = []
        ksStr = []
        kpStr = []
        kStr = []
        keq = []

        for index, r in enumerate(reactionListCopy):

            ks = []
            kp = []
            k = []

            if r[0] == TReactionType.UNIUNI:

                V.append('v' + str(index) + ' = ' + str(r[3]['v'][0]) + '\n')

                if str(r[1][0]) not in ks and 'ks' in r[3]:
                    ksStr.append('ks_' + str(index) + '_' + str(r[1][0]) + ' = ' + str(r[3]['ks'][0]) + '\n')
                    ks.append(str(r[1][0]))

                if str(r[2][0]) not in kp and 'kp' in r[3]:
                    kpStr.append('kp_' + str(index) + '_' + str(r[2][0]) + ' = ' + str(r[3]['kp'][0]) + '\n')
                    kp.append(str(r[2][0]))

                if str(r[1][0]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[1][0]) + ' = ' + str(r[3]['k'][0]) + '\n')
                    k.append(str(r[1][0]))

                if str(r[2][0]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[2][0]) + ' = ' + str(r[3]['k'][1]) + '\n')
                    k.append(str(r[2][0]))

                keq.append('keq' + str(index) + ' = ' + str(r[3]['keq'][0]) + '\n')

            if r[0] == TReactionType.BIUNI:

                V.append('v' + str(index) + ' = ' + str(r[3]['v'][0]) + '\n')

                if str(r[1][0]) not in ks and 'ks' in r[3]:
                    ksStr.append('ks_' + str(index) + '_' + str(r[1][0]) + ' = ' + str(r[3]['ks'][0]) + '\n')
                    ks.append(str(r[1][0]))

                if str(r[1][1]) not in ks and 'ks' in r[3]:
                    ksStr.append('ks_' + str(index) + '_' + str(r[1][1]) + ' = ' + str(r[3]['ks'][1]) + '\n')
                    ks.append(str(r[1][1]))

                if str(r[2][0]) not in kp and 'kp' in r[3]:
                    kpStr.append('kp_' + str(index) + '_' + str(r[2][0]) + ' = ' + str(r[3]['kp'][0]) + '\n')
                    kp.append(str(r[2][0]))

                if str(r[1][0]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[1][0]) + ' = ' + str(r[3]['k'][0]) + '\n')
                    k.append(str(r[1][0]))

                if str(r[1][1]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[1][1]) + ' = ' + str(r[3]['k'][1]) + '\n')
                    k.append(str(r[1][1]))

                if str(r[2][0]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[2][0]) + ' = ' + str(r[3]['k'][2]) + '\n')
                    k.append(str(r[2][0]))

                keq.append('keq' + str(index) + ' = ' + str(r[3]['keq'][0]) + '\n')

            if r[0] == TReactionType.UNIBI:

                V.append('v' + str(index) + ' = ' + str(r[3]['v'][0]) + '\n')

                if str(r[1][0]) not in ks and 'ks' in r[3]:
                    ksStr.append('ks_' + str(index) + '_' + str(r[1][0]) + ' = ' + str(r[3]['ks'][0]) + '\n')
                    ks.append(str(r[1][0]))

                if str(r[2][0]) not in kp and 'kp' in r[3]:
                    kpStr.append('kp_' + str(index) + '_' + str(r[2][0]) + ' = ' + str(r[3]['kp'][0]) + '\n')
                    kp.append(str(r[2][0]))

                if str(r[2][1]) not in kp and 'kp' in r[3]:
                    kpStr.append('kp_' + str(index) + '_' + str(r[2][1]) + ' = ' + str(r[3]['kp'][1]) + '\n')
                    kp.append(str(r[2][1]))

                if str(r[1][0]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[1][0]) + ' = ' + str(r[3]['k'][0]) + '\n')
                    k.append(str(r[1][0]))

                if str(r[2][0]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[2][0]) + ' = ' + str(r[3]['k'][1]) + '\n')
                    k.append(str(r[2][0]))

                if str(r[2][1]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[2][1]) + ' = ' + str(r[3]['k'][2]) + '\n')
                    k.append(str(r[2][1]))

                keq.append('keq' + str(index) + ' = ' + str(r[3]['keq'][0]) + '\n')

            if r[0] == TReactionType.BIBI:

                V.append('v' + str(index) + ' = ' + str(r[3]['v'][0]) + '\n')

                if str(r[1][0]) not in ks and 'ks' in r[3]:
                    ksStr.append('ks_' + str(index) + '_' + str(r[1][0]) + ' = ' + str(r[3]['ks'][0]) + '\n')
                    ks.append(str(r[1][0]))

                if str(r[1][1]) not in ks and 'ks' in r[3]:
                    ksStr.append('ks_' + str(index) + '_' + str(r[1][1]) + ' = ' + str(r[3]['ks'][1]) + '\n')
                    ks.append(str(r[1][1]))

                if str(r[2][0]) not in kp and 'kp' in r[3]:
                    kpStr.append('kp_' + str(index) + '_' + str(r[2][0]) + ' = ' + str(r[3]['kp'][0]) + '\n')
                    kp.append(str(r[2][0]))

                if str(r[2][1]) not in kp and 'kp' in r[3]:
                    kpStr.append('kp_' + str(index) + '_' + str(r[2][1]) + ' = ' + str(r[3]['kp'][1]) + '\n')
                    kp.append(str(r[2][1]))

                if str(r[1][0]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[1][0]) + ' = ' + str(r[3]['k'][0]) + '\n')
                    k.append(str(r[1][0]))

                if str(r[1][1]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[1][1]) + ' = ' + str(r[3]['k'][1]) + '\n')
                    k.append(str(r[1][1]))

                if str(r[2][0]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[2][0]) + ' = ' + str(r[3]['k'][2]) + '\n')
                    k.append(str(r[2][0]))

                if str(r[2][1]) not in k and 'k' in r[3]:
                    kStr.append('k_' + str(index) + '_' + str(r[2][1]) + ' = ' + str(r[3]['k'][3]) + '\n')
                    k.append(str(r[2][1]))

                keq.append('keq' + str(index) + ' = ' + str(r[3]['keq'][0]) + '\n')

        for each in V:
            antStr = antStr + each
        antStr = antStr + '\n'
        if ksStr:
            for each in ksStr:
                antStr = antStr + each
            antStr = antStr + '\n'
        if kpStr:
            for each in kpStr:
                antStr = antStr + each
            antStr = antStr + '\n'
        if kStr:
            for each in kStr:
                antStr = antStr + each
            antStr = antStr + '\n'
        for each in keq:
            antStr = antStr + each
        antStr = antStr + '\n'

    def reversibility(rxnType):

        rev = False
        if rev_prob and isinstance(rev_prob, list):
            rev = random.choices([True, False], [rev_prob[rxnType], 1.0 - rev_prob[rxnType]])[0]
        if isinstance(rev_prob, float) or isinstance(rev_prob, int):
            rev = random.choices([True, False], [rev_prob, 1 - rev_prob])[0]

        # todo: add straight Boolean case

        return rev

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
                antStr = antStr + '; ' + 'v' + str(reactionIndex) + '*(1'

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
                antStr = antStr + '; ' + 'v' + str(reactionIndex) + '*(1'

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
                antStr = antStr + '; ' + 'v' + str(reactionIndex) + '*(1'

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
                antStr = antStr + '; ' + 'v' + str(reactionIndex) + '*(1'

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
            antStr = antStr + ';\n'
        antStr = antStr + '\n'

        for index, r in enumerate(reactionListCopy):
            if kinetics[1] is 'trivial':
                antStr = antStr + 'v' + str(index) + ' = 1\n'

            if kinetics[1] is 'uniform':
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0], scale=kinetics[3][kinetics[2].index('v')][1]
                                    - kinetics[3][kinetics[2].index('v')][0])
                antStr = antStr + 'v' + str(index) + ' = ' + str(const) + '\n'

            if kinetics[1] is 'loguniform':
                const = loguniform.rvs(kinetics[3][kinetics[2].index('v')][0], kinetics[3][kinetics[2].index('v')][1])
                antStr = antStr + 'v' + str(index) + ' = ' + str(const) + '\n'

            if kinetics[1] is 'normal':
                const = None
                while True:
                    const = norm.rvs(loc=kinetics[3][kinetics[2].index('v')][0], scale=kinetics[3][kinetics[2].index('v')][1])
                    if const >= 0:
                        antStr = antStr + 'v' + str(index) + ' = ' + str(const) + '\n'
                        break

            if kinetics[1] is 'lognormal':
                const = lognorm.rvs(scale=kinetics[3][kinetics[2].index('v')][0], s=kinetics[3][kinetics[2].index('v')][1])
                antStr = antStr + 'v' + str(index) + ' = ' + str(const) + '\n'

        antStr = antStr + '\n'

        for each in hs:
            if kinetics[1] is 'trivial':
                antStr = antStr + each + ' = 1\n'
            else:
                const = uniform.rvs(loc=kinetics[3][kinetics[2].index('hs')][0], scale=kinetics[3][kinetics[2].index('hs')][1]
                                    - kinetics[3][kinetics[2].index('hs')][0])
                antStr = antStr + each + ' = ' + str(const) + '\n'

            # todo: Save this later. Allow for different distributions types depending on parameter type
            # if kinetics[1] is 'uniform':
            #     const = uniform.rvs(loc=kinetics[3][kinetics[2].index('hs')][0], scale=kinetics[3][kinetics[2].index('hs')][1]
            #                         - kinetics[3][kinetics[2].index('hs')][0])
            #     antStr = antStr + each + ' = ' + str(const) + '\n'
            #
            # if kinetics[1] is 'loguniform':
            #     const = uniform.rvs(kinetics[3][kinetics[2].index('hs')][0], kinetics[3][kinetics[2].index('hs')][1])
            #     antStr = antStr + each + ' = ' + str(const) + '\n'
            #
            # if kinetics[1] is 'normal':
            #     const = uniform.rvs(loc=kinetics[3][kinetics[2].index('hs')][0], scale=kinetics[3][kinetics[2].index('hs')][1])
            #     antStr = antStr + each + ' = ' + str(const) + '\n'
            #
            # if kinetics[1] is 'lognormal':
            #     const = uniform.rvs(loc=kinetics[3][kinetics[2].index('v')][0], scale=kinetics[3][kinetics[2].index('v')][1])
            #     antStr = antStr + each + ' = ' + str(const) + '\n'

    # if Settings.addDegradationSteps:
    #     reactionIndex += 1
    #     parameterIndex = reactionIndex
    #     for sp in floatingIds:
    #         antStr = antStr + 'S' + str(sp) + ' ->; ' + 'k' + str(reactionIndex) + '*' + 'S' + str(sp) + '\n'
    #         reactionIndex += 1

    # if Settings.addDegradationSteps:
    #     # Next the degradation rate constants
    #     for sp in floatingIds:
    #         # antStr = antStr + 'k' + str (parameterIndex) + ' = ' + str (random.random()*Settings.rateConstantScale) + '\n'
    #         antStr = antStr + 'k' + str(parameterIndex) + ' = ' + '0.01' + '\n'
    #         parameterIndex += 1

    # antStr = antStr + '\n'
    # for index, r in enumerate(reactionListCopy):
    #     antStr = antStr + 'E' + str(index) + ' = 1\n'

    def getICvalue(ICind):

        # todo: add additional distributions (maybe, maybe not)

        IC = None
        if ICparams == 'trivial':
            IC = 1
        if isinstance(ICparams, list) and ICparams[0] is 'dist':
            IC = uniform.rvs(loc=ICparams[1], scale=ICparams[2]-ICparams[1])
        if isinstance(ICparams, list) and ICparams[0] is 'list':
            IC = ICparams[1][ICind]
        if ICparams is None:
            IC = uniform.rvs(loc=0, scale=10)

        return IC

    for index, b in enumerate(boundaryIds):
        ICvalue = getICvalue(b)
        antStr = antStr + 'S' + str(b) + ' = ' + str(ICvalue) + '\n'

    antStr = antStr + '\n'
    for index, b in enumerate(floatingIds):
        ICvalue = getICvalue(b)
        antStr = antStr + 'S' + str(b) + ' = ' + str(ICvalue) + '\n'

    return antStr
