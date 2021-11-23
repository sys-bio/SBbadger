
import sys
import tellurium as te
from teRNG import buildNetworks
import os
import shutil
import glob


def runRNG(group_name=None, start_over=False, n_models=None, n_species=None, kinetics='massAction', rxnProb=False,
           revProb=False, constDist=None, constParams=None, inDist='random', outDist='random', jointDist=None,
           inRange=None, outRange=None, jointRange=None, cutOff=1.0, ICparams=None):

    if group_name is None:
        print('Please provide group_name.')
        sys.exit(1)
    if n_models is None:
        print('Please provide n_models.')
        sys.exit(1)
    if n_species is None:
        print('Please provide n_species.')
        sys.exit(1)

    if jointDist and (inDist is not 'random' or outDist is not 'random'):
        print("You have provided both a joint distribution and onr or both of the input and output distributions")
        sys.exit(1)

    if rxnProb:
        try:
            if round(sum(rxnProb), 10) != 1:
                raise ValueError
        except ValueError:
            print('Your stated probabilities are', rxnProb, 'and they do not add to 1.')
            sys.exit(1)

    if isinstance(revProb, list):
        try:
            if any(x < 0.0 for x in revProb) or any(x > 1.0 for x in revProb):
                raise ValueError
        except ValueError:
            print('One or more of your reversibility probabilities is not between 0 and 1')
            sys.exit(1)

    if isinstance(revProb, float):
        try:
            if revProb < 0.0 or revProb > 1.0:
                raise ValueError
        except ValueError:
            print('Your reversibility probability is not between 0 and 1')
            sys.exit(1)

    if isinstance(jointRange, list) and jointRange[0] < 1:
        print("Node degree cannot be less than 1.")
        sys.exit(1)
    if isinstance(inRange, list) and inRange[0] < 1:
        print("Node degree cannot be less than 1.")
        sys.exit(1)
    if isinstance(outRange, list) and outRange[0] < 1:
        print("Node degree cannot be less than 1.")
        sys.exit(1)

    num_existing_models = 0

    if start_over:
        if os.path.exists('models/' + group_name + '/'):
            shutil.rmtree('models/' + group_name + '/')
            os.makedirs('models/' + group_name + '/' + 'antimony')
            os.makedirs('models/' + group_name + '/' + 'distributions')
            os.makedirs('models/' + group_name + '/' + 'sbml')
        else:
            os.makedirs('models/' + group_name + '/' + 'antimony')
            os.makedirs('models/' + group_name + '/' + 'distributions')
            os.makedirs('models/' + group_name + '/' + 'sbml')
    else:
        if os.path.exists('models/' + group_name + '/'):
            gd = glob.glob('models/' + group_name + '/antimony/*')
            num_existing_models = len(gd)
        else:
            os.makedirs('models/' + group_name + '/' + '/' + 'antimony')
            os.makedirs('models/' + group_name + '/' + 'distributions')
            os.makedirs('models/' + group_name + '/' + 'sbml')

    i = num_existing_models
    while i < n_models:

        rl, dists = buildNetworks._generateReactionList(n_species, kinetics='massAction',
                                                        rxnProb=rxnProb, revProb=revProb, constDist=constDist,
                                                        constParams=constParams)

        st = buildNetworks._getFullStoichiometryMatrix(rl)
        stt = buildNetworks._removeBoundaryNodes(st)
        antStr = buildNetworks._getAntimonyScript(stt[1], stt[2], rl, ICparams=ICparams)
        print(antStr)

        anti_dir = 'models/' + group_name + '/antimony/' + str(i) + '.txt'
        f = open(anti_dir, 'w')
        f.write(antStr)
        f.close()

        dist_dir = 'models/' + group_name + '/distributions/' + str(i) + '.cvs'
        f = open(dist_dir, 'w')
        f.write('out distribution\n')
        for each in dists[0]:
            f.write(str(each[0]) + ',' + str(each[1]) + '\n')
        f.write('\n')
        f.write('in distribution\n')
        for each in dists[1]:
            f.write(str(each[0]) + ',' + str(each[1]) + '\n')
        f.write('\n')
        f.write('joint distribution\n')
        for each in dists[2]:
            f.write(str(each[0]) + ',' + str(each[1]) + ',' + str(each[2]) + '\n')
        f.write('\n')
        f.close()

        sbml_dir = 'models/' + group_name + '/sbml/' + str(i) + '.sbml'
        r = te.loada(antStr)
        r.exportToSBML(sbml_dir)

        i += 1
