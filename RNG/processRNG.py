
import sys
import tellurium as te
import buildNetworks
import os
import shutil
import glob
from multiprocessing import Pool, cpu_count, Queue, SimpleQueue, JoinableQueue, Process, Lock
import multiprocessing as mp
import gc
import antimony


def runRNG(verbose_exceptions=False, add_deg=False, add_E=False, group_name=None, n_models=None, 
           n_species=None, kinetics=None, inDist='random', outDist='random', output_dir=None, overwrite=False, 
           rxn_prob=False, rev_prob=False, jointDist=None, inRange=None, outRange=None, jointRange=None, cutOff=1.0, 
           ICparams=None):

    # print('kinetics', kinetics)

    if not verbose_exceptions:
        sys.tracebacklimit = 0

    if group_name is None:
        raise Exception('Please provide a group_name.')
    if n_models is None:
        raise Exception('Please provide n_models (the number of models).')
    if n_species is None:
        raise Exception('Please provide n_species (the number of species).')
    if kinetics is None:
        raise Exception('Please provide the type of kinetics to use. See example run file for available options')

    if jointDist and (inDist is not 'random' or outDist is not 'random'):
        raise Exception("You have provided both a joint distribution and onr or both of the input and output distributions")

    if rxn_prob:
        if round(sum(rxn_prob), 10) != 1:
            raise Exception(f"Your stated probabilities are {rxn_prob} and they do not add to 1.")

    if isinstance(rev_prob, list):
        if any(x < 0.0 for x in rev_prob) or any(x > 1.0 for x in rev_prob):
            raise Exception('One or more of your reversibility probabilities is not between 0 and 1')

    if isinstance(rev_prob, float):
        if rev_prob < 0.0 or rev_prob > 1.0:
            raise Exception('Your reversibility probability is not between 0 and 1')

    if isinstance(jointRange, list) and jointRange[0] < 1:
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(inRange, list) and inRange[0] < 1:
        raise Exception("Node degree cannot be less than 1.")

    if isinstance(outRange, list) and outRange[0] < 1:
        raise Exception("Node degree cannot be less than 1.")

    sys.tracebacklimit = 1000

    num_existing_models = 0
    if output_dir:
        path = os.path.join(output_dir, 'models', group_name, '')
        if overwrite:
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(os.path.join(path, 'antimony'))
                os.makedirs(os.path.join(path, 'distributions'))
                os.makedirs(os.path.join(path, 'sbml'))
            else:
                os.makedirs(os.path.join(path, 'antimony'))
                os.makedirs(os.path.join(path, 'distributions'))
                os.makedirs(os.path.join(path, 'sbml'))
        else:
            if os.path.exists(os.path.join(path)):
                gd = glob.glob(os.path.join(path, 'antimony', '*'))
                num_existing_models = len(gd)
            else:
                os.makedirs(os.path.join(path, 'antimony'))
                os.makedirs(os.path.join(path, 'distributions'))
                os.makedirs(os.path.join(path, 'sbml'))

    else:
        path = os.path.join('models', group_name, '')
        if overwrite:
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(os.path.join(path, 'antimony'))
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

    # processes = []

    # mp.set_start_method("spawn", force=True)
    # manager = mp.Manager()

    i = num_existing_models

    while i < n_models:

        print(i)
        # print('kinetics', kinetics)
        rl, dists = buildNetworks._generateReactionList(n_species, kinetics, inDist, outDist, jointDist=jointDist,
                                                        inRange=inRange, outRange=outRange, jointRange=jointRange,
                                                        cutOff=cutOff, rxn_prob=rxn_prob, rev_prob=rev_prob)

        st = buildNetworks._getFullStoichiometryMatrix(rl)
        stt = buildNetworks._removeBoundaryNodes(st)
        antStr = buildNetworks._getAntimonyScript(stt[1], stt[2], rl, ICparams=ICparams, kinetics=kinetics,
                                                  rev_prob=rev_prob, add_deg=add_deg, add_E=add_E)

        if output_dir:
            anti_dir = os.path.join(output_dir, 'models', group_name, 'antimony', str(i) + '.txt')
        else:
            anti_dir = os.path.join('models', group_name, 'antimony', str(i) + '.txt')
        with open(anti_dir, 'w') as f:
            f.write(antStr)

        if output_dir:
            dist_dir = os.path.join(output_dir, 'models', group_name, 'distributions', str(i) + '.cvs')
        else:
            dist_dir = os.path.join('models', group_name, 'distributions', str(i) + '.cvs')
        with open(dist_dir, 'w') as f:
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

        # --------------------------------------------------------------------------------------------------------

        if output_dir:
            sbml_dir = os.path.join(output_dir, 'models', group_name, 'sbml', str(i) + '.sbml')
        else:
            sbml_dir = os.path.join('models', group_name, 'sbml', str(i) + '.sbml')

        # astr = "a=3"  # Replace with your own string
        antimony.loadAntimonyString(antStr)
        sbml = antimony.getSBMLString()
        # print(sbml)

        with open(sbml_dir, 'w') as f:
            f.write(sbml)

        antimony.clearPreviousLoads()

        # r = te.loada(antStr)
        # r.exportToSBML(sbml_dir)

        i += 1
