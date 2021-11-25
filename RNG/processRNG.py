import sys
import tellurium as te
from RNG import buildNetworks
import os
import shutil
import glob
from multiprocessing import Pool, cpu_count, Queue, SimpleQueue, JoinableQueue, Process
import multiprocessing as mp

def _run1(i, n_species, kinetics, rxn_prob, rev_prob, constDist, constParams, ICparams, group_name):
    rl, dists = buildNetworks._generateReactionList(n_species, kinetics=kinetics,
                                                    rxn_prob=rxn_prob, rev_prob=rev_prob, constDist=constDist,
                                                    constParams=constParams)

    st = buildNetworks._getFullStoichiometryMatrix(rl)
    stt = buildNetworks._removeBoundaryNodes(st)
    antStr = buildNetworks._getAntimonyScript(stt[1], stt[2], rl, ICparams=ICparams, kinetics=kinetics)

    # todo always use the os module for handling file paths.
    #   os.path.join(absolute_directory, "models")
    #   os.path.join(os.path.dirname(__file__), "models") # absolute path to current dir
    anti_dir = 'models/' + group_name + '/antimony/' + str(i) + '.txt'

    # todo always use the "with" construct ( context manager) in Python for opening files
    # with open(anti_dir, "w") as f:
    #   f.write(antStr) # handles close automatically
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

    # i += 1

def _run1FromQueue(q: Queue):
    _run1(*q.get())

def runRNG(group_name=None, overwrite=False, n_models=None, n_species=None, kinetics='mass_action', rxn_prob=False,
           rev_prob=False, constDist=None, constParams=None, inDist='random', outDist='random', jointDist=None,
           inRange=None, outRange=None, jointRange=None, cutOff=1.0, ICparams=None):
    # todo if these function parameters are not optional parameters, then do not have keyword args:
    #   - Organize the function so that requires parameters come first and then keywords come afterwards.
    #   - Then you do not need validation for every parameter, as python will handle this automatically.

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

    """
    todo figure out how exceptions in Python work. Here's an example: 
    
    This code: 
    ```
        if rxn_prob:
        try:
            if round(sum(rxn_prob), 10) != 1:
                raise ValueError
        except ValueError:
            print('Your stated probabilities are', rxn_prob, 'and they do not add to 1.')
            sys.exit(1)
    ```
    Should be     
    ```
    if rxn_prob:
        if round(sum(rxn_prob), 10) != 1:
            raise ValueError(f"Your stated probabilities are {rxn_prob} and they do not add to 1.")
    Notes: 
        - I've used a formatted "f" string.
        - You do not need sys.exit. Python will handle the errors and give you a traceback -- this is 
          far superior to anything you (or I) can come up with so you should use it.  
    ```
    """
    if rxn_prob:
        try:
            if round(sum(rxn_prob), 10) != 1:
                raise ValueError
        except ValueError:
            print('Your stated probabilities are', rxn_prob, 'and they do not add to 1.')
            sys.exit(1)

    # todo one way of handling a optional list vs float parameter is to convert a float to a list of 1,
    #  then in later code you'll only have to deal with the list case

    if isinstance(rev_prob, list):
        try:
            if any(x < 0.0 for x in rev_prob) or any(x > 1.0 for x in rev_prob):
                raise ValueError
        except ValueError:
            print('One or more of your reversibility probabilities is not between 0 and 1')
            sys.exit(1)

    if isinstance(rev_prob, float):
        try:
            if rev_prob < 0.0 or rev_prob > 1.0:
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

    if overwrite:
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

    # todo consider parallelising this loop.
    """
    What goes in the queue? 
        - Arguments to the function for a single iteration of RNG
    What comes out? 
        - Using the arguments in a function. 
    
    """

    # a list to store the processes
    processes = []

    # manager.Queue is a *proxy* to a queue, not a queue. Think of this as a reference or pointer like in c/c++.
    # we need this so we can have a *shared* queue, i.e. a single queue shared between all processes.
    # The input to the queue is the arguments for each iteration. The first item in the queue is the first out (FIFO)
    manager = mp.Manager()
    q = manager.Queue(maxsize=cpu_count()) # You could change this to a parameter (sometimes called j) that defaults to cpu_count().

    for i in range(n_models):
        print("running iteration", i, "Is queue full?: ", q.full())
        # Add items to the queue. These are arguments to the function we want to run.
        # Importantly, q.put will *block* program execution when it is full. This is why
        # the code halts when the queue is full.
        args = (i, n_species, kinetics, rxn_prob, rev_prob, constDist, constParams, ICparams, group_name)
        q.put(args)

        # create a new process to run this iteration.
        p = Process(target=_run1FromQueue, args=(q,))

        # it begins executing when you call the start method.
        p.start()
        processes.append(p)

    # processes keep executing until they call join --
    # Think of this as consolidating all processes. Main program execution continues when all
    # processes have joined (or program will hang). Not completely sure why its called "join"
    for p in processes:
        p.join()

    # i = num_existing_models
    # while i < n_models:
    #     run1(i, n_species, kinetics, rxn_prob, rev_prob, constDist, constParams, ICparams, group_name)
    #     i += 1

    # rl, dists = buildNetworks._generateReactionList(n_species, kinetics=kinetics,
        #                                                 rxn_prob=rxn_prob, rev_prob=rev_prob, constDist=constDist,
        #                                                 constParams=constParams)
        #
        # st = buildNetworks._getFullStoichiometryMatrix(rl)
        # stt = buildNetworks._removeBoundaryNodes(st)
        # antStr = buildNetworks._getAntimonyScript(stt[1], stt[2], rl, ICparams=ICparams, kinetics=kinetics)
        #
        # # todo always use the os module for handling file paths.
        # #   os.path.join(absolute_directory, "models")
        # #   os.path.join(os.path.dirname(__file__), "models") # absolute path to current dir
        # anti_dir = 'models/' + group_name + '/antimony/' + str(i) + '.txt'
        #
        # # todo always use the "with" construct ( context manager) in Python for opening files
        # # with open(anti_dir, "w") as f:
        # #   f.write(antStr) # handles close automatically
        # f = open(anti_dir, 'w')
        # f.write(antStr)
        # f.close()
        #
        # dist_dir = 'models/' + group_name + '/distributions/' + str(i) + '.cvs'
        # f = open(dist_dir, 'w')
        # f.write('out distribution\n')
        # for each in dists[0]:
        #     f.write(str(each[0]) + ',' + str(each[1]) + '\n')
        # f.write('\n')
        # f.write('in distribution\n')
        # for each in dists[1]:
        #     f.write(str(each[0]) + ',' + str(each[1]) + '\n')
        # f.write('\n')
        # f.write('joint distribution\n')
        # for each in dists[2]:
        #     f.write(str(each[0]) + ',' + str(each[1]) + ',' + str(each[2]) + '\n')
        # f.write('\n')
        # f.close()
        #
        # sbml_dir = 'models/' + group_name + '/sbml/' + str(i) + '.sbml'
        # r = te.loada(antStr)
        # r.exportToSBML(sbml_dir)
        #
        # i += 1
