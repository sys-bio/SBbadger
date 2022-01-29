Usage
=====

There are three steps to model generation with SBbadger: generation of frequency distributions,
network construction, and imposition of rate laws. For greater flexibility these steps can be executed
individually. The generate.models() function effectively strings these components together so the
the options for each of these components translate directly to the generation of models.

Generating distributions
------------------------

To generate distributions we call the ``generate.distributions`` function from SBbadger.  There are two options for
generating degree frequency distributions. The first is to supply a function.

.. code-block:: console

    import SBbadger

    def in_dist(k):
        return k ** (-2)

    SBbadger.generate.distributions(
        group_name="test",
        n_models=10,
        n_species=50,
        in_dist=in_dist,
        min_freq=1.0
        )

``in_dist(k)`` is an un-normalized continuous power law function that is handed to SBbadger and subsequently
discretized, truncated, and normalized. Truncation and normalization depend on the number of species (``n_species``)
and the minimum expected number of nodes per degree (``min_freq``). Here , for example, we have ``min_freq=1.0``,
meaning that the expected number of nodes with degree X must be greater than 1. For the above example we obtain
degree probabilities and expected frequencies found in the following table.

.. list-table::
   :widths: 20, 5, 5, 5, 5, 5

   * - Edge Degree
     - 1
     - 2
     - 3
     - 4
     - 5
   * - Probabilities
     - 0.683
     - 0.171
     - 0.076
     - 0.043
     - 0.027
   * - Expected Frequencies
     - 34.162
     - 8.541
     - 3.796
     - 2.135
     - 1.366

If an edge degree of 6 were allowed the probability mass would be redistributed and the degree 6 bin would
have an expected node frequency less than the cutoff of 1.Once the probability distribution is determined it
is sampled up to the number of desired species and an output file is deposited into the ``distributions``
directory. For the above example a sample looks like the following:

.. code-block:: console

    out distribution

    in distribution
    1,67
    2,17
    3,10
    4,2
    5,2
    6,1
    7,1

    joint distribution

Note that this example only results in an in-degree sampling as there is no out-degree or joint-degree functions
provided.

The second way to generate a frequency sampling is to directly provide a probability list. This takes the form
*[(degree_1, prob_1), (degree_2, prob_2), ... (degree_n, prob_n)]* such as

.. code-block:: console

    in_dist = [(1, 0.6), (2, 0.3), (3, 0.1)]

A third option is to simply provide the frequency distribution directly. This takes the form
*[(degree_1, freq_1), (degree_2, freq_2), ... (degree_n, freq_n)]* such as

.. code-block:: console

    in_dist = [(1, 6), (2, 3), (3, 1)]

Note that in this last case, if 10 models are desired SBbadger will produce 10 output files with the exact same
frequency distributions. Currently this is necessary to produce the same number of networks in the next step.

Although the absence of one of the distributions is valid, mixing methods is not. Providing a function for the indegree
distribution and a list for the outdegree distribution is not currently supported.

Generating Networks
-------------------

The ``generate.networks`` function reads the output of the ``generate.distributions`` function and constructs
reaction networks based on any distributions it finds, or randomly if it finds none. In the simplest case one just
calls the function with the ``group_name`` argument as shown here:

.. code-block:: console

    SBbadger.generate.networks(group_name="test")

If used with the ``in_dist`` example above the result is a set of files that look like the following:

.. code-block:: console

    50
    2,(21),(9,12),(),(),()
    0,(46),(1),(),(),()
    2,(31),(39,17),(),(),()
    2,(29),(31,20),(),(),()
    0,(13),(29),(),(),()
    2,(24),(41,32),(),(),()
    0,(23),(6),(),(),()
    2,(35),(4,38),(),(),()
    0,(2),(23),(),(),()
    2,(24),(28,22),(),(),()
    0,(0),(15),(),(),()
    3,(41,22),(22,39),(),(),()
    2,(44),(8,46),(),(),()
    1,(15,43),(49),(),(),()
    2,(12),(42,14),(),(),()
    2,(32),(13,24),(),(),()
    2,(39),(10,21),(),(),()
    2,(6),(5,36),(),(),()
    1,(45,35),(10),(),(),()
    2,(28),(38,31),(),(),()
    0,(35),(33),(),(),()
    0,(20),(31),(),(),()
    2,(12),(39,35),(),(),()
    2,(18),(33,33),(),(),()
    1,(18,23),(13),(),(),()
    0,(28),(9),(),(),()
    0,(17),(34),(),(),()
    0,(38),(11),(),(),()
    1,(47,20),(11),(),(),()
    0,(13),(40),(),(),()
    2,(42),(21,7),(),(),()
    0,(1),(14),(),(),()
    0,(9),(47),(),(),()
    2,(29),(15,23),(),(),()
    0,(9),(39),(),(),()
    0,(24),(19),(),(),()
    2,(31),(16,2),(),(),()
    2,(24),(30,26),(),(),()
    2,(13),(48,49),(),(),()
    0,(37),(41),(),(),()
    0,(10),(25),(),(),()
    0,(35),(12),(),(),()
    2,(34),(13,37),(),(),()
    0,(44),(0),(),(),()
    2,(32),(18,3),(),(),()
    0,(38),(43),(),(),()
    2,(7),(44,27),(),(),()
    0,(41),(45),(),(),()

The first is the number of species in the network. The subsequent lines represent the reactions. The reactions are
formatted as *reaction type, (reactants), (products), (modifiers), (activator/inhibitor), (modifier type)*.
The reactant types are designated as UNI-UNI: 0, BI_UNI: 1, UNI-BI: 2, and BI-BI: 3. The last three entries are for
modifiers that are available when using modular kinetics. They describe the modifying species, their role as activator
or inhibitor, and the type (allosteric or specific, please see **supplementary material** for more information). The
additional argument ``mod_reg`` is needed to incorporate regulators. An example is thus

.. code-block:: console

    generate.networks(
        group_name="test",
        mod_reg=[[0.25, 0.25, 0.25, 0.25], 0.5, 0.5],
        )

The ``mod_reg`` argument has three parts: a list of probabilities for finding 0, 1, 2, or 3 modifiers, the probability
that a modifier is an activator (as opposed to an inhibitor), and the probability that it is an allosteric
regulator (as opposed to specific). An example of the output is

.. code-block:: console

    50
    1,(23,23),(4),(32,39),(1,-1),(s,s)
    1,(1,40),(29),(),(),()
    1,(40,39),(48),(19,40),(1,1),(a,a)
    3,(0,25),(6,19),(),(),()
    0,(24),(8),(37,41),(-1,-1),(s,a)
    2,(46),(41,14),(),(),()
    1,(19,29),(46),(),(),()
    0,(30),(14),(49,47,16),(1,1,1),(a,a,a)
    0,(42),(12),(28),(1),(s)
    0,(40),(9),(),(),()
    2,(17),(18,1),(47),(1),(s)
    2,(49),(26,34),(29),(1),(a)
    2,(6),(41,21),(23),(1),(s)
    0,(13),(31),(),(),()
    1,(24,28),(31),(),(),()
    1,(33,9),(39),(),(),()
    2,(42),(20,33),(),(),()
    0,(47),(10),(1),(1),(s)
    0,(30),(36),(),(),()
    2,(0),(9,7),(),(),()
    0,(35),(43),(),(),()
    0,(14),(45),(),(),()
    0,(38),(23),(31),(1),(s)
    2,(6),(15,19),(),(),()
    0,(5),(24),(44),(1),(a)
    2,(25),(17,38),(),(),()
    0,(49),(45),(),(),()
    2,(32),(3,44),(),(),()
    2,(39),(18,13),(),(),()
    0,(7),(36),(),(),()
    0,(22),(16),(),(),()
    2,(15),(28,4),(),(),()
    0,(15),(43),(),(),()
    0,(44),(5),(),(),()
    0,(5),(11),(),(),()
    2,(35),(42,21),(),(),()
    2,(30),(47,27),(),(),()
    0,(20),(22),(),(),()
    2,(30),(2,40),(),(),()
    2,(10),(32,35),(),(),()
    0,(45),(25),(),(),()
    2,(35),(0,37),(),(),()
    2,(41),(49,30),(),(),()

As many as three modifiers are currently supported. Note that the modifiers tend to stop getting added as the
algorithm progresses. This is because modifiers count against the edge distributions and this power law distribution
has relatively few high edge nodes. Thus, it becomes less and less likely that nodes will have enough edges to
support additional modifiers.

Two additional options are available at this stage. The first is an option to eliminate reactions that appear to violate
mass balance, such as ``A + B -> A``. This is done with the argument ``mass_violating_reactions=False``. The second
is limit how edges are counted against the distributions to only those with reactants and products that are consumed
and produced in a reaction. Thus, in the reaction A + B -> A + C, only B -> C would be added to the edge network. Note
that the full reaction would still be added to the model. This is done to better simulate metabolic networks.

