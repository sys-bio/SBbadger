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

    if __name__ == "__main__":

        SBbadger.generate.distributions(
            group_name="dist1",
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
*[(degree_1, prob_1), (degree_2, prob_1), ... (degree_n, prob_n)]* such as

.. code-block:: console

    in_dist = [(1, 0.6), (2, 0.3), (3, 0.1)]

A third option is to simply provide the frequency distribution directly. This takes the form
*[(degree_1, freq_1), (degree_2, freq_1), ... (degree_n, freq_n)]* such as

.. code-block:: console

    in_dist = [(1, 6), (2, 3), (3, 1)]

Note that in this last case, if 10 models are desired SBbadger will produce 10 output files with the exact same
frequency distributions. Currently this is necessary to produce the same number of networks in the next step.

Although the absence of one of the distributions is valid, mixing methods is not. Providing a function for the indegree
distribution and a list for the outdegree distribution is not currently supported.