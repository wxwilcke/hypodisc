Examples
========

Two small datasets are provided in the ``test`` directory for testing and debugging purposes. Below are a few examples that highlight several key features of the program.

Run HypoDisc on one dataset: ::

    hypodisc/run.py --depth 1:3 --min_support 2 --namespace ex:http://example.org/ --namespace w3:http://w3.org/ tests/test_dataset_a.nt

Run HypoDisc on two or more datasets: ::

    hypodisc/run.py --depth 1:3 --min_support 2 --namespace ex:http://example.org/ --namespace w3:http://w3.org/ tests/test_dataset_a.nt tests/test_dataset_b.nt

Let HypoDisc choose which branches to explore and/or extend with a certain probability: ::

    hypodisc/run.py --depth 1:3 --min_support 2 --p_explore 0.7 --p_extend 0.4 --namespace ex:http://example.org/ --namespace w3:http://w3.org/ tests/test_dataset_a.nt tests/test_dataset_b.nt

Do not compute clusters on text attributes: ::

    hypodisc/run.py --depth 1:3 --min_support 2 --no-textual_support --namespace ex:http://example.org/ --namespace w3:http://w3.org/ tests/test_dataset_a.nt tests/test_dataset_b.nt

Employ a depth-first search strategy (uses less memory, but algorithm does no longer possess the `anytime` property): ::

    hypodisc/run.py --depth 1:3 --min_support 2 --strategy DFS --namespace ex:http://example.org/ --namespace w3:http://w3.org/ tests/test_dataset_a.nt tests/test_dataset_b.nt

View discovered patterns in browser: ::

    hypodisc/browse.py out.nt 
