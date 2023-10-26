# HypoDisc

On one dataset:

    hypodisc/run.py --depth 1:3 --min_support 2 --multimodal  --namespace ex:http://www.example.org/ --namespace w3:http://w3.org/ tests/test_dataset_a.nt
    
On two or more datasets:

    hypodisc/run.py --depth 1:3 --min_support 2 --multimodal  --namespace ex:http://www.example.org/ --namespace w3:http://w3.org/ tests/test_dataset_a.nt tests/test_dataset_b.nt
