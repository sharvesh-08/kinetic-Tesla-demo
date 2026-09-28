"""Compatibility entry point: datasets now come exclusively from Layers 4 and 5.

See layer6.build_feature_datasets for source and output arguments. The former
independent summary-column generator is retired because it violated the live
FeatureWindow contract.
"""
from .build_feature_datasets import main

if __name__ == '__main__':
    main()
