from cdt.data import CausalPairGenerator
import numpy as np

generator = CausalPairGenerator('linear')
data, labels = generator.generate(100, npoints=500)
# generator.to_csv('generated_pairs')
d = data.values

a = 0
