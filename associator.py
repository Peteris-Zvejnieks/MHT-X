import itertools as itt

class Associator():
    def __init__(self, association_condition,
                 combination_constraint, /,
                 max_k = 3, for_optimizer = True):
        self.association_condition   = association_condition
        self.combination_constraint  = combination_constraint
        self.max_k          = max_k
        self.for_optimizer  = for_optimizer

    def __call__(self, group1, group2):
        all_associations1, all_associations2 = [], []

        self.offset = len(group1)
        for i, obj in enumerate(group1):
            associables     = [j for j, y in enumerate(group2) if self.association_condition(obj, y)]
            combinations    = self._getAllCombinations(associables, 'Exit')
            all_associations2.extend(combinations)
            all_associations1.extend([(i,) for x in combinations])

        for j, obj in enumerate(group2):
            associables     = [i for i, x in enumerate(group1) if self.association_condition(x, obj)]
            combinations    = self._getAllCombinations(associables, 'Entry', 2)
            all_associations1.extend(combinations)
            all_associations2.extend([(j,) for x in combinations])

        all_Y1 = [self._map_adresses_to_data(group1, asc) for asc in all_associations1]
        all_Y2 = [self._map_adresses_to_data(group2, asc) for asc in all_associations2]
        Ys, ascs = tuple(), tuple()
        for Y1, Y2, asc1, asc2 in zip(all_Y1, all_Y2, all_associations1, all_associations2):
            if self.combination_constraint(Y1, Y2):
                Ys   += ((Y1, Y2),)
                ascs += ((asc1, asc2),)
        if self.for_optimizer: 
            associations_for_optimizer = list(map(self._for_optimizer, ascs))
            return (associations_for_optimizer, Ys, ascs)
        else: return (Ys, ascs)

    def _getAllCombinations(self, things, none_str, min_k = 1):
        assert (type(none_str) is str), 'Incorrect type for none_str'
        combinations = [(none_str,)]
        for k in range(min_k, min(len(things), self.max_k) + 1):
            combinations.extend(list(itt.combinations(things, k)))
        return combinations

    def _map_adresses_to_data(self, group, asc):
        try:    return [group[i] for i in asc]
        except TypeError: return []

    def _for_optimizer(self, x):
        a, b = x
        f0 = lambda x: tuple(map(lambda y: y + self.offset, x))
        if   type(a[0]) is str : return (tuple(), f0(b))
        elif type(b[0]) is str : return (a, tuple())
        else                   : return (a, f0(b))
        
class Combination_constraint():
    def __init__(self, f):
        self.f = f
    def __call__(self, Y1, Y2):
        return self.f(Y1, Y2)
        
class Association_condition():
    def __init__(self, f):
        self.f = f
    def __call__(self, Y1, Y2):
        return self.f(Y1, Y2)