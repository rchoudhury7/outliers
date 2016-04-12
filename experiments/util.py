
import copy

def make_one_vs_all(Y, target_class, label_class_target=1, label_class_others=2):

    selector_target = (Y == target_class)
    selector_others = (Y != target_class)

    Y = copy.deepcopy(Y)
    X = copy.deepcopy(Y)

    Y[selector_target] = label_class_target
    Y[selector_others] = label_class_others

    return Y
    
    
