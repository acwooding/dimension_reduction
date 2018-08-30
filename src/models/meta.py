from sklearn.model_selection import GridSearchCV

DR_META_ESTIMATORS = {
    'grid_search': GridSearchCV
}

def available_meta_estimators():
    """Valid Meta-estimators for dimension reduction applications
    This function simply returns the list of known dimension reduction
    algorithms.

    It exists to allow for a description of the mapping for
    each of the valid strings.

    ============     ====================================
    Meta-est         Function
    ============     ====================================
    grid_search      sklearn.model_selection.GridSearchCV
    ============     ====================================
    """
    return DR_META_ESTIMATORS

