import pytest


@pytest.fixture(autouse=True)
def _restore_window_globals():
    """asym_* code sets window globals process-wide (asym_pilot.set_windows on width2_geometry; asym_register on
    width2_train's dense grids); restore them after every test so no test sees another's windows."""
    from src import width2_geometry as g
    from src import width2_train as wt
    saved = (g.INNER, g.OUTER, g._XI, g._XO, wt._XI, wt._XO)
    yield
    g.INNER, g.OUTER, g._XI, g._XO, wt._XI, wt._XO = saved
