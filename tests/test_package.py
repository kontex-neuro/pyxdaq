def test_package():
    import pyxdaq
    import pyxdaq.xdaq


def test_resource():
    from pyxdaq import resources
    for attr in ['bitfile_path', 'isa_path', 'reg_path']:
        assert getattr(resources.rhd, attr).exists()
        assert getattr(resources.rhs, attr).exists()
