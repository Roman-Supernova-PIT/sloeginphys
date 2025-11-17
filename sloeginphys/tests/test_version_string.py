def test_version_is_string():
    from sloeginphys import __version__
    assert isinstance(__version__, str)
