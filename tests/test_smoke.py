from tests.conftest import make_obs, factory_robot


def test_agent_returns_dict_for_lone_factory(config):
    from main import agent

    uid, data = factory_robot(uid="me", col=5, row=2, owner=0)
    obs = make_obs(robots={uid: data})

    actions = agent(obs, config)

    assert isinstance(actions, dict)
    assert set(actions.keys()) == {"me"}
    assert isinstance(actions["me"], str)


def test_agent_skips_enemy_units(config):
    from main import agent

    me_uid, me = factory_robot(uid="me", col=5, row=2, owner=0)
    enemy_uid, enemy = factory_robot(uid="enemy", col=14, row=2, owner=1)
    obs = make_obs(robots={me_uid: me, enemy_uid: enemy})

    actions = agent(obs, config)

    assert "enemy" not in actions
    assert "me" in actions


def test_agent_full_pipeline_two_units(config):
    from main import agent

    me_fac_uid, me_fac = factory_robot(uid="f", col=5, row=2,
                                       energy=1000, owner=0)
    scout_uid, scout = "s", [1, 5, 3, 100, 0, 0, 0, 0]
    obs = make_obs(robots={me_fac_uid: me_fac, scout_uid: scout})

    actions = agent(obs, config)

    assert set(actions.keys()) == {"f", "s"}
    valid_prefixes = (
        "NORTH", "SOUTH", "EAST", "WEST", "IDLE",
        "BUILD_", "JUMP_", "REMOVE_", "TRANSFER_", "TRANSFORM",
    )
    for v in actions.values():
        assert any(v.startswith(p) for p in valid_prefixes), v


def test_agent_does_not_walk_off_board(config):
    from main import agent

    # Factory at the very north edge — must not pick NORTH.
    me_fac_uid, me_fac = factory_robot(uid="f", col=5, row=19,
                                       energy=1000, owner=0)
    obs = make_obs(robots={me_fac_uid: me_fac}, north=19)
    actions = agent(obs, config)
    assert actions["f"] != "NORTH"
