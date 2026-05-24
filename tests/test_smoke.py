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
