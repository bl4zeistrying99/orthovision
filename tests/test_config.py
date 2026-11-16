from orthovision.config import _merge, load_config


def test_default_config_loads():
    cfg = load_config()
    assert cfg["data"]["labels"] == ["abnormal", "acl", "meniscus"]
    assert set(cfg["models"]["names"]) == {"resnet50", "efficientnet_b0", "vit", "swin"}


def test_merge_is_deep():
    assert _merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}}
