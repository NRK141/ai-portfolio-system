from src.strategy.weights import make_target_weights


def test_target_weights_sum_to_one():
    weights = make_target_weights(["A", "B"], 0.6, {"IAU": 0.4})
    assert abs(sum(weights.values()) - 1.0) < 1e-8
