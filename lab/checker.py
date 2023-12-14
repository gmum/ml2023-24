import os
from typing import Callable, Tuple, List, Type

import numpy as np
import torch

import utils
from types import SimpleNamespace
from torch.optim import SGD
from torch.optim import Adagrad as torch_adagrad
from torch.optim import RMSprop as torch_rmsprop
from torch.optim import Adadelta as torch_adadelta
from torch.optim import Adam as torch_adam


def check_closest(fn: Callable) -> None:
    inputs = [
        (6, np.array([5, 3, 4])),
        (10, np.array([12, 2, 8, 9, 13, 14])),
        (-2, np.array([-5, 12, 6, 0, -14, 3])),
    ]
    assert np.isclose(fn(*inputs[0]), 5), "Jest błąd w funkcji closest!"
    assert np.isclose(fn(*inputs[1]), 9), "Jest błąd w funkcji closest!"
    assert np.isclose(fn(*inputs[2]), 0), "Jest błąd w funkcji closest!"


def check_poly(fn: Callable) -> None:
    inputs = [
        (6, np.array([5.5, 3, 4])),
        (10, np.array([12, 2, 8, 9, 13, 14])),
        (-5, np.array([6, 3, -12, 9, -15])),
    ]
    assert np.isclose(fn(*inputs[0]), 167.5), "Jest błąd w funkcji poly!"
    assert np.isclose(fn(*inputs[1]), 1539832), "Jest błąd w funkcji poly!"
    assert np.isclose(fn(*inputs[2]), -10809), "Jest błąd w funkcji poly!"


def check_multiplication_table(fn: Callable) -> None:
    inputs = [3, 5]
    assert np.all(
        fn(inputs[0]) == np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    ), "Jest błąd w funkcji multiplication_table!"
    assert np.all(
        fn(inputs[1])
        == np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 4, 6, 8, 10],
                [3, 6, 9, 12, 15],
                [4, 8, 12, 16, 20],
                [5, 10, 15, 20, 25],
            ]
        )
    ), "Jest błąd w funkcji multiplication_table!"


def check_1_1(
        mean_error: Callable,
        mean_squared_error: Callable,
        max_error: Callable,
        train_sets: List[np.ndarray],
) -> None:
    train_set_1d, train_set_2d, train_set_10d = train_sets
    assert np.isclose(mean_error(train_set_1d, np.array([8])), 8.897352)
    assert np.isclose(mean_error(train_set_2d, np.array([2.5, 5.2])), 7.89366)
    assert np.isclose(mean_error(train_set_10d, np.array(np.arange(10))), 14.16922)

    assert np.isclose(mean_squared_error(train_set_1d, np.array([3])), 23.03568)
    assert np.isclose(mean_squared_error(train_set_2d, np.array([2.4, 8.9])), 124.9397)
    assert np.isclose(mean_squared_error(train_set_10d, -np.arange(10)), 519.1699)

    assert np.isclose(max_error(train_set_1d, np.array([3])), 7.89418)
    assert np.isclose(max_error(train_set_2d, np.array([2.4, 8.9])), 14.8628)
    assert np.isclose(max_error(train_set_10d, -np.linspace(0, 5, num=10)), 23.1727)


def check_1_2(
        minimize_me: Callable, minimize_mse: Callable, minimize_max: Callable, train_set_1d: np.ndarray
) -> None:
    assert np.isclose(minimize_mse(train_set_1d), -0.89735)
    assert np.isclose(minimize_mse(train_set_1d * 2), -1.79470584)
    assert np.isclose(minimize_me(train_set_1d), -1.62603)
    assert np.isclose(minimize_me(train_set_1d ** 2), 3.965143)
    assert np.isclose(minimize_max(train_set_1d), 0.0152038)
    assert np.isclose(minimize_max(train_set_1d / 2), 0.007601903895526174)


def check_1_3(
        me_grad: Callable, mse_grad: Callable, max_grad: Callable, train_sets: List[np.ndarray]
) -> None:
    train_set_1d, train_set_2d, train_set_10d = train_sets
    assert all(np.isclose(me_grad(train_set_1d, np.array([0.99])), [0.46666667]))
    assert all(np.isclose(me_grad(train_set_2d, np.array([0.99, 8.44])), [0.21458924, 0.89772834]))
    assert all(
        np.isclose(
            me_grad(train_set_10d, np.linspace(0, 10, num=10)),
            [
                -0.14131273,
                -0.031631,
                0.04742431,
                0.0353542,
                0.16364242,
                0.23353252,
                0.30958123,
                0.35552034,
                0.4747464,
                0.55116738,
            ],
        )
    )

    assert all(np.isclose(mse_grad(train_set_1d, np.array([1.24])), [4.27470585]))
    assert all(
        np.isclose(mse_grad(train_set_2d, np.array([-8.44, 10.24])), [-14.25378235, 21.80373175])
    )
    assert all(np.isclose(max_grad(train_set_1d, np.array([5.25])), [1.0]))
    assert all(
        np.isclose(max_grad(train_set_2d, np.array([-6.28, -4.45])), [-0.77818704, -0.62803259])
    )


def check_02_linear_regression(lr_cls: Type) -> None:
    from sklearn import datasets

    np.random.seed(54)

    input_dataset = datasets.load_diabetes()
    lr = lr_cls()
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    expected = np.load(".checker/05/lr_diabetes.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(
        loss, 26004.287402, rtol=1e-03, atol=1e-06
    ), "Wrong value of the loss function!"


def check_02_regularized_linear_regression(lr_cls: Type) -> None:
    from sklearn import datasets

    np.random.seed(54)

    input_dataset = datasets.load_diabetes()
    lr = lr_cls(lr=1e-2, alpha=1e-4)
    lr.fit(input_dataset.data, input_dataset.target)
    returned = lr.predict(input_dataset.data)
    # np.savez_compressed(".checker/05/rlr_diabetes.out.npz", data=returned)
    expected = np.load(".checker/05/rlr_diabetes.out.npz")["data"]
    assert np.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    loss = lr.loss(input_dataset.data, input_dataset.target)
    assert np.isclose(
        loss, 26111.08336411, rtol=1e-03, atol=1e-06
    ), "Wrong value of the loss function!"


def check_4_1_mse(fn: Callable, datasets: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    results = [torch.tensor(13.8520), torch.tensor(31.6952)]
    for (data, param), loss in zip(datasets, results):
        result = fn(data, param)
        assert torch.allclose(fn(data, param), loss, atol=1e-3), "Wrong loss returned!"


def check_4_1_me(fn: Callable, datasets: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    results = [torch.tensor(3.6090), torch.tensor(5.5731)]
    for (data, param), loss in zip(datasets, results):
        assert torch.allclose(fn(data, param), loss, atol=1e-3), "Wrong loss returned!"


def check_4_1_max(fn: Callable, datasets: List[Tuple[np.ndarray, np.ndarray]]) -> None:
    results = [torch.tensor(7.1878), torch.tensor(7.5150)]
    for (data, param), loss in zip(datasets, results):
        assert torch.allclose(fn(data, param), loss, atol=1e-3), "Wrong loss returned!"


def check_4_1_lin_reg(fn: Callable, data: List[np.ndarray]) -> None:
    X, y, w = data
    assert torch.allclose(fn(X, w, y), torch.tensor(29071.6699), atol=1e-3), "Wrong loss returned!"


def check_4_1_reg_reg(fn: Callable, data: List[np.ndarray]) -> None:
    X, y, w = data
    assert torch.allclose(fn(X, w, y), torch.tensor(29073.4551)), "Wrong loss returned!"


def check_04_logistic_reg(lr_cls: Type) -> None:
    np.random.seed(10)
    torch.manual_seed(10)

    # **** First dataset ****
    input_dataset = utils.get_classification_dataset_1d()
    lr = lr_cls(1)
    lr.fit(input_dataset.data, input_dataset.target, lr=1e-3, num_steps=int(1e4))
    returned = lr.predict(input_dataset.data)
    save_path = ".checker/04/lr_dataset_1d.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    returned = lr.predict_proba(input_dataset.data)
    save_path = ".checker/04/lr_dataset_1d_proba.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    returned = lr.predict(input_dataset.data)
    save_path = ".checker/04/lr_dataset_1d_preds.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    # **** Second dataset ****
    input_dataset = utils.get_classification_dataset_2d()
    lr = lr_cls(2)
    lr.fit(input_dataset.data, input_dataset.target, lr=1e-2, num_steps=int(1e4))
    returned = lr.predict(input_dataset.data)
    save_path = ".checker/04/lr_dataset_2d.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    returned = lr.predict_proba(input_dataset.data)
    save_path = ".checker/04/lr_dataset_2d_proba.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"

    returned = lr.predict(input_dataset.data)
    save_path = ".checker/04/lr_dataset_2d_preds.out.torch"
    # torch.save(returned, save_path)
    expected = torch.load(save_path)
    assert torch.allclose(expected, returned, rtol=1e-03, atol=1e-06), "Wrong prediction returned!"


def optim_f(w: torch.Tensor) -> torch.Tensor:
    x = torch.tensor([0.2, 2], dtype=torch.float)
    return torch.sum(x * w ** 2)


def optim_g(w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x = torch.tensor([0.2, 2], dtype=torch.float)
    return torch.sum(x * w + b)


opt_checker_1 = SimpleNamespace(
    f=optim_f, params=[torch.tensor([-6, 2], dtype=torch.float, requires_grad=True)]
)
opt_checker_2 = SimpleNamespace(
    f=optim_g,
    params=[
        torch.tensor([-6, 2], dtype=torch.float, requires_grad=True),
        torch.tensor([1, -1], dtype=torch.float, requires_grad=True),
    ],
)

test_params = {
    "Momentum": {
        "torch_cls": SGD,
        "torch_params": {"lr": 0.1, "momentum": 0.9},
        "params": {"learning_rate": 0.1, "gamma": 0.9},
    },
    "Adagrad": {
        "torch_cls": torch_adagrad,
        "torch_params": {"lr": 0.5, "eps": 1e-8},
        "params": {"learning_rate": 0.5, "epsilon": 1e-8},
    },
    "RMSProp": {
        "torch_cls": torch_rmsprop,
        "torch_params": {
            "lr": 0.5,
            "alpha": 0.9,
            "eps": 1e-08,
        },
        "params": {"learning_rate": 0.5, "gamma": 0.9, "epsilon": 1e-8},
    },
    "Adadelta": {
        "torch_cls": torch_adadelta,
        "torch_params": {"rho": 0.9, "eps": 1e-1},
        "params": {"gamma": 0.9, "epsilon": 1e-1},
    },
    "Adam": {
        "torch_cls": torch_adam,
        "torch_params": {"lr": 0.5, "betas": (0.9, 0.999), "eps": 1e-08},
        "params": {"learning_rate": 0.5, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
    },
}


def test_optimizer(optim_cls: Type, num_steps: int = 10) -> None:
    test_dict = test_params[optim_cls.__name__]

    for ns in [opt_checker_1, opt_checker_2]:
        torch_params = [p.clone().detach().requires_grad_(True) for p in ns.params]
        torch_opt = test_dict["torch_cls"](torch_params, **test_dict["torch_params"])
        for _ in range(num_steps):
            torch_opt.zero_grad()

            loss = ns.f(*torch_params)
            loss.backward()
            torch_opt.step()

        params = [p.clone().detach().requires_grad_(True) for p in ns.params]
        opt = optim_cls(params, **test_dict["params"])

        for _ in range(num_steps):
            opt.zero_grad()

            loss = ns.f(*params)
            loss.backward()
            opt.step()

        for p, tp in zip(params, torch_params):
            assert torch.allclose(p, tp)


def test_droput(dropout_cls: Type) -> None:
    drop = dropout_cls(0.5)
    drop.train()
    x = torch.randn(10, 30)
    out = drop(x)

    for row, orig_row in zip(out, x):
        zeros_in_row = torch.where(row == 0.0)[0]
        non_zeros_in_row = torch.where(row != 0.0)[0]
        non_zeros_scaled = (row[non_zeros_in_row] == 2 * orig_row[non_zeros_in_row]).all()
        assert len(zeros_in_row) > 0 and len(zeros_in_row) < len(row) and non_zeros_scaled

    drop_eval = dropout_cls(0.5)
    drop_eval.eval()
    x = torch.randn(10, 30)
    out_eval = drop_eval(x)

    for row in out_eval:
        zeros_in_row = len(torch.where(row == 0.0)[0])
        assert zeros_in_row == 0


def test_bn(bn_cls: Type) -> None:
    torch.manual_seed(42)
    bn = bn_cls(num_features=100)

    opt = torch.optim.SGD(bn.parameters(), lr=0.1)

    bn.train()
    x = torch.rand(20, 100)
    out = bn(x)

    assert out.mean().abs().item() < 1e-4
    assert abs(out.var().item() - 1) < 1e-1

    assert (bn.sigma != 1).all()
    assert (bn.mu != 1).all()

    loss = 1 - out.mean()
    loss.backward()
    opt.step()

    assert (bn.beta != 0).all()

    n_steps = 10

    for i in range(n_steps):
        x = torch.rand(20, 100)
        out = bn(x)
        loss = 1 - out.mean()
        loss.backward()
        opt.step()

    torch.manual_seed(43)
    test_x = torch.randn(20, 100)
    bn.eval()
    test_out = bn(test_x)

    assert abs(test_out.mean() + 0.5) < 1e-1


expected_mean_readout = torch.tensor(
    [[-0.0035, 0.0505, -0.2221, 0.1404, 0.1922, -0.3736, -0.0672, 0.0752,
      -0.0613, 0.0439, -0.1307, -0.0752, -0.0310, 0.0081, -0.0553, -0.1734],
     [-0.0054, -0.0144, -0.3113, 0.1665, 0.0738, -0.3303, 0.0420, 0.0668,
      0.0494, 0.2648, -0.0478, 0.0550, -0.1923, -0.0157, 0.0508, 0.0148],
     [-0.1912, 0.0309, -0.1512, 0.1283, 0.1120, -0.4540, -0.0644, 0.1378,
      -0.0194, 0.0103, -0.1713, 0.0175, -0.0604, -0.0193, -0.0208, -0.0822]]
)
expected_attention_readout = torch.Tensor(
    [[-0.0083, 0.0499, -0.2197, 0.1380, 0.1921, -0.3753, -0.0669, 0.0771,
      -0.0592, 0.0411, -0.1317, -0.0769, -0.0299, 0.0074, -0.0568, -0.1741],
     [-0.0068, -0.0131, -0.3102, 0.1656, 0.0736, -0.3312, 0.0410, 0.0670,
      0.0485, 0.2635, -0.0479, 0.0544, -0.1933, -0.0162, 0.0508, 0.0150],
     [-0.1911, 0.0308, -0.1514, 0.1271, 0.1100, -0.4542, -0.0658, 0.1376,
      -0.0215, 0.0099, -0.1723, 0.0164, -0.0618, -0.0209, -0.0217, -0.0817]],
)
expected_sage_layer_output = torch.tensor(
    [[-5.0965e-01, -4.5482e-01, -8.1451e-01, 5.4286e-03],
     [-5.6737e-01, -5.9137e-01, -7.9304e-01, 7.5955e-02],
     [-4.6768e-01, -5.0346e-01, -7.2765e-01, 5.0357e-02],
     [-6.4185e-01, -5.0983e-01, -8.6305e-01, 1.3008e-02],
     [-5.0465e-01, -3.5816e-01, -8.7864e-01, -3.1902e-02],
     [-5.6591e-01, -4.2403e-01, -8.7506e-01, 2.9357e-02],
     [-6.4185e-01, -5.0983e-01, -8.6305e-01, 1.3008e-02],
     [-5.7196e-01, -3.5674e-01, -9.4769e-01, -4.9931e-03],
     [-6.4185e-01, -5.0983e-01, -8.6305e-01, 1.3008e-02],
     [-5.2655e-01, -5.1094e-01, -8.3806e-01, -1.8521e-02],
     [-6.4185e-01, -5.0983e-01, -8.6305e-01, 1.3008e-02],
     [-5.7628e-01, -5.5394e-01, -8.7300e-01, -7.6976e-03],
     [-4.6768e-01, -5.0346e-01, -7.2765e-01, 5.0357e-02],
     [-5.4808e-01, -5.3204e-01, -7.8906e-01, 4.2878e-02],
     [-5.3417e-01, -3.5912e-01, -9.5030e-01, 2.3648e-05],
     [-6.2538e-01, -2.9249e-01, -1.1233e+00, 1.0970e-01],
     [-6.5214e-01, -3.8342e-01, -1.0136e+00, -1.6424e-02],
     [-6.5214e-01, -3.8342e-01, -1.0136e+00, -1.6424e-02]],
)
expected_gin_layer_output = torch.tensor(
    [[-0.4516, -0.3673, -0.5313, 0.3170],
     [-0.4524, -0.3760, -0.5243, 0.3249],
     [-0.4570, -0.3747, -0.5313, 0.3221],
     [-0.4763, -0.4030, -0.5390, 0.3335],
     [-0.4481, -0.3855, -0.5187, 0.3295],
     [-0.4545, -0.3838, -0.5245, 0.3276],
     [-0.4763, -0.4030, -0.5390, 0.3335],
     [-0.4390, -0.4001, -0.4973, 0.3446],
     [-0.4763, -0.4030, -0.5390, 0.3335],
     [-0.4683, -0.3882, -0.5400, 0.3248],
     [-0.4763, -0.4030, -0.5390, 0.3335],
     [-0.4682, -0.3921, -0.5374, 0.3277],
     [-0.4570, -0.3747, -0.5313, 0.3221],
     [-0.4225, -0.3671, -0.4928, 0.3295],
     [-0.3760, -0.3700, -0.4407, 0.3489],
     [-0.2646, -0.3342, -0.3357, 0.3683],
     [-0.3859, -0.3950, -0.4392, 0.3624],
     [-0.3859, -0.3950, -0.4392, 0.3624]],
)
expected_simple_mpnn_output = torch.tensor(
    [[-0.1990, -0.2007, -0.7749, -0.2355],
     [-0.5297, -0.4750, -0.8783, -0.0762],
     [-0.3664, -0.4155, -0.7463, -0.0573],
     [-0.5217, -0.3488, -0.9198, -0.1840],
     [0.1237, -0.0524, -0.5546, -0.1867],
     [-0.3597, -0.2378, -0.8626, -0.1551],
     [-0.5217, -0.3488, -0.9198, -0.1840],
     [-0.3358, -0.2634, -0.8318, -0.0586],
     [-0.5217, -0.3488, -0.9198, -0.1840],
     [-0.2175, -0.2724, -0.7910, -0.2460],
     [-0.5217, -0.3488, -0.9198, -0.1840],
     [-0.3758, -0.3293, -0.9195, -0.2665],
     [-0.3664, -0.4155, -0.7463, -0.0573],
     [-0.3907, -0.4223, -0.7682, -0.0586],
     [-0.2049, -0.2482, -0.7605, -0.0309],
     [-0.1718, 0.0814, -1.0231, -0.2095],
     [-0.3551, -0.2676, -0.8502, -0.0614],
     [-0.3551, -0.2676, -0.8502, -0.0614]]
)
expected_sum_readout = torch.tensor(
    [[-0.0451, 0.6570, -2.8874, 1.8256, 2.4987, -4.8573, -0.8733, 0.9780,
      -0.7967, 0.5701, -1.6988, -0.9777, -0.4033, 0.1053, -0.7191, -2.2545],
     [-0.0268, -0.0720, -1.5565, 0.8324, 0.3692, -1.6515, 0.2101, 0.3342,
      0.2468, 1.3238, -0.2389, 0.2752, -0.9615, -0.0785, 0.2541, 0.0741],
     [-0.9559, 0.1545, -0.7560, 0.6414, 0.5598, -2.2701, -0.3222, 0.6888,
      -0.0969, 0.0516, -0.8565, 0.0875, -0.3022, -0.0964, -0.1039, -0.4109]],
)
expected_gine_layer_output = torch.tensor(
    [[-0.4502, -0.3649, -0.5164, 0.3204],
     [-0.4699, -0.3855, -0.5349, 0.3266],
     [-0.4746, -0.3870, -0.5404, 0.3256],
     [-0.4300, -0.3570, -0.5018, 0.3213],
     [-0.3783, -0.3281, -0.4547, 0.3227],
     [-0.4541, -0.3734, -0.5200, 0.3237],
     [-0.4300, -0.3570, -0.5018, 0.3213],
     [-0.4720, -0.3863, -0.5399, 0.3247],
     [-0.4300, -0.3570, -0.5018, 0.3213],
     [-0.3145, -0.2931, -0.3972, 0.3250],
     [-0.4300, -0.3570, -0.5018, 0.3213],
     [-0.3138, -0.2965, -0.3943, 0.3277],
     [-0.4746, -0.3870, -0.5404, 0.3256],
     [-0.4471, -0.3719, -0.5137, 0.3254],
     [-0.4353, -0.3751, -0.4988, 0.3315],
     [-0.4574, -0.3719, -0.5230, 0.3219],
     [-0.4700, -0.3940, -0.5322, 0.3306],
     [-0.4700, -0.3940, -0.5322, 0.3306]],
)
