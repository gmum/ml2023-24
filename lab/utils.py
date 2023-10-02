from typing import Callable, List, Any, Type, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from torch import nn

try:
    import torchvision
    from torchvision.transforms import Compose, Lambda, ToTensor
    import torch
except:
    pass

from collections import namedtuple
import matplotlib.animation as animation
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

import sys

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
elif sys.version_info[1] < 7:
    Dataset = namedtuple(
        "Dataset",
        ["data", "target", "target_names", "filename"],
    )
    Dataset.__new__.__defaults__ = (None,) * len(Dataset._fields)
else:
    Dataset = namedtuple(
        "Dataset", ["data", "target", "target_names", "filename"], defaults=(None, None, None, None)
    )


def get_fn_values(points: np.ndarray, fn: Callable, X_vals: np.ndarray) -> np.ndarray:
    return np.array([fn(points, v) for v in X_vals])


def plot_1d_set(
    dataset: np.ndarray, ax: plt.Axes, loss_fns: List[Callable], show_title: bool = False
) -> None:
    linspace = np.linspace(dataset.min(), dataset.max(), num=200)
    ax.set_xlabel("v")
    ax.set_ylabel("Loss val")
    ax.scatter(dataset, [0] * len(dataset))
    for idx, loss_fn in enumerate(loss_fns):
        y_vals = get_fn_values(dataset, loss_fn, linspace)
        if show_title:
            ax.set_title(loss_fn.__name__)
        ax.plot(linspace, y_vals, label=loss_fn.__name__)


def plot_2d_set(dataset: np.ndarray, ax: plt.Axes, loss_fn: Callable) -> None:
    dataset_mins = dataset.min(0)
    dataset_maxs = dataset.max(0)
    first_linspace = np.linspace(dataset_mins[0], dataset_maxs[0], num=40)
    second_linspace = np.linspace(dataset_mins[1], dataset_maxs[1], num=40)
    X, Y = np.meshgrid(first_linspace, second_linspace)
    Z = np.empty_like(X)

    for row_idx, first_coord in enumerate(first_linspace):
        for col_idx, second_coord in enumerate(second_linspace):
            Z[row_idx][col_idx] = loss_fn(dataset, np.array([first_coord, second_coord]))
    ax.plot_surface(X, Y, Z)

    ax.scatter(dataset[:, 0], dataset[:, 1], np.zeros((dataset.shape[0],)))


def contour_2d_set(
    dataset: np.ndarray, ax: plt.Axes, loss_fn: Callable, linspaces: np.ndarray | None = None
) -> None:
    dataset_mins = dataset.min(0)
    dataset_maxs = dataset.max(0)
    if linspaces is None:
        first_linspace = np.linspace(dataset_mins[0], dataset_maxs[0], num=25)
        second_linspace = np.linspace(dataset_mins[1], dataset_maxs[1], num=25)
    else:
        first_linspace, second_linspace = linspaces
    X, Y = np.meshgrid(first_linspace, second_linspace, indexing="xy")
    Z = np.empty_like(X)

    for row_idx, first_coord in enumerate(first_linspace):
        for col_idx, second_coord in enumerate(second_linspace):
            Z[col_idx][row_idx] = loss_fn(dataset, np.array([first_coord, second_coord]))

    ax.contour(X, Y, Z, levels=20)
    if linspaces is None:
        ax.scatter(dataset[:, 0], dataset[:, 1])
    else:
        ax.contourf(first_linspace, second_linspace, Z, levels=300, cmap=cm.PuBu_r)
    #    plt.colorbar()


def plot_2d_loss_fn(loss_fn: Callable, title: str, dataset: np.ndarray) -> None:
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle(title)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    plot_2d_set(dataset, ax, loss_fn)
    ax = fig.add_subplot(1, 2, 2)
    contour_2d_set(dataset, ax, loss_fn)
    plt.show(fig)
    plt.close(fig)


def plot_minimums(
    dataset: np.ndarray, loss_fns: List[Callable], loss_fns_mins: List[Any], title: str
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title)

    min_vals = []
    for loss_fn, loss_fn_min, ax in zip(loss_fns, loss_fns_mins, axes):
        min_val = loss_fn_min(dataset)
        min_vals += [min_val]
        ax.scatter(min_val, loss_fn(dataset, min_val), color="black")
        plot_1d_set(dataset, ax, [loss_fn], show_title=True)

    plt.show(fig)
    plt.close(fig)
    print("ME minimum: {:.2f} MSE minimum: {:.2f} Max Error minimum: {:.2f}".format(*min_vals))


def plot_gradient_steps_1d(
    ax: plt.Axes,
    dataset: np.ndarray,
    gradient_descent_fn: Callable,
    grad_fn: Callable,
    loss_fn: Callable,
    num_steps: int = 100,
    learning_rate: float = 1e-1,
) -> np.ndarray:
    final_v, final_grad, all_v = gradient_descent_fn(
        grad_fn, dataset, num_steps=num_steps, learning_rate=learning_rate
    )
    plot_1d_set(dataset, ax, [loss_fn])
    y_vals = get_fn_values(dataset, loss_fn, all_v)
    ax.scatter(all_v, y_vals, c=np.arange(len(all_v)), cmap=plt.cm.viridis)
    return final_v


def plot_gradient_steps_2d(
    ax: plt.Axes,
    dataset: np.ndarray,
    gradient_descent_fn: Callable,
    grad_fn: Callable,
    loss_fn: Callable,
    num_steps: int = 100,
    learning_rate: float = 1e-2,
    linspaces: np.ndarray | None = None,
) -> np.ndarray:
    final_v, final_grad, all_v = gradient_descent_fn(
        grad_fn, dataset, num_steps=num_steps, learning_rate=learning_rate
    )
    contour_2d_set(dataset, ax, loss_fn, linspaces)
    ax.scatter(all_v[:, 0], all_v[:, 1], c=np.arange(len(all_v)), cmap=plt.cm.viridis)

    print("Final grad value for {}: {}".format(loss_fn.__name__, final_grad))
    return final_v


def visualize_normal_dist(X: np.ndarray, loc: float, scale: float) -> None:
    peak = 1 / np.sqrt(2 * np.pi * (scale**2))
    plt.hist(X, bins=50, density=True)
    plt.plot([loc - scale, loc - scale], [0, peak], color="r", label="1 sigma")
    plt.plot([loc + scale, loc + scale], [0, peak], color="r")

    plt.plot([loc - 2 * scale, loc - 2 * scale], [0, peak], color="b", label="2 sigma")
    plt.plot([loc + 2 * scale, loc + 2 * scale], [0, peak], color="b")

    plt.plot([loc - 3 * scale, loc - 3 * scale], [0, peak], color="g", label="3 sigma")
    plt.plot([loc + 3 * scale, loc + 3 * scale], [0, peak], color="g")
    plt.legend()


def scatter_with_whiten(
    X: np.ndarray, whiten: Callable, name: str, standarize: bool = False
) -> None:
    plt.title(name)
    plt.scatter(X[:, 0], X[:, 1], label="Before whitening")
    white_X = whiten(X)
    plt.axis("equal")
    plt.scatter(white_X[:, 0], white_X[:, 1], label="After whitening")

    if standarize:
        X_standarized = (X - X.mean(axis=0)) / X.std(axis=0)
        plt.scatter(X_standarized[:, 0], X_standarized[:, 1], label="Standarized")

    plt.legend()
    plt.show()


def plot_clustering(X: np.ndarray, y: np.ndarray, k: int = 3) -> None:
    assert X.shape[0] == y.shape[0]

    f = plt.figure(figsize=(8, 8))
    ax = f.add_subplot(111)
    ax.axis("equal")

    for i in range(k):
        ax.scatter(X[y == i, 0], X[y == i, 1])


def animate_clustering(X: np.ndarray, ys: np.ndarray) -> animation.FuncAnimation:
    def update_colors(i: int, ys: np.ndarray, scat: Any) -> Tuple[Any]:
        scat.set_array(ys[i])
        return (scat,)

    n_frames = len(ys)

    colors = ys[0]

    fig = plt.figure(figsize=(8, 8))
    scat = plt.scatter(X[:, 0], X[:, 1], c=colors)

    ani = animation.FuncAnimation(fig, update_colors, frames=range(n_frames), fargs=(ys, scat))
    return ani


def plot_cluster_comparison(datasets: List[np.ndarray], results: List[np.ndarray]) -> None:
    assert len(results) == len(datasets), "`results` list length does not match the dataset length!"

    n_rows = len(results)
    n_cols = len(results[0])

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_rows, 4 * n_cols))

    for ax, col in zip(axes[0], ["K-Means", "DBSCAN", "Agglomerative", "GMM"]):
        ax.set_title(col, size=24)

    for row, X, y_row in zip(axes, datasets, results):
        for ax, y in zip(row, y_row):
            ax.scatter(X[:, 0], X[:, 1], c=y.astype(np.int64))


def get_clustering_data() -> List[np.ndarray]:
    def standarize(X: np.ndarray) -> np.ndarray:
        return StandardScaler().fit_transform(X)

    n_samples = 1500
    noisy_circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

    noisy_moons = make_moons(n_samples=n_samples, noise=0.05)
    # Anisotropicly distributed data
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)

    # blobs with varied variances
    varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    datasets = [noisy_circles[0], noisy_moons[0], X_aniso, varied[0]]

    datasets = [standarize(X) for X in datasets]

    return datasets


def get_toy_dataset() -> Dataset:
    first_example_cov = np.array([[1, 0.99], [0.99, 1]])
    second_example_cov = np.array([[1, -0.99], [-0.99, 1]])
    X1 = np.random.multivariate_normal([0, 0], first_example_cov, size=1000)
    X2 = np.random.multivariate_normal([8, 8], second_example_cov, size=1000)
    X = np.concatenate([X1, X2])
    Y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])

    toy_dataset = Dataset(X, Y, Y, "Toy dataset")
    return toy_dataset


def test_pca(
    name: str,
    pca_cls: Type,
    dataset: Dataset,
    n_components: int | None = None,
    var_to_explain: float | None = None,
) -> None:
    X = dataset.data
    y = dataset.target
    y_names = dataset.target_names

    pca = pca_cls(n_components=n_components, var_to_explain=var_to_explain)
    pca.fit(X)
    B = pca.transform(X)
    print(f"Dataset {name}, Data dimension after the projection: {B.shape[1]}")

    if B.shape[1] == 1:
        B = np.concatenate([B, np.zeros_like(B)], 1)

    scatter = plt.scatter(B[:, 0], B[:, 1], c=y)
    scatter_objects, _ = scatter.legend_elements()
    plt.title(name)
    plt.legend(scatter_objects, y_names, loc="lower left", title="Classes")
    plt.show()


def create_regression_dataset(
    func: Callable,
    sample_size: int = 10,
    embed_func: Callable | None = None,
    embed_kwargs: Dict[str, Any] | None = None,
) -> Dataset:
    dataset_X = np.random.uniform(-2.5, 2.5, size=sample_size).reshape(-1, 1)
    dataset_Y_clean = func(dataset_X)
    dataset_Y = dataset_Y_clean + np.random.normal(0, 0.2, size=dataset_Y_clean.shape)
    dataset_Y = dataset_Y.squeeze()
    if embed_func is not None and embed_kwargs is not None:
        dataset_X = embed_func(dataset_X, **embed_kwargs)
    return Dataset(dataset_X, dataset_Y)


def plot_regression_dataset(dataset: Dataset, name: str) -> None:
    plt.plot([dataset.data.min(), dataset.data.max()], [0, 0], "k--")
    plt.plot([0, 0], [dataset.target.min(), dataset.target.max()], "k--")
    plt.title(name)
    plt.scatter(dataset.data, dataset.target)
    plt.show()


def plot_regression_results(
    dataset: Dataset,
    regression_cls: Type,
    name: str,
    embed_func: Callable | None = None,
    regression_kwargs: Dict[str, Any] | None = None,
    **embed_kwargs: Any,
) -> None:
    if embed_func is None:
        embed_func = lambda x: x
    if regression_kwargs is None:
        regression_kwargs = dict()

    X = embed_func(dataset.data, **embed_kwargs)
    regression = regression_cls(**regression_kwargs)
    regression.fit(X, dataset.target)

    loss_val = regression.loss(X, dataset.target)
    linspace_X = np.linspace(-2.5, 2.5)
    embedded_linspace = embed_func(linspace_X.reshape(-1, 1), **embed_kwargs)
    predicted_Y = regression.predict(embedded_linspace)

    plt.title(name)
    plt.plot(linspace_X, np.zeros_like(linspace_X), "k--")
    print(f"Dataset {name}\nWartość funkcji kosztu: {regression.loss(X, dataset.target)}")

    plot_min = min(predicted_Y.min(), dataset.target.min())
    plot_max = max(predicted_Y.max(), dataset.target.max())
    plt.plot([0, 0], [plot_min, plot_max], "k--")
    plt.plot(linspace_X, predicted_Y, c="C0", label="Regression results", linewidth=3)
    plt.scatter(dataset.data, dataset.target, c="C1", label="Samples")
    plt.legend()
    plt.show()


def plot_torch_fn(
    complex_fn: Callable, a: torch.Tensor, x: torch.Tensor, result: torch.Tensor
) -> None:
    linspace = torch.linspace(-5, 5, steps=400)
    vals = complex_fn(a, linspace)
    plt.plot(linspace.numpy(), vals.detach().numpy())
    plt.scatter(x.detach().numpy(), complex_fn(a, x).detach().numpy(), label="Starting point")
    plt.scatter(result, complex_fn(a, result), label="End point")
    plt.legend()


def get_classification_dataset_1d() -> Dataset:
    torch.manual_seed(8)
    X = torch.cat(
        [
            torch.randn(10, 1) * 3 + 10,
            torch.randn(10, 1) * 3 + 1,
        ]
    )

    y = torch.cat([torch.zeros(10), torch.ones(10)])
    return Dataset(X, y)


def get_classification_dataset_2d() -> Dataset:
    torch.manual_seed(4)
    X = torch.cat(
        [
            torch.randn(50, 2) * 2 + torch.tensor([4.0, 2.0]),
            torch.randn(50, 2) * 0.5 + torch.tensor([2.0, -4.0]),
        ]
    )

    y = torch.cat([torch.zeros(50), torch.ones(50)])
    return Dataset(X, y)


def visualize_optimizer(
    optim: torch.optim.Optimizer, n_steps: int, title: str | None = None, **params: Dict[str, Any]
) -> None:
    def f(w: torch.Tensor) -> torch.Tensor:
        x = torch.tensor([0.2, 2], dtype=torch.float)
        return torch.sum(x * w**2)

    w = torch.tensor([-6, 2], dtype=torch.float, requires_grad=True)

    optimizer = optim([w], **params)

    history = [w.clone().detach().numpy()]

    for i in range(n_steps):
        optimizer.zero_grad()

        loss = f(w)
        loss.backward()
        optimizer.step()
        history.append(w.clone().detach().numpy())

    delta = 0.01
    x = np.arange(-7.0, 7.0, delta)
    y = np.arange(-4.0, 4.0, delta)
    X, Y = np.meshgrid(x, y)

    Z = 0.2 * X**2 + 2 * Y**2

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.contour(X, Y, Z, 20)

    h = np.array(history)

    ax.plot(h[:, 0], h[:, 1], "x-")

    if title is not None:
        ax.set_title(title)


class ModelTrainer:
    def __init__(
        self,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        batch_size: int = 128,
    ):
        self.batch_size = batch_size
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable = torch.nn.functional.cross_entropy,
        n_epochs: int = 100,
    ) -> Dict[str, List[float]]:
        self.logs: Dict[str, List[float]] = {
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }
        model = model.to(self.device)
        correct, numel = 0, 0
        for e in range(1, n_epochs + 1):
            model.train()
            for x, y in self.train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model(x)
                y_pred = torch.argmax(output, dim=1)
                correct += torch.sum(y_pred == y).item()
                numel += self.batch_size
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

            self.logs["train_loss"].append(loss.item())
            self.logs["train_accuracy"].append(correct / numel)
            correct, numel = 0, 0

            model.eval()
            with torch.no_grad():
                for x_test, y_test in self.test_loader:
                    x_test = x_test.to(self.device)
                    y_test = y_test.to(self.device)
                    output = model(x_test)
                    y_pred = torch.argmax(output, dim=1)
                    correct += torch.sum(y_pred == y_test).item()
                    numel += self.batch_size
                loss = loss_fn(output, y_test)

            self.logs["test_loss"].append(loss.item())
            self.logs["test_accuracy"].append(correct / numel)
            correct, numel = 0, 0

        return self.logs


def load_mnist(train: bool = True, shrinkage: float | None = None) -> torch.utils.data.Dataset:
    dataset = torchvision.datasets.MNIST(
        root=".", download=True, train=train, transform=Compose([ToTensor(), Lambda(torch.flatten)])
    )
    if shrinkage:
        dataset_size = len(dataset)
        perm = torch.randperm(dataset_size)
        idx = perm[: int(dataset_size * shrinkage)]
        return torch.utils.data.Subset(dataset, idx)
    return dataset


def show_results(
    orientation: str = "horizontal",
    accuracy_bottom: Any = None,
    loss_top: Any = None,
    **histories: Dict[str, Any],
) -> None:
    if orientation == "horizontal":
        f, ax = plt.subplots(1, 2, figsize=(16, 5))
    else:
        f, ax = plt.subplots(2, 1, figsize=(16, 16))
    for i, (name, h) in enumerate(histories.items()):
        if len(histories) == 1:
            ax[0].set_title(
                "Best test accuracy: {:.2f}% (train: {:.2f}%)".format(
                    max(h["test_accuracy"]) * 100, max(h["train_accuracy"]) * 100
                )
            )
        else:
            ax[0].set_title("Accuracy")
        ax[0].plot(h["train_accuracy"], color="C%s" % i, linestyle="--", label="%s train" % name)
        ax[0].plot(h["test_accuracy"], color="C%s" % i, label="%s test" % name)
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("accuracy")
        if accuracy_bottom:
            ax[0].set_ylim(bottom=accuracy_bottom)
        ax[0].legend()

        if len(histories) == 1:
            ax[1].set_title(
                "Minimal train loss: {:.4f} (test: {:.4f})".format(
                    min(h["train_loss"]), min(h["test_loss"])
                )
            )
        else:
            ax[1].set_title("Loss")
        ax[1].plot(h["train_loss"], color="C%s" % i, linestyle="--", label="%s train" % name)
        ax[1].plot(h["test_loss"], color="C%s" % i, label="%s test" % name)
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("loss")
        if loss_top:
            ax[1].set_ylim(top=loss_top)
        ax[1].legend()

    plt.show()
