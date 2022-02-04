import numpy as np
from scipy import optimize
from scipy import special


class FocalLoss:
    """Class for custom objective loss function implementation."""

    def __init__(self, gamma, alpha=None):
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        """Compute weight vector at."""
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(
            y == 1, self.alpha, 1 - self.alpha
        )  # where y=1 set alpha else 1-alpha

    def pt(self, y, p):
        """Compute predicted vector pt."""
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y == 1, p, 1 - p)  # where y=1 set alpha else 1-alpha

    def __call__(self, y_true, y_pred):
        """Compute loss function."""
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return -at * (1 - pt) ** g * np.log(pt)

    def grad(self, y_true, y_pred):
        """Compute gradient of loss function."""
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        """Compute hessian of loss function."""
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(
        self, y_true
    ):  # initialization of first fit p (b=0) which minimizes custom objective loss fct -> so not simply #min/(#min + #maj) # noqa
        """Initialize first fit F_{0}(X) which will be improved on."""
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(), bounds=(0, 1), method="bounded"
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, y_true, preds):
        """Define function which returns needed gradient and hessian."""
        y = y_true
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, y_true, preds):
        """Define function which returns needed name, loss function and boolean."""
        y = y_true
        p = special.expit(preds)
        is_higher_better = False
        return "focal_loss", self(y, p).mean(), is_higher_better

    def lgb_score(self, y_true, preds):
        """Define function which returns loss function."""
        y = y_true
        p = special.expit(preds)
        return self(y, p).mean()


class WeightedLoss:
    """Class for custom objective loss function implementation."""

    def __init__(self, weight_maj, weight_min):
        self.weight_maj = weight_maj
        self.weight_min = weight_min

    def wt(self, y):
        """Compute weight vector wt."""
        return np.where(
            y == 1, self.weight_min, self.weight_maj
        )  # where y=1 set w1 else w0

    def pt(self, y, p):
        """Compute predicted vector pt."""
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y == 1, p, 1 - p)  # where y=1 set p else 1-p

    def __call__(self, y_true, y_pred):
        """Compute loss function."""
        wt = self.wt(y_true)
        pt = self.pt(y_true, y_pred)
        return -wt * np.log(pt)

    def grad(self, y_true, y_pred):
        """Compute gradient of loss function."""
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        wt = self.wt(y_true)
        pt = self.pt(y_true, y_pred)
        return -wt * y * (1 - pt)

    def hess(self, y_true, y_pred):
        """Compute hessian of loss function."""
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        wt = self.wt(y_true)
        pt = self.pt(y_true, y_pred)

        return wt * (pt * (1 - pt)) * y ** 2

    def init_score(self, y_true):  # initialization of first fit (b=0)
        """Initialize first fit F_{0}(X) which will be improved on."""
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(), bounds=(0, 1), method="bounded"
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, y_true, preds):
        """Define function which returns needed gradient and hessian."""
        y = y_true
        p = special.expit(preds)
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, y_true, preds):
        """Define function which returns needed name, loss function and boolean."""
        y = y_true
        p = special.expit(preds)
        is_higher_better = False
        return "weighted_loss", self(y, p).mean(), is_higher_better

    def lgb_score(self, y_true, preds):
        """Define function which returns loss function."""
        y = y_true
        p = special.expit(preds)
        return self(y, p).mean()